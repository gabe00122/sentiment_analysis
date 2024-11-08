import time
from functools import partial
from pathlib import Path

import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp, random

from sentiment_lm.common.checkpointer import Checkpointer
from sentiment_lm.common.dataset_iterator import (
    read_training_data,
    TrainingData,
    load_training_data,
)
from sentiment_lm.utils.logger import JaxLogger, Metrics
from sentiment_lm.experiment import Experiment
from sentiment_lm.optimizer import create_optimizer
from sentiment_lm.util import set_flags, count_params
from sentiment_lm.constants import EMPTY_TOKEN


def train(experiment: Experiment):
    set_flags()

    settings = experiment.settings
    seed = random.PRNGKey(settings.seed)

    init_key, train_key, training_shuffle_key, validation_shuffle_key = random.split(
        seed, num=4
    )
    init_rngs = nnx.Rngs(init_key)
    rngs = nnx.Rngs(train_key)

    training_data = load_training_data(
        settings.training_file, training_shuffle_key)
    validation_data = load_training_data(
        settings.validation_file, validation_shuffle_key
    )

    samples = training_data.tokens.shape[0]

    print(f"Total Samples: {samples}")

    optimizer = create_optimizer(settings, init_rngs, training_data)
    print(f"Param Count: {count_params(optimizer.model)}")

    checkpoints = Checkpointer(experiment.checkpoint_path)

    writer = JaxLogger(experiment.settings, experiment.unique_token)

    steps = samples // settings.batch_size
    total_steps = steps * settings.epochs
    checkpoint_rate = total_steps // 10

    validation_rate = total_steps // 200
    logging_rate = total_steps // 200

    training_metrics = create_metrics()
    validation_metrics = create_metrics()

    for global_step in range(total_steps):
        training_data = train_step_accumulate(
            optimizer, rngs, settings.batch_size, settings.accumulation_steps, training_data, True, training_metrics
        )

        if global_step % validation_rate == validation_rate - 1:
            validation_data = train_step_accumulate(
                optimizer, rngs, settings.batch_size, settings.accumulation_steps, validation_data, False, validation_metrics
            )
            writer.log(validation_metrics.compute(), global_step, "validation")
            validation_metrics.reset()

        if global_step % logging_rate == logging_rate - 1:
            writer.log(training_metrics.compute(), global_step, "training")
            training_metrics.reset()

        if global_step % checkpoint_rate == checkpoint_rate - 1:
            print("Saving checkpoint")
            checkpoints.save(optimizer.model, global_step)

    print("Saving final checkpoint")
    checkpoints.save(optimizer.model, total_steps - 1)
    checkpoints.close()
    writer.close()

    print("Running validation")
    validation_accuracy = validate(
        optimizer, rngs, settings.batch_size, validation_data
    )
    print(f"Validation accuracy {validation_accuracy:.1f}%")
    experiment.save_results({"validation_accuracy": validation_accuracy})

    print("Experiment complete 🎉")

from sentiment_lm.constants import STAR_TOKENS

def autoregressive_loss(model, tokens, lengths):
    segment_position = jnp.arange(tokens.shape[-1], dtype=model.dtype)
    logit_pred = model(tokens, segment_position)

    logit_pred = logit_pred[:, :-1, :]
    tokens = tokens[:, 1:]

    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logit_pred, tokens),
        where=tokens != EMPTY_TOKEN,
    )

    batch_index = jnp.arange(tokens.shape[0])

    first_star = STAR_TOKENS[0]
    last_star = STAR_TOKENS[-1]

    star_logits = logit_pred[batch_index, lengths - 2, first_star:last_star]
    star_labels = tokens[batch_index, lengths - 2] - first_star

    # limit it to valid star ratings
    predicted_stars = jnp.argmax(star_logits, axis=-1)
    percent_correct = jnp.mean(predicted_stars == star_labels, dtype=jnp.float32)
    metrics = {"percent_correct": percent_correct, "loss": loss}

    return loss, metrics


def train_step_accumulate(optimizer: nnx.Optimizer, rngs: nnx.Rngs, batch_size: int, accumulation_steps: int, training_data: TrainingData, training: bool, metrics: nnx.MultiMetric) -> TrainingData:
    # this could probably use jax.scan and be jitted
    for _ in range(accumulation_steps):
        training_data = train_step(optimizer, rngs, batch_size//accumulation_steps, training_data, training, metrics)
    
    return training_data


@partial(nnx.jit, static_argnums=(2, 4), donate_argnums=3)
def train_step(
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch_size: int,
    training_data: TrainingData,
    training: bool,
    metrics: nnx.MultiMetric,
) -> TrainingData:
    training_data, tokens, lengths = read_training_data(
        training_data, rngs.shuffle(), batch_size
    )

    if training:
        grads, m = nnx.grad(autoregressive_loss, has_aux=True, wrt=nnx.Param)(
            optimizer.model, tokens, lengths
        )
        optimizer.update(grads)
        metrics.update(**m)
    else:
        _, m = autoregressive_loss(optimizer.model, tokens, lengths)
        metrics.update(**m)

    return training_data


def validate(
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch_size: int,
    validation_data: TrainingData,
) -> float:
    steps = validation_data.tokens.shape[0] // batch_size

    metrics = create_metrics()

    for _ in range(steps):
        validation_data = train_step(
            optimizer, rngs, batch_size, validation_data, False, metrics
        )

    return metrics.compute()["accuracy"].item()


def create_metrics():
    return nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
        percent_correct=nnx.metrics.Average('percent_correct'),
    )
