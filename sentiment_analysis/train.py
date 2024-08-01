import time
from functools import partial
from pathlib import Path

import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp, random

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.common.dataset_iterator import (
    read_training_data,
    TrainingData,
    load_training_data,
)
from sentiment_analysis.common.metrics import TensorboardWriter, Metrics
from sentiment_analysis.experiment import Experiment
from sentiment_analysis.optimizer import create_optimizer
from sentiment_analysis.util import set_flags, count_params


def train(experiment: Experiment):
    set_flags()

    settings = experiment.settings
    seed = random.PRNGKey(settings.seed)

    init_key, train_key, training_shuffle_key, validation_shuffle_key = random.split(
        seed, num=4
    )
    init_rngs = nnx.Rngs(init_key)
    rngs = nnx.Rngs(train_key)

    training_data = load_training_data(settings.training_file, training_shuffle_key)
    validation_data = load_training_data(
        settings.validation_file, validation_shuffle_key
    )

    samples = training_data.tokens.shape[0]
    context_size = training_data.tokens.shape[-1]

    if context_size != settings.model.context_size:
        print(
            f"Model context size {settings.model.context_size} and data context size {context_size} don't match"
        )
        return

    print(f"Total Samples: {samples}")

    optimizer = create_optimizer(settings, init_rngs, training_data)
    print(f"Param Count: {count_params(optimizer.model)}")

    checkpoints = Checkpointer(experiment.checkpoint_path)

    writer = TensorboardWriter(Path("./tensorboard"), experiment.run_name)

    steps = samples // settings.batch_size
    total_steps = steps * settings.epochs
    checkpoint_rate = 100_000
    validation_rate = settings.accumulation_steps

    start_time = time.time()
    for global_step in range(total_steps):
        training_data, metrics = train_step(
            optimizer, rngs, settings.batch_size, training_data, True
        )

        if global_step % validation_rate == validation_rate - 1:
            validation_data, val_metrics = train_step(
                optimizer, rngs, settings.batch_size, validation_data, False
            )
            writer.write(val_metrics, global_step, "validation")

            val_loss = val_metrics["loss"].item()
            val_correct = val_metrics["percent_correct"].item() * 100
            print(
                f"validation loss = {val_loss:.4f}, validation correct = {val_correct:.1f}%"
            )

            writer.write(metrics, global_step, "training")

            loss = metrics["loss"].item()
            percent_correct = metrics["percent_correct"].item() * 100

            end_time = time.time()
            delta_time = end_time - start_time
            samples_per_second = settings.model.context_size * (
                (settings.batch_size / delta_time) * validation_rate
            )

            epoch = global_step // steps
            step = global_step % steps

            print(
                f"epoch = {epoch}/{settings.epochs}, step = {step}/{steps}, loss = {loss:.4f}, correct = {percent_correct:.1f}%, perf = {samples_per_second:.2f}"
            )
            start_time = time.time()

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


def autoregressive_loss(model, tokens, lengths):
    segment_position = jnp.arange(tokens.shape[-1])
    logit_pred = model(tokens, segment_position)

    logit_pred = logit_pred[:, :-1, :]
    tokens = tokens[:, 1:]

    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logit_pred, tokens),
        where=tokens != -1,
    )

    batch_index = jnp.arange(tokens.shape[0])
    star_logits = logit_pred[batch_index, lengths - 2, 1:6]
    star_labels = tokens[batch_index, lengths - 2] - 1

    predicted_stars = jnp.argmax(star_logits, axis=-1)  # limit it to valid star ratings
    percent_correct = jnp.mean(predicted_stars == star_labels, dtype=jnp.float32)
    # jax.debug.breakpoint()
    metrics = {"percent_correct": percent_correct, "loss": loss}

    return loss, metrics


@partial(nnx.jit, static_argnums=(2, 4), donate_argnums=3)
def train_step(
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch_size: int,
    training_data: TrainingData,
    training: bool,
) -> tuple[TrainingData, Metrics]:
    training_data, tokens, lengths = read_training_data(
        training_data, rngs.shuffle(), batch_size
    )

    loss_fn = autoregressive_loss

    if training:
        grads, metrics = nnx.grad(loss_fn, has_aux=True, wrt=nnx.Param)(
            optimizer.model, tokens, lengths
        )
        optimizer.update(grads)
    else:
        _, metrics = loss_fn(optimizer.model, tokens, lengths)

    return training_data, metrics


def validate(
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    batch_size: int,
    validation_data: TrainingData,
) -> float:
    steps = validation_data.tokens.shape[0] // batch_size

    output = np.zeros((steps,), dtype=np.float32)
    for step in range(steps):
        validation_data, metrics = train_step(
            optimizer, rngs, batch_size, validation_data, False
        )
        percent_correct = metrics["percent_correct"].item() * 100
        output[step] = percent_correct
        print(f"percent_correct = {percent_correct:.1f}%")

    return output.mean().item()
