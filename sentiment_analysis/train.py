import os
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import optax
from flax import nnx
from jax import numpy as jnp, random

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.common.metrics import TensorboardWriter, Metrics
from sentiment_analysis.experiment import Experiment
from sentiment_analysis.model import Model
from sentiment_analysis.types import ExperimentSettings
from sentiment_analysis.util import count_params
from sentiment_analysis.common.dataset_iterator import read_training_data, TrainingData, load_training_data


def set_flags():
    os.environ['XLA_FLAGS'] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
    )


def create_optimizer(settings: ExperimentSettings, total_steps: int):
    learning_rate = optax.warmup_cosine_decay_schedule(
        0.0, settings.optimizer.learning_rate, settings.optimizer.warmup_steps, total_steps
    )

    if settings.optimizer.weight_decay > 0:
        return optax.adamw(
            learning_rate,
            b1=settings.optimizer.beta1,
            b2=settings.optimizer.beta2,
            eps=settings.optimizer.eps,
            weight_decay=settings.optimizer.weight_decay
        )
    else:
        return optax.adam(
            learning_rate,
            b1=settings.optimizer.beta1,
            b2=settings.optimizer.beta2,
            eps=settings.optimizer.eps,
        )


def train(experiment: Experiment):
    set_flags()

    settings = experiment.settings
    seed = random.PRNGKey(settings.seed)

    init_key, train_key, training_shuffle_key, validation_shuffle_key = random.split(seed, num=4)
    rngs = nnx.Rngs(train_key)

    training_data = load_training_data(settings.training_file, training_shuffle_key)
    validation_data = load_training_data(settings.validation_file, validation_shuffle_key)

    samples = training_data.tokens.shape[0]
    context_size = training_data.tokens.shape[-1]

    if context_size != settings.model.context_size:
        print(f"Model context size {settings.model.context_size} and data context size {context_size} don't match")
        return

    print(f"Total Samples: {samples}")

    steps = samples // settings.batch_size
    total_steps = steps * settings.epochs

    optimizer = nnx.Optimizer(
        Model(settings.model, nnx.Rngs(init_key)),
        create_optimizer(settings, total_steps)
    )
    print(f"Param Count: {count_params(optimizer.model)}")

    checkpoints = Checkpointer(experiment.checkpoint_path)
    writer = TensorboardWriter(Path("./tensorboard"))


    for epoch in range(settings.epochs):
        for step in range(steps):
            start_time = time.time()
            training_data, metrics = train_step(optimizer, rngs, settings.batch_size, training_data, False)

            validation_data, val_metrics = train_step(optimizer, rngs, settings.batch_size, validation_data, True)

            writer.write(metrics)

            loss = metrics["loss"].item()
            percent_correct = metrics["percent_correct"].item() * 100

            val_loss = val_metrics["loss"].item()
            val_correct = val_metrics["percent_correct"].item() * 100

            end_time = time.time()
            delta_time = end_time - start_time
            samples_per_second = settings.batch_size / delta_time

            print(f"epoch = {epoch}/{settings.epochs}, step = {step}/{steps}, loss = {loss}, correct = {percent_correct:.0f}%, perf = {samples_per_second:.2f}")
            print(f"val-loss = {val_loss}, val-correct = {val_correct:.0f}%")


        checkpoints.save(optimizer.model, epoch)

    writer.close()
    checkpoints.close()


def loss_fn(model, tokens, labels, rngs, training: bool):
    logit_pred = model(tokens, deterministic=not training, rngs=rngs)

    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logit_pred, labels))

    predicted_labels = jnp.argmax(logit_pred, axis=-1)
    percent_correct = jnp.mean(predicted_labels == labels, dtype=jnp.float32)
    metrics = {"percent_correct": percent_correct, "loss": loss}

    return loss, metrics


@partial(nnx.jit, static_argnums=(2, 4), donate_argnums=3)
def train_step(optimizer: nnx.Optimizer, rngs: nnx.Rngs, batch_size: int, training_data: TrainingData, training: bool) -> tuple[TrainingData, Metrics]:
    training_data, tokens, labels = read_training_data(training_data, rngs.shuffle(), batch_size)

    if training:
        grads, metrics = nnx.grad(loss_fn, has_aux=True, wrt=nnx.Param)(optimizer.model, tokens, labels, rngs, training)
        optimizer.update(grads)
    else:
        _, metrics = loss_fn(optimizer.model, tokens, labels, rngs, training)

    return training_data, metrics

