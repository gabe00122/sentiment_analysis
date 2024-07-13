import os
from functools import partial
from pathlib import Path

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


def set_flags():
    os.environ['XLA_FLAGS'] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=true "
    )

def load_data(path):
    data = jnp.load(path)
    tokens = data['tokens']
    labels = data['labels']
    return tokens, labels


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
    init_key, train_key, shuffle_key = random.split(seed, num=3)

    tokens, labels = load_data(settings.training_file)

    if tokens.shape[-1] != settings.model.context_size:
        print(f"Model context size {settings.model.context_size} and data context size {tokens.shape[-1]} don't match")
        return

    samples = tokens.shape[0]
    print(f"Total Samples: {samples}")

    steps = samples // settings.batch_size
    total_steps = steps * settings.epochs

    optimizer = nnx.Optimizer(
        Model(settings.model, nnx.Rngs(init_key)),
        create_optimizer(settings, total_steps)
    )
    print(f"Param Count: {count_params(optimizer.model)}")

    checkpoints = Checkpointer(experiment.checkpoint_path)
    indices = jnp.arange(samples, dtype=jnp.int32)
    rngs = nnx.Rngs(train_key)

    writer = TensorboardWriter(Path("./tensorboard"))


    for epoch in range(settings.epochs):
        shuffle_key, consume_rng = random.split(shuffle_key)
        indices = random.permutation(consume_rng, indices)

        for step in range(steps):
            metrics = train_step(optimizer, indices, tokens, labels, settings.batch_size, step, rngs)
            writer.write(metrics)

            loss = metrics["loss"].mean().item()
            percent_correct = metrics["percent_correct"].mean().item() * 100

            print(f"epoch = {epoch}/{settings.epochs}, step = {step}/{steps}, loss = {loss}, correct = {percent_correct:.0f}%")

        checkpoints.save(optimizer.model, epoch)

    writer.close()
    checkpoints.close()


@partial(nnx.jit, donate_argnums=(0, 6), static_argnums=4)
def train_step(optimizer: nnx.Optimizer, indices: jax.Array, tokens: jax.Array, labels: jax.Array, batch_size: int, step: int, rngs: nnx.Rngs) -> Metrics:
    model = optimizer.model

    indices = jax.lax.dynamic_slice(
        indices, (batch_size * step,), (batch_size,)
    )
    tokens = tokens[indices]
    labels = labels[indices]

    def loss_fn(model):
        logit_pred = model(tokens, deterministic=False, rngs=rngs)

        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logit_pred, labels))

        predicted_labels = jnp.argmax(logit_pred, axis=-1)
        percent_correct = jnp.mean(
            predicted_labels == labels, dtype=jnp.float32
        )
        metrics = {"percent_correct": percent_correct, "loss": loss}

        return loss, metrics

    grads, metrics = nnx.grad(loss_fn, has_aux=True, wrt=nnx.Param)(model)
    optimizer.update(grads)

    return metrics
