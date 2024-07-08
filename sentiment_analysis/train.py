from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import optax
from flax import nnx
from jax import numpy as jnp, random

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.common.metrics import TensorboardWriter, create_metrics_buffer, append_buffer, MetricsBuffer
from sentiment_analysis.model import Model
from sentiment_analysis.types import ExperimentSettings
from sentiment_analysis.util import get_calls_per_epoch, get_total_steps, count_params


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


def train(settings: ExperimentSettings):
    seed = random.PRNGKey(settings.seed)
    init_key, train_key, shuffle_key = random.split(seed, num=3)

    tokens, labels = load_data(settings.training_file)

    if tokens.shape[-1] != settings.model.context_size:
        print(f"Model context size {settings.model.context_size} and data context size {tokens.shape[-1]} don't match")
        return

    samples = tokens.shape[0]
    print(f"Total Samples: {samples}")

    calls_per_epoch = get_calls_per_epoch(samples, settings)
    total_steps = get_total_steps(samples, settings)

    optimizer = nnx.Optimizer(
        Model(settings.model, nnx.Rngs(init_key)),
        create_optimizer(settings, total_steps)
    )
    print(f"Param Count: {count_params(optimizer.model)}")

    checkpoints = Checkpointer("checkpoints")
    indices = jnp.arange(samples, dtype=jnp.int32)

    optimizer_graph, optimizer = nnx.split(optimizer)
    rngs_graph, rngs = nnx.split(nnx.Rngs(train_key))

    static = StaticState(
        optimizer=optimizer_graph,
        rngs=rngs_graph,
        batch_size=settings.batch_size,
    )

    dynamic = DynamicState(
        step=0,
        optimizer=optimizer,
        rngs=rngs,
        indices=indices,
        tokens=tokens,
        labels=labels
    )

    writer = TensorboardWriter(Path("./tensorboard"))

    for epoch in range(settings.epochs):
        shuffle_key, consume_rng = random.split(shuffle_key)
        dynamic = dynamic._replace(
            step=0,
            indices=random.permutation(consume_rng, dynamic.indices)
        )

        for call in range(calls_per_epoch):
            dynamic, metrics = multi_training_step(static, dynamic, settings.batch_per_call)
            writer.write(metrics)

            loss = metrics.values["loss"].mean().item()
            percent_correct = metrics.values["percent_correct"].mean().item() * 100

            print(f"epoch = {epoch}/{settings.epochs}, step = {call}/{calls_per_epoch}, loss = {loss}, correct = {percent_correct:.0f}%")

        checkpoints.save(epoch, nnx.merge(optimizer_graph, dynamic.optimizer).model)

    writer.close()
    checkpoints.close()


def train_step(model: nnx.Module, optimizer: nnx.Optimizer, indices, tokens, labels, batch_size: int, step: int, rngs: nnx.Rngs):
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


class StaticState(NamedTuple):
    optimizer: nnx.GraphDef
    rngs: nnx.GraphDef
    batch_size: int


class DynamicState(NamedTuple):
    step: jax.typing.ArrayLike
    optimizer: nnx.State
    rngs: nnx.State
    indices: jax.Array
    tokens: jax.Array
    labels: jax.Array


def training_step_wrapper(static: StaticState, dynamic: DynamicState) -> tuple[DynamicState, dict[str, jax.Array]]:
    optimizer = nnx.merge(static.optimizer, dynamic.optimizer)
    rngs = nnx.merge(static.rngs, dynamic.rngs)

    metrics = train_step(optimizer.model, optimizer, dynamic.indices, dynamic.tokens, dynamic.labels, static.batch_size, dynamic.step, rngs)

    dynamic = dynamic._replace(
        optimizer=nnx.state(optimizer),
        rngs=nnx.state(rngs),
        step=dynamic.step + 1
    )

    return dynamic, metrics


@partial(jax.jit, static_argnums=(0, 2), donate_argnums=1)
def multi_training_step(static: StaticState, dynamic: DynamicState, batches_per_call: int) -> tuple[DynamicState, MetricsBuffer]:
    dynamic, metrics = training_step_wrapper(static, dynamic)
    metrics_buffer = create_metrics_buffer(metrics, batches_per_call)

    if batches_per_call <= 1:
        return dynamic, metrics_buffer

    def loop_body(i, curry) -> tuple[DynamicState, MetricsBuffer]:
        dynamic, metrics_buffer = curry

        dynamic, metrics = training_step_wrapper(static, dynamic)
        metrics_buffer = append_buffer(metrics_buffer, metrics)

        return dynamic, metrics_buffer

    return jax.lax.fori_loop(1, batches_per_call, loop_body, (dynamic, metrics_buffer))
