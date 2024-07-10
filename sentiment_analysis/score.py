from functools import partial
from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.common.metrics import create_metrics_buffer, append_buffer, MetricsBuffer
from sentiment_analysis.model import Model
from sentiment_analysis.train_cli import load_settings


def train():
    settings = load_settings("./experiment_settings/tiny.json")


    batch_size = settings.batch_size
    batch_per_call = settings.batch_per_call

    data = jnp.load("./data/test.npz")
    tokens = data['tokens']
    labels = data['labels']

    sample_count = tokens.shape[0]
    print(f"Total Samples: {sample_count}")

    samples_per_call = batch_size * batch_per_call
    gpu_calls_per_epoch = (sample_count // samples_per_call)

    sample_count = gpu_calls_per_epoch * samples_per_call
    print(f"Rounding To: {sample_count}")

    tokens = tokens[:sample_count]
    labels = labels[:sample_count]

    model = Model(settings.model, rngs=nnx.Rngs(0))

    checkpoints = Checkpointer("checkpoints")
    model = checkpoints.restore(model, 0)

    indices = jnp.arange(sample_count, dtype=jnp.int32)

    model_graph, model_state = nnx.split(model)

    rngs = nnx.Rngs(0)
    rngs_graph, rngs = nnx.split(rngs)

    static = StaticState(
        model=model_graph,
        rngs=rngs_graph,
        batch_size=batch_size,
    )

    dynamic = DynamicState(
        step=0,
        model=model_state,
        rngs=rngs,
        indices=indices,
        tokens=tokens,
        labels=labels
    )

    output = jnp.zeros(gpu_calls_per_epoch, jnp.float32)
    for call in range(gpu_calls_per_epoch):
        dynamic, metrics = multi_training_step(static, dynamic, batch_per_call)

        percent_correct = metrics.values["percent_correct"].mean().item() * 100
        output = output.at[call].set(percent_correct)

        print(f"step = {call}/{gpu_calls_per_epoch}, correct = {percent_correct:.0f}%")

    print(f"output = {output.mean().item():.2f}%")

    checkpoints.close()

def train_step(model: nnx.Module, indices, tokens, labels, batch_size: int, step: int, rngs: nnx.Rngs):
    indices = jax.lax.dynamic_slice(
        indices, (batch_size * step,), (batch_size,)
    )
    tokens = tokens[indices]
    labels = labels[indices]

    logit_pred = model(tokens, deterministic=False, rngs=rngs)

    logits_indices = jnp.argmax(logit_pred, axis=1)
    percent_correct = jnp.mean(
        logits_indices == labels, dtype=jnp.float32
    )
    metrics = {"percent_correct": percent_correct}


    return metrics


class StaticState(NamedTuple):
    model: nnx.GraphDef
    rngs: nnx.GraphDef
    batch_size: int

class DynamicState(NamedTuple):
    step: jax.typing.ArrayLike
    model: nnx.State
    rngs: nnx.State
    indices: jax.Array
    tokens: jax.Array
    labels: jax.Array


def training_step_wrapper(static: StaticState, dynamic: DynamicState) -> tuple[DynamicState, dict[str, jax.Array]]:
    model = nnx.merge(static.model, dynamic.model)
    rngs = nnx.merge(static.rngs, dynamic.rngs)

    metrics = train_step(model, dynamic.indices, dynamic.tokens, dynamic.labels, static.batch_size, dynamic.step, rngs)

    dynamic = dynamic._replace(
        model=nnx.state(model),
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


if __name__ == '__main__':
    train()
