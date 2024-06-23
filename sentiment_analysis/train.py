from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import optax
from flax import nnx
from jax import numpy as jnp, random

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.common.metrics import TensorboardWriter, create_metrics_buffer, append_buffer, MetricsBuffer
from sentiment_analysis.model.network import Network


def train():
    seed = 123
    batch_size = 64
    batch_per_call = 250
    epochs = 20

    data = jnp.load("./data/training.npz")
    tokens = data['tokens']
    labels = data['labels']

    sample_count = tokens.shape[0]
    print(f"Total Samples: {sample_count}")

    samples_per_call = batch_size * batch_per_call
    gpu_calls_per_epoch = (sample_count // samples_per_call)

    sample_count = gpu_calls_per_epoch * samples_per_call
    print(f"Rounding To: {sample_count}")

    tokens = tokens[:sample_count]
    labels = labels[:sample_count] - 1

    total_steps = (sample_count // batch_size) * epochs

    rngs = nnx.Rngs(seed)
    model = Network(
        vocab_size=16000,
        seq_length=115,
        output_tokens=8,
        embedding_features=512,
        transformer_layers=12,
        transformer_heads=8,
        mlp_features=(2048,),
        max_position_offset=10,
        activation=nnx.relu,
        output_classes=5,
        dropout_rate=0.1,
        layer_norm=True,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    optimizer = nnx.Optimizer(model, optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            0.0, 0.0001, 6000, total_steps
        ),
        weight_decay=0.0001,
    ))
    count_params(model)

    checkpoints = Checkpointer("checkpoints5")
    indices = jnp.arange(sample_count, dtype=jnp.int32)

    optimizer_graph, optimizer = nnx.split(optimizer)
    rngs_graph, rngs = nnx.split(rngs)

    static = StaticState(
        optimizer=optimizer_graph,
        rngs=rngs_graph,
        batch_size=batch_size,
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
    indices_rng = random.PRNGKey(42)

    for epoch in range(epochs):
        rng_key, indices_rng = random.split(indices_rng)
        dynamic = dynamic._replace(
            step=0,
            indices=random.permutation(rng_key, dynamic.indices)
        )

        for call in range(gpu_calls_per_epoch):
            dynamic, metrics = multi_training_step(static, dynamic, batch_per_call)
            writer.write(metrics)

            loss = metrics.values["loss"].mean().item()
            percent_correct = metrics.values["percent_correct"].mean().item() * 100

            print(f"epoch = {epoch}/{epochs}, step = {call}/{gpu_calls_per_epoch}, loss = {loss}, correct = {percent_correct:.0f}%")

        checkpoints.save(epoch, nnx.merge(optimizer_graph, dynamic.optimizer).model)

    writer.flush()
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

        logits_indices = jnp.argmax(logit_pred, axis=1)
        percent_correct = jnp.mean(
            logits_indices == labels, dtype=jnp.float32
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


def count_params(model):
    params = nnx.state(model, nnx.Param)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Param Count: {param_count}")


if __name__ == '__main__':
    train()
