import os
from functools import partial
from pathlib import Path
from typing import NamedTuple, Any

import jax
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
from jax import random, Array, numpy as jnp
from jax.typing import ArrayLike
from optax.losses import softmax_cross_entropy

from sentiment_analysis.metrics import (
    Metrics,
    PandasWriter,
    MetricsBuffer,
    create_metrics_buffer,
    append_buffer,
)
from sentiment_analysis.network import Network
from sentiment_analysis.positional_embeddings import get_positional_embeddings
from sentiment_analysis.transformer import Transformer


def main():
    total_steps = 25_000
    epochs = 10

    rng_key = random.PRNGKey(0)
    vocab_size = 2000
    embedding_features = 128 * 2

    batch_size = 128

    sequence_length = 128
    num_heads = 8
    num_layers = 12

    data_size = total_steps * sequence_length

    training_data = jnp.load("./data/training_data.npz")
    print(training_data)
    print(training_data["tokens"].shape)
    labels = training_data["stars"][:data_size]  # should be stars or labels
    tokens = training_data["tokens"][:data_size]
    del training_data

    transformer = Transformer(
        num_heads=num_heads,
        token_features=embedding_features,
        num_layers=num_layers,
    )

    network = Network(
        transformer=transformer,
        vocab_size=vocab_size,
        embedding_features=embedding_features,
        position_embeddings=get_positional_embeddings(
            sequence_length, embedding_features
        ),
    )

    param_key, rng_key = random.split(rng_key)
    dummy_tokens = jnp.zeros(sequence_length, jnp.int16)

    network_params = network.init(param_key, dummy_tokens)

    optimizer = optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            0.000005,0.00125, 2000, total_steps * epochs
        ),
        b1=0.9,
        b2=0.98,
        weight_decay=0.0001
    )
    opt_state = optimizer.init(network_params)

    indices = jnp.arange(data_size)

    static_state = StaticState(network, optimizer, batch_size)
    training_state = TrainingState(rng_key, network_params, opt_state, indices, 0, tokens, labels)

    writer = PandasWriter(Path("./metrics_large_high_lr.parquet"))

    for _ in range(epochs):
        rng_key, indices_key = random.split(training_state.rng_key)
        indices = random.permutation(indices_key, training_state.indices)

        training_state = training_state._replace(batch_index=0, indices=indices, rng_key=rng_key)

        for i in range(total_steps // 500):
            if i == 0:
                print("Compilation started")

            # training_state, metrics = training_step(static_state, training_state)
            training_state, metrics = train_loop(static_state, training_state)
            writer.write(metrics)

            if i == 0:
                print("Compilation finished")

            # if i % 100 == 99:
            print(f"{i}: {metrics.values["loss"][metrics.length - 1].item()}")
            print(f"{metrics.values["percent_correct"][metrics.length - 1].item()}")

    writer.flush()
    save_model("models", training_state.params)


class TrainingSample(NamedTuple):
    tokens: Array
    mask: Array
    labels: Array


def loss(network: Network, params, training_batch: TrainingSample, rng_key):
    #vec_network = jax.vmap(network.apply, in_axes=(None, 0, 0))

    logits = network.apply(params, training_batch.tokens, training_batch.mask, rngs=rng_key)
    hot_labels = nn.one_hot(training_batch.labels - 1, 5)
    mean_cross_entropy = jnp.mean(
        softmax_cross_entropy(logits, hot_labels)
    )

    logits_indices = jnp.argmax(logits, axis=1)
    percent_correct = jnp.mean(
        logits_indices == training_batch.labels - 1, dtype=jnp.float32
    )
    metrics = {"percent_correct": percent_correct}

    return mean_cross_entropy, metrics


class StaticState(NamedTuple):
    network: Network
    solver: Any
    batch_size: int


class TrainingState(NamedTuple):
    rng_key: Array
    params: Any
    opt_state: Any
    indices: Array
    batch_index: ArrayLike
    tokens: Array
    labels: Array


# @partial(jax.jit, static_argnums=0)
def training_step(
    static_state: StaticState, state: TrainingState
) -> tuple[TrainingState, Metrics]:
    rng_key = state.rng_key
    rng_key, dropout_key = random.split(rng_key)

    start_slice = state.batch_index * static_state.batch_size

    indices = jax.lax.dynamic_slice(state.indices, (start_slice,), (static_state.batch_size,))
    # jax.debug.breakpoint()

    tokens = state.tokens[indices]
    labels = state.labels[indices]
    mask = tokens != -1

    sample = TrainingSample(tokens=tokens, labels=labels, mask=mask)

    (loss_value, metrics), grad = jax.value_and_grad(loss, argnums=1, has_aux=True)(
        static_state.network, state.params, sample, rng_key
    )
    updates, opt_state = static_state.solver.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    # jax.debug.breakpoint()

    state = TrainingState(
        rng_key=rng_key,
        params=params,
        opt_state=opt_state,
        indices=state.indices,
        batch_index=state.batch_index + 1,
        tokens=state.tokens,
        labels=state.labels,
    )

    # for path, leaf in jax.tree_util.tree_leaves_with_path(grad):
    #     name = jax.tree_util.keystr(path)
    #     metrics |= grad_stats(name, leaf)

    metrics |= {
        "loss": loss_value,
    }

    return state, metrics


@partial(jax.jit, static_argnums=0, donate_argnums=1)
def train_loop(
    static_state: StaticState, state: TrainingState
) -> tuple[TrainingState, MetricsBuffer]:
    loop_count = 500
    state, metrics = training_step(static_state, state)
    metrics_buffer = create_metrics_buffer(metrics, loop_count)

    def loop_body(i, curry) -> tuple[TrainingState, MetricsBuffer]:
        state, metrics_buffer = curry

        state, metrics = training_step(static_state, state)
        metrics_buffer = append_buffer(metrics_buffer, metrics)

        return state, metrics_buffer

    return jax.lax.fori_loop(1, loop_count, loop_body, (state, metrics_buffer))


def grad_stats(name: str, leaf) -> Metrics:
    leaf = jnp.array(leaf)
    return {
        name: jnp.linalg.norm(leaf),
    }


def save_model(file_name, params):
    abs_parh = os.path.abspath(file_name)
    path = ocp.test_utils.erase_and_create_empty(abs_parh)
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(path / "1", args=ocp.args.StandardSave(params))
    ckptr.wait_until_finished()


if __name__ == "__main__":
    main()
