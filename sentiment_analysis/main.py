from pathlib import Path
from typing import NamedTuple, Any
from functools import partial
import os

import jax
from jax import random, Array, numpy as jnp
from jax.typing import ArrayLike
from flax import linen as nn
import optax
from optax.losses import softmax_cross_entropy
import numpy as np
import orbax.checkpoint as ocp

from sentiment_analysis.network import Network
from sentiment_analysis.positional_embeddings import get_positional_embeddings
from sentiment_analysis.transformer import Transformer
from sentiment_analysis.metrics import (
    Metrics,
    PandasWriter,
    MetricsBuffer,
    create_metrics_buffer,
    append_buffer,
)


def main():
    training_data = jnp.load("./data/training_data.npz")
    print(training_data)
    labels = training_data["starts"]  # should be stars or labels
    tokens = training_data["tokens"]

    rng_key = random.PRNGKey(42)
    vocab_size = 2000
    embedding_features = 32
    sequence_length = 128
    num_heads = 8
    batch_size = 64

    transformer = Transformer(
        num_heads=num_heads,
        token_features=embedding_features,
        num_layers=6,
    )

    network = Network(
        transformer=transformer,
        vocab_size=vocab_size,
        embedding_features=embedding_features,
        position_embeddings=get_positional_embeddings(
            sequence_length, embedding_features
        ),
    )

    param_key, dummy_key, rng_key = random.split(rng_key, 3)
    dummy_tokens = jnp.zeros(sequence_length, jnp.int16)

    network_params = network.init(param_key, dummy_tokens)

    optimizer = optax.adam(
        learning_rate=optax.warmup_cosine_decay_schedule(
            0.0025 / 2, 0.0025, 1_000, 9_000
        )
    )
    opt_state = optimizer.init(network_params)

    rng_key, indices_key = random.split(rng_key)
    indices = jnp.arange(tokens.size)
    indices = random.permutation(indices_key, indices)

    static_state = StaticState(network, optimizer, batch_size, tokens, labels)
    training_state = TrainingState(rng_key, network_params, opt_state, indices, 0)

    total_steps = 10_000
    writter = PandasWriter(Path("./metrics_glort.parquet"))

    for i in range(total_steps // 500):
        if i == 0:
            print("Compilation started")

        # training_state, metrics = training_step(static_state, training_state)
        training_state, metrics = train_loop(static_state, training_state)
        writter.write(metrics)

        # if i == 0:
        print("Compilation finished")

        # if i % 100 == 99:
        print(f"{i}: {metrics.values["loss"][metrics.length - 1].item()}")

    writter.flush()
    save_model("models", training_state.params)


class TrainingSample(NamedTuple):
    tokens: Array
    mask: Array
    labels: Array


def loss(network: Network, params, training_batch: TrainingSample):
    vec_network = jax.vmap(network.apply, in_axes=(None, 0, 0))

    logits = vec_network(params, training_batch.tokens, training_batch.mask)
    mean_cross_entropy = jnp.mean(
        softmax_cross_entropy(logits, nn.one_hot(training_batch.labels, 5))
    )

    logits_indices = jnp.argmax(logits, axis=1)
    percent_correct = jnp.mean(
        logits_indices == training_batch.labels, dtype=jnp.float32
    )
    metrics = {"percent_correct": percent_correct}

    return mean_cross_entropy, metrics


class StaticState(NamedTuple):
    network: Network
    solver: Any
    batch_size: int
    tokens: Array
    labels: Array


class TrainingState(NamedTuple):
    rng_key: Array
    params: Any
    opt_state: Any
    indices: Array
    batch_index: ArrayLike


# @partial(jax.jit, static_argnums=0)
def training_step(
    static_state: StaticState, state: TrainingState
) -> tuple[TrainingState, Metrics]:
    rng_key = state.rng_key
    keys = random.split(rng_key, static_state.batch_size + 1)
    rng_key = keys[0]
    sample_keys = keys[1:]

    start_slice = state.batch_index * static_state.batch_size
    end_slice = start_slice + static_state.batch_size

    indices = state.indices[start_slice:end_slice]
    tokens = static_state.tokens[indices]
    labels = static_state.labels[indices]
    mask = tokens != -1

    sample = TrainingSample(tokens=tokens, labels=labels, mask=mask)

    (loss_value, metrics), grad = jax.value_and_grad(loss, argnums=1, has_aux=True)(
        static_state.network, state.params, sample
    )
    updates, opt_state = static_state.solver.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    state = TrainingState(
        rng_key=rng_key,
        params=params,
        opt_state=opt_state,
        indices=state.indices,
        batch_index=state.batch_index + 1,
    )

    # for path, leaf in jax.tree_util.tree_leaves_with_path(grad):
    #     name = jax.tree_util.keystr(path)
    #     metrics |= grad_stats(name, leaf)

    metrics |= {
        "loss": loss_value,
    }

    return state, metrics


@partial(jax.jit, static_argnums=0)
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
