import jax
from jax import Array, random, numpy as jnp
from typing import NamedTuple


class TrainingData(NamedTuple):
    step: Array
    tokens: Array
    labels: Array
    indices: Array

def reset_batch(batch: TrainingData, rng_key):
    indices = random.permutation(rng_key, batch.indices)
    return batch._replace(step=jnp.uint32(0), indices=indices)

def read_training_data(batch: TrainingData, rng_key, batch_size: int) -> tuple[TrainingData, jax.Array, jax.Array]:
    steps = batch.indices.shape[0] // batch_size

    batch = jax.lax.cond(batch.step >= steps, lambda: reset_batch(batch, rng_key), lambda: batch)

    indices = jax.lax.dynamic_slice(batch.indices, (batch_size * batch.step,), (batch_size,))
    tokens = batch.tokens[indices]
    labels = batch.labels[indices]
    batch = batch._replace(step=batch.step + 1)

    return batch, tokens, labels


def create_training_data(tokens: jax.Array, labels: jax.Array, shuffle_key) -> TrainingData:
    indices = jnp.arange(tokens.shape[0], dtype=jnp.uint32)
    indices = random.permutation(shuffle_key, indices)

    return TrainingData(jnp.uint32(0), tokens, labels, indices)


def load_training_data(path: str, shuffle_key) -> TrainingData:
    data = jnp.load(path)
    tokens = data['tokens']
    labels = data['length']

    return create_training_data(tokens, labels, shuffle_key)
