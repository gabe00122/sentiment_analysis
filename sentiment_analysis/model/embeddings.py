import math

import jax
from flax import nnx
from jax import numpy as jnp, random
from jax.typing import DTypeLike


def get_positional_embeddings(
    seq_length, features, n=10000, dtype: DTypeLike = jnp.float32
):
    output = []
    for k in range(seq_length):
        token = []
        for i in range(features // 2):
            denominator = n ** (2 * i / features)
            token.append(math.sin(k / denominator))
            token.append(math.cos(k / denominator))
        output.append(token)
    return jnp.asarray(output, dtype=dtype)


def randomize_offsets(rngs, position_embeddings, seq_length, max_offset):
    features = position_embeddings.shape[-1]

    offset = random.randint(rngs, (), 0, max_offset)
    return jax.lax.dynamic_slice(
        position_embeddings, (offset, 0), (seq_length, features)
    )


class PositionalEmbeddings(nnx.Module):
    def __init__(
        self,
        seq_length: int,
        embedding_features: int,
        max_offset: int,
        scale: float,
        dtype: DTypeLike,
    ):
        self.max_offset = max_offset
        self.seq_length = seq_length
        self.embedding = nnx.Param(
            get_positional_embeddings(
                seq_length + max_offset, embedding_features, dtype=dtype
            )
            * jnp.array(scale, dtype=dtype)
        )

    def __call__(self, batch_size: int, deterministic: bool, rngs: nnx.Rngs):
        embeddings = self.embedding.value

        if self.max_offset == 0:
            return embeddings

        if deterministic:
            half_offset = self.max_offset // 2
            return embeddings[half_offset : half_offset + self.seq_length]

        if batch_size > 0:
            rng_batch = random.split(rngs.position(), batch_size)
            return jax.vmap(randomize_offsets, in_axes=(0, None, None, None))(
                rng_batch, embeddings, self.seq_length, self.max_offset
            )

        return randomize_offsets(
            rngs.position(), embeddings, self.seq_length, self.max_offset
        )
