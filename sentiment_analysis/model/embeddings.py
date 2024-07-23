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


def randomize_offsets(rngs, position_embeddings, context_size, max_offset):
    features = position_embeddings.shape[-1]

    offset = random.randint(rngs, (), 0, max_offset)
    return jax.lax.dynamic_slice(
        position_embeddings, (offset, 0), (context_size, features)
    )


class PositionalEmbeddings(nnx.Module):
    def __init__(
        self,
        context_size: int,
        embedding_features: int,
        max_offset: int,
        scale: float,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
    ):
        self.dtype = dtype
        self.max_offset = max_offset
        self.context_size = context_size
        self.embedding = nnx.Variable(
            get_positional_embeddings(
                context_size + max_offset, embedding_features, dtype=param_dtype
            )
            * jnp.array(scale, dtype=param_dtype)
        )

    def __call__(self, batch_size: int, deterministic: bool, rngs: nnx.Rngs):
        embeddings = self.embedding.value

        if self.max_offset == 0:
            out = embeddings

        elif deterministic:
            half_offset = self.max_offset // 2
            out = embeddings[half_offset: half_offset + self.context_size]

        elif batch_size > 0:
            rng_batch = random.split(rngs.position(), batch_size)
            out = jax.vmap(randomize_offsets, in_axes=(0, None, None, None))(
                rng_batch, embeddings, self.context_size, self.max_offset
            )
        else:
            out = randomize_offsets(
                rngs.position(), embeddings, self.context_size, self.max_offset
            )

        return jnp.asarray(out, dtype=self.dtype)
