import jax
from flax import nnx
from jax import numpy as jnp, random
from jax.typing import DTypeLike


class Embedder(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_features: int,
        *,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs
    ):
        self.dtype = dtype
        self.embedding_features = embedding_features
        self.param_dtype = param_dtype

        key = rngs.param()
        self.embedding_table = nnx.Param(
            random.normal(key, (vocab_size, embedding_features)) * 0.01,
            dtype=param_dtype,
        )

    def encode(self, x: jax.Array):
        x = jnp.take(self.embedding_table.value, x, axis=0, fill_value=0)

        x = jnp.asarray(x, dtype=self.dtype)
        x *= jnp.sqrt(self.embedding_features).astype(self.dtype)
        return x

    def decode(self, x: jax.Array):
        return jnp.dot(x, self.embedding_table.value.T)
