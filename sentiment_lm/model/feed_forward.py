import functools

from flax import nnx
from jax import numpy as jnp
from jax.typing import DTypeLike


class FFBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        hidden_features: int,
        activation,
        *,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.hidden_features = hidden_features

        linear = functools.partial(
            nnx.Linear,
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.activation = activation
        self.up_proj = linear(d_model, hidden_features)
        self.down_proj = linear(d_model, hidden_features)

    def __call__(self, inputs):
        x = self.up_proj(inputs)
        x = self.activation(x)
        out = self.down_proj(x)
        return out


class GLUBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        hidden_features: int,
        activation,
        *,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.hidden_features = hidden_features

        linear = functools.partial(
            nnx.Linear,
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.activation = activation
        self.up_proj = linear(d_model, hidden_features * 2)
        self.down_proj = linear(hidden_features, d_model)

    def __call__(self, inputs):
        x, gate = jnp.split(self.up_proj(inputs), 2, axis=-1)
        x = self.activation(x) * gate
        out = self.down_proj(x)
        return out
