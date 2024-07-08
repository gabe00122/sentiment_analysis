from typing import Callable

from flax import nnx
from jax import Array
from jax.typing import DTypeLike


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        features: int,
        mlp_features: int,
        kernel_init,
        mlp_activation: Callable[[Array], Array],
        dtype: DTypeLike,
        dropout_rate: float,
        decode: bool,
        rngs: nnx.Rngs,
    ):
        self.pre_attention_norm = nnx.LayerNorm(features, param_dtype=dtype, rngs=rngs)
        self.pre_mlp_norm = nnx.LayerNorm(features, param_dtype=dtype, rngs=rngs)

        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(dropout_rate)

        self.attention = nnx.MultiHeadAttention(
            num_heads,
            features,
            decode=decode,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.mlp_in = nnx.Linear(
            features,
            mlp_features,
            kernel_init=kernel_init,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.mlp_activation = mlp_activation

        self.mlp_out = nnx.Linear(
            mlp_features,
            features,
            kernel_init=kernel_init,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, inputs, mask, deterministic: bool, rngs: nnx.Rngs):
        x = inputs
        res = x

        x = self.pre_attention_norm(x)
        x = self.attention(x, mask=mask, deterministic=deterministic, rngs=rngs)
        if self.dropout:
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        x += res
        res = x

        x = self.pre_mlp_norm(x)
        x = self.mlp_in(x)
        x = self.mlp_activation(x)
        if self.dropout:
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)
        x = self.mlp_out(x)
        if self.dropout:
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        x += res

        return x
