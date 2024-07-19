from typing import Callable

from flax import nnx
from jax import Array
from jax.typing import DTypeLike
from sentiment_analysis.model.feed_forward import GLUFeedForwardBlock


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        features: int,
        mlp_features: int,
        kernel_init,
        mlp_activation: Callable[[Array], Array],
        normalization,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        dropout_rate: float,
        decode: bool,
        rngs: nnx.Rngs,
    ):
        self.norm = normalization(features, dtype=dtype, rngs=rngs)

        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(dropout_rate)

        self.attention = nnx.MultiHeadAttention(
            num_heads,
            features,
            decode=decode,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            normalize_qk=True,
            rngs=rngs,
        )

        self.ff_block = GLUFeedForwardBlock(features, mlp_features, mlp_activation, kernel_init, dtype, param_dtype, dropout_rate, rngs)

    def __call__(self, inputs, mask, deterministic: bool, rngs: nnx.Rngs):
        x = inputs

        norm_x = self.norm(x)

        a_x = self.attention(norm_x, mask=mask, deterministic=deterministic, rngs=rngs, sow_weights=True)
        if hasattr(self, 'dropout'):
            a_x = self.dropout(a_x, deterministic=deterministic, rngs=rngs)

        ff_x = self.ff_block(norm_x, deterministic, rngs)

        x += a_x + ff_x

        return x
