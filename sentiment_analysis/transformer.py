from jax import numpy as jnp
import flax.linen as nn
from sentiment_analysis.attention import MultiHeadDotProductAttention


class TransformerLayer(nn.Module):
    kernel_init: nn.initializers.Initializer
    num_heads: int = 8
    token_features: int = 16
    training = True

    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        res = x

        #x = nn.LayerNorm()(x)
        x = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.token_features,
            kernel_init=self.kernel_init,
            attention_init=nn.initializers.glorot_normal(),
            dropout_rate=0.1,
            deterministic=not self.training,
        )(x, mask=mask)
        x = nn.Dropout(rate=0.1, deterministic=True)(x)

        x += res
        res = x

        #x = nn.LayerNorm()(x)
        x = nn.Dense(
            features=self.token_features * 4,
            kernel_init=self.kernel_init,
        )(x)
        x = nn.Dropout(rate=0.1, deterministic=not self.training)(x)

        x = nn.relu(x)
        x = nn.Dense(
            features=self.token_features,
            kernel_init=self.kernel_init,
        )(x)
        x = nn.Dropout(rate=0.1, deterministic=not self.training)(x)

        x += res

        return x


def get_init_scale(n):
    return (9 * n) ** -(1 / 4)


class Transformer(nn.Module):
    num_heads: int = 8
    token_features: int = 16
    vocab_size: int = 10
    num_layers: int = 6

    @nn.compact
    def __call__(self, inputs, mask=None):
        kernel_init = nn.initializers.variance_scaling(
            scale=get_init_scale(self.num_layers),
            mode="fan_avg",
            distribution="truncated_normal",
        )
        if mask is not None:
            mask = nn.make_attention_mask(mask, mask, jnp.logical_and)

        x = inputs
        for i in range(self.num_layers):
            x = TransformerLayer(
                num_heads=self.num_heads,
                token_features=self.token_features,
                kernel_init=kernel_init,
            )(x, mask)

        # x = nn.LayerNorm()(x)
        x = nn.DenseGeneral(
            features=5,
            axis=(-2, -1),
            kernel_init=kernel_init,
        )(x)

        # jax.debug.breakpoint()

        return x
