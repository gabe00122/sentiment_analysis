import jax
import flax.linen as nn


class TransformerLayer(nn.Module):
    kernel_init: nn.initializers.Initializer
    num_heads: int = 8
    token_features: int = 16

    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        res = x

        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.token_features,
            kernel_init=self.kernel_init,
        )(x, mask=mask)

        x += res
        res = x

        x = nn.LayerNorm()(x)
        x = nn.Dense(
            features=self.token_features,
            kernel_init=self.kernel_init,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.token_features,
            kernel_init=self.kernel_init,
        )(x)

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
            axis=(0, 1),
            kernel_init=kernel_init,
        )(x)

        # jax.debug.breakpoint()

        return x
