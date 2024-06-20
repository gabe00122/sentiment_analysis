from typing import Sequence, Callable

from flax import nnx
from jax import Array
from jax.typing import DTypeLike


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        features: int,
        hidden_features: Sequence[int],
        kernel_init,
        mlp_activation: Callable[[Array], Array],
        dtype: DTypeLike,
        dropout_rate: float,
        use_layer_norm: bool,
        rngs: nnx.Rngs,
    ):
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate

        if use_layer_norm:
            self.pre_attention_layer_norm = nnx.LayerNorm(
                features, param_dtype=dtype, rngs=rngs
            )
            self.pre_mlp_layer_norm = nnx.LayerNorm(
                features, param_dtype=dtype, rngs=rngs
            )

        if dropout_rate > 0.0:
            self.post_attention_dropout = nnx.Dropout(dropout_rate)
            self.post_mlp_dropout = nnx.Dropout(dropout_rate)

        self.attention = nnx.MultiHeadAttention(
            num_heads,
            features,
            decode=False,
            dropout_rate=dropout_rate,
            kernel_init=kernel_init,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.mlp_activation = mlp_activation
        self.mlp_layers = []

        in_features = features
        for hidden_feature in hidden_features:
            self.mlp_layers.append(
                nnx.Linear(
                    in_features,
                    hidden_feature,
                    kernel_init=kernel_init,
                    param_dtype=dtype,
                    rngs=rngs,
                )
            )
            in_features = hidden_feature

        self.mlp_layers.append(
            nnx.Linear(
                in_features,
                features,
                kernel_init=kernel_init,
                param_dtype=dtype,
                rngs=rngs,
            )
        )

    def __call__(self, inputs, mask, deterministic: bool, rngs: nnx.Rngs):
        x = inputs
        res = x

        if self.use_layer_norm:
            x = self.pre_attention_layer_norm(x)
        x = self.attention(x, mask=mask, deterministic=deterministic, rngs=rngs) # please remove the sow!
        if self.dropout_rate > 0.0:
            x = self.post_attention_dropout(x, deterministic=deterministic, rngs=rngs)

        x += res
        res = x

        if self.use_layer_norm:
            x = self.pre_mlp_layer_norm(x)

        for i, hidden_layer in enumerate(self.mlp_layers):
            x = hidden_layer(x)
            if i < len(self.mlp_layers) - 1:
                x = self.mlp_activation(x)

        if self.dropout_rate > 0.0:
            x = self.post_mlp_dropout(x, deterministic=deterministic, rngs=rngs)

        x += res

        return x
