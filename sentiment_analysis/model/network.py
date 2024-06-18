from typing import Sequence
from functools import partial

from flax import nnx
from jax.typing import DTypeLike
from jax import numpy as jnp

from sentiment_analysis.model.transformer import TransformerLayer
from sentiment_analysis.model.embeddings import PositionalEmbeddings


def get_fixup_scale(transformer_layers: int, fixup_constant: float = 0.67) -> float:
    return (fixup_constant * transformer_layers) ** -(1 / 4)


def get_embed_scale(embedding_features: int) -> float:
    return embedding_features ** -(1 / 2)


class Network(nnx.Module):
    def __init__(self,
                 vocab_size: int,
                 seq_length: int,
                 output_tokens: int,
                 embedding_features: int,
                 transformer_layers: int,
                 transformer_heads: int,
                 mlp_features: Sequence[int],
                 max_position_offset: int,
                 output_classes: int,
                 dropout_rate: float,
                 layer_norm: bool,
                 fixup_constant: float,
                 dtype: DTypeLike,
                 rngs: nnx.Rngs):
        self.dropout_rate = dropout_rate
        self.output_tokens = output_tokens

        if fixup_constant > 0.0:
            kernel_scale = get_fixup_scale(transformer_layers, fixup_constant)
            embedding_scale = get_embed_scale(embedding_features) * kernel_scale
        else:
            # default flax init
            kernel_scale = 1.0
            embedding_scale = 1.0  # this is actually a smaller value for fan_in

        embedding_init = nnx.initializers.normal(embedding_scale)

        self.token_embedding = nnx.Embed(
            vocab_size,
            embedding_features,
            param_dtype=dtype,
            embedding_init=embedding_init,
            rngs=rngs,
        )

        self.position_embedding = PositionalEmbeddings(seq_length, embedding_features, max_position_offset, embedding_scale, dtype)
        self.output_embedding = nnx.Param(embedding_init(rngs.params(), (output_tokens, embedding_features), dtype))

        if dropout_rate > 0.0:
            self.embedding_dropout = nnx.Dropout(dropout_rate)

        kernel_init = nnx.initializers.variance_scaling(kernel_scale, "fan_avg", "truncated_normal")
        mlp_activation = nnx.relu

        self.transformer_layers = []
        for _ in range(transformer_layers):
            self.transformer_layers.append(TransformerLayer(
                num_heads=transformer_heads,
                features=embedding_features,
                hidden_features=mlp_features,
                kernel_init=kernel_init,
                mlp_activation=mlp_activation,
                dtype=dtype,
                dropout_rate=dropout_rate,
                use_layer_norm=layer_norm,
                rngs=rngs))

        self.output_layer = nnx.LinearGeneral((output_tokens, embedding_features), output_classes, axis=(-2, -1), kernel_init=kernel_init, param_dtype=dtype, rngs=rngs)

    def __call__(self, inputs, mask, deterministic: bool, rngs: nnx.Rngs):
        batch_size = inputs.shape[0] if len(inputs.shape) > 1 else 0

        x = self.token_embedding(inputs)
        x += self.position_embedding(batch_size, deterministic, rngs)

        output_embedding = self.output_embedding.value
        if batch_size > 0:
            output_embedding = jnp.tile(output_embedding, (batch_size, 1, 1))

        x = jnp.concatenate([output_embedding, x], axis=-2)

        if self.dropout_rate > 0.0:
            x = self.embedding_dropout(x, deterministic=deterministic, rngs=rngs)

        for transformer in self.transformer_layers:
            x = transformer(x, mask, deterministic, rngs)

        output_token = x[..., 0:self.output_tokens, :]
        out = self.output_layer(output_token)

        return out


def main():
    rngs = nnx.Rngs(0)
    network = Network(True, 10, 12, 4, 2, (0,), 6, 5, 0.1, False, True, dtype=jnp.float16, rngs=rngs)
    print(network)

    output = network(jnp.zeros((4, 12), dtype=jnp.int16), jnp.zeros((12,), dtype=jnp.bool), rngs)
    print(output)


if __name__ == '__main__':
    main()
