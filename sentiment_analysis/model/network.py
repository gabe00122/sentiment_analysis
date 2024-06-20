from typing import Sequence

import jax.debug
from flax import nnx
from jax import numpy as jnp
from jax.typing import DTypeLike
import math

from sentiment_analysis.model.embeddings import PositionalEmbeddings
from sentiment_analysis.model.transformer import TransformerLayer


def get_fixup_scale(transformer_layers: int, fixup_constant: float = 0.67) -> float:
    return (fixup_constant * transformer_layers) ** -(1 / 4)


def get_embed_scale(embedding_features: int) -> float:
    return embedding_features ** -(1 / 2)


class Network(nnx.Module):
    def __init__(
        self,
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
        rngs: nnx.Rngs,
    ):
        self.dropout_rate = dropout_rate
        self.output_tokens = output_tokens

        if fixup_constant > 0.0:
            kernel_scale = get_fixup_scale(transformer_layers, fixup_constant)
            embedding_scale = get_embed_scale(embedding_features) * kernel_scale
        else:
            # default flax init
            kernel_scale = 1.0
        embedding_scale = 1.0 / embedding_features  # this is actually a smaller value for fan_in

        embedding_init = nnx.initializers.normal(embedding_scale)

        self.token_embedding = nnx.Embed(
            vocab_size,
            embedding_features,
            param_dtype=dtype,
            embedding_init=embedding_init,
            rngs=rngs,
        )

        self.position_embedding = PositionalEmbeddings(
            seq_length, embedding_features, max_position_offset, embedding_scale, dtype
        )
        self.output_embedding = nnx.Param(
            embedding_init(rngs.params(), (output_tokens, embedding_features), dtype)
        )

        if dropout_rate > 0.0:
            self.embedding_dropout = nnx.Dropout(dropout_rate)

        mlp_activation = nnx.relu

        self.transformer_layers = []
        for i in range(transformer_layers):
            kernel_init = nnx.initializers.variance_scaling(
                kernel_scale, "fan_avg", "truncated_normal"
            )

            self.transformer_layers.append(
                TransformerLayer(
                    num_heads=transformer_heads,
                    features=embedding_features,
                    hidden_features=mlp_features,
                    kernel_init=kernel_init,
                    mlp_activation=mlp_activation,
                    dtype=dtype,
                    dropout_rate=dropout_rate,
                    use_layer_norm=layer_norm,
                    rngs=rngs,
                )
            )

        if layer_norm:
            self.output_layer_norm = nnx.LayerNorm(embedding_features, rngs=rngs)

        self.output_layer = nnx.LinearGeneral(
            (output_tokens, embedding_features),
            output_tokens * embedding_features,
            axis=(-2, -1),
            kernel_init=nnx.initializers.variance_scaling(
                kernel_scale, "fan_avg", "truncated_normal"
            ),
            param_dtype=dtype,
            rngs=rngs,
        )

        self.final_layer = nnx.Linear(
            output_tokens * embedding_features,
            output_classes,
            kernel_init=nnx.initializers.variance_scaling(
                kernel_scale, "fan_avg", "truncated_normal"
            ),
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, inputs, deterministic: bool, rngs: nnx.Rngs):
        batch_size = inputs.shape[0] if len(inputs.shape) > 1 else 0

        x = self.token_embedding(inputs)
        x += self.position_embedding(batch_size, deterministic, rngs)

        output_embedding = self.output_embedding.value
        if batch_size > 0:
            output_embedding = jnp.tile(output_embedding, (batch_size, 1, 1))

        x = jnp.concatenate([output_embedding, x], axis=-2)

        if self.dropout_rate > 0.0:
            x = self.embedding_dropout(x, deterministic=deterministic, rngs=rngs)

        mask = make_mask(inputs, self.output_tokens, batch_size)
        for transformer in self.transformer_layers:
            x = transformer(x, mask, deterministic, rngs)

        output_tokens = x[..., 0 : self.output_tokens, :]

        if self.output_layer_norm:
            output_tokens = self.output_layer_norm(output_tokens)

        out = self.output_layer(output_tokens)
        out = nnx.relu(out)
        out = self.final_layer(out)

        # jax.debug.breakpoint()

        return out


def make_mask(inputs, output_tokens, batch_size):
    mask = inputs != -1
    if batch_size > 0:
        output_mask = jnp.ones((batch_size, output_tokens), dtype=jnp.bool)
    else:
        output_mask = jnp.ones(output_tokens, dtype=jnp.bool)

    mask = jnp.concatenate([output_mask, mask], axis=-1)
    mask = nnx.make_attention_mask(mask, mask, jnp.logical_and)
    return mask
