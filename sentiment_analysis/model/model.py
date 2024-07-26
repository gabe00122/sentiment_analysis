import math
from typing import Callable

import jax.debug
from flax import nnx
from jax import numpy as jnp, Array

from sentiment_analysis.model.types import ModelSettings
from sentiment_analysis.model.embeddings import PositionalEmbeddings, Embedder
from sentiment_analysis.model.transformer import TransformerLayer


class Model(nnx.Module):
    def __init__(
        self,
        settings: ModelSettings,
        rngs: nnx.Rngs,
    ):
        self.settings = settings

        dtype = dtype_by_name(self.settings.dtype)
        param_dtype = dtype_by_name(self.settings.dtype)

        self.activation = activation_by_name(self.settings.activation)
        normalization = norm_by_name(self.settings.normalization)
        kernel_init = nnx.initializers.glorot_normal()

        vocab_size = settings.vocab.size + 6
        context_size = settings.context_size

        self.token_embedding = Embedder(vocab_size, settings.hidden_features, dtype, param_dtype, rngs)

        self.position_embedding = PositionalEmbeddings(
            context_size,
            settings.hidden_features,
            settings.max_position_offset,
            0.05,
            dtype,
            param_dtype
        )

        self.transformer_layers = []
        for i in range(settings.transformer_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    num_heads=settings.transformer_heads,
                    features=settings.hidden_features,
                    mlp_features=settings.mlp_feature,
                    kernel_init=kernel_init,
                    mlp_activation=self.activation,
                    normalization=normalization,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    dropout_rate=settings.dropout_rate,
                    decode=False,
                    rngs=rngs,
                )
            )

        #self.input_norm = normalization(settings.hidden_features, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        self.output_norm = normalization(settings.hidden_features, rngs=rngs, dtype=dtype, param_dtype=param_dtype)

    def __call__(self, inputs, deterministic: bool, rngs: nnx.Rngs):
        batch_size = inputs.shape[0] if len(inputs.shape) > 1 else 0

        token_embed = self.token_embedding.encode(inputs)
        position_embed = self.position_embedding(batch_size, deterministic, rngs)

        x = token_embed + position_embed
        #x = self.input_norm(x)

        mask = nnx.make_causal_mask(inputs)

        for transformer in self.transformer_layers:
            x = transformer(x, mask, deterministic, rngs)

        x = self.output_norm(x)
        x = self.token_embedding.decode(x)

        x = jnp.asarray(x, dtype=jnp.float32)

        return x


def make_mask(inputs):
    mask = inputs != -1
    mask = nnx.make_attention_mask(mask, mask, jnp.logical_and)
    return mask


def relu2(x):
    x = nnx.relu(x)
    return x * x


def activation_by_name(name: str) -> Callable[[Array], Array]:
    match name:
        case 'relu':
            return nnx.relu
        case 'relu2':
            return relu2
        case 'gelu':
            return nnx.gelu


def dtype_by_name(name: str):
    match name:
        case 'float32':
            return jnp.float32
        case 'float16':
            return jnp.float16
        case 'bfloat16':
            return jnp.bfloat16

def norm_by_name(name: str):
    match name:
        case 'rms':
            return nnx.RMSNorm
        case 'layer':
            return nnx.LayerNorm
