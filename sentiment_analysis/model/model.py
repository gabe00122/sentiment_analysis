import math
from typing import Callable

from flax import nnx
from jax import numpy as jnp, Array

from sentiment_analysis.model.types import ModelSettings
from sentiment_analysis.model.embeddings import PositionalEmbeddings
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

        embedding_scale = math.sqrt(1.0 / settings.hidden_features)
        embedding_init = nnx.initializers.normal(embedding_scale)

        vocab_size = settings.vocab.size
        context_size = settings.context_size
        if settings.output.type == 'classification_tokens':
            vocab_size += settings.output.output_tokens
            context_size += settings.output.output_tokens

        self.token_embedding = nnx.Embed(
            vocab_size,
            settings.hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=embedding_init,
            rngs=rngs,
        )

        self.position_embedding = PositionalEmbeddings(
            context_size,
            settings.hidden_features,
            settings.max_position_offset,
            embedding_scale,
            dtype,
            param_dtype
        )

        if settings.dropout_rate > 0.0:
            self.dropout = nnx.Dropout(settings.dropout_rate)

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

        self.output_norm = normalization(settings.hidden_features, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        if settings.output.type == 'classification_tokens':
            self.output_layer = nnx.LinearGeneral(
                (settings.output.output_tokens, settings.hidden_features),
                settings.output.output_classes if settings.output.format == 'softmax' else 1,
                axis=(-2, -1),
                kernel_init=kernel_init,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        elif settings.output.type == 'mean' or settings.output.type == 'max':
            self.output_layer = nnx.Linear(
                settings.hidden_features,
                settings.output.output_classes if settings.output.format == 'softmax' else 1,
                kernel_init=kernel_init,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def __call__(self, inputs, deterministic: bool, rngs: nnx.Rngs):
        batch_size = inputs.shape[0] if len(inputs.shape) > 1 else 0

        if self.settings.output.type == 'classification_tokens':
            inputs += self.settings.output.output_tokens
            output_tokens = jnp.arange(self.settings.output.output_tokens, dtype=inputs.dtype)

            if batch_size > 0:
                output_tokens = jnp.tile(output_tokens, (batch_size, 1))
            inputs = jnp.concatenate([output_tokens, inputs], axis=-1)

        x = self.token_embedding(inputs)
        x += self.position_embedding(batch_size, deterministic, rngs)

        if hasattr(self, 'dropout'):
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        mask = make_mask(inputs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask, deterministic, rngs)

        # for classification tokens
        if self.settings.output.type == 'classification_tokens':
            x = x[..., 0: self.settings.output.output_tokens, :]
        elif self.settings.output.type == 'mean':
            x = jnp.mean(x, axis=-2, where=(inputs != -1)[:, :, jnp.newaxis])
        elif self.settings.output.type == 'max':
            x = jnp.max(x, axis=-2, where=(inputs != -1)[:, :, jnp.newaxis], initial=-1000)

        x = self.output_norm(x)
        x = self.output_layer(x)

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
