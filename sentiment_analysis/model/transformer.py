import functools
import typing

import jax
from jax import numpy as jnp
from jax.typing import DTypeLike
from flax import nnx

from sentiment_analysis.model import positional_embeddings
from sentiment_analysis.model.embeddings import Embedder
from sentiment_analysis.model.feed_forward import GLUBlock, FFBlock


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.dtype = dtype
        self.param_dtype = param_dtype

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.d_model // self.num_heads

        linear_general = functools.partial(
            nnx.LinearGeneral,
            in_features=self.d_model,
            out_features=(self.num_heads, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            use_bias=False,
            rngs=rngs,
        )

        self.query = linear_general()
        self.key = linear_general()
        self.value = linear_general()
        self.out = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.d_model,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, inputs, segment_positions, *, mask):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        query = positional_embeddings.apply_rope(
            query, segment_positions, head_dim=self.head_dim
        )
        key = positional_embeddings.apply_rope(
            key, segment_positions, head_dim=self.head_dim
        )

        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(self.dtype)

        attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key)

        if mask is not None:
            big_neg = jnp.finfo(self.dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)

        # attn_weights = attn_weights.astype(jnp.float32)
        attn_weights = jax.nn.softmax(attn_weights).astype(self.dtype)

        x = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        out = self.out(x)

        return out


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        d_model,
        block,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.RMSNorm(
            d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.block = block

    def __call__(self, inputs, *args, **kwargs):
        return inputs + self.block(self.norm(inputs), *args, **kwargs)


ActivationName = typing.Literal["relu", "silu", "gelu", "mish"]


def get_activation(name: ActivationName):
    match name:
        case "relu":
            return jax.nn.relu
        case "silu":
            return jax.nn.silu
        case "gelu":
            return jax.nn.gelu
        case "mish":
            return jax.nn.mish


class TransformerLayer(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        ffn_size: int,
        *,
        activation_name: ActivationName = "gelu",
        glu: bool = True,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        self.attention = ResidualBlock(
            d_model,
            AttentionBlock(
                num_heads,
                d_model,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                rngs=rngs,
            ),
            rngs=rngs,
        )

        ff_block = GLUBlock if glu else FFBlock
        self.ffn = ResidualBlock(
            d_model,
            ff_block(
                d_model,
                ffn_size,
                get_activation(activation_name),
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                rngs=rngs,
            ),
            rngs=rngs,
        )

    def __call__(self, inputs, segment_positions, mask):
        x = self.attention(inputs, segment_positions, mask=mask)
        x = self.ffn(x)
        return x


class TransformerModel(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        ffn_size: int,
        *,
        activation_name: ActivationName = "gelu",
        glu: bool = True,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.dtype = dtype

        kernel_init = nnx.initializers.normal()

        self.embedder = Embedder(
            vocab_size, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TransformerLayer(
                num_heads,
                d_model,
                ffn_size,
                activation_name=activation_name,
                glu=glu,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                rngs=rngs,
            ))

        self.output_norm = nnx.RMSNorm(
            d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(self, inputs, segment_positions):
        x = self.embedder.encode(inputs)

        mask = nnx.make_causal_mask(inputs, dtype=jnp.bool)

        for layer in self.layers:
            x = layer(x, segment_positions, mask)

        x = self.output_norm(x)
        x = self.embedder.decode(x)

        x = jnp.asarray(x, dtype=jnp.float32)

        return x
