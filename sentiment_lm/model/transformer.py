import typing

import jax
from jax import numpy as jnp
from jax.typing import DTypeLike
from flax import nnx

from sentiment_lm.model import positional_embeddings
from sentiment_lm.model.embeddings import Embedder
from sentiment_lm.model.feed_forward import GLUBlock, FFBlock


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        *,
        attention_softcap: float | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.attention_softcap = attention_softcap
        self.dtype = dtype
        self.param_dtype = param_dtype

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.d_model // self.num_heads

        self.in_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self.num_heads, self.head_dim * 3),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            use_bias=False,
            rngs=rngs,
        )

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
        in_proj = self.in_proj(inputs)

        query, key, value = jnp.split(in_proj, 3, -1)

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

        if self.attention_softcap is not None:
            attn_weights = softcap(attn_weights, self.attention_softcap)

        attn_weights = jax.nn.softmax(attn_weights).astype(self.dtype)

        x = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        out = self.out(x)

        return out


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
        attention_softcap: float | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        self.attention_norm = nnx.RMSNorm(
            d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.attention = AttentionBlock(
            num_heads,
            d_model,
            attention_softcap=attention_softcap,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.ffn_norm = nnx.RMSNorm(
            d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        ff_block = GLUBlock if glu else FFBlock
        self.ffn = ff_block(
            d_model,
            ffn_size,
            get_activation(activation_name),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

    def __call__(self, x, segment_positions, mask):
        x += self.attention(self.attention_norm(x), segment_positions, mask=mask)
        x += self.ffn(self.ffn_norm(x))
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
        attention_softcap: float,
        output_softcap: float,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.output_softcap = output_softcap
        self.dtype = dtype

        kernel_init = nnx.initializers.normal()

        self.embedder = Embedder(
            vocab_size, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    num_heads,
                    d_model,
                    ffn_size,
                    attention_softcap=attention_softcap,
                    activation_name=activation_name,
                    glu=glu,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=kernel_init,
                    rngs=rngs,
                )
            )

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

        if self.output_softcap is not None:
            x = softcap(x, self.output_softcap)

        return x


def softcap(x, cap):
    return jnp.tanh(x / cap) * cap
