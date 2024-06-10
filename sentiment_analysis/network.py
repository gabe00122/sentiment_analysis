import jax
from flax import linen as nn
from jax import Array, numpy as jnp

from .transformer import Transformer


class Network(nn.Module):
    vocab_size: int
    embedding_features: int
    transformer: Transformer
    position_embeddings: Array

    @nn.compact
    def __call__(self, inputs, mask=None):
        token_embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_features,
            embedding_init=nn.initializers.variance_scaling(
                self.embedding_features ** -(1 / 2), "fan_in", "normal", out_axis=0
            ),
            # param_dtype=jnp.float16,
            # dtype=jnp.float16
        )(inputs)
        embeddings = token_embeddings + self.position_embeddings

        return self.transformer(embeddings, mask)

    def __hash__(self):
        return id(self)
