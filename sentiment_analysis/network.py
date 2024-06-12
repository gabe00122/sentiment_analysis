from flax import linen as nn
from jax import Array

from .transformer import Transformer, get_init_scale


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
            embedding_init=nn.initializers.normal(1.0),
            # param_dtype=jnp.float16,
            # dtype=jnp.float16
        )(inputs)

        embeddings = token_embeddings + self.position_embeddings
        embeddings *= (self.embedding_features ** -(1 / 2)) * get_init_scale(self.transformer.num_layers)

        return self.transformer(embeddings, mask)

    def __hash__(self):
        return id(self)
