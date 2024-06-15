import jax.lax
from flax import linen as nn
from jax import Array, random, numpy as jnp
from functools import partial

from .transformer import Transformer, get_init_scale


@partial(jax.vmap, in_axes=(0, None))
def get_offsets(rngs, position_embeddings):
    half_size = 128
    offset = random.randint(rngs, (), 0, half_size)
    return jax.lax.dynamic_slice(position_embeddings, (offset, 0), (half_size, 128 * 2))


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
        has_batch = len(inputs.shape) > 1
        position_embeddings = self.position_embeddings

        if has_batch:
            rng = self.make_rng("params")
            rngs = random.split(rng, inputs.shape[0])
            position_embeddings = get_offsets(rngs, position_embeddings)
        else:
            position_embeddings = position_embeddings[64:128+64]


        embeddings = token_embeddings + position_embeddings

        # if mask is not None:
        #     embeddings = jax.vmap(lambda a, b, c: jnp.where(a[..., jnp.newaxis], b, c), in_axes=(0, 0, None))(mask, embeddings, empty_embeddings)

        embeddings *= (self.embedding_features ** -(1 / 2)) * get_init_scale(self.transformer.num_layers)

        return self.transformer(embeddings, mask)

    def __hash__(self):
        return id(self)
