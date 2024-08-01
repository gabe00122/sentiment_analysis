import jax
import tokenmonster
from jax import numpy as jnp

offset = 6


def encode(vocab: tokenmonster.Vocab, text: str, context_size: int) -> tuple[jax.Array, int]:
    tokens = list(vocab.tokenize(text))
    length = len(tokens)

    tokens += [-1] * (context_size - len(tokens))
    tokens = jnp.array(tokens, jnp.int32)
    tokens = tokens + offset

    return tokens, length


def decode(vocab: tokenmonster.Vocab, tokens: jax.Array) -> str:
    tokens = tokens - offset
    tokens = tokens.tolist()
    tokens = list(filter(lambda x: x >= 0, tokens))

    return vocab.decode(tokens)
