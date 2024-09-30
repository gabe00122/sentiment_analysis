import jax
import tokenmonster
from jax import numpy as jnp
from sentiment_lm.constants import SPECIAL_TOKENS, EMPTY_TOKEN


class Tokenizer:
    def __init__(self, vocab_path: str, context_size: int):
        self.vocab = tokenmonster.Vocab(vocab_path)
        self.context_size = context_size

    def encode(self, text: str) -> tuple[jax.Array, int]:
        tokens = list(self.vocab.tokenize(text))
        length = len(tokens)

        tokens = [t + SPECIAL_TOKENS for t in tokens]
        tokens += [EMPTY_TOKEN] * (self.context_size - len(tokens))
        tokens = jnp.array(tokens, jnp.int32)

        return tokens, length


    def decode(self, tokens: jax.Array) -> str:
        tokens = tokens - SPECIAL_TOKENS
        tokens = tokens.tolist()
        tokens = [x for x in tokens if x >= 0]
        return self.vocab.decode(tokens)
