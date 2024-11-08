import jax
import tokenmonster
from jax import numpy as jnp
from sentiment_lm.constants import END_TOKEN, SPECIAL_TOKENS, EMPTY_TOKEN, STAR_TOKENS, START_TOKEN
import sentencepiece as spm


class Tokenizer:
    def __init__(self, vocab_path: str, context_size: int):
        self.vocab = spm.SentencePieceProcessor(model_file=vocab_path)
        self.context_size = context_size

    def encode(self, text: str, stars: int|None=None) -> tuple[jax.Array, int]:
        vocab_tokens = self.vocab.encode(text, out_type=int)

        tokens = [START_TOKEN]
        tokens.extend([t + SPECIAL_TOKENS for t in vocab_tokens])

        if stars is not None:
            tokens.append(END_TOKEN)
            tokens.append(STAR_TOKENS[stars - 1])

        # pad
        pad_size = len(tokens)
        tokens.extend([EMPTY_TOKEN] * (self.context_size - pad_size))
        # tokens = jnp.array(tokens, jnp.uint16)

        return tokens, pad_size

    def decode_context(self, tokens: jax.Array) -> str:
        tokens = tokens - SPECIAL_TOKENS
        tokens_list = tokens.tolist()
        tokens_list = [x for x in tokens_list if x >= 0]
        return self.vocab.decode(tokens_list)

    def decode_token(self, token: jax.Array) -> str:
        int_token: int = token.item()
        int_token -= SPECIAL_TOKENS
        return self.decoder.decode(int_token)
