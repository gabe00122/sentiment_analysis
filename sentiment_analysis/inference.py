import argparse
from pathlib import Path

import jax
from flax import nnx
from jax import numpy as jnp, random

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.experiment import load_settings
from sentiment_analysis.tokenizer import Tokenizer
from sentiment_analysis.util import count_params


def main():
    path = Path("results/small_generative_2024-08-04_23-50-54")
    settings = load_settings(path / "settings.json")
    checkpointer = Checkpointer(path / "checkpoints")

    model = settings.model.create_model(settings.vocab.size, nnx.Rngs(0))
    model = checkpointer.restore_latest(model)
    print(count_params(model))
    checkpointer.close()

    @nnx.jit
    def predict_token(tokens, i, rng_key):
        temp = 0.7
        top_k = 50
        top_p = 0.90

        logits = model(tokens[jnp.newaxis, :], jnp.arange(context_size))[0, i - 1]
        logits /= temp
        probs = nnx.softmax(logits)

        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        top_p_probs = probs[top_k_indices]

        cumsum_top_p = jnp.cumsum(top_p_probs) - top_p_probs
        top_k_logits = jnp.where(cumsum_top_p < top_p, top_k_logits, -jnp.inf)

        sample_index = random.categorical(rng_key, top_k_logits)
        return top_k_indices[sample_index].astype(jnp.int16)

    @nnx.jit
    def predict_rating(tokens: jax.Array, length: jax.Array):
        tokens = tokens.at[length].set(0)
        logits = model(tokens[jnp.newaxis, :], jnp.arange(context_size))[0, length]

        return jnp.argmax(logits)

    context_size = 128
    tokenizer = Tokenizer(settings.vocab.path, context_size)
    rng_key = random.key(0)

    while True:
        prompt = input("prompt: ")

        context, length = tokenizer.encode(prompt)

        stars = None

        for i in range(length, context_size):
            rng_key, sample_key = random.split(rng_key)

            pred_token = predict_token(context, i, sample_key)

            if pred_token == 0:
                stars = predict_rating(context, i)
                break

            context = context.at[i].set(pred_token)

        output = tokenizer.decode(context)
        print(output)

        if stars is not None:
            print("\nStars: " + ("⭐" * stars))


if __name__ == "__main__":
    main()
