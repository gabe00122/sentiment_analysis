
import jax
from flax import nnx
from jax import numpy as jnp, random

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.experiment import load_settings, Experiment
from sentiment_analysis.tokenizer import Tokenizer
from sentiment_analysis.util import count_params
from sentiment_analysis.constants import CONTEXT_SIZE


def inference_cli(model_path: str):
    experiment = Experiment.load(model_path)
    model = experiment.restore_last_checkpoint()
    print(count_params(model))

    @nnx.jit
    def predict_token(tokens, i, rng_key):
        temp = 0.7
        top_k = 50
        top_p = 0.90

        logits = model(tokens[jnp.newaxis, :], jnp.arange(CONTEXT_SIZE))[0, i - 1]
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
        logits = model(tokens[jnp.newaxis, :], jnp.arange(CONTEXT_SIZE))[0, length]

        return jnp.argmax(logits)

    tokenizer = Tokenizer(experiment.settings.vocab.path, CONTEXT_SIZE)
    rng_key = random.key(0)

    while True:
        prompt = input("prompt: ")

        context, length = tokenizer.encode(prompt)

        stars = None

        for i in range(length, CONTEXT_SIZE):
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