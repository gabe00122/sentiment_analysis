from pathlib import Path

from flax import nnx
from tokenmonster import Vocab
from jax import numpy as jnp, random
import jax

from sentiment_analysis.experiment import Experiment, load_settings
from sentiment_analysis.common.checkpointer import Checkpointer
# from sentiment_analysis.model.transformer import TransformerModel
from sentiment_analysis.vocab import encode, decode
from sentiment_analysis.model.mamba import Mamba2Model


def count_params(model) -> int:
    params = nnx.state(model, nnx.Param)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def main():
    path = Path("results/small_genrative_2024-08-04_23-50-54")
    settings = load_settings(path / "settings.json")
    checkpointer = Checkpointer(path / "checkpoints")

    model = settings.model.create_model(settings.vocab.size + 6, nnx.Rngs(0)) #Model(settings.model, nnx.Rngs(0))
    model = checkpointer.restore_latest(model)
    print(count_params(model))

    @nnx.jit
    def inference(tokens, i, rng_key):
        temp = 0.7
        top_k = 50
        top_p = 0.90

        logits = model(tokens[jnp.newaxis, :], jnp.arange(128))[0, i - 1]
        logits /= temp
        probs = nnx.softmax(logits)

        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        top_p_probs = probs[top_k_indices]

        cumsum_top_p = jnp.cumsum(top_p_probs) - top_p_probs
        top_k_logits = jnp.where(cumsum_top_p < top_p, top_k_logits, -jnp.inf)

        sample_index = random.categorical(rng_key, top_k_logits)
        return top_k_indices[sample_index]

    vocab = Vocab(settings.vocab.path)
    rng_key = random.key(0)

    while True:
        prompt = input("prompt: ")

        context_size = 128
        context, length = encode(vocab, prompt, context_size)

        stars = None

        for i in range(length, context_size):
            rng_key, sample_key = random.split(rng_key)

            pred_token = inference(context, i, sample_key)

            if pred_token < 5:
                stars = pred_token
                break

            context = context.at[i].set(pred_token)

        output = decode(vocab, context)
        print(output)

        if stars is not None:
            print("\nStars: " + str(stars.item()))


if __name__ == "__main__":
    main()
