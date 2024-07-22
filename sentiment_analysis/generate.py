from pathlib import Path

from flax import nnx
from tokenmonster import Vocab
from jax import numpy as jnp, random
import jax

from sentiment_analysis.experiment import Experiment, load_settings
from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.model import Model
from sentiment_analysis.vocab import encode, decode

def count_params(model) -> int:
    params = nnx.state(model, nnx.Param)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def main():
    path = Path("results/large_mixed_single_2024-07-22_09-00-53")
    settings = load_settings(path / "settings.json")
    checkpointer = Checkpointer(path / "checkpoints")

    model = Model(settings.model, nnx.Rngs(0))
    model = checkpointer.restore_latest(model)
    print(count_params(model))

    @nnx.jit
    def inference(tokens, i, rng_key):
        predictions =  model(tokens, True, nnx.Rngs(0))[i - 1]
        values, indices = jax.lax.top_k(predictions, 50)
        pred_token = random.categorical(rng_key, values / 0.4)
        return indices[pred_token]


    vocab = Vocab(settings.model.vocab.path)
    rng_key = random.key(0)

    while True:
        prompt = input("prompt: ")

        context_size = settings.model.context_size
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




if __name__ == '__main__':
    main()
