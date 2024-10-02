
from pathlib import Path
import jax
from flax import nnx
from jax import numpy as jnp, random

from rich.console import Console
from rich.prompt import Prompt

from sentiment_lm.experiment import Experiment
from sentiment_lm.tokenizer import Tokenizer
from sentiment_lm.util import count_params
from sentiment_lm.constants import CONTEXT_SIZE


def inference_cli(
        model_path: Path,
        temperature: float,
        top_k: int,
        top_p: float
    ):
    console = Console()

    experiment = Experiment.load(model_path)
    
    console.print("[1/3] Loading checkpoint")
    model = experiment.restore_last_checkpoint()
    console.print(f"Finished loading {count_params(model)} parameters")

    @nnx.jit
    def predict_token(tokens, i, rng_key):
        logits = model(tokens[jnp.newaxis, :], jnp.arange(CONTEXT_SIZE))[0, i - 1]
        logits /= temperature
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

    console.print("[2/3] [orange]Warming[/orange] up token prediction")
    predict_token(jnp.zeros(CONTEXT_SIZE, jnp.int16), jnp.int32(0), rng_key)
    console.print("[3/3] [orange]Warming[/orange] up rating prediction")
    predict_rating(jnp.zeros(CONTEXT_SIZE, jnp.int16), jnp.int32(0))

    while True:
        prompt = Prompt.ask("\n[blue]prompt[/blue]")

        if prompt == '/bye':
            console.print("[green]Goodbye![/green]")
            return

        console.print(f"[green]{prompt}[/green]", end="")
        context, length = tokenizer.encode(prompt)
        stars = None

        for i in range(length, CONTEXT_SIZE):
            rng_key, sample_key = random.split(rng_key)

            pred_token = predict_token(context, jnp.int32(i), sample_key)

            if pred_token == 0:
                stars = predict_rating(context, jnp.int32(i))
                break

            text = tokenizer.decode2(pred_token)
            console.print(text, end="")

            context = context.at[i].set(pred_token)

        if stars is not None:
            print("\n\n[cyan]Stars[/cyan]: " + ("⭐" * stars))
