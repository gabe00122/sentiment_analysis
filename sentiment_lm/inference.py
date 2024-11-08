from pathlib import Path
import jax
from flax import nnx
from jax import numpy as jnp, random

from rich.console import Console
from rich.prompt import Prompt
import rich.table

from sentiment_lm.experiment import Experiment
from sentiment_lm.tokenizer import Tokenizer
from sentiment_lm.util import count_params
from sentiment_lm.constants import END_TOKEN, STAR_TOKENS


def inference_cli(
    model_path: Path,
    temperature: float,
    top_k: int,
    top_p: float,
):
    console = Console()

    experiment = Experiment.load(model_path)
    context_size = experiment.settings.context_size

    console.print(_print_model_card(experiment))

    console.print("[1/3] Loading checkpoint")
    model = experiment.restore_last_checkpoint()
    console.print(f"Finished loading {count_params(model)} parameters")

    @nnx.jit
    def predict_token(tokens, i, rng_key):
        logits = model(tokens[jnp.newaxis, :], jnp.arange(context_size))[0, i - 1]
        logits /= temperature
        probs = nnx.softmax(logits)

        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        top_p_probs = probs[top_k_indices]

        cumsum_top_p = jnp.cumsum(top_p_probs) - top_p_probs
        top_k_logits = jnp.where(cumsum_top_p < top_p, top_k_logits, -jnp.inf)

        sample_index = random.categorical(rng_key, top_k_logits)
        return top_k_indices[sample_index].astype(jnp.uint16)

    @nnx.jit
    def predict_rating(tokens: jax.Array, length: jax.Array):
        tokens = tokens.at[length].set(END_TOKEN)
        logits = model(tokens[jnp.newaxis, :], jnp.arange(context_size))[0, length]

        return jnp.argmax(logits) - STAR_TOKENS[0] + 1

    tokenizer = Tokenizer(experiment.settings.vocab.path, context_size)
    rng_key = random.key(0)

    console.print("[2/3] [orange]Warming[/orange] up token prediction")
    predict_token(jnp.zeros(context_size, jnp.uint16), jnp.uint32(0), rng_key)
    console.print("[3/3] [orange]Warming[/orange] up rating prediction")
    predict_rating(jnp.zeros(context_size, jnp.uint16), jnp.uint32(0))

    while True:
        prompt = Prompt.ask("\n[blue]Prompt[/blue]")

        if prompt == "":
            continue

        if prompt == "/bye":
            console.print("[green]Goodbye![/green]")
            break

        console.print(f"\n[green]{prompt}[/green]", end="")
        context, length = tokenizer.encode(prompt)
        context = jnp.array(context, dtype=jnp.uint16)

        stars = None

        for i in range(length, context_size):
            rng_key, sample_key = random.split(rng_key)

            pred_token = predict_token(context, jnp.uint32(i), sample_key)

            if pred_token == END_TOKEN:
                stars = predict_rating(context, jnp.uint32(i))
                break

            text = tokenizer.decode_token(pred_token)
            console.print(text, end="")

            context = context.at[i].set(pred_token)

        if stars is not None:
            console.print("\n\n[yellow]Stars[/yellow]: " + ("⭐" * stars))


def _print_model_card(experiment: Experiment):
    return experiment.settings
