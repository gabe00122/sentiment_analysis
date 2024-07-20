from functools import partial
from pathlib import Path
from typing import NamedTuple
import time
import os

import jax
from flax import nnx
from jax import numpy as jnp

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.model import Model
from sentiment_analysis.experiment import load_settings
from sentiment_analysis.common.dataset_iterator import TrainingData


def score():
    os.environ['XLA_FLAGS'] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
    )

    path = Path("results/small_mixed_single_2024-07-19_02-05-01")
    settings = load_settings(path / "settings.json")

    data = jnp.load("./data/validation.npz")
    tokens = data['tokens']
    labels = data['length']

    samples = tokens.shape[0]
    steps = samples // settings.batch_size
    samples = steps * settings.batch_size

    model = Model(settings.model, rngs=nnx.Rngs(0))

    checkpoints = Checkpointer(path / "checkpoints")
    model = checkpoints.restore(model, 49999)

    indices = jnp.arange(samples, dtype=jnp.uint32)
    batch = TrainingData(jnp.uint32(0), tokens, labels, indices)

    output = jnp.zeros(steps, jnp.float32)

    for step in range(steps):
        start_time = time.time()

        batch, output = eval_step(model, settings.batch_size, batch, output)

        output[step].block_until_ready()
        end_time = time.time()
        delta_time = end_time - start_time
        samples_per_second = settings.batch_size / delta_time

        print(f"step = {step}/{steps}, perf = {samples_per_second:.2f}")

    print(f"output = {(output.mean().item() * 100):.2f}%")

    # jax.profiler.save_device_memory_profile("memory.prof")

    checkpoints.close()


@partial(nnx.jit, static_argnums=1, donate_argnums=(2, 3))
def eval_step(model, batch_size: int, batch: TrainingData, output: jax.Array) -> jax.Array:
    step = batch.step
    indices = jax.lax.dynamic_slice(
        batch.indices, (batch_size * step,), (batch_size,)
    )
    tokens = batch.tokens[indices]
    labels = batch.labels[indices]

    logit_pred = model(tokens, deterministic=True, rngs=None)

    logit_pred = logit_pred[:, :-1, :]
    tokens = tokens[:, 1:]
    predicted_labels = jnp.argmax(logit_pred[labels][:, :, :5], axis=-1)
    percent_correct = jnp.mean(predicted_labels == tokens[labels], dtype=jnp.float32)

    output = output.at[step].set(percent_correct)

    batch = batch._replace(step=step+1)

    return batch, output


if __name__ == '__main__':
    score()
