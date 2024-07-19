import os
import time
from functools import partial
from pathlib import Path

import optax
from flax import nnx
import jax
from jax import numpy as jnp, random
import numpy as np

from sentiment_analysis.common.checkpointer import Checkpointer
from sentiment_analysis.common.dataset_iterator import read_training_data, TrainingData, load_training_data
from sentiment_analysis.common.metrics import TensorboardWriter, Metrics
from sentiment_analysis.experiment import Experiment
from sentiment_analysis.optimizer import create_optimizer


def count_params(model) -> int:
    params = nnx.state(model, nnx.Param)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def set_flags():
    os.environ['XLA_FLAGS'] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
    )


def train(experiment: Experiment):
    set_flags()

    settings = experiment.settings
    seed = random.PRNGKey(settings.seed)

    init_key, train_key, training_shuffle_key, validation_shuffle_key = random.split(seed, num=4)
    init_rngs = nnx.Rngs(init_key)
    rngs = nnx.Rngs(train_key)

    training_data = load_training_data(settings.training_file, training_shuffle_key)
    validation_data = load_training_data(settings.validation_file, validation_shuffle_key)

    samples = training_data.tokens.shape[0]
    context_size = training_data.tokens.shape[-1]

    if context_size != settings.model.context_size:
        print(f"Model context size {settings.model.context_size} and data context size {context_size} don't match")
        return

    print(f"Total Samples: {samples}")

    optimizer = create_optimizer(settings, init_rngs, training_data)
    print(f"Param Count: {count_params(optimizer.model)}")

    checkpoints = Checkpointer(experiment.checkpoint_path)
    writer = TensorboardWriter(Path("./tensorboard"), experiment.run_name)

    steps = samples // settings.batch_size
    total_steps = steps * settings.epochs
    checkpoint_rate = 10_000
    validation_rate = 20

    for global_step in range(total_steps):
        start_time = time.time()
        training_data, metrics = train_step(optimizer, rngs, settings.batch_size, training_data, True)

        if global_step % validation_rate == validation_rate - 1:
            validation_data, val_metrics = train_step(optimizer, rngs, settings.batch_size, validation_data, False)
            writer.write(val_metrics, global_step, "validation")

            val_loss = val_metrics["loss"].item()
            val_correct = val_metrics["percent_correct"].item() * 100
            print(f"validation loss = {val_loss:.4f}, validation correct = {val_correct:.1f}%")

        writer.write(metrics, global_step, "training")

        loss = metrics["loss"].item()
        percent_correct = metrics["percent_correct"].item() * 100

        end_time = time.time()
        delta_time = end_time - start_time
        samples_per_second = settings.batch_size / delta_time

        epoch = global_step // steps
        step = global_step % steps

        print(f"epoch = {epoch}/{settings.epochs}, step = {step}/{steps}, loss = {loss:.4f}, correct = {percent_correct:.1f}%, perf = {samples_per_second:.2f}")

        if global_step % checkpoint_rate == checkpoint_rate - 1:
            print("Saving checkpoint")
            checkpoints.save(optimizer.model, global_step)


    print("Saving final checkpoint")
    checkpoints.save(optimizer.model, total_steps - 1)
    checkpoints.close()
    writer.close()

    print("Running validation")
    validation_accuracy = validate(optimizer, rngs, settings.batch_size, validation_data)
    print(f"Validation accuracy {validation_accuracy:.1f}%")
    experiment.save_results({"validation_accuracy": validation_accuracy})

    print("Experiment complete 🎉")


def classification_loss_fn(model, tokens, labels, rngs, training: bool):
    logit_pred = model(tokens, deterministic=not training, rngs=rngs)

    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logit_pred, labels))

    predicted_labels = jnp.argmax(logit_pred, axis=-1)
    percent_correct = jnp.mean(predicted_labels == labels, dtype=jnp.float32)
    metrics = {"percent_correct": percent_correct, "loss": loss}

    return loss, metrics


def regressive_loss_fn(model, tokens, labels, rngs, training: bool):
    class_scale = model.settings.output.output_classes

    pred = nnx.sigmoid(jnp.squeeze(model(tokens, deterministic=not training, rngs=rngs)))

    scaled_down_labels = (labels + 0.5) / class_scale
    loss = jnp.mean((pred - scaled_down_labels) ** 2)

    predicted_labels = jnp.floor(pred * class_scale)
    percent_correct = jnp.mean(predicted_labels == labels, dtype=jnp.float32)
    metrics = {"percent_correct": percent_correct, "loss": loss}

    return loss, metrics


@partial(nnx.jit, static_argnums=(2, 4), donate_argnums=3)
def train_step(optimizer: nnx.Optimizer, rngs: nnx.Rngs, batch_size: int, training_data: TrainingData, training: bool) -> tuple[TrainingData, Metrics]:
    training_data, tokens, labels = read_training_data(training_data, rngs.shuffle(), batch_size)

    if optimizer.model.settings.output.format == 'regression':
        loss_fn = regressive_loss_fn
    else:
        loss_fn = classification_loss_fn

    if training:
        grads, metrics = nnx.grad(loss_fn, has_aux=True, wrt=nnx.Param)(optimizer.model, tokens, labels, rngs, training)
        optimizer.update(grads)
    else:
        _, metrics = loss_fn(optimizer.model, tokens, labels, rngs, training)

    return training_data, metrics


def validate(optimizer: nnx.Optimizer, rngs: nnx.Rngs, batch_size: int, validation_data: TrainingData) -> float:
    steps = validation_data.tokens.shape[0] // batch_size

    output = np.zeros((steps,), dtype=np.float32)
    for step in range(steps):
        validation_data, metrics = train_step(optimizer, rngs, batch_size, validation_data, False)
        percent_correct = metrics["percent_correct"].item() * 100
        output[step] = percent_correct
        print(f"percent_correct = {percent_correct:.1f}%")

    return output.mean().item()
