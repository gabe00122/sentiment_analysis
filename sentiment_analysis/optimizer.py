import optax
from flax import nnx

from sentiment_analysis.types import ExperimentSettings
from sentiment_analysis.common.dataset_iterator import TrainingData
from sentiment_analysis.model import Model


def create_optax_optimizer(settings: ExperimentSettings, total_steps: int):
    # min_rate = settings.optimizer.learning_rate / 10
    every_k_schedule = settings.batch_per_call
    learning_rate = optax.linear_onecycle_schedule(total_steps // every_k_schedule, settings.optimizer.learning_rate)

    if settings.optimizer.weight_decay > 0:
        return optax.MultiSteps(
            optax.chain(
                optax.clip(1.0),
                optax.adamw(
                    learning_rate,
                    b1=settings.optimizer.beta1,
                    b2=settings.optimizer.beta2,
                    eps=settings.optimizer.eps,
                    weight_decay=settings.optimizer.weight_decay
                )
            ), every_k_schedule=every_k_schedule)
    else:
        return optax.adam(
            learning_rate,
            b1=settings.optimizer.beta1,
            b2=settings.optimizer.beta2,
            eps=settings.optimizer.eps,
        )


def create_optimizer(settings: ExperimentSettings, rngs: nnx.Rngs, training_data: TrainingData) -> nnx.Optimizer:
    model = Model(settings.model, rngs)
    total_steps = (training_data.tokens.shape[0] // settings.batch_size) * settings.epochs
    tx = create_optax_optimizer(settings, total_steps)

    return nnx.Optimizer(model, tx)
