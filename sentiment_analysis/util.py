import jax
from flax import nnx
from sentiment_analysis.types import ExperimentSettings


def get_call_size(settings: ExperimentSettings) -> int:
    return settings.batch_size * settings.batch_per_call


def get_calls_per_epoch(samples: int, settings: ExperimentSettings) -> int:
    call_size = get_call_size(settings)
    return samples // call_size


def get_rounded_steps(samples: int, settings: ExperimentSettings) -> int:
    call_size = get_call_size(settings)
    return (samples // call_size) * call_size


def get_total_steps(sample: int, settings: ExperimentSettings) -> int:
    return get_rounded_steps(sample, settings) * settings.epochs


def count_params(model) -> int:
    params = nnx.state(model, nnx.Param)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
