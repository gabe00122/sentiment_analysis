import jax
from flax import nnx
from typing import Callable
from functools import partial
from collections.abc import Sequence
from sentiment_analysis.common.metrics import Metrics, MetricsBuffer


@partial(jax.jit)
def _jitted_loop(graphdefs):
    pass


def common_loop[S, D](
        models: Sequence[nnx.Module],
        static: S,
        dynamic: D,
        body: Callable[[models, S, D], (D, Metrics)],
        calls: int
) -> (D, MetricsBuffer):
    model_graphdefs = []
    model_params = []

    for model in models:
        graphdef, params = nnx.split(model)
        model_graphdefs.append(graphdef)
        model_params.append(params)
