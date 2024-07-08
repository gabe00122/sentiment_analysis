from abc import ABC, abstractmethod
from pathlib import Path
import os
from typing import Any, NamedTuple
from jax import Array, numpy as jnp
from jax.typing import ArrayLike
from tensorboardX import SummaryWriter

type Metrics = dict[str, Array]


class MetricsBuffer(NamedTuple):
    values: Metrics
    length: Array


class Writer(ABC):
    @abstractmethod
    def write(self, metrics: Metrics | MetricsBuffer) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class TensorboardWriter(Writer):
    data: dict[str, Any] = {}

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.writer = SummaryWriter(os.path.abspath(output_path))
        self.global_step = 0

    def write(self, metrics: Metrics | MetricsBuffer) -> None:
        if isinstance(metrics, MetricsBuffer):
            for key, value in metrics.values.items():
                value_list = value[: metrics.length].tolist()

                for i, value_item in enumerate(value_list):
                    self.writer.add_scalar(key, value_item, self.global_step + i)

            self.global_step += metrics.length
        else:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value.item(), self.global_step)

            self.global_step += 1

    def close(self) -> None:
        self.writer.close()


def create_metrics_buffer(metrics: Metrics, capacity: ArrayLike) -> MetricsBuffer:
    def create_metric(value: Array) -> Array:
        shape = (capacity, *value.shape)
        metric = jnp.zeros(shape, dtype=value.dtype)
        metric = metric.at[0].set(value)

        return metric

    values = {k: create_metric(v) for k, v in metrics.items()}

    return MetricsBuffer(values=values, length=jnp.int32(1))


def append_buffer(metrics_buffer: MetricsBuffer, metrics: Metrics) -> MetricsBuffer:
    def append_value(buffer_value: Array, update: Array):
        return buffer_value.at[metrics_buffer.length].set(update)

    values = {k: append_value(v, metrics[k]) for k, v in metrics_buffer.values.items()}

    return MetricsBuffer(values=values, length=metrics_buffer.length + 1)
