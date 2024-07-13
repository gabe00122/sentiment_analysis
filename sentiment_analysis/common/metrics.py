import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from jax import Array
from tensorboardX import SummaryWriter

type Metrics = dict[str, Array]

class Writer(ABC):
    @abstractmethod
    def write(self, metrics: Metrics) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class TensorboardWriter(Writer):
    data: dict[str, Any] = {}

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.writer = SummaryWriter(os.path.abspath(output_path))
        self.global_step = 0

    def write(self, metrics: Metrics) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value.item(), self.global_step)

        self.global_step += 1

    def close(self) -> None:
        self.writer.close()
