import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from jax import Array
from tensorboardX import SummaryWriter

type Metrics = dict[str, Array]


class Writer(ABC):
    @abstractmethod
    def write(
        self, metrics: Metrics, global_step: int, prefix: Optional[str] = None
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class TensorboardWriter(Writer):
    data: dict[str, Any] = {}

    def __init__(self, output_path: Path, run_name: str) -> None:
        self.output_path = output_path
        self.writer = SummaryWriter(os.path.abspath(output_path / run_name))

    def write(
        self, metrics: Metrics, global_step: int, prefix: Optional[str] = None
    ) -> None:
        for key, value in metrics.items():
            if prefix is not None:
                key = f"{key}/{prefix}"

            self.writer.add_scalar(key, value.item(), global_step)

    def close(self) -> None:
        self.writer.close()
