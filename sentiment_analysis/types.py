from sentiment_analysis.model.types import ModelSettings
from typing import Literal
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerSettings:
    type: Literal['adamw']
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    eps: float
    beta1: float
    beta2: float


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int | Literal['random']
    training_file: str
    validation_file: str
    epochs: int
    batch_size: int
    batch_per_call: int
    optimizer: OptimizerSettings
    model: ModelSettings
