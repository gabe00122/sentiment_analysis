from pydantic import TypeAdapter
from sentiment_lm.model.transformer_settings import ModelSettings
from typing import Any, Literal
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class LoggerConfig:
    use_console: bool = True
    use_tb: bool = False
    use_csv: bool = False
    use_wandb: bool = False
    use_neptune: bool = False


@dataclass(frozen=True)
class VocabSettings:
    path: str
    size: int


@dataclass(frozen=True)
class OptimizerSettings:
    type: Literal["adamw"]
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    eps: float
    beta1: float
    beta2: float


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int | Literal["random"]
    training_file: str
    validation_file: str
    epochs: int
    batch_size: int
    accumulation_steps: int
    context_size: int
    vocab: VocabSettings
    optimizer: OptimizerSettings
    model: ModelSettings
    logger: LoggerConfig
