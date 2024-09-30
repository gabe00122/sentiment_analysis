from sentiment_lm.model.transformer_settings import ModelSettings
from typing import Literal
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class VocabSettings:
    type: Literal["token_monster"]
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
    vocab: VocabSettings
    optimizer: OptimizerSettings
    model: ModelSettings
