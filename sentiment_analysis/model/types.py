from typing import Sequence, Literal, Tuple

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class VocabSettings:
    type: Literal['token_monster']
    path: str
    size: int


@dataclass(frozen=True)
class OutputSettings:
    type: Literal['classification_tokens']
    format: Literal['regression', 'softmax']
    output_tokens: int
    output_classes: int


@dataclass(frozen=True)
class ModelSettings:
    vocab: VocabSettings
    context_size: int
    hidden_features: int
    transformer_layers: int
    transformer_heads: int
    mlp_feature: int
    activation: Literal['relu', 'relu2']
    max_position_offset: int
    output: OutputSettings
    dropout_rate: float
    dtype: Literal['float32', 'float16']
