from typing import Literal

from pydantic.dataclasses import dataclass
from flax import nnx

from sentiment_analysis.model.transformer import ActivationName, TransformerModel


@dataclass(frozen=True)
class ModelSettings:
    num_layers: int
    num_heads: int
    d_model: int
    ffn_size: int
    activation_name: ActivationName
    glu: bool
    dtype: Literal['float32', 'bfloat16']
    param_dtype: Literal['float32', 'bfloat16']

    def create_model(self, vocab_size: int, rngs: nnx.Rngs) -> TransformerModel:
        return TransformerModel(
            vocab_size=vocab_size,
            d_model=self.d_model,
            ffn_size=self.ffn_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            activation_name=self.activation_name,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            glu=self.glu,
            rngs=rngs
        )
