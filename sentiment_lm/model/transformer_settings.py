from typing import Literal, Optional

from pydantic.dataclasses import dataclass
from flax import nnx

from sentiment_lm.constants import SPECIAL_TOKENS
from sentiment_lm.model.transformer import ActivationName, TransformerModel


@dataclass(frozen=True)
class ModelSettings:
    num_layers: int
    num_heads: int
    d_model: int
    ffn_size: int
    activation_name: ActivationName
    glu: bool
    dtype: Literal["float32", "bfloat16"]
    param_dtype: Literal["float32", "bfloat16"]
    attention_softcap: Optional[float] = None
    output_softcap: Optional[float] = None

    def create_model(self, vocab_size: int, rngs: nnx.Rngs) -> TransformerModel:
        return TransformerModel(
            vocab_size=vocab_size + SPECIAL_TOKENS,
            d_model=self.d_model,
            ffn_size=self.ffn_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            activation_name=self.activation_name,
            glu=self.glu,
            attention_softcap=self.attention_softcap,
            output_softcap=self.output_softcap,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            rngs=rngs,
        )
