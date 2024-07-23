from flax import nnx
from jax.typing import DTypeLike

class FeedForwardBlock(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, activation, kernel_init: nnx.Initializer, dtype: DTypeLike, param_dtype: DTypeLike, dropout_rate: float, rngs: nnx.Rngs):
        self.up_linear = nnx.Linear(in_features, hidden_features, kernel_init=kernel_init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.down_linear = nnx.Linear(hidden_features, in_features, kernel_init=kernel_init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.activation = activation
        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(dropout_rate)

    def __call__(self, x, deterministic: bool, rngs: nnx.Rngs):
        x = self.up_linear(x)
        if self.dropout is not None:
            x = self.dropout(x, deterministic=deterministic, rngs=rngs)
        x = self.activation(x)
        x = self.down_linear(x)
        return x


class GLUFeedForwardBlock(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, activation, kernel_init: nnx.Initializer,
                 dtype: DTypeLike, param_dtype: DTypeLike, dropout_rate: float, rngs: nnx.Rngs):
        self.activation_linear = nnx.Linear(in_features, hidden_features, kernel_init=kernel_init, dtype=dtype,
                                    param_dtype=param_dtype, rngs=rngs)
        self.gate_linear = nnx.Linear(in_features, hidden_features, kernel_init=kernel_init, dtype=dtype,
                                            param_dtype=param_dtype, rngs=rngs)

        self.down_linear = nnx.Linear(hidden_features, in_features, kernel_init=kernel_init, dtype=dtype,
                                      param_dtype=param_dtype, rngs=rngs)
        self.activation = activation
        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(dropout_rate)

    def __call__(self, x, deterministic: bool, rngs: nnx.Rngs):
        a_x = self.activation_linear(x)
        g_x = self.gate_linear(x)

        if hasattr(self, 'dropout'):
            a_x = self.dropout(a_x, deterministic=deterministic, rngs=rngs)
            g_x = self.dropout(g_x, deterministic=deterministic, rngs=rngs)

        a_x = self.activation(a_x)
        x = a_x * g_x

        x = self.down_linear(x)
        return x
