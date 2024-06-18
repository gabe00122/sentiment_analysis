from typing import NamedTuple, Any
from .network import Network
from jax import Array
from jax.typing import ArrayLike


class StaticState(NamedTuple):
    network: Network
    solver: Any
    batch_size: int

class TrainingState(NamedTuple):
    rng_key: Array
    params: Any
    opt_state: Any
    indices: Array
    batch_index: ArrayLike
    tokens: Array
    labels: Array
