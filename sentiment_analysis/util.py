import os
import jax
from flax import nnx


def count_params(model) -> int:
    params = nnx.state(model, nnx.Param)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def set_flags():
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=true "
        # "--xla_gpu_enable_custom_fusions=true "
        # "--xla_gpu_enable_address_computation_fusion=true "
    )
