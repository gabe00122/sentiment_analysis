import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
import jax


class Checkpointer:
    def __init__(self, directory: Path):
        self.mngr = ocp.CheckpointManager(directory)

    def save(self, step: int, model: nnx.Module):
        _, state = nnx.split(model)
        self.mngr.save(step, args=ocp.args.StandardSave(state))

    def restore(self, step: int, model: nnx.Module) -> nnx.Module:
        _, state = nnx.split(model)
        abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)

        return self.mngr.restore(step, args=ocp.args.StandardRestore(abstract_state))

    def restore_latest(self, model: nnx.Module) -> nnx.Module:
        return self.restore(self.mngr.latest_step(), model)

    def close(self):
        self.mngr.close()
