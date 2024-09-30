import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
import jax


class Checkpointer:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        directory = directory.absolute()
        self.mngr = ocp.CheckpointManager(directory)

    def save(self, model: nnx.Module, step: int):
        state = nnx.state(model, nnx.Param)
        self.mngr.save(step, args=ocp.args.StandardSave(state))

    def restore(self, model: nnx.Module, step: int) -> nnx.Module:
        graphdef, state, other_state = nnx.split(model, nnx.Param, ...)
        abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)

        restored_state = self.mngr.restore(
            step, args=ocp.args.StandardRestore(abstract_state)
        )
        return nnx.merge(graphdef, restored_state, other_state)

    def restore_latest(self, model: nnx.Module) -> nnx.Module:
        return self.restore(model, self.mngr.latest_step())

    def close(self):
        self.mngr.close()
