import json
from pathlib import Path
from random import getrandbits
from dataclasses import replace
import subprocess
import datetime
from functools import cached_property

from pydantic import TypeAdapter, BaseModel
from flax import nnx

from sentiment_lm.common.checkpointer import Checkpointer
from sentiment_lm.training_settings import ExperimentSettings


class ExperimentMetadata(BaseModel):
    start_time: datetime.datetime
    git_hash: str


def _create_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        start_time=_get_iso_time(), git_hash=_get_git_revision_hash()
    )


def _get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def _get_iso_time():
    return datetime.datetime.now()


class Experiment:
    def __init__(
        self,
        path: Path,
        settings: ExperimentSettings,
        metadata: ExperimentMetadata,
    ):
        self.path = path
        self.settings = settings
        self.metadata = metadata

    def init_dir(self):
        self.path.mkdir(parents=True)
        self.checkpoint_path.mkdir()

        settings_bytes = TypeAdapter(ExperimentSettings).dump_json(
            self.settings, indent=2
        )
        settings_path = self.path / Path("settings.json")
        settings_path.write_bytes(settings_bytes)

        metadata_text = self.metadata.model_dump_json(indent=2)
        metadata_path = self.path / Path("metadata.json")
        metadata_path.write_text(metadata_text)

    @cached_property
    def _file_time_format(self) -> str:
        return self.metadata.start_time.strftime("%Y-%m-%d_%H-%M-%S")

    @cached_property
    def checkpoint_path(self) -> Path:
        return self.path / Path("checkpoints")

    def restore_last_checkpoint(self) -> nnx.Module:
        ckptr = Checkpointer(self.checkpoint_path)
        model = self.settings.model.create_model(self.settings.vocab.size, nnx.Rngs(0))
        model = ckptr.restore_latest(model)
        ckptr.close()

        return model

    def restore_checkpoint(self, step: int) -> nnx.Module:
        ckptr = Checkpointer(self.checkpoint_path)
        model = self.settings.model.create_model(self.settings.vocab.size, nnx.Rngs(0))
        model = ckptr.restore(model, step)
        ckptr.close()

        return model

    @classmethod
    def load(cls, path: Path) -> "Experiment":
        settings = load_settings(path / "settings.json")

        raw_metadata = (path / "metadata.json").read_bytes()
        metadata = ExperimentMetadata.model_validate_json(raw_metadata)

        experiment = cls(path, settings, metadata)

        return experiment

    @classmethod
    def create_experiment(cls, settings_file: Path) -> "Experiment":
        settings = load_settings(settings_file)
        metadata = _create_metadata()

        experiment_path = _get_experiment_path(settings_file, metadata.start_time)

        experiment = cls(experiment_path, settings, metadata)
        experiment.init_dir()

        return experiment

    def save_results(self, data):
        path = self.path / "results.json"
        path.write_text(json.dumps(data))


def load_settings(file: str | Path) -> ExperimentSettings:
    settings = TypeAdapter(ExperimentSettings).validate_json(Path(file).read_bytes())

    if settings.seed == "random":
        settings = replace(settings, seed=getrandbits(32))

    return settings


def _get_experiment_path(settings_file: Path, start_time: datetime.datetime) -> Path:
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{settings_file.stem}_{timestamp}"
    experiment_path = Path("results") / experiment_name

    return experiment_path
