from pathlib import Path
from random import getrandbits
from dataclasses import replace
import subprocess
import datetime
from functools import cached_property

from pydantic import TypeAdapter, BaseModel
from sentiment_analysis.types import ExperimentSettings


class ExperimentMetadata(BaseModel):
    start_time: datetime.datetime
    git_hash: str


def _create_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        start_time = _get_iso_time(),
        git_hash = _get_git_revision_hash()
    )


def _get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def _get_iso_time():
    return datetime.datetime.now()


class Experiment:
    def __init__(self, name: str, settings: ExperimentSettings, metadata: ExperimentMetadata = _create_metadata()):
        self.name = name
        self.settings = settings
        self.metadata = metadata

    def init_dir(self):
        self.path.mkdir(parents=True)
        self.checkpoint_path.mkdir()

        settings_bytes = TypeAdapter(ExperimentSettings).dump_json(self.settings, indent=2)
        settings_path = self.path / Path("settings.json")
        settings_path.write_bytes(settings_bytes)

        metadata_bytes = self.metadata.model_dump_json(indent=2)
        metadata_path = self.path / Path("metadata.json")
        metadata_path.write_text(metadata_bytes)


    @cached_property
    def id(self):
        return f"{self.name}_{self._file_time_format}"

    @cached_property
    def path(self) -> Path:
        return Path("results") / Path(self.id)

    @cached_property
    def _file_time_format(self) -> str:
        return self.metadata.start_time.strftime("%Y-%m-%d_%H-%M-%S")

    @cached_property
    def checkpoint_path(self) -> Path:
        return self.path / Path("checkpoints")

    @classmethod
    def create_experiment(cls, settings_file: str | Path) -> 'Experiment':
        settings_file = Path(settings_file)
        settings = load_settings(settings_file)

        experiment = cls(settings_file.stem, settings)
        experiment.init_dir()

        return experiment


def load_settings(file: str | Path) -> ExperimentSettings:
    settings = TypeAdapter(ExperimentSettings).validate_json(Path(file).read_bytes())

    if settings.seed == 'random':
        settings = replace(settings, seed=getrandbits(32))

    return settings


def main():
    Experiment.create_experiment("./experiment_settings/tiny.json")


if __name__ == '__main__':
    main()
