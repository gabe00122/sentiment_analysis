from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class VocabSettings:
    type: Literal['token_monster']
    path: str
    size: int
