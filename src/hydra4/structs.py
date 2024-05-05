from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReturnMode(str, Enum):
    FIRST = "first"
    STRONGET = "strongest"
    LAST = "last"
    LAST_STRONGEST = "last_strongest"
    LAST_FIRST = "last_first"
    FIRST_STRONGEST = "first_strongest"

    def is_dual(self) -> bool:
        return self in ("last_strongest", "last_first", "first_strongest")

    def __str__(self) -> str:
        return self.value


@dataclass()
class Block:
    azimuth: int
    points: list[tuple]
