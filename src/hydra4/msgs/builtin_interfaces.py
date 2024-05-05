from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class builtin_interfaces__msg__Time:
    sec: int
    nanosec: int

    __msgtype__: ClassVar[str] = "builtin_interfaces/msg/Time"
