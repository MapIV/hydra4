from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .builtin_interfaces import builtin_interfaces__msg__Time


@dataclass
class std_msgs__msg__Header:
    stamp: builtin_interfaces__msg__Time
    frame_id: str

    __msgtype__: ClassVar[str] = "std_msgs/msg/Header"
