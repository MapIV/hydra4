from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from .builtin_interfaces import builtin_interfaces__msg__Time
from .std_msgs import std_msgs__msg__Header


@dataclass
class pandar_msgs__msg__PandarPacket:
    stamp: builtin_interfaces__msg__Time
    data: npt.NDArray[np.uint8]
    size: int

    __msgtype__: ClassVar[str] = "pandar_msgs/msg/PandarPacket"


@dataclass
class pandar_msgs__msg__PandarScan:
    header: std_msgs__msg__Header
    packets: list[pandar_msgs__msg__PandarPacket]

    __msgtype__: ClassVar[str] = "pandar_msgs/msg/PandarScan"
