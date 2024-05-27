from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from .builtin_interfaces import builtin_interfaces__msg__Time
from .std_msgs import std_msgs__msg__Header


@dataclass
class velodyne_msgs__msg__VelodynePacket:
    stamp: builtin_interfaces__msg__Time
    data: npt.NDArray[np.uint8]

    __msgtype__: ClassVar[str] = "velodyne_msgs/msg/VelodynePacket"


@dataclass
class velodyne_msgs__msg__VelodyneScan:
    header: std_msgs__msg__Header
    packets: list[velodyne_msgs__msg__VelodynePacket]

    __msgtype__: ClassVar[str] = "velodyne_msgs/msg/VelodyneScan"
