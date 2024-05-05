from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pytest
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.typesys.stores.latest import builtin_interfaces__msg__Time, std_msgs__msg__Header

PANDAR_PACKET = """
builtin_interfaces/Time stamp
uint8[1500]             data
uint32                  size
"""

PANDAR_SCAN = """
std_msgs/Header   header
PandarPacket[]    packets
"""

class PandarPacket(Protocol):
  stamp: builtin_interfaces__msg__Time
  data: npt.NDArray[np.uint8]
  size: int

class PandarScan(Protocol):
  header: std_msgs__msg__Header
  packets: list[PandarPacket]

typestore = get_typestore(Stores.LATEST)
typestore.register(get_types_from_msg(PANDAR_PACKET, 'pandar_msgs/msg/PandarPacket'))
typestore.register(get_types_from_msg(PANDAR_SCAN, 'pandar_msgs/msg/PandarScan'))


@pytest.fixture
def pandar_scans_xt32() -> list[PandarScan]:
  path = Path(__file__).parent / "rosbags" / "pandar_xt32"
  with AnyReader([path], default_typestore=typestore) as reader:
    msgs = []
    for connection, _, rawdata in reader.messages(connections=reader.connections):
      msgs.append(reader.deserialize(rawdata, connection.msgtype))

    return msgs
