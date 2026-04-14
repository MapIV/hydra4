from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np

from ..msgs import pandar_msgs__msg__PandarPacket
from ..structs import Block, ReturnMode
from .base import PandarBase

_BLOCK_SIZE: int = 130  # 2 bytes azimuth + 32 channels × 4 bytes
_CHANNEL_SIZE: int = 4  # 2 bytes distance + 1 byte intensity + 1 byte reserved
_NUM_BLOCKS: int = 8
_NUM_CHANNELS: int = 32
_DIS_UNIT_M: float = 0.004


class PandarXT32(PandarBase):
    def __init__(self, calibration_path: Path, min_distance: float, max_distance: float) -> None:
        """
        Initializes the PandarXT32 object with calibration path, minimum distance,
        and maximum distance.

        Parameters:
            calibration_path (Path): The path to the calibration data file.
            min_distance (float): The minimum distance.
            max_distance (float): The maximum distance.

        Returns:
            None
        """
        super().__init__(calibration_path, min_distance, max_distance)

        self.firing_offset = [(1.512 * channel + 0.28) * 1e-6 for channel in range(32)]
        self.block_offset_single = [3.28 - 50.0 * (8 - block - 1) for block in range(8)]
        self.block_offset_dual = [3.28 - 50.0 * (8 - block - 1) / 2 for block in range(8)]

    def __call__(self, packet: pandar_msgs__msg__PandarPacket) -> Generator[Block, None, None]:
        """
        Processes the Pandar packet data to extract blocks of data with specific attributes.

        Parameters:
            packet (pandar_msgs__msg__PandarPacket): The Pandar packet containing lidar data.

        Yields:
            Block: A block of lidar data with associated attributes.
        """
        data: bytes = bytes(packet.data)

        sop_0, sop_1, protocol_version_major, protocol_version_minor, _ = struct.unpack_from(
            "! B B B B H", data, 0
        )

        if sop_0 != 0xEE or sop_1 != 0xFF:
            raise ValueError("Invalid start of packet")

        if protocol_version_major != 0x06 or protocol_version_minor != 0x01:
            raise ValueError("Unsupported protocol version")

        bodies_offset: int = 12
        tail_offset: int = bodies_offset + _NUM_BLOCKS * _BLOCK_SIZE  # 12 + 1040 = 1052

        (
            _,
            _,
            return_mode_,
            _,
            year_,
            month,
            day,
            hour,
            minute,
            second,
            *usecond_,
            _,
        ) = struct.unpack_from("! 9s B B H B B B B B B 4B B", data, tail_offset)

        if return_mode_ == 0x33:
            return_mode = ReturnMode.FIRST
        elif return_mode_ == 0x37:
            return_mode = ReturnMode.STRONGEST
        elif return_mode_ == 0x38:
            return_mode = ReturnMode.LAST
        elif return_mode_ == 0x39:
            return_mode = ReturnMode.LAST_STRONGEST
        elif return_mode_ == 0x3B:
            return_mode = ReturnMode.LAST_FIRST
        elif return_mode_ == 0x3C:
            return_mode = ReturnMode.FIRST_STRONGEST
        else:
            raise ValueError("Unsupported return mode")

        year = year_ + 1900
        usecond = int.from_bytes(bytes(usecond_), byteorder="little")
        timestamp = datetime(year, month, day, hour, minute, second, usecond)

        block_ids = range(0, _NUM_BLOCKS, 2) if return_mode.is_dual() else range(_NUM_BLOCKS)
        block_offset = self.block_offset_dual if return_mode.is_dual() else self.block_offset_single

        for seq, block_id in enumerate(block_ids):
            block_off = bodies_offset + block_id * _BLOCK_SIZE
            (azimuth,) = struct.unpack_from("< H", data, block_off)
            block = Block(azimuth, points=[])

            for ring in range(_NUM_CHANNELS):
                ch_off = block_off + 2 + ring * _CHANNEL_SIZE
                distance_, intensity, _ = struct.unpack_from("< H B B", data, ch_off)

                if distance_ == 0:
                    continue
                distance = distance_ * _DIS_UNIT_M
                if not self.min_distance <= distance < self.max_distance:
                    continue

                xy_distance = distance * np.cos(self.elevations[ring])
                x = xy_distance * np.sin(np.deg2rad(self.azimuths[ring] + azimuth * 0.01))
                y = xy_distance * np.cos(np.deg2rad(self.azimuths[ring] + azimuth * 0.01))
                z = distance * np.sin(self.elevations[ring])

                block.points.append(
                    (
                        x,
                        y,
                        z,
                        intensity,
                        ring,
                        azimuth,
                        timestamp.timestamp() + block_offset[seq] + self.firing_offset[ring],
                    )
                )

            yield block
