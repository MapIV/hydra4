from __future__ import annotations

import struct
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np

from ..msgs import pandar_msgs__msg__PandarPacket
from ..structs import Block, ReturnMode
from .base import PandarBase

# fmt: off
# Appendix B.4: Laser firing time offsets Δt(n) [μs], indexed [channel_0_indexed][azimuth_state].
# High Resolution mode — Azimuth States 0–3.
# None indicates the channel does not fire in that azimuth state (distance will be 0).
_FIRING_OFFSET_HIGH_RES: tuple[tuple[float | None, ...], ...] = (
    # ch  1
    (None,   18.867, None,   18.867),
    # ch  2
    (None,    6.289, None,    6.289),
    # ch  3
    (18.867,  None,  21.011,  None),
    # ch  4
    ( 6.289,  None,   6.289,  None),
    # ch  5
    (None,   12.578, None,   12.578),
    # ch  6
    (None,    0.0,   None,    0.0),
    # ch  7
    (12.578,  None,  14.722,  None),
    # ch  8
    ( 0.0,    None,   0.0,    None),
    # ch  9
    (None,   18.867, None,   18.867),
    # ch 10
    (None,    6.289, None,    6.289),
    # ch 11
    (18.867,  None,  21.011,  None),
    # ch 12
    ( 6.289,  None,   6.289,  None),
    # ch 13
    (None,   12.578, None,   12.578),
    # ch 14
    (None,    0.0,   None,    0.0),
    # ch 15
    (12.578,  None,  14.722,  None),
    # ch 16
    ( 0.0,    None,   0.0,    None),
    # ch 17
    (None,   18.867, None,   18.867),
    # ch 18
    (None,    6.289, None,    6.289),
    # ch 19
    (18.867,  None,  21.011,  None),
    # ch 20
    ( 6.289,  None,   6.289,  None),
    # ch 21
    (None,   12.578, None,   12.578),
    # ch 22
    (None,    0.0,   None,    0.0),
    # ch 23
    (12.578,  None,  14.722,  None),
    # ch 24
    ( 0.0,    None,   0.0,    None),
    # ch 25
    (20.52,  20.52,  22.664, 20.52),
    # ch 26
    (16.549, 16.549, 18.693, 16.549),
    # ch 27
    (10.26,  10.26,  10.26,  10.26),
    # ch 28
    (16.549, 16.549, 18.693, 16.549),
    # ch 29
    (20.52,  20.52,  22.664, 20.52),
    # ch 30
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 31
    (14.231, 14.231, 16.375, 14.231),
    # ch 32
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 33
    (14.231, 14.231, 16.375, 14.231),
    # ch 34
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 35
    (10.26,  10.26,  10.26,  10.26),
    # ch 36
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 37
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 38
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 39
    (22.838, 22.838, 24.982, 22.838),
    # ch 40
    (22.838, 22.838, 24.982, 22.838),
    # ch 41
    (14.231, 14.231, 16.375, 14.231),
    # ch 42
    (16.549, 16.549, 18.693, 16.549),
    # ch 43
    (20.52,  20.52,  22.664, 20.52),
    # ch 44
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 45
    (10.26,  10.26,  10.26,  10.26),
    # ch 46
    (16.549, 16.549, 18.693, 16.549),
    # ch 47
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 48
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 49
    (10.26,  10.26,  10.26,  10.26),
    # ch 50
    (22.838, 22.838, 24.982, 22.838),
    # ch 51
    (14.231, 14.231, 16.375, 14.231),
    # ch 52
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 53
    (20.52,  20.52,  22.664, 20.52),
    # ch 54
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 55
    (14.231, 14.231, 16.375, 14.231),
    # ch 56
    (16.549, 16.549, 18.693, 16.549),
    # ch 57
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 58
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 59
    (10.26,  10.26,  10.26,  10.26),
    # ch 60
    (22.838, 22.838, 24.982, 22.838),
    # ch 61
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 62
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 63
    (20.52,  20.52,  22.664, 20.52),
    # ch 64
    (22.838, 22.838, 24.982, 22.838),
    # ch 65
    (14.231, 14.231, 16.375, 14.231),
    # ch 66
    (16.549, 16.549, 18.693, 16.549),
    # ch 67
    (20.52,  20.52,  22.664, 20.52),
    # ch 68
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 69
    (10.26,  10.26,  10.26,  10.26),
    # ch 70
    (16.549, 16.549, 18.693, 16.549),
    # ch 71
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 72
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 73
    (10.26,  10.26,  10.26,  10.26),
    # ch 74
    (22.838, 22.838, 24.982, 22.838),
    # ch 75
    (14.231, 14.231, 16.375, 14.231),
    # ch 76
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 77
    (20.52,  20.52,  22.664, 20.52),
    # ch 78
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 79
    (14.231, 14.231, 16.375, 14.231),
    # ch 80
    (16.549, 16.549, 18.693, 16.549),
    # ch 81
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 82
    ( 7.942,  7.942,  7.942,  7.942),
    # ch 83
    (10.26,  10.26,  10.26,  10.26),
    # ch 84
    (22.838, 22.838, 24.982, 22.838),
    # ch 85
    ( 1.653,  1.653,  1.653,  1.653),
    # ch 86
    ( 3.971,  3.971,  3.971,  3.971),
    # ch 87
    (20.52,  20.52,  22.664, 20.52),
    # ch 88
    (22.838, 22.838, 24.982, 22.838),
    # ch 89
    (None,   18.867, None,   18.867),
    # ch 90
    (None,    6.289, None,    6.289),
    # ch 91
    (18.867,  None,  21.011,  None),
    # ch 92
    ( 6.289,  None,   6.289,  None),
    # ch 93
    (None,   12.578, None,   12.578),
    # ch 94
    (None,    0.0,   None,    0.0),
    # ch 95
    (12.578,  None,  14.722,  None),
    # ch 96
    ( 0.0,    None,   0.0,    None),
    # ch 97
    (None,   18.867, None,   18.867),
    # ch 98
    (None,    6.289, None,    6.289),
    # ch 99
    (18.867,  None,  21.011,  None),
    # ch100
    ( 6.289,  None,   6.289,  None),
    # ch101
    (None,   12.578, None,   12.578),
    # ch102
    (None,    0.0,   None,    0.0),
    # ch103
    (12.578,  None,  14.722,  None),
    # ch104
    ( 0.0,    None,   0.0,    None),
    # ch105
    (None,   18.867, None,   18.867),
    # ch106
    (None,    6.289, None,    6.289),
    # ch107
    (18.867,  None,  21.011,  None),
    # ch108
    ( 6.289,  None,   6.289,  None),
    # ch109
    (None,   12.578, None,   12.578),
    # ch110
    (None,    0.0,   None,    0.0),
    # ch111
    (12.578,  None,  14.722,  None),
    # ch112
    ( 0.0,    None,   0.0,    None),
    # ch113
    (None,   18.867, None,   18.867),
    # ch114
    (None,    6.289, None,    6.289),
    # ch115
    (18.867,  None,  21.011,  None),
    # ch116
    ( 6.289,  None,   6.289,  None),
    # ch117
    (None,   12.578, None,   12.578),
    # ch118
    (None,    0.0,   None,    0.0),
    # ch119
    (12.578,  None,  14.722,  None),
    # ch120
    ( 0.0,    None,   0.0,    None),
    # ch121
    (None,   18.867, None,   18.867),
    # ch122
    (None,    6.289, None,    6.289),
    # ch123
    (18.867,  None,  21.011,  None),
    # ch124
    ( 6.289,  None,   6.289,  None),
    # ch125
    (None,   12.578, None,   12.578),
    # ch126
    (None,    0.0,   None,    0.0),
    # ch127
    (12.578,  None,  14.722,  None),
    # ch128
    ( 0.0,    None,   0.0,    None),
)

# Appendix B.4: Standard mode — Azimuth States 0–1.
_FIRING_OFFSET_STANDARD: tuple[tuple[float, float], ...] = (
    # ch  1
    (46.645, 46.645),
    # ch  2
    (34.067, 34.067),
    # ch  3
    (18.867, 21.011),
    # ch  4
    ( 6.289,  6.289),
    # ch  5
    (40.356, 40.356),
    # ch  6
    (27.778, 27.778),
    # ch  7
    (12.578, 14.722),
    # ch  8
    ( 0.0,    0.0),
    # ch  9
    (46.645, 46.645),
    # ch 10
    (34.067, 34.067),
    # ch 11
    (18.867, 21.011),
    # ch 12
    ( 6.289,  6.289),
    # ch 13
    (40.356, 40.356),
    # ch 14
    (27.778, 27.778),
    # ch 15
    (12.578, 14.722),
    # ch 16
    ( 0.0,    0.0),
    # ch 17
    (46.645, 46.645),
    # ch 18
    (34.067, 34.067),
    # ch 19
    (18.867, 21.011),
    # ch 20
    ( 6.289,  6.289),
    # ch 21
    (40.356, 40.356),
    # ch 22
    (27.778, 27.778),
    # ch 23
    (12.578, 14.722),
    # ch 24
    ( 0.0,    0.0),
    # ch 25
    (20.52,  22.664),
    # ch 26
    (16.549, 18.693),
    # ch 27
    (10.26,  10.26),
    # ch 28
    (16.549, 18.693),
    # ch 29
    (20.52,  22.664),
    # ch 30
    ( 3.971,  3.971),
    # ch 31
    (14.231, 16.375),
    # ch 32
    ( 7.942,  7.942),
    # ch 33
    (14.231, 16.375),
    # ch 34
    ( 7.942,  7.942),
    # ch 35
    (10.26,  10.26),
    # ch 36
    ( 1.653,  1.653),
    # ch 37
    ( 1.653,  1.653),
    # ch 38
    ( 3.971,  3.971),
    # ch 39
    (22.838, 24.982),
    # ch 40
    (22.838, 24.982),
    # ch 41
    (14.231, 16.375),
    # ch 42
    (16.549, 18.693),
    # ch 43
    (20.52,  22.664),
    # ch 44
    ( 7.942,  7.942),
    # ch 45
    (10.26,  10.26),
    # ch 46
    (16.549, 18.693),
    # ch 47
    ( 1.653,  1.653),
    # ch 48
    ( 3.971,  3.971),
    # ch 49
    (10.26,  10.26),
    # ch 50
    (22.838, 24.982),
    # ch 51
    (14.231, 16.375),
    # ch 52
    ( 3.971,  3.971),
    # ch 53
    (20.52,  22.664),
    # ch 54
    ( 7.942,  7.942),
    # ch 55
    (14.231, 16.375),
    # ch 56
    (16.549, 18.693),
    # ch 57
    ( 1.653,  1.653),
    # ch 58
    ( 7.942,  7.942),
    # ch 59
    (10.26,  10.26),
    # ch 60
    (22.838, 24.982),
    # ch 61
    ( 1.653,  1.653),
    # ch 62
    ( 3.971,  3.971),
    # ch 63
    (20.52,  22.664),
    # ch 64
    (22.838, 24.982),
    # ch 65
    (14.231, 16.375),
    # ch 66
    (16.549, 18.693),
    # ch 67
    (20.52,  22.664),
    # ch 68
    ( 7.942,  7.942),
    # ch 69
    (10.26,  10.26),
    # ch 70
    (16.549, 18.693),
    # ch 71
    ( 1.653,  1.653),
    # ch 72
    ( 3.971,  3.971),
    # ch 73
    (10.26,  10.26),
    # ch 74
    (22.838, 24.982),
    # ch 75
    (14.231, 16.375),
    # ch 76
    ( 3.971,  3.971),
    # ch 77
    (20.52,  22.664),
    # ch 78
    ( 7.942,  7.942),
    # ch 79
    (14.231, 16.375),
    # ch 80
    (16.549, 18.693),
    # ch 81
    ( 1.653,  1.653),
    # ch 82
    ( 7.942,  7.942),
    # ch 83
    (10.26,  10.26),
    # ch 84
    (22.838, 24.982),
    # ch 85
    ( 1.653,  1.653),
    # ch 86
    ( 3.971,  3.971),
    # ch 87
    (20.52,  22.664),
    # ch 88
    (22.838, 24.982),
    # ch 89
    (46.645, 46.645),
    # ch 90
    (34.067, 34.067),
    # ch 91
    (18.867, 21.011),
    # ch 92
    ( 6.289,  6.289),
    # ch 93
    (40.356, 40.356),
    # ch 94
    (27.778, 27.778),
    # ch 95
    (12.578, 14.722),
    # ch 96
    ( 0.0,    0.0),
    # ch 97
    (46.645, 46.645),
    # ch 98
    (34.067, 34.067),
    # ch 99
    (18.867, 21.011),
    # ch100
    ( 6.289,  6.289),
    # ch101
    (40.356, 40.356),
    # ch102
    (27.778, 27.778),
    # ch103
    (12.578, 14.722),
    # ch104
    ( 0.0,    0.0),
    # ch105
    (46.645, 46.645),
    # ch106
    (34.067, 34.067),
    # ch107
    (18.867, 21.011),
    # ch108
    ( 6.289,  6.289),
    # ch109
    (40.356, 40.356),
    # ch110
    (27.778, 27.778),
    # ch111
    (12.578, 14.722),
    # ch112
    ( 0.0,    0.0),
    # ch113
    (46.645, 46.645),
    # ch114
    (34.067, 34.067),
    # ch115
    (18.867, 21.011),
    # ch116
    ( 6.289,  6.289),
    # ch117
    (40.356, 40.356),
    # ch118
    (27.778, 27.778),
    # ch119
    (12.578, 14.722),
    # ch120
    ( 0.0,    0.0),
    # ch121
    (46.645, 46.645),
    # ch122
    (34.067, 34.067),
    # ch123
    (18.867, 21.011),
    # ch124
    ( 6.289,  6.289),
    # ch125
    (40.356, 40.356),
    # ch126
    (27.778, 27.778),
    # ch127
    (12.578, 14.722),
    # ch128
    ( 0.0,    0.0),
)
# fmt: on

_NUM_CHANNELS: int = 128
_DIS_UNIT_M: float = 0.004  # Dis Unit fixed at 4 mm (Section 3.1.2.2)

# Tail field byte offsets (relative to tail start)
_TAIL_OFF_AZ_STATE: int = 9  # Azimuth State (2 bytes, little-endian)
_TAIL_OFF_OP_STATE: int = 11  # Operational State (1 byte)
_TAIL_OFF_RET_MODE: int = 12  # Return Mode (1 byte)
_TAIL_OFF_DATETIME: int = 15  # Year … Second (6 bytes)
_TAIL_OFF_TIMESTAMP: int = 21  # Timestamp μs (4 bytes, little-endian)

_RETURN_MODE_MAP: dict[int, ReturnMode] = {
    0x33: ReturnMode.FIRST,
    0x37: ReturnMode.STRONGEST,
    0x38: ReturnMode.LAST,
    0x39: ReturnMode.LAST_STRONGEST,
    0x3B: ReturnMode.LAST_FIRST,
    0x3C: ReturnMode.FIRST_STRONGEST,
}


class PandarOT128(PandarBase):
    def __init__(self, calibration_path: Path, min_distance: float, max_distance: float) -> None:
        """
        Initializes the PandarOT128 decoder.

        Parameters:
            calibration_path (Path): Path to the 128-channel calibration CSV.
            min_distance (float): Minimum valid distance in metres.
            max_distance (float): Maximum valid distance in metres.
        """
        super().__init__(calibration_path, min_distance, max_distance)

    def __call__(self, packet: pandar_msgs__msg__PandarPacket) -> Generator[Block, None, None]:
        """
        Decodes one PandarOT128 UDP packet into a sequence of Blocks.

        The packet layout (Section 3.1.2, little-endian throughout):
          Pre-Header : 6 bytes  (0xEE 0xFF 0x01 0x04 reserved×2)
          Header     : 6 bytes  (channel_num block_num fbr dis_unit return_num flags)
          Body       : 776 bytes (noise OFF) / 1032 bytes (noise ON)
                         azimuth_1(2) + block_1(128×3or4) +
                         azimuth_2(2) + block_2(128×3or4) + CRC(4)
          Func-Safety: 17 bytes (optional, present when flags bit[2] = 1)
          Tail       : 56 bytes

        Firing times follow Appendix B.3 (block start) + B.4 (channel offset).

        Parameters:
            packet (pandar_msgs__msg__PandarPacket): Raw UDP packet.

        Yields:
            Block: One Block per body block (2 per packet).
        """
        data: bytes = bytes(packet.data)

        # ── Pre-Header (offset 0, 6 bytes) ──────────────────────────────────
        sop_0, sop_1, proto_major, proto_minor = struct.unpack_from("< BBBB", data, 0)
        if sop_0 != 0xEE or sop_1 != 0xFF:
            raise ValueError(f"Invalid SOP bytes: 0x{sop_0:02X} 0x{sop_1:02X}")
        if proto_major != 0x01 or proto_minor != 0x04:
            raise ValueError(f"Unsupported protocol version: {proto_major}.{proto_minor}")

        # ── Header (offset 6, 6 bytes) ───────────────────────────────────────
        flags: int = data[11]  # 6th byte of header
        discrete_noise: bool = bool(flags & 0x20)  # bit[5]
        has_func_safety: bool = bool(flags & 0x04)  # bit[2]

        bytes_per_ch: int = 4 if discrete_noise else 3
        block_data_size: int = _NUM_CHANNELS * bytes_per_ch  # 384 or 512

        # ── Body layout (offset 12) ──────────────────────────────────────────
        # azimuth_1(2) | block_1(block_data_size) | azimuth_2(2) | block_2(block_data_size) | CRC(4)
        body_offset: int = 12
        tail_offset: int = body_offset + 2 + block_data_size + 2 + block_data_size + 4
        if has_func_safety:
            tail_offset += 17  # Section 3.1.2.4: 17 bytes

        # ── Tail (56 bytes) ──────────────────────────────────────────────────
        azimuth_state_word: int = struct.unpack_from("< H", data, tail_offset + _TAIL_OFF_AZ_STATE)[
            0
        ]
        # bits [15:14] = Block 1 azimuth state, [13:12] = Block 2 azimuth state
        az_states: tuple[int, int] = (
            (azimuth_state_word >> 14) & 0x3,
            (azimuth_state_word >> 12) & 0x3,
        )

        operational_state: int = data[tail_offset + _TAIL_OFF_OP_STATE]  # 0=HighRes, 2=Standard
        return_mode_byte: int = data[tail_offset + _TAIL_OFF_RET_MODE]

        year_: int
        month: int
        day: int
        hour: int
        minute: int
        second: int
        year_, month, day, hour, minute, second = struct.unpack_from(
            "< 6B", data, tail_offset + _TAIL_OFF_DATETIME
        )
        timestamp_us: int = struct.unpack_from("< I", data, tail_offset + _TAIL_OFF_TIMESTAMP)[0]

        year: int = year_ + 1900
        t0: float = datetime(year, month, day, hour, minute, second, timestamp_us % 1_000_000).timestamp()

        # ── Return mode ──────────────────────────────────────────────────────
        if return_mode_byte not in _RETURN_MODE_MAP:
            raise ValueError(f"Unsupported return mode: 0x{return_mode_byte:02X}")
        return_mode: ReturnMode = _RETURN_MODE_MAP[return_mode_byte]
        is_dual: bool = return_mode.is_dual()
        is_high_res: bool = operational_state == 0

        # ── Block start times (Appendix B.3) ─────────────────────────────────
        # Single return — High Resolution: Block1 = t0−27.778 μs, Block2 = t0
        # Single return — Standard:        Block1 = t0−55.556 μs, Block2 = t0
        # Dual return:                     Both blocks = t0
        if is_dual:
            block_start_offsets: tuple[float, float] = (0.0, 0.0)
        elif is_high_res:
            block_start_offsets = (-27.778e-6, 0.0)
        else:
            block_start_offsets = (-55.556e-6, 0.0)

        # ── Firing offset table (Appendix B.4) ───────────────────────────────
        firing_table: tuple[tuple[float | None, ...], ...] | tuple[tuple[float, float], ...] = (
            _FIRING_OFFSET_HIGH_RES if is_high_res else _FIRING_OFFSET_STANDARD
        )

        # ── Parse body blocks ────────────────────────────────────────────────
        pos: int = body_offset
        for blk in range(2):
            azimuth: int = struct.unpack_from("< H", data, pos)[0]
            pos += 2

            block: Block = Block(azimuth=azimuth, points=[])
            block_t: float = t0 + block_start_offsets[blk]
            az_state: int = az_states[blk]

            for ring in range(_NUM_CHANNELS):
                if bytes_per_ch == 3:
                    distance_raw, intensity = struct.unpack_from("< HB", data, pos)
                else:  # 4 bytes: distance(2) + reflectivity(1) + weight_factor(1)
                    distance_raw, intensity, _ = struct.unpack_from("< HBB", data, pos)
                pos += bytes_per_ch

                if distance_raw == 0:  # No valid return
                    continue

                distance: float = distance_raw * _DIS_UNIT_M
                if not (self.min_distance <= distance <= self.max_distance):
                    continue

                az_deg: float = azimuth * 0.01 + self.azimuths[ring]
                xy_distance: float = distance * np.cos(self.elevations[ring])
                x: float = float(xy_distance * np.sin(np.deg2rad(az_deg)))
                y: float = float(xy_distance * np.cos(np.deg2rad(az_deg)))
                z: float = float(distance * np.sin(self.elevations[ring]))

                dt: float | None = firing_table[ring][az_state]
                firing_time: float = block_t + (dt if dt is not None else 0.0) * 1e-6

                block.points.append((x, y, z, intensity, ring, azimuth, firing_time))

            yield block
