from __future__ import annotations

import math

import pytest

from hydra4 import Hydra
from hydra4.decoders import PandarOT128
from hydra4.structs import Block

_N_PACKETS = 20  # keep integration tests fast


@pytest.fixture
def decoder() -> PandarOT128:
    model = Hydra.Model.PandarOT128
    return PandarOT128(
        model.calibration_path, model.default_min_distance, model.default_max_distance
    )


@pytest.fixture
def sample_packets(pandar_packets_ot128: list) -> list:
    return pandar_packets_ot128[:_N_PACKETS]


@pytest.fixture
def sample_blocks(decoder: PandarOT128, sample_packets: list) -> list[Block]:
    blocks = []
    for packet in sample_packets:
        blocks.extend(decoder(packet))
    return blocks


@pytest.fixture
def sample_points(sample_blocks: list[Block]) -> list[tuple]:
    pts: list[tuple] = []
    for block in sample_blocks:
        pts.extend(block.points)
    return pts


class TestOT128BlockStructure:
    def test_yields_block_instances(self, decoder: PandarOT128, sample_packets: list) -> None:
        for block in decoder(sample_packets[0]):
            assert isinstance(block, Block)

    def test_dual_return_yields_two_blocks_per_packet(
        self, decoder: PandarOT128, sample_packets: list
    ) -> None:
        """In dual-return mode the OT128 decoder yields exactly 2 blocks per packet."""
        for packet in sample_packets[:5]:
            count = sum(1 for _ in decoder(packet))
            assert count == 2, f"Expected 2 blocks, got {count}"

    def test_dual_return_blocks_share_azimuth(
        self, decoder: PandarOT128, sample_packets: list
    ) -> None:
        """Both blocks from a dual-return packet carry the same azimuth value."""
        for packet in sample_packets[:5]:
            blocks = list(decoder(packet))
            assert len(blocks) == 2
            assert blocks[0].azimuth == blocks[1].azimuth, (
                f"Blocks have different azimuths: {blocks[0].azimuth} vs {blocks[1].azimuth}"
            )

    def test_block_azimuth_in_range(self, sample_blocks: list[Block]) -> None:
        for block in sample_blocks:
            assert 0 <= block.azimuth <= 35999, f"Block azimuth {block.azimuth} out of range"

    def test_consecutive_packet_azimuths_non_decreasing(
        self, decoder: PandarOT128, sample_packets: list
    ) -> None:
        """Azimuth should increase from packet to packet (within same rotation)."""
        prev_az = None
        for packet in sample_packets[:10]:
            blocks = list(decoder(packet))
            az = blocks[0].azimuth
            if prev_az is not None:
                diff = (az - prev_az) % 36000
                assert diff < 18000, f"Unexpected azimuth decrease: {prev_az} → {az}"
            prev_az = az


class TestOT128PointValues:
    def test_ring_indices_in_range(self, sample_points: list[tuple]) -> None:
        for _, _, _, _, ring, _, _ in sample_points:
            assert 0 <= ring <= 127, f"Ring {ring} out of [0, 127]"

    def test_xyz_are_finite(self, sample_points: list[tuple]) -> None:
        for x, y, z, _, _, _, _ in sample_points:
            assert math.isfinite(x) and math.isfinite(y) and math.isfinite(z)

    def test_intensity_in_valid_range(self, sample_points: list[tuple]) -> None:
        for _, _, _, intensity, _, _, _ in sample_points:
            assert 0 <= intensity <= 255, f"Intensity {intensity} out of [0, 255]"

    def test_distances_at_least_min_distance(self, sample_points: list[tuple]) -> None:
        min_dist = Hydra.Model.PandarOT128.default_min_distance
        for x, y, z, _, _, _, _ in sample_points:
            dist = math.sqrt(x**2 + y**2 + z**2)
            assert dist >= min_dist, f"Point distance {dist:.4f} < min {min_dist}"

    def test_distances_below_max_distance(self, sample_points: list[tuple]) -> None:
        max_dist = Hydra.Model.PandarOT128.default_max_distance
        for x, y, z, _, _, _, _ in sample_points:
            dist = math.sqrt(x**2 + y**2 + z**2)
            assert dist < max_dist, f"Point distance {dist:.4f} >= max {max_dist}"

    def test_no_zero_distance_points(self, sample_points: list[tuple]) -> None:
        for x, y, z, _, _, _, _ in sample_points:
            assert x**2 + y**2 + z**2 > 0.0, "Zero-distance point leaked through filter"

    def test_point_azimuth_matches_block_azimuth(
        self, decoder: PandarOT128, sample_packets: list
    ) -> None:
        for packet in sample_packets[:3]:
            for block in decoder(packet):
                for _, _, _, _, _, pt_az, _ in block.points:
                    assert pt_az == block.azimuth

    def test_both_returns_present(self, decoder: PandarOT128, sample_packets: list) -> None:
        """First and last return blocks should both have points (at close range)."""
        nonempty_pairs = 0
        for packet in sample_packets[:10]:
            blocks = list(decoder(packet))
            if len(blocks[0].points) > 0 and len(blocks[1].points) > 0:
                nonempty_pairs += 1
        assert nonempty_pairs > 0, "No packets had points in both return blocks"


class TestOT128DistanceFilter:
    def test_tighter_min_distance_filters_more(self, pandar_packets_ot128: list) -> None:
        model = Hydra.Model.PandarOT128
        default_d = PandarOT128(
            model.calibration_path, model.default_min_distance, model.default_max_distance
        )
        strict_d = PandarOT128(model.calibration_path, 1.0, model.default_max_distance)
        packets = pandar_packets_ot128[:_N_PACKETS]
        default_count = sum(len(b.points) for p in packets for b in default_d(p))
        strict_count = sum(len(b.points) for p in packets for b in strict_d(p))
        assert strict_count <= default_count
