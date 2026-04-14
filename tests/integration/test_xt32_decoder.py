from __future__ import annotations

import math

import pytest

from hydra4 import Hydra
from hydra4.decoders import PandarXT32
from hydra4.structs import Block

_N_PACKETS = 20  # keep integration tests fast


@pytest.fixture
def decoder() -> PandarXT32:
    model = Hydra.Model.PandarXT32
    return PandarXT32(
        model.calibration_path, model.default_min_distance, model.default_max_distance
    )


@pytest.fixture
def sample_packets(pandar_packets_xt32: list) -> list:
    return pandar_packets_xt32[:_N_PACKETS]


@pytest.fixture
def sample_blocks(decoder: PandarXT32, sample_packets: list) -> list[Block]:
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


class TestXT32BlockStructure:
    def test_yields_block_instances(self, decoder: PandarXT32, sample_packets: list) -> None:
        for block in decoder(sample_packets[0]):
            assert isinstance(block, Block)

    def test_block_count_per_packet(self, decoder: PandarXT32, sample_packets: list) -> None:
        """XT32 yields 8 blocks (single-return) or 4 blocks (dual-return) per packet."""
        for packet in sample_packets[:5]:
            count = sum(1 for _ in decoder(packet))
            assert count in (4, 8), f"Unexpected block count {count}"

    def test_block_azimuth_in_range(self, sample_blocks: list[Block]) -> None:
        for block in sample_blocks:
            assert 0 <= block.azimuth <= 35999, f"Block azimuth {block.azimuth} out of range"

    def test_within_packet_azimuths_increase(
        self, decoder: PandarXT32, sample_packets: list
    ) -> None:
        """Within a single packet, consecutive block azimuths should increase."""
        for packet in sample_packets[:5]:
            azimuths = [b.azimuth for b in decoder(packet)]
            for i in range(1, len(azimuths)):
                diff = (azimuths[i] - azimuths[i - 1]) % 36000
                # A proper step is much less than half a rotation
                assert diff < 18000, f"Unexpected azimuth jump in packet: {azimuths}"


class TestXT32PointValues:
    def test_ring_indices_in_range(self, sample_points: list[tuple]) -> None:
        for _, _, _, _, ring, _, _ in sample_points:
            assert 0 <= ring <= 31, f"Ring {ring} out of [0, 31]"

    def test_xyz_are_finite(self, sample_points: list[tuple]) -> None:
        for x, y, z, _, _, _, _ in sample_points:
            assert math.isfinite(x) and math.isfinite(y) and math.isfinite(z)

    def test_intensity_in_valid_range(self, sample_points: list[tuple]) -> None:
        for _, _, _, intensity, _, _, _ in sample_points:
            assert 0 <= intensity <= 255, f"Intensity {intensity} out of [0, 255]"

    def test_distances_at_least_min_distance(self, sample_points: list[tuple]) -> None:
        min_dist = Hydra.Model.PandarXT32.default_min_distance
        for x, y, z, _, _, _, _ in sample_points:
            dist = math.sqrt(x**2 + y**2 + z**2)
            assert dist >= min_dist, f"Point distance {dist:.4f} < min {min_dist}"

    def test_distances_below_max_distance(self, sample_points: list[tuple]) -> None:
        max_dist = Hydra.Model.PandarXT32.default_max_distance
        for x, y, z, _, _, _, _ in sample_points:
            dist = math.sqrt(x**2 + y**2 + z**2)
            assert dist < max_dist, f"Point distance {dist:.4f} >= max {max_dist}"

    def test_no_zero_distance_points(self, sample_points: list[tuple]) -> None:
        """Raw distance=0 (invalid return) must never reach the output."""
        for x, y, z, _, _, _, _ in sample_points:
            assert x**2 + y**2 + z**2 > 0.0, "Zero-distance point leaked through filter"

    def test_point_azimuth_matches_block_azimuth(
        self, decoder: PandarXT32, sample_packets: list
    ) -> None:
        """Each point's stored azimuth equals the azimuth of its containing block."""
        for packet in sample_packets[:3]:
            for block in decoder(packet):
                for _, _, _, _, _, pt_az, _ in block.points:
                    assert pt_az == block.azimuth


class TestXT32DistanceFilter:
    def test_tighter_min_distance_filters_more(self, pandar_packets_xt32: list) -> None:
        model = Hydra.Model.PandarXT32
        default_decoder = PandarXT32(
            model.calibration_path, model.default_min_distance, model.default_max_distance
        )
        strict_decoder = PandarXT32(model.calibration_path, 1.0, model.default_max_distance)
        packets = pandar_packets_xt32[:_N_PACKETS]
        default_count = sum(len(b.points) for p in packets for b in default_decoder(p))
        strict_count = sum(len(b.points) for p in packets for b in strict_decoder(p))
        assert strict_count <= default_count

    def test_custom_max_distance_respected(self, pandar_packets_xt32: list) -> None:
        model = Hydra.Model.PandarXT32
        custom_max = 5.0
        decoder = PandarXT32(model.calibration_path, model.default_min_distance, custom_max)
        for packet in pandar_packets_xt32[:_N_PACKETS]:
            for block in decoder(packet):
                for x, y, z, *_ in block.points:
                    dist = math.sqrt(x**2 + y**2 + z**2)
                    assert dist < custom_max, (
                        f"Distance {dist:.4f} exceeded custom max {custom_max}"
                    )
