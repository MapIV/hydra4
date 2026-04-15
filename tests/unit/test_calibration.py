from __future__ import annotations

import math

import pytest

from hydra4 import Hydra
from hydra4.decoders import PandarOT128, PandarXT32


class TestXT32Calibration:
    @pytest.fixture
    def decoder(self) -> PandarXT32:
        model = Hydra.Model.PandarXT32
        return PandarXT32(
            model.calibration_path, model.default_min_distance, model.default_max_distance
        )

    def test_channel_count(self, decoder: PandarXT32) -> None:
        assert len(decoder.azimuths) == 32
        assert len(decoder.elevations) == 32

    def test_elevations_are_in_radians(self, decoder: PandarXT32) -> None:
        """Elevations must be in radians and within ±90°."""
        for el in decoder.elevations:
            assert -math.pi / 2 <= el <= math.pi / 2, f"Elevation {el:.4f} rad out of ±π/2"

    def test_elevations_span_expected_range(self, decoder: PandarXT32) -> None:
        """XT32 covers roughly −16° to +15°."""
        min_deg = math.degrees(min(decoder.elevations))
        max_deg = math.degrees(max(decoder.elevations))
        assert min_deg < -10.0
        assert max_deg > 10.0

    def test_elevations_strictly_decreasing(self, decoder: PandarXT32) -> None:
        """Channels are ordered top-to-bottom (decreasing elevation)."""
        elevs = decoder.elevations
        for i in range(len(elevs) - 1):
            assert elevs[i] > elevs[i + 1], f"Channel {i+1} elevation not > channel {i+2}"

    def test_azimuth_offsets_are_small(self, decoder: PandarXT32) -> None:
        """Azimuth correction values are small (< ±5°)."""
        for az in decoder.azimuths:
            assert -5.0 <= az <= 5.0, f"Azimuth offset {az}° unusually large"

    def test_min_distance_stored(self, decoder: PandarXT32) -> None:
        assert decoder.min_distance == pytest.approx(Hydra.Model.PandarXT32.default_min_distance)

    def test_max_distance_stored(self, decoder: PandarXT32) -> None:
        assert decoder.max_distance == pytest.approx(Hydra.Model.PandarXT32.default_max_distance)


class TestOT128Calibration:
    @pytest.fixture
    def decoder(self) -> PandarOT128:
        model = Hydra.Model.PandarOT128
        return PandarOT128(
            model.calibration_path, model.default_min_distance, model.default_max_distance
        )

    def test_channel_count(self, decoder: PandarOT128) -> None:
        assert len(decoder.azimuths) == 128
        assert len(decoder.elevations) == 128

    def test_elevations_are_in_radians(self, decoder: PandarOT128) -> None:
        for el in decoder.elevations:
            assert -math.pi / 2 <= el <= math.pi / 2, f"Elevation {el:.4f} rad out of ±π/2"

    def test_elevations_span_expected_range(self, decoder: PandarOT128) -> None:
        """OT128 covers roughly −24.7° to +15.1°."""
        min_deg = math.degrees(min(decoder.elevations))
        max_deg = math.degrees(max(decoder.elevations))
        assert min_deg < -20.0
        assert max_deg > 10.0

    def test_azimuth_offsets_are_small(self, decoder: PandarOT128) -> None:
        for az in decoder.azimuths:
            assert -5.0 <= az <= 5.0, f"Azimuth offset {az}° unusually large"

    def test_min_distance_stored(self, decoder: PandarOT128) -> None:
        assert decoder.min_distance == pytest.approx(Hydra.Model.PandarOT128.default_min_distance)

    def test_max_distance_stored(self, decoder: PandarOT128) -> None:
        assert decoder.max_distance == pytest.approx(Hydra.Model.PandarOT128.default_max_distance)
