from __future__ import annotations

import numpy as np

from hydra4 import Hydra

# ─── helpers ─────────────────────────────────────────────────────────────────


def _all_frames(hydra: Hydra, scans: list) -> list:
    return [pc for msg in scans for pc in hydra.to_pypcd4(msg)]


def _total_points(hydra: Hydra, scans: list) -> int:
    return sum(pc.points for pc in _all_frames(hydra, scans))


# ─── PandarOT128 E2E ─────────────────────────────────────────────────────────


class TestPandarOT128:
    # Point counts with default dual-return deduplication (threshold=0.1 m).
    _EXPECTED_DEDUP = (70612, 72241)
    # Point counts without deduplication (threshold=0.0 m): both returns kept.
    _EXPECTED_NO_DEDUP = (139086, 142307)

    def test_frame_count(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        assert len(_all_frames(hydra, pandar_scans_ot128)) == 2

    def test_point_counts_with_dedup(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128, dual_return_distance_threshold=0.1)
        for i, pc in enumerate(_all_frames(hydra, pandar_scans_ot128)):
            assert pc.points == self._EXPECTED_DEDUP[i], (
                f"Frame {i}: got {pc.points}, expected {self._EXPECTED_DEDUP[i]}"
            )

    def test_point_counts_without_dedup(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128, dual_return_distance_threshold=0.0)
        for i, pc in enumerate(_all_frames(hydra, pandar_scans_ot128)):
            assert pc.points == self._EXPECTED_NO_DEDUP[i], (
                f"Frame {i}: got {pc.points}, expected {self._EXPECTED_NO_DEDUP[i]}"
            )

    def test_dedup_reduces_point_count(self, pandar_scans_ot128: list) -> None:
        pts_dedup = _total_points(
            Hydra(Hydra.Model.PandarOT128, dual_return_distance_threshold=0.1),
            pandar_scans_ot128,
        )
        pts_no_dedup = _total_points(
            Hydra(Hydra.Model.PandarOT128, dual_return_distance_threshold=0.0),
            pandar_scans_ot128,
        )
        assert pts_dedup < pts_no_dedup

    def test_dedup_keeps_roughly_half(self, pandar_scans_ot128: list) -> None:
        """After dedup, point count should be roughly half of the no-dedup count."""
        pts_dedup = _total_points(
            Hydra(Hydra.Model.PandarOT128, dual_return_distance_threshold=0.1),
            pandar_scans_ot128,
        )
        pts_no_dedup = _total_points(
            Hydra(Hydra.Model.PandarOT128, dual_return_distance_threshold=0.0),
            pandar_scans_ot128,
        )
        ratio = pts_dedup / pts_no_dedup
        assert 0.45 <= ratio <= 0.65, f"Unexpected dedup ratio: {ratio:.3f}"

    def test_metadata_fields(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        pc = _all_frames(hydra, pandar_scans_ot128)[0]
        assert pc.metadata.fields == ("x", "y", "z", "intensity", "ring", "azimuth", "stamp")

    def test_metadata_sizes(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        pc = _all_frames(hydra, pandar_scans_ot128)[0]
        assert pc.metadata.size == (4, 4, 4, 4, 2, 4, 8)

    def test_metadata_types(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        pc = _all_frames(hydra, pandar_scans_ot128)[0]
        assert pc.metadata.type == ("F", "F", "F", "F", "U", "F", "F")

    def test_xyz_all_finite(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        for pc in _all_frames(hydra, pandar_scans_ot128):
            data = pc.pc_data
            assert np.all(np.isfinite(data["x"]))
            assert np.all(np.isfinite(data["y"]))
            assert np.all(np.isfinite(data["z"]))

    def test_ring_indices_in_range(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        for pc in _all_frames(hydra, pandar_scans_ot128):
            rings = pc.pc_data["ring"].astype(np.int32)
            assert int(rings.min()) >= 0
            assert int(rings.max()) <= 127

    def test_distances_within_model_limits(self, pandar_scans_ot128: list) -> None:
        model = Hydra.Model.PandarOT128
        hydra = Hydra(model)
        # XYZ is stored as float32; reconstructed distances may be slightly below the
        # decoder's float64 threshold due to rounding. Allow 1e-4 m tolerance.
        tol = 1e-4
        for pc in _all_frames(hydra, pandar_scans_ot128):
            d = pc.pc_data
            dists = np.sqrt(
                d["x"].astype(np.float64) ** 2
                + d["y"].astype(np.float64) ** 2
                + d["z"].astype(np.float64) ** 2
            )
            assert float(dists.min()) >= model.default_min_distance - tol
            assert float(dists.max()) < model.default_max_distance

    def test_intensity_in_range(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        for pc in _all_frames(hydra, pandar_scans_ot128):
            intensities = pc.pc_data["intensity"]
            assert float(intensities.min()) >= 0.0
            assert float(intensities.max()) <= 255.0

    def test_to_pypcd4_returns_list(self, pandar_scans_ot128: list) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        result = hydra.to_pypcd4(pandar_scans_ot128[0])
        assert isinstance(result, list)

    def test_tighter_min_distance_reduces_points(self, pandar_scans_ot128: list) -> None:
        pts_default = _total_points(Hydra(Hydra.Model.PandarOT128), pandar_scans_ot128)
        pts_strict = _total_points(
            Hydra(Hydra.Model.PandarOT128, min_distance=1.0), pandar_scans_ot128
        )
        assert pts_strict <= pts_default
