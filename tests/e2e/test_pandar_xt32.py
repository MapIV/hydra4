from __future__ import annotations

import numpy as np

from hydra4 import Hydra

# ─── helpers ─────────────────────────────────────────────────────────────────


def _all_frames(hydra: Hydra, scans: list) -> list:
    return [pc for msg in scans for pc in hydra.to_pypcd4(msg)]


def _total_points(hydra: Hydra, scans: list) -> int:
    return sum(pc.points for pc in _all_frames(hydra, scans))


# ─── PandarXT32 E2E ──────────────────────────────────────────────────────────


class TestPandarXT32:
    # Counts reflect corrected frame-boundary accumulation (no boundary-block double-count).
    _EXPECTED = (60644, 60815, 60828, 60824)

    def test_frame_count(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        assert len(_all_frames(hydra, pandar_scans_xt32)) == len(self._EXPECTED)

    def test_point_counts_per_frame(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        for i, pc in enumerate(_all_frames(hydra, pandar_scans_xt32)):
            assert pc.points == self._EXPECTED[i], (
                f"Frame {i}: got {pc.points}, expected {self._EXPECTED[i]}"
            )

    def test_metadata_fields(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        pc = _all_frames(hydra, pandar_scans_xt32)[0]
        assert pc.metadata.fields == ("x", "y", "z", "intensity", "ring", "azimuth", "stamp")

    def test_metadata_sizes(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        pc = _all_frames(hydra, pandar_scans_xt32)[0]
        assert pc.metadata.size == (4, 4, 4, 4, 2, 4, 8)

    def test_metadata_types(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        pc = _all_frames(hydra, pandar_scans_xt32)[0]
        assert pc.metadata.type == ("F", "F", "F", "F", "U", "F", "F")

    def test_xyz_all_finite(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        for pc in _all_frames(hydra, pandar_scans_xt32):
            data = pc.pc_data
            assert np.all(np.isfinite(data["x"]))
            assert np.all(np.isfinite(data["y"]))
            assert np.all(np.isfinite(data["z"]))

    def test_ring_indices_in_range(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        for pc in _all_frames(hydra, pandar_scans_xt32):
            rings = pc.pc_data["ring"].astype(np.int32)
            assert int(rings.min()) >= 0
            assert int(rings.max()) <= 31

    def test_distances_within_model_limits(self, pandar_scans_xt32: list) -> None:
        model = Hydra.Model.PandarXT32
        hydra = Hydra(model)
        for pc in _all_frames(hydra, pandar_scans_xt32):
            d = pc.pc_data
            dists = np.sqrt(
                d["x"].astype(np.float64) ** 2
                + d["y"].astype(np.float64) ** 2
                + d["z"].astype(np.float64) ** 2
            )
            assert float(dists.min()) >= model.default_min_distance
            assert float(dists.max()) < model.default_max_distance

    def test_intensity_in_range(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        for pc in _all_frames(hydra, pandar_scans_xt32):
            intensities = pc.pc_data["intensity"]
            assert float(intensities.min()) >= 0.0
            assert float(intensities.max()) <= 255.0

    def test_tighter_max_distance_reduces_total_points(self, pandar_scans_xt32: list) -> None:
        pts_120 = _total_points(
            Hydra(Hydra.Model.PandarXT32, max_distance=120.0), pandar_scans_xt32
        )
        pts_10 = _total_points(Hydra(Hydra.Model.PandarXT32, max_distance=10.0), pandar_scans_xt32)
        assert pts_10 <= pts_120

    def test_dedup_threshold_has_no_effect_on_single_return(self, pandar_scans_xt32: list) -> None:
        """Single-return XT32 data: no same-azimuth block pairs → threshold makes no difference."""
        pts_with = _total_points(
            Hydra(Hydra.Model.PandarXT32, dual_return_distance_threshold=0.1),
            pandar_scans_xt32,
        )
        pts_without = _total_points(
            Hydra(Hydra.Model.PandarXT32, dual_return_distance_threshold=0.0),
            pandar_scans_xt32,
        )
        assert pts_with == pts_without

    def test_to_pypcd4_returns_list(self, pandar_scans_xt32: list) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        result = hydra.to_pypcd4(pandar_scans_xt32[0])
        assert isinstance(result, list)
