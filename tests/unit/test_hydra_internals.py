from __future__ import annotations

import numpy as np
import pytest

from hydra4 import Hydra

# Structured dtype matching Hydra metadata fields (x f32, y f32, z f32,
# intensity f32, ring u16, azimuth f32, stamp f64).
_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("intensity", "f4"), ("ring", "u2"),
    ("azimuth", "f4"), ("stamp", "f8"),
])


def _pt(x: float, y: float, z: float, ring: int) -> np.ndarray:
    """Build a single-element structured array for use in deduplication tests."""
    arr = np.zeros(1, dtype=_DTYPE)
    arr["x"] = x
    arr["y"] = y
    arr["z"] = z
    arr["ring"] = ring
    return arr


class TestDeduplicate:
    def test_empty_blk0_returns_empty(self) -> None:
        blk0 = np.empty(0, dtype=_DTYPE)
        blk1 = _pt(x=5.0, y=0.0, z=0.0, ring=0)
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 0

    def test_identical_distance_filters_all(self) -> None:
        """Both returns at the same distance → all first-return points removed."""
        blk0 = _pt(x=5.0, y=0.0, z=0.0, ring=0)
        blk1 = _pt(x=5.0, y=0.0, z=0.0, ring=0)
        # |5.0 - 5.0| = 0.0 < 0.1 → filtered
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 0

    def test_large_gap_keeps_all(self) -> None:
        """Large distance gap → first return kept."""
        blk0 = _pt(x=2.0, y=0.0, z=0.0, ring=0)
        blk1 = _pt(x=5.0, y=0.0, z=0.0, ring=0)
        # |2.0 - 5.0| = 3.0 >= 0.1 → kept
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 1

    def test_empty_blk1_keeps_all_blk0(self) -> None:
        """No second return → NaN lookup → first return always kept."""
        blk0 = _pt(x=3.0, y=0.0, z=0.0, ring=7)
        blk1 = np.empty(0, dtype=_DTYPE)
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 1

    def test_different_rings_kept(self) -> None:
        """blk0 ring≠blk1 ring → no match → NaN dist → blk0 kept."""
        blk0 = _pt(x=3.0, y=0.0, z=0.0, ring=0)
        blk1 = _pt(x=3.0, y=0.0, z=0.0, ring=1)
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 1

    def test_mixed_filtering(self) -> None:
        """ring=0 same distance (filtered), ring=1 large gap (kept)."""
        blk0 = np.concatenate([
            _pt(x=5.0, y=0.0, z=0.0, ring=0),  # |5-5|=0 < 0.1 → filtered
            _pt(x=2.0, y=0.0, z=0.0, ring=1),  # |2-9|=7 >= 0.1 → kept
        ])
        blk1 = np.concatenate([
            _pt(x=5.0, y=0.0, z=0.0, ring=0),
            _pt(x=9.0, y=0.0, z=0.0, ring=1),
        ])
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 1
        assert int(result["ring"][0]) == 1

    def test_well_below_threshold_filtered(self) -> None:
        """|diff| = 0.05 < 0.1 → filtered."""
        blk0 = _pt(x=2.0, y=0.0, z=0.0, ring=0)
        blk1 = _pt(x=2.05, y=0.0, z=0.0, ring=0)
        # dist0≈2.0, dist1≈2.05; |diff|≈0.05 < 0.1 → filtered
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 0

    def test_well_above_threshold_kept(self) -> None:
        """|diff| = 0.2 >= 0.1 → kept."""
        blk0 = _pt(x=2.0, y=0.0, z=0.0, ring=0)
        blk1 = _pt(x=2.2, y=0.0, z=0.0, ring=0)
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 1

    def test_multiple_points_same_ring_all_kept(self) -> None:
        """Multiple blk0 points at ring=0, blk1 at very different distance → all kept."""
        blk0 = np.concatenate([
            _pt(x=2.0, y=0.0, z=0.0, ring=0),
            _pt(x=3.0, y=0.0, z=0.0, ring=0),
        ])
        blk1 = _pt(x=8.0, y=0.0, z=0.0, ring=0)
        # blk1 dist=8.0; |2-8|=6 and |3-8|=5, both >= 0.1 → both kept
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert len(result) == 2

    def test_return_dtype_preserved(self) -> None:
        """Output array has the same dtype as input."""
        blk0 = _pt(x=2.0, y=0.0, z=0.0, ring=0)
        blk1 = _pt(x=8.0, y=0.0, z=0.0, ring=0)
        result = Hydra._deduplicate(blk0, blk1, threshold=0.1)
        assert result.dtype == _DTYPE

    def test_zero_threshold_keeps_all(self) -> None:
        """threshold=0 means any distance diff (even 0) triggers keep (>= 0)."""
        blk0 = _pt(x=5.0, y=0.0, z=0.0, ring=0)
        blk1 = _pt(x=5.0, y=0.0, z=0.0, ring=0)
        # |0| >= 0 → True → kept
        result = Hydra._deduplicate(blk0, blk1, threshold=0.0)
        assert len(result) == 1


class TestBlockToArray:
    def test_empty_points_returns_empty_array(self) -> None:
        result = Hydra._block_to_array([], _DTYPE)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.dtype == _DTYPE

    def test_single_point_field_values(self) -> None:
        pt = (1.5, 2.5, -0.5, 100.0, 3, 12345.0, 1_234_567_890.0)
        result = Hydra._block_to_array([pt], _DTYPE)
        assert len(result) == 1
        assert result["x"][0] == pytest.approx(1.5, abs=1e-4)
        assert result["y"][0] == pytest.approx(2.5, abs=1e-4)
        assert result["z"][0] == pytest.approx(-0.5, abs=1e-4)
        assert int(result["ring"][0]) == 3
        assert result["intensity"][0] == pytest.approx(100.0, abs=1e-4)

    def test_multiple_points_count(self) -> None:
        pts = [(float(i), 0.0, 0.0, 0.0, i % 32, 0.0, 0.0) for i in range(32)]
        result = Hydra._block_to_array(pts, _DTYPE)
        assert len(result) == 32

    def test_output_dtype_matches(self) -> None:
        pt = (1.0, 2.0, 3.0, 50.0, 0, 100.0, 0.0)
        result = Hydra._block_to_array([pt], _DTYPE)
        assert result.dtype == _DTYPE

    def test_ring_values_preserved(self) -> None:
        pts = [(0.0, 0.0, 0.0, 0.0, ring, 0.0, 0.0) for ring in range(32)]
        result = Hydra._block_to_array(pts, _DTYPE)
        for i, ring in enumerate(range(32)):
            assert int(result["ring"][i]) == ring
