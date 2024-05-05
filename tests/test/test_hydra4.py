from __future__ import annotations

from hydra4 import Hydra
from hydra4.msgs import pandar_msgs__msg__PandarScan


def test_pandar_xt32(pandar_scans_xt32: list[pandar_msgs__msg__PandarScan]) -> None:
    hydra = Hydra(Hydra.Model.PandarXT32)

    expect_points = (60644, 60847, 60860, 60856)

    counter = 0
    for msg in pandar_scans_xt32:
        for pc in hydra.to_pypcd4(msg):
            assert pc.metadata.fields == ('x', 'y', 'z', 'intensity', 'ring', 'azimuth', 'stamp')
            assert pc.metadata.size == (4, 4, 4, 4, 2, 4, 8)
            assert pc.metadata.type == ('F', 'F', 'F', 'F', 'U', 'F', 'F')
            assert pc.points == expect_points[counter]
            counter += 1
