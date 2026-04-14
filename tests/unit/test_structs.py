from __future__ import annotations

from hydra4.structs import Block, ReturnMode


class TestReturnMode:
    def test_first_is_not_dual(self) -> None:
        assert ReturnMode.FIRST.is_dual() is False

    def test_strongest_is_not_dual(self) -> None:
        assert ReturnMode.STRONGET.is_dual() is False

    def test_last_is_not_dual(self) -> None:
        assert ReturnMode.LAST.is_dual() is False

    def test_last_strongest_is_dual(self) -> None:
        assert ReturnMode.LAST_STRONGEST.is_dual() is True

    def test_last_first_is_dual(self) -> None:
        assert ReturnMode.LAST_FIRST.is_dual() is True

    def test_first_strongest_is_dual(self) -> None:
        assert ReturnMode.FIRST_STRONGEST.is_dual() is True

    def test_str_first(self) -> None:
        assert str(ReturnMode.FIRST) == "first"

    def test_str_last_strongest(self) -> None:
        assert str(ReturnMode.LAST_STRONGEST) == "last_strongest"

    def test_all_dual_modes(self) -> None:
        dual = {ReturnMode.LAST_STRONGEST, ReturnMode.LAST_FIRST, ReturnMode.FIRST_STRONGEST}
        non_dual = {ReturnMode.FIRST, ReturnMode.STRONGET, ReturnMode.LAST}
        for mode in dual:
            assert mode.is_dual(), f"{mode} should be dual"
        for mode in non_dual:
            assert not mode.is_dual(), f"{mode} should not be dual"


class TestBlock:
    def test_creation_with_empty_points(self) -> None:
        block = Block(azimuth=1000, points=[])
        assert block.azimuth == 1000
        assert block.points == []

    def test_creation_with_points(self) -> None:
        pt = (1.0, 2.0, 3.0, 100.0, 5, 0, 0.0)
        block = Block(azimuth=500, points=[pt])
        assert block.azimuth == 500
        assert len(block.points) == 1
        assert block.points[0] == pt

    def test_points_list_is_mutable(self) -> None:
        block = Block(azimuth=0, points=[])
        pt = (1.0, 2.0, 3.0, 50.0, 3, 100, 1234.0)
        block.points.append(pt)
        assert len(block.points) == 1

    def test_azimuth_boundary_values(self) -> None:
        for az in (0, 1, 18000, 35998, 35999):
            block = Block(azimuth=az, points=[])
            assert block.azimuth == az
