from __future__ import annotations

import pytest

from hydra4 import Hydra
from hydra4.decoders import PandarOT128, PandarXT32


class TestModelDecoder:
    def test_xt32_decoder_class(self) -> None:
        assert Hydra.Model.PandarXT32.decoder is PandarXT32

    def test_ot128_decoder_class(self) -> None:
        assert Hydra.Model.PandarOT128.decoder is PandarOT128


class TestModelCalibrationPath:
    def test_xt32_calibration_path_exists(self) -> None:
        assert Hydra.Model.PandarXT32.calibration_path.exists()

    def test_ot128_calibration_path_exists(self) -> None:
        assert Hydra.Model.PandarOT128.calibration_path.exists()

    def test_xt32_calibration_is_csv(self) -> None:
        assert Hydra.Model.PandarXT32.calibration_path.suffix == ".csv"

    def test_ot128_calibration_is_csv(self) -> None:
        assert Hydra.Model.PandarOT128.calibration_path.suffix == ".csv"


class TestModelDistances:
    def test_xt32_default_min_distance(self) -> None:
        assert Hydra.Model.PandarXT32.default_min_distance == pytest.approx(0.1)

    def test_xt32_default_max_distance(self) -> None:
        assert Hydra.Model.PandarXT32.default_max_distance == pytest.approx(120.0)

    def test_ot128_default_min_distance(self) -> None:
        assert Hydra.Model.PandarOT128.default_min_distance == pytest.approx(0.3)

    def test_ot128_default_max_distance(self) -> None:
        assert Hydra.Model.PandarOT128.default_max_distance == pytest.approx(230.0)

    def test_xt32_min_less_than_max(self) -> None:
        model = Hydra.Model.PandarXT32
        assert model.default_min_distance < model.default_max_distance

    def test_ot128_min_less_than_max(self) -> None:
        model = Hydra.Model.PandarOT128
        assert model.default_min_distance < model.default_max_distance


class TestModelStr:
    def test_xt32_str(self) -> None:
        assert str(Hydra.Model.PandarXT32) == "pandar_xt32"

    def test_ot128_str(self) -> None:
        assert str(Hydra.Model.PandarOT128) == "pandar_ot128"


class TestHydraInit:
    def test_default_min_distance_xt32(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        assert hydra.decoder.min_distance == pytest.approx(0.1)

    def test_default_max_distance_xt32(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        assert hydra.decoder.max_distance == pytest.approx(120.0)

    def test_default_min_distance_ot128(self) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        assert hydra.decoder.min_distance == pytest.approx(0.3)

    def test_default_max_distance_ot128(self) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128)
        assert hydra.decoder.max_distance == pytest.approx(230.0)

    def test_custom_min_distance(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32, min_distance=0.5)
        assert hydra.decoder.min_distance == pytest.approx(0.5)

    def test_custom_max_distance(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32, max_distance=50.0)
        assert hydra.decoder.max_distance == pytest.approx(50.0)

    def test_default_dedup_threshold(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        assert hydra.dual_return_distance_threshold == pytest.approx(0.1)

    def test_custom_dedup_threshold(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32, dual_return_distance_threshold=0.5)
        assert hydra.dual_return_distance_threshold == pytest.approx(0.5)

    def test_zero_dedup_threshold_disables_dedup(self) -> None:
        hydra = Hydra(Hydra.Model.PandarOT128, dual_return_distance_threshold=0.0)
        assert hydra.dual_return_distance_threshold == 0.0

    def test_metadata_fields(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        assert hydra.metadata.fields == ("x", "y", "z", "intensity", "ring", "azimuth", "stamp")

    def test_metadata_sizes(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        assert hydra.metadata.size == (4, 4, 4, 4, 2, 4, 8)

    def test_metadata_types(self) -> None:
        hydra = Hydra(Hydra.Model.PandarXT32)
        assert hydra.metadata.type == ("F", "F", "F", "F", "U", "F", "F")
