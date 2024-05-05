from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pypcd4 import MetaData, PointCloud

from .decoders import PandarBase, PandarXT32
from .msgs import pandar_msgs__msg__PandarScan


class Hydra:
    """
    A class for decoding and converting messages from supported lidar models
    into a pypcd4.PointCloud.
    """

    class Model(str, Enum):
        """
        An enumeration of the supported lidar models.
        """

        PandarXT32 = "pandar_xt32"

        @property
        def decoder(self) -> type[PandarBase]:
            """
            Returns the decoder class for the given model.

            Parameters:
                self (Model): The model to get the decoder for.

            Returns:
                type[PandarBase]: The decoder class for the given model.

            Raises:
                ValueError: If the given model is not supported.
            """
            if self == "pandar_xt32":
                return PandarXT32

            raise ValueError(f"Unknown model: {self}")

        @property
        def calibration_path(self) -> Path:
            """
            Returns the path to the default calibration file for the given model.

            Parameters:
                self (Model): The model to get the default calibration path for.

            Returns:
                Path: The path to the default calibration file for the given model.

            Raises:
                ValueError: If the given model is not supported.
            """
            if self == "pandar_xt32":
                return Path(__file__).parent / "calibrations" / "pandar_xt32.csv"

            raise ValueError(f"Unknown model: {self}")

        def __str__(self) -> str:
            """
            Returns the string representation of the model.

            Parameters:
                self (Model): The model to convert to a string.

            Returns:
                str: The string representation of the model.
            """
            return self.value

    def __init__(
        self,
        model: Model,
        calibration_path: Optional[Union[str, Path]] = None,
        min_distance: float = 0.0,
        max_distance: float = 120.0,
    ) -> None:
        """
        Initializes a new instance of the Hydra class.

        Parameters:
            model (Model): The model to create an instance for.
            calibration_path (Optional[Union[str, Path]]): The path to the calibration file.
                If None, the default calibration file for the given model is used.
            min_distance (float): The minimum distance from the sensor to consider a point valid.
            max_distance (float): The maximum distance from the sensor to consider a point valid.
        """
        self.model = model

        self.calibration_path = (
            model.calibration_path if calibration_path is None else Path(calibration_path)
        )

        self.decoder = self.model.decoder(self.calibration_path, min_distance, max_distance)

        self.metadata = MetaData(
            fields=("x", "y", "z", "intensity", "ring", "azimuth", "stamp"),
            size=(4, 4, 4, 4, 2, 4, 8),
            type=("F", "F", "F", "F", "U", "F", "F"),
            count=(1, 1, 1, 1, 1, 1, 1),
            points=0,
            width=0,
        )

        self.last_phase = 0
        self.points: list[tuple] = []

    def to_pypcd4(self, in_msg: pandar_msgs__msg__PandarScan) -> list[PointCloud]:
        """
        Converts a PandarScan message into a pypcd4.PointCloud.
        See https://github.com/MapIV/pypcd4 for more information about the pypcd4 library.

        Parameters:
            in_msg (pandar_msgs__msg__PandarScan): The message to convert.

        Returns:
            list[PointCloud]: The converted PointClouds.
        """
        pcs: list[PointCloud] = []
        for packet in in_msg.packets:
            for block in self.decoder(packet):
                if (current_phase := int(block.azimuth + 36000) % 36000) <= self.last_phase:
                    self.metadata.points = len(self.points)
                    self.metadata.width = len(self.points)

                    pc = PointCloud(
                        self.metadata,
                        np.array(self.points, dtype=self.metadata.build_dtype()),
                    )

                    self.points = block.points
                    self.last_phase = current_phase

                    pcs.append(pc)

                self.points.extend(block.points)
                self.last_phase = current_phase

        return pcs
