from __future__ import annotations

from abc import ABCMeta, abstractmethod
from csv import DictReader
from pathlib import Path
from typing import Generator

import numpy as np

from ..msgs import pandar_msgs__msg__PandarPacket
from ..structs import Block


class PandarBase(metaclass=ABCMeta):
    def __init__(self, calibration_path: Path, min_distance: float, max_distance: float) -> None:
        """
        Initializes the PandarBase object with calibration data,
        minimum distance, and maximum distance.

        Parameters:
            calibration_path (Path): The path to the calibration data file.
            min_distance (float): The minimum distance.
            max_distance (float): The maximum distance.

        Returns:
            None
        """
        with calibration_path.open("r") as fp:
            reader = DictReader(fp)

            self.azimuths = []
            self.elevations = []
            for row in reader:
                self.azimuths.append(float(row["Azimuth"]))
                self.elevations.append(np.deg2rad(float(row["Elevation"])))

        self.min_distance = min_distance
        self.max_distance = max_distance

    @abstractmethod
    def __call__(self, packet: pandar_msgs__msg__PandarPacket) -> Generator[Block, None, None]: ...
