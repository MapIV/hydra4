from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from pypcd4 import MetaData, PointCloud

from .decoders import PandarBase, PandarOT128, PandarXT32
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
        PandarOT128 = "pandar_ot128"

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
            if self == "pandar_ot128":
                return PandarOT128

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
            if self == "pandar_ot128":
                return Path(__file__).parent / "calibrations" / "pandar_ot128.csv"

            raise ValueError(f"Unknown model: {self}")

        @property
        def default_min_distance(self) -> float:
            """
            Returns the recommended minimum detection range for the given model.

            Parameters:
                self (Model): The model to get the default minimum distance for.

            Returns:
                float: The default minimum distance in metres.

            Raises:
                ValueError: If the given model is not supported.
            """
            if self == "pandar_xt32":
                return 0.1
            if self == "pandar_ot128":
                return 0.3

            raise ValueError(f"Unknown model: {self}")

        @property
        def default_max_distance(self) -> float:
            """
            Returns the recommended maximum detection range for the given model.

            Parameters:
                self (Model): The model to get the default maximum distance for.

            Returns:
                float: The default maximum distance in metres.

            Raises:
                ValueError: If the given model is not supported.
            """
            if self == "pandar_xt32":
                return 120.0
            if self == "pandar_ot128":
                return 230.0

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
        calibration_path: str | Path | None = None,
        min_distance: float | None = None,
        max_distance: float | None = None,
        dual_return_distance_threshold: float = 0.1,
    ) -> None:
        """
        Initializes a new instance of the Hydra class.

        Parameters:
            model (Model): The model to create an instance for.
            calibration_path (str | Path | None): The path to the calibration file.
                If None, the default calibration file for the given model is used.
            min_distance (float | None): The minimum distance from the sensor to consider a point
                valid. If None, the model's recommended minimum range is used
                (0.1 m for PandarXT32, 0.3 m for PandarOT128).
            max_distance (float | None): The maximum distance from the sensor to consider a point
                valid. If None, the model's recommended maximum range is used
                (120.0 m for PandarXT32, 230.0 m for PandarOT128).
            dual_return_distance_threshold (float): In dual-return mode, if the distance
                difference between the first and last return for the same channel is smaller
                than this threshold (metres), the first return is discarded. Set to 0.0 to
                disable deduplication. Defaults to 0.1 m (matching Nebula behaviour).
        """
        self.model = model

        self.calibration_path = (
            model.calibration_path if calibration_path is None else Path(calibration_path)
        )

        resolved_min_distance = model.default_min_distance if min_distance is None else min_distance
        resolved_max_distance = model.default_max_distance if max_distance is None else max_distance

        self.decoder = self.model.decoder(
            self.calibration_path, resolved_min_distance, resolved_max_distance
        )

        self.metadata = MetaData(
            fields=("x", "y", "z", "intensity", "ring", "azimuth", "stamp"),
            size=(4, 4, 4, 4, 2, 4, 8),
            type=("F", "F", "F", "F", "U", "F", "F"),
            count=(1, 1, 1, 1, 1, 1, 1),
            points=0,
            width=0,
        )

        self.dual_return_distance_threshold = dual_return_distance_threshold
        self.last_phase = 0
        self.arrays: list[np.ndarray] = []
        # True when the last appended element in self.arrays was the immediately preceding block.
        # Used to safely pop it for dual-return deduplication.
        self._last_appended: bool = False

    @staticmethod
    def _block_to_array(points: list, dtype: np.dtype) -> np.ndarray:
        if points:
            return np.array(points, dtype=dtype)
        return np.empty(0, dtype=dtype)

    @staticmethod
    def _deduplicate(
        blk0: np.ndarray, blk1: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Return the subset of blk0 whose per-ring distance differs from blk1 by >= threshold."""
        if len(blk0) == 0:
            return blk0

        # Build a distance-by-ring lookup from blk1 (256 covers both OT128 and XT32).
        blk1_dist_by_ring = np.full(256, np.nan, dtype=np.float64)
        if len(blk1) > 0:
            blk1_d = np.sqrt(
                blk1["x"].astype(np.float64) ** 2
                + blk1["y"].astype(np.float64) ** 2
                + blk1["z"].astype(np.float64) ** 2
            )
            blk1_dist_by_ring[blk1["ring"]] = blk1_d

        blk0_d = np.sqrt(
            blk0["x"].astype(np.float64) ** 2
            + blk0["y"].astype(np.float64) ** 2
            + blk0["z"].astype(np.float64) ** 2
        )
        blk1_d_for_blk0 = blk1_dist_by_ring[blk0["ring"]]

        # Keep a blk0 point when blk1 has no matching ring (NaN) or the gap meets the threshold.
        keep = np.isnan(blk1_d_for_blk0) | (np.abs(blk0_d - blk1_d_for_blk0) >= threshold)
        return blk0[keep]

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
        dtype = self.metadata.build_dtype()
        threshold = self.dual_return_distance_threshold

        for packet in in_msg.packets:
            for block in self.decoder(packet):
                current_phase: int = int(block.azimuth + 36000) % 36000
                # A true 360° wrap is detected when the azimuth decreases by more than
                # half a revolution.  This avoids spurious splits in dual-return mode,
                # where both blocks share the same azimuth value.
                if self.last_phase - current_phase > 18000:
                    if self.arrays:
                        frame_data = np.concatenate(self.arrays)
                        n = len(frame_data)
                        meta = MetaData(
                            fields=self.metadata.fields,
                            size=self.metadata.size,
                            type=self.metadata.type,
                            count=self.metadata.count,
                            points=n,
                            width=n,
                        )
                        self.arrays = []
                        self._last_appended = False
                        pcs.append(PointCloud(meta, frame_data))

                blk_arr = self._block_to_array(block.points, dtype)

                if threshold > 0.0 and current_phase == self.last_phase and self._last_appended:
                    # Second return of a dual-return pair (same azimuth as previous block).
                    # Pop the first-return block and filter it against the second return.
                    blk0 = self.arrays.pop()
                    blk0_filtered = self._deduplicate(blk0, blk_arr, threshold)
                    if len(blk0_filtered) > 0:
                        self.arrays.append(blk0_filtered)
                    if len(blk_arr) > 0:
                        self.arrays.append(blk_arr)
                    self._last_appended = len(blk_arr) > 0
                elif len(blk_arr) > 0:
                    self.arrays.append(blk_arr)
                    self._last_appended = True
                else:
                    self._last_appended = False

                self.last_phase = current_phase

        return pcs
