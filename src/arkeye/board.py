"""Board-grid conventions and perspective transforms.

Logical tile coordinates use (column, row), with row zero at the bottom. This
matches level JSON routes and Theresa.wiki's reverse-wrapped tile scene.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]
PointArray = NDArray[np.float64]


class BoardCalibrationError(ValueError):
    """Raised for invalid grids, point sets, or calibration data."""


@dataclass(frozen=True, slots=True)
class BoardGrid:
    width: int
    height: int
    tile_indices: NDArray[np.int64]
    tiles: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_map_data(cls, map_data: Mapping[str, Any]) -> BoardGrid:
        raw_map = map_data.get("map")
        raw_tiles = map_data.get("tiles")
        if not isinstance(raw_map, list) or not raw_map or not isinstance(raw_tiles, list):
            raise BoardCalibrationError("Invalid mapData: expected non-empty map and tiles")
        width = len(raw_map[0])
        if not width or any(len(row) != width for row in raw_map):
            raise BoardCalibrationError("Invalid mapData: rows must have equal non-zero width")
        tile_indices = np.asarray(raw_map, dtype=np.int64)
        if np.any(tile_indices < 0) or np.any(tile_indices >= len(raw_tiles)):
            raise BoardCalibrationError("Invalid mapData: tile index is out of range")
        return cls(
            width=width,
            height=len(raw_map),
            tile_indices=tile_indices,
            tiles=tuple(raw_tiles),
        )

    def tile_index(self, column: int, row: int) -> int:
        if not (0 <= column < self.width and 0 <= row < self.height):
            raise BoardCalibrationError(f"Tile coordinate out of range: {(column, row)}")
        # mapData.map is stored top-to-bottom; routes use bottom-to-top rows.
        return int(self.tile_indices[self.height - 1 - row, column])

    def tile(self, column: int, row: int) -> Mapping[str, Any]:
        return self.tiles[self.tile_index(column, row)]


@dataclass(frozen=True, slots=True)
class Homography:
    matrix: FloatMatrix

    def __post_init__(self) -> None:
        if self.matrix.shape != (3, 3) or not np.isfinite(self.matrix).all():
            raise BoardCalibrationError("A homography must be a finite 3x3 matrix")
        if abs(np.linalg.det(self.matrix)) < 1e-12:
            raise BoardCalibrationError("A homography must be invertible")

    @classmethod
    def identity(cls) -> Homography:
        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def from_points(cls, source: Any, destination: Any) -> Homography:
        source_points = _as_points(source)
        destination_points = _as_points(destination)
        if len(source_points) != len(destination_points) or len(source_points) < 4:
            raise BoardCalibrationError("At least four corresponding point pairs are required")
        matrix, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 3.0)
        if matrix is None:
            raise BoardCalibrationError("Unable to fit a homography to these points")
        return cls(np.asarray(matrix, dtype=np.float64))

    def transform(self, points: Any) -> PointArray:
        point_array = _as_points(points)
        homogeneous = np.concatenate([point_array, np.ones((len(point_array), 1))], axis=1)
        transformed = (self.matrix @ homogeneous.T).T
        if np.any(np.abs(transformed[:, 2]) < 1e-12):
            raise BoardCalibrationError("A point maps to infinity under this homography")
        result = transformed[:, :2] / transformed[:, 2:3]
        if not np.isfinite(result).all():
            raise BoardCalibrationError("Homography produced a non-finite point")
        return result

    def inverse(self) -> Homography:
        return Homography(np.linalg.inv(self.matrix))

    def after(self, earlier: Homography) -> Homography:
        """Compose transforms: apply ``earlier`` first, then this transform."""

        return Homography(self.matrix @ earlier.matrix)


@dataclass(frozen=True, slots=True)
class BoardCalibration:
    preview_size: tuple[int, int]
    map_to_preview: Homography
    preview_to_frame: Homography

    def __post_init__(self) -> None:
        if len(self.preview_size) != 2 or min(self.preview_size) <= 0:
            raise BoardCalibrationError("Preview size must contain positive width and height")

    @property
    def map_to_frame(self) -> Homography:
        return self.preview_to_frame.after(self.map_to_preview)

    def map_points_to_frame(self, points: Any) -> PointArray:
        return self.map_to_frame.transform(points)

    def frame_points_to_map(self, points: Any) -> PointArray:
        return self.map_to_frame.inverse().transform(points)

    def warp_frame_to_preview(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return cv2.warpPerspective(
            frame,
            self.preview_to_frame.inverse().matrix,
            self.preview_size,
            flags=cv2.INTER_LINEAR,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "preview_size": list(self.preview_size),
            "map_to_preview": self.map_to_preview.matrix.tolist(),
            "preview_to_frame": self.preview_to_frame.matrix.tolist(),
        }

    def save(self, path: str | PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BoardCalibration:
        if value.get("version") != 1:
            raise BoardCalibrationError("Unsupported calibration version")
        try:
            preview_size = tuple(int(item) for item in value["preview_size"])
            map_to_preview = np.asarray(value["map_to_preview"], dtype=np.float64)
            preview_to_frame = np.asarray(value["preview_to_frame"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise BoardCalibrationError("Invalid calibration data") from exc
        return cls(
            preview_size=preview_size,
            map_to_preview=Homography(map_to_preview),
            preview_to_frame=Homography(preview_to_frame),
        )

    @classmethod
    def load(cls, path: str | PathLike[str]) -> BoardCalibration:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BoardCalibrationError(f"Unable to load calibration: {path}") from exc
        if not isinstance(value, Mapping):
            raise BoardCalibrationError("Calibration root must be a JSON object")
        return cls.from_dict(value)


def _as_points(points: Any) -> PointArray:
    try:
        point_array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BoardCalibrationError("Points must be numeric") from exc
    if point_array.ndim == 1 and point_array.shape == (2,):
        point_array = point_array.reshape(1, 2)
    if point_array.ndim != 2 or point_array.shape[1] != 2 or not len(point_array):
        raise BoardCalibrationError("Points must have shape (N, 2), with N >= 1")
    if not np.isfinite(point_array).all():
        raise BoardCalibrationError("Points must be finite")
    return np.ascontiguousarray(point_array)
