"""Project level routes into battle-frame masks for focused enemy detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from arkeye.board import BoardCalibration


def _route_map_points(route: Mapping[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    def append(position: Mapping[str, Any] | None) -> None:
        if position is None:
            return
        point = float(position["col"]), float(position["row"])
        if not points or point != points[-1]:
            points.append(point)

    append(route.get("startPosition"))
    for checkpoint in route.get("checkpoints") or []:
        # Waiting checkpoints commonly use (0, 0) as a dummy position.
        if checkpoint.get("type") == "MOVE":
            append(checkpoint.get("position"))
    append(route.get("endPosition"))
    return points


def route_frame_points(level: Mapping[str, Any], calibration: BoardCalibration) -> tuple[NDArray[np.int32], ...]:
    projected: list[NDArray[np.int32]] = []
    for route in level.get("routes") or []:
        map_points = _route_map_points(route)
        if len(map_points) < 2:
            continue
        frame_points = calibration.map_points_to_frame(map_points)
        projected.append(np.rint(frame_points).astype(np.int32))
    return tuple(projected)


def build_route_mask(
    level: Mapping[str, Any],
    calibration: BoardCalibration,
    frame_shape: Sequence[int],
    *,
    width_tiles: float = 1.2,
) -> NDArray[np.uint8]:
    if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0 or width_tiles <= 0:
        raise ValueError("Invalid frame shape or route width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    origin, x_unit, y_unit = calibration.map_points_to_frame([[0, 0], [1, 0], [0, 1]])
    tile_size = (float(np.linalg.norm(x_unit - origin)) + float(np.linalg.norm(y_unit - origin))) / 2.0
    thickness = max(5, round(tile_size * width_tiles))
    for points in route_frame_points(level, calibration):
        cv2.polylines(mask, [points], False, 255, thickness, cv2.LINE_AA)
        for point in points:
            cv2.circle(mask, tuple(point), max(2, thickness // 2), 255, -1, cv2.LINE_AA)
    return mask


def build_deployable_mask(
    level: Mapping[str, Any],
    calibration: BoardCalibration,
    frame_shape: Sequence[int],
    *,
    allowed_types: frozenset[str] = frozenset({"MELEE", "RANGED"}),
) -> NDArray[np.uint8]:
    """Project deployable tiles into frame space for friendly-unit validation."""

    map_data = level.get("mapData") or {}
    tiles = map_data.get("tiles") or []
    grid = map_data.get("map") or []
    if not tiles or not grid:
        raise ValueError("Level mapData is missing tiles")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    origin, x_unit, y_unit = calibration.map_points_to_frame([[0, 0], [1, 0], [0, 1]])
    tile_size = (float(np.linalg.norm(x_unit - origin)) + float(np.linalg.norm(y_unit - origin))) / 2.0
    radius = max(3, round(tile_size * 0.45))
    for row, grid_row in enumerate(grid):
        for col, tile_index in enumerate(grid_row):
            tile = tiles[tile_index]
            if tile.get("buildableType") not in allowed_types:
                continue
            map_row = len(grid) - 1 - row
            center = calibration.map_points_to_frame([[col, map_row]])[0]
            cv2.circle(mask, tuple(np.rint(center).astype(int)), radius, 255, -1, cv2.LINE_AA)
    return mask


def build_ui_mask(
    frame_shape: Sequence[int],
    config: Mapping[str, Any] | None = None,
) -> NDArray[np.uint8]:
    """Mask user-labelled fixed UI rectangles without covering spawn entrances."""

    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    if config is None:
        mask[: round(height * 130 / 1080), :] = 255
        mask[round(height * 900 / 1080) :, :] = 255
        return mask
    scale_x = width / float(config["frame_width"])
    scale_y = height / float(config["frame_height"])
    for rectangle in config.get("rectangles") or []:
        left = max(0, round(float(rectangle["x"]) * scale_x))
        top = max(0, round(float(rectangle["y"]) * scale_y))
        right = min(width, round((float(rectangle["x"]) + float(rectangle["width"])) * scale_x))
        bottom = min(height, round((float(rectangle["y"]) + float(rectangle["height"])) * scale_y))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 255
    return mask
