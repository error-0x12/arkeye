from __future__ import annotations

import cv2
import numpy as np
import pytest

from arkeye.board import BoardCalibration, BoardCalibrationError, BoardGrid, Homography


def test_grid_uses_bottom_zero_logical_rows() -> None:
    grid = BoardGrid.from_map_data(
        {
            "map": [[4, 5], [2, 3], [0, 1]],
            "tiles": [{"tileKey": f"tile_{index}"} for index in range(6)],
        }
    )

    assert grid.width == 2
    assert grid.height == 3
    assert grid.tile(0, 0)["tileKey"] == "tile_0"
    assert grid.tile(1, 2)["tileKey"] == "tile_5"


def test_homography_round_trip() -> None:
    source = [[0, 0], [8, 0], [8, 6], [0, 6]]
    destination = [[100, 500], [500, 500], [500, 100], [100, 100]]
    transform = Homography.from_points(source, destination)

    result = transform.transform([[4, 3]])
    restored = transform.inverse().transform(result)

    np.testing.assert_allclose(result, [[300, 300]])
    np.testing.assert_allclose(restored, [[4, 3]])


def test_board_calibration_composes_map_preview_and_frame_transforms() -> None:
    map_points = [[0, 0], [8, 0], [8, 6], [0, 6]]
    preview_points = [[100, 500], [500, 500], [500, 100], [100, 100]]
    frame_points = [[210, 1020], [1010, 1020], [1010, 220], [210, 220]]
    calibration = BoardCalibration(
        preview_size=(600, 600),
        map_to_preview=Homography.from_points(map_points, preview_points),
        preview_to_frame=Homography.from_points(preview_points, frame_points),
    )

    frame_point = calibration.map_points_to_frame([[4, 3]])
    map_point = calibration.frame_points_to_map(frame_point)

    np.testing.assert_allclose(frame_point, [[610, 620]])
    np.testing.assert_allclose(map_point, [[4, 3]])


def test_calibration_save_load_and_warp(tmp_path) -> None:
    calibration = BoardCalibration(
        preview_size=(64, 48),
        map_to_preview=Homography.identity(),
        preview_to_frame=Homography.identity(),
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = BoardCalibration.load(path)

    frame = np.full((48, 64, 3), (10, 20, 30), dtype=np.uint8)
    warped = loaded.warp_frame_to_preview(frame)

    assert loaded.to_dict() == calibration.to_dict()
    assert warped.shape == frame.shape
    assert np.array_equal(warped, frame)


def test_homography_rejects_invalid_points() -> None:
    with pytest.raises(BoardCalibrationError, match="At least four"):
        Homography.from_points([[0, 0]], [[0, 0]])
    with pytest.raises(BoardCalibrationError, match="finite"):
        Homography.from_points(
            [[0, 0], [1, 0], [1, 1], [0, np.nan]],
            [[0, 0], [1, 0], [1, 1], [0, 1]],
        )


def test_identity_calibration_uses_opencv_warp_contract() -> None:
    calibration = BoardCalibration((32, 24), Homography.identity(), Homography.identity())
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 255

    warped = calibration.warp_frame_to_preview(frame)

    assert cv2.countNonZero(warped[:, :, 1]) == 32 * 24
