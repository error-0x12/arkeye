from __future__ import annotations

import numpy as np

from arkeye.board import BoardCalibration, Homography
from arkeye.routes import build_deployable_mask, build_route_mask, route_frame_points


def identity_calibration() -> BoardCalibration:
    return BoardCalibration((100, 100), Homography.identity(), Homography.identity())


def test_route_projection_ignores_dummy_wait_positions() -> None:
    level = {
        "routes": [
            {
                "startPosition": {"col": 10, "row": 20},
                "checkpoints": [
                    {"type": "WAIT_FOR_SECONDS", "position": {"col": 0, "row": 0}},
                    {"type": "MOVE", "position": {"col": 30, "row": 20}},
                ],
                "endPosition": {"col": 50, "row": 20},
            }
        ]
    }

    routes = route_frame_points(level, identity_calibration())

    assert len(routes) == 1
    np.testing.assert_array_equal(routes[0], [[10, 20], [30, 20], [50, 20]])


def test_route_mask_covers_route_but_not_distant_or_dummy_points() -> None:
    level = {
        "routes": [
            {
                "startPosition": {"col": 10, "row": 20},
                "checkpoints": [
                    {"type": "WAIT_FOR_SECONDS", "position": {"col": 0, "row": 0}},
                    {"type": "MOVE", "position": {"col": 50, "row": 20}},
                ],
                "endPosition": {"col": 80, "row": 20},
            }
        ]
    }

    mask = build_route_mask(level, identity_calibration(), (100, 100))

    assert mask[20, 10] == 255
    assert mask[20, 50] == 255
    assert mask[60, 50] == 0
    assert mask[0, 0] == 0


def test_deployable_mask_only_contains_allowed_tiles() -> None:
    level = {
        "mapData": {
            "map": [[0, 1], [2, 3]],
            "tiles": [
                {"buildableType": "NONE"},
                {"buildableType": "MELEE"},
                {"buildableType": "RANGED"},
                {"buildableType": "NONE"},
            ],
        }
    }

    mask = build_deployable_mask(level, identity_calibration(), (100, 100))

    assert mask[1, 1] == 255
    assert mask[0, 0] == 255
    empty = build_deployable_mask(level, identity_calibration(), (100, 100), allowed_types=frozenset())
    assert not np.any(empty)
