from __future__ import annotations

import cv2
import numpy as np

from arkeye.speed import detect_battle_speed, detect_deployment_slowdown


def test_detects_both_battle_speeds_at_full_and_half_resolution() -> None:
    for timestamp, expected in ((8, 2), (52, 1)):
        frame = cv2.imread(f"tests/fixtures/t_{timestamp:02}.jpg")
        assert detect_battle_speed(frame) == expected
        assert detect_battle_speed(cv2.resize(frame, (960, 540))) == expected


def test_speed_button_absence_returns_none() -> None:
    assert detect_battle_speed(np.zeros((1080, 1920, 3), dtype=np.uint8)) is None


def test_deployment_slowdown_requires_a_large_green_range_overlay() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:15, 10:20] = (0, 200, 0)
    assert not detect_deployment_slowdown(frame)

    frame[20:70, 20:70] = (0, 200, 0)
    assert detect_deployment_slowdown(frame)
