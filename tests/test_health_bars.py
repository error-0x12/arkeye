from __future__ import annotations

import cv2
import numpy as np

from arkeye.health_bars import HealthBarDetector
from arkeye.routes import build_ui_mask


def test_health_bar_detector_distinguishes_friendly_enemy_and_device() -> None:
    frame = np.zeros((240, 300, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 90), (169, 97), (200, 200, 0), -1)
    cv2.rectangle(frame, (50, 102), (169, 108), (50, 200, 50), -1)
    cv2.rectangle(frame, (50, 140), (109, 148), (0, 0, 220), -1)
    cv2.rectangle(frame, (50, 180), (149, 187), (50, 200, 50), -1)
    detector = HealthBarDetector()

    evidence = detector.detect(frame)

    kinds = {item.kind for item in evidence}
    assert kinds == {"friendly", "enemy", "device"}
    matched = detector.match((110, 100), (70, 30, 80, 70), evidence)
    assert matched is not None
    assert matched.kind == "friendly"


def test_real_frame_detects_red_bar_and_paired_operator_bars() -> None:
    frame = cv2.imread("tests/fixtures/61.jpg")
    detector = HealthBarDetector()

    evidence = detector.detect(frame)

    assert any(item.kind == "enemy" and abs(item.bbox[0] - 521) <= 3 for item in evidence)
    assert sum(item.kind == "friendly" for item in evidence) >= 4
    assert any(item.kind == "device" and 405 <= item.bbox[1] <= 420 for item in evidence)


def test_paired_operator_bar_has_priority_over_green_fragment() -> None:
    frame = cv2.imread("tests/fixtures/source_frame.jpg")
    detector = HealthBarDetector()
    evidence = detector.detect(frame)

    matched = detector.match((600, 650), (540, 500, 120, 160), evidence)

    assert matched is not None
    assert matched.kind == "friendly"


def test_user_ui_mask_config_scales_rectangles() -> None:
    config = {
        "frame_width": 1920,
        "frame_height": 1080,
        "rectangles": [{"x": 100, "y": 200, "width": 300, "height": 100}],
    }

    mask = build_ui_mask((540, 960, 3), config)

    assert mask[100, 50] == 255
    assert mask[99, 50] == 0
    assert mask[149, 199] == 255
    assert mask[150, 200] == 0
