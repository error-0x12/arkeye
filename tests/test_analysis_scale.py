from __future__ import annotations

import cv2
import numpy as np
import pytest

from arkeye.assets import BattleTemplate, TemplateLibrary
from arkeye.detector import BattlefieldDetector, TemplateClassifier


def test_scaled_detection_maps_coordinates_back_to_original_frame() -> None:
    sprite = np.zeros((40, 40, 4), dtype=np.uint8)
    cv2.circle(sprite, (20, 20), 14, (20, 20, 220, 255), -1)
    template = BattleTemplate.from_bgra("enemy_a", sprite, source="game_sprite")
    background = np.full((100, 140, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(
        background,
        TemplateClassifier(TemplateLibrary([template])),
        confidence_threshold=0.6,
        min_area=20,
        analysis_scale=0.5,
    )
    frame = background.copy()
    height, width = template.alpha.shape
    frame[30 : 30 + height, 50 : 50 + width] = template.image

    detections = detector.detect(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.enemy_id == "enemy_a"
    assert detection.bbox[0] == pytest.approx(50, abs=2)
    assert detection.bbox[1] == pytest.approx(30, abs=2)
    assert detection.foot_point[0] == pytest.approx(50 + width / 2, abs=2)
    assert detection.foot_point[1] == pytest.approx(30 + height - 1, abs=2)
