from __future__ import annotations

from unittest.mock import Mock

import cv2
import numpy as np

from arkeye.assets import BattleTemplate, TemplateLibrary
from arkeye.detector import BattlefieldDetector, LocalizedMatch, TemplateClassifier


def square_template() -> BattleTemplate:
    sprite = np.zeros((48, 48, 4), dtype=np.uint8)
    cv2.rectangle(sprite, (8, 8), (40, 40), (20, 20, 220, 255), -1)
    return BattleTemplate.from_bgra("enemy_square", sprite, source="game_sprite", name="square")


def composite(frame: np.ndarray, template: BattleTemplate, x: int, y: int) -> None:
    height, width = template.alpha.shape
    alpha = template.alpha.astype(np.float32)[:, :, None] / 255.0
    region = frame[y : y + height, x : x + width]
    region[:] = (template.image * alpha + region * (1 - alpha)).astype(np.uint8)


def test_first_same_class_overlap_splits_without_track_history() -> None:
    template = square_template()
    background = np.full((100, 140, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(background, TemplateClassifier(TemplateLibrary([template])), min_area=20)
    frame = background.copy()
    composite(frame, template, 20, 20)
    composite(frame, template, 50, 20)

    detections = detector.detect(frame, overlapping_tracks=())

    assert len(detections) == 2
    assert {detection.enemy_id for detection in detections} == {"enemy_square"}
    assert detections[0].foot_point[0] < detections[1].foot_point[0]


def test_unexplained_overlap_residual_becomes_unknown() -> None:
    template = square_template()
    classifier = TemplateClassifier(TemplateLibrary([template]))
    background = np.full((100, 140, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(background, classifier, min_area=20)
    frame = background.copy()
    composite(frame, template, 20, 20)
    composite(frame, template, 50, 20)
    localized = LocalizedMatch(
        bbox=(0, 0, 33, 33),
        enemy_id="enemy_square",
        template_name="square",
        confidence=0.9,
        alpha=template.alpha,
    )
    classifier.locate = Mock(return_value=(localized,))

    detections = detector.detect(frame, overlapping_tracks=())

    assert {detection.enemy_id for detection in detections} == {"enemy_square", "unknown"}
