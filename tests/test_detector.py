from __future__ import annotations

import cv2
import numpy as np

from arkeye.assets import BattleTemplate, TemplateLibrary
from arkeye.detector import BattlefieldDetector, ForegroundDetector, TemplateClassifier


def make_template(shape: str, color: tuple[int, int, int]) -> BattleTemplate:
    image = np.zeros((48, 48, 4), dtype=np.uint8)
    if shape == "circle":
        cv2.circle(image, (24, 24), 16, (*color, 255), thickness=-1)
    else:
        cv2.rectangle(image, (8, 8), (40, 40), (*color, 255), thickness=-1)
    return BattleTemplate.from_bgra(f"enemy_{shape}", image, source="game_sprite", name=shape)


def composite(background: np.ndarray, template: BattleTemplate, x: int, y: int) -> None:
    height, width = template.alpha.shape
    alpha = template.alpha.astype(np.float32)[:, :, None] / 255.0
    region = background[y : y + height, x : x + width]
    blended = template.image.astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
    region[:] = blended.astype(np.uint8)


def test_template_classifier_uses_color_and_silhouette() -> None:
    circle = make_template("circle", (20, 20, 220))
    square = make_template("square", (220, 20, 20))
    classifier = TemplateClassifier(TemplateLibrary([circle, square]))
    background = np.full((80, 80, 3), 100, dtype=np.uint8)
    composite(background, circle, 16, 16)
    candidate = background[16 : 16 + circle.alpha.shape[0], 16 : 16 + circle.alpha.shape[1]]

    match = classifier.classify(candidate, circle.alpha)

    assert match is not None
    assert match.enemy_id == "enemy_circle"
    assert match.confidence > 0.99
    assert classifier.classify(candidate, circle.alpha, allowed_enemy_ids=["enemy_square"]).enemy_id == "enemy_square"


def test_template_classifier_marks_close_different_classes_as_ambiguous() -> None:
    circle = make_template("circle", (20, 20, 220))
    similar = BattleTemplate("enemy_circle_similar", "similar", circle.image, circle.alpha, circle.source)
    classifier = TemplateClassifier(TemplateLibrary([circle, similar]))
    background = np.full((80, 80, 3), 100, dtype=np.uint8)
    composite(background, circle, 16, 16)
    height, width = circle.alpha.shape
    candidate = background[16 : 16 + height, 16 : 16 + width]

    match = classifier.classify(candidate, circle.alpha)

    assert match is not None
    assert match.enemy_id == "enemy_circle"
    assert match.confidence <= 0.59


def test_foreground_detector_returns_candidates_only_for_changes() -> None:
    circle = make_template("circle", (20, 20, 220))
    background = np.full((100, 120, 3), 100, dtype=np.uint8)
    detector = ForegroundDetector(background, min_area=20)

    assert detector.detect(background) == ()

    frame = background.copy()
    composite(frame, circle, 30, 40)
    candidates = detector.detect(frame)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.area > 500
    assert candidate.bbox[0] >= 30
    assert candidate.foot_point[1] == candidate.bbox[1] + candidate.bbox[3] - 1


def test_red_guide_is_removed_without_removing_overlapped_target() -> None:
    background = np.full((120, 220, 3), 100, dtype=np.uint8)
    detector = ForegroundDetector(background, min_area=20, suppress_red_guides=True)
    frame = background.copy()
    cv2.line(frame, (10, 55), (210, 55), (20, 20, 220), 2)
    cv2.rectangle(frame, (95, 45), (115, 65), (220, 80, 40), -1)

    candidates = detector.detect(frame)

    assert len(candidates) == 1
    assert candidates[0].bbox[2] <= 25
    assert 40 <= candidates[0].foot_point[0] <= 120


def test_solid_red_target_is_not_classified_as_a_guide() -> None:
    background = np.full((100, 120, 3), 100, dtype=np.uint8)
    detector = ForegroundDetector(background, min_area=20, suppress_red_guides=True)
    frame = background.copy()
    cv2.circle(frame, (60, 50), 18, (20, 20, 220), -1)

    candidates = detector.detect(frame)

    assert len(candidates) == 1
    assert candidates[0].area > 500


def test_battlefield_detector_recognizes_known_and_marks_unknown() -> None:
    circle = make_template("circle", (20, 20, 220))
    classifier = TemplateClassifier(TemplateLibrary([circle]))
    background = np.full((100, 120, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(background, classifier, min_area=20)
    frame = background.copy()
    composite(frame, circle, 20, 30)

    known = detector.detect(frame)

    assert len(known) == 1
    assert known[0].enemy_id == "enemy_circle"
    assert known[0].confidence > 0.9

    other = make_template("square", (20, 220, 20))
    composite(frame, other, 70, 20)
    detections = detector.detect(frame)

    assert len(detections) == 2
    assert {detection.enemy_id for detection in detections} == {"enemy_circle", "unknown"}


def test_friendly_outside_deployable_mask_remains_unknown() -> None:
    sprite = np.zeros((48, 48, 4), dtype=np.uint8)
    cv2.circle(sprite, (24, 24), 16, (20, 20, 220, 255), -1)
    template = BattleTemplate.from_bgra("friendly", sprite, source="game_sprite")
    background = np.full((100, 120, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(
        background,
        TemplateClassifier(TemplateLibrary([template])),
        deployable_mask=np.zeros((100, 120), dtype=np.uint8),
        min_area=20,
    )
    frame = background.copy()
    composite(frame, template, 20, 30)

    detection = detector.detect(frame)[0]

    assert detection.enemy_id == "unknown"


def test_yellow_diamond_is_masked_but_unit_is_not() -> None:
    background = np.full((100, 160, 3), 100, dtype=np.uint8)
    detector = ForegroundDetector(background)
    frame = background.copy()
    cv2.fillConvexPoly(frame, np.array([[60, 20], [80, 40], [60, 60], [40, 40]]), (0, 210, 240))
    cv2.circle(frame, (110, 40), 14, (20, 220, 20), -1)

    mask = detector._artifact_mask(frame)

    assert mask[40, 60] == 255
    assert mask[40, 110] == 0
