from __future__ import annotations

import cv2
import numpy as np

from arkeye.assets import TemplateLibrary
from arkeye.detector import TemplateClassifier


def composite(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, mask: np.ndarray) -> None:
    height, width = rgba.shape[:2]
    alpha = rgba[:, :, 3].astype(np.float32)[:, :, None] / 255.0
    region = canvas[y : y + height, x : x + width]
    region[:] = (rgba[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    mask[y : y + height, x : x + width] = np.maximum(mask[y : y + height, x : x + width], rgba[:, :, 3])


def test_locate_splits_friend_and_enemy_inside_one_blob() -> None:
    friendly = cv2.imread("assets/0_1/friendly_t12.png", cv2.IMREAD_UNCHANGED)
    slime = cv2.imread("assets/0_1/slime_t12.png", cv2.IMREAD_UNCHANGED)
    classifier = TemplateClassifier(TemplateLibrary.load_manifest("assets/0_1/manifest.json"))
    canvas = np.full((300, 220, 3), 100, dtype=np.uint8)
    silhouette = np.zeros((300, 220), dtype=np.uint8)
    composite(canvas, friendly, 15, 10, silhouette)
    composite(canvas, slime, 60, 190, silhouette)

    matches = classifier.locate(canvas, silhouette, confidence_threshold=0.6)

    classes = {match.enemy_id for match in matches}
    assert "friendly" in classes
    assert "enemy_1007_slime" in classes
    centers = [match.bbox[0] + match.bbox[2] / 2 for match in matches]
    assert len(set(centers)) > 1


def test_locate_does_not_turn_friend_into_soldier() -> None:
    frame = cv2.imread("tests/fixtures/t_24.jpg")
    x, y, width, height = 413, 335, 178, 275
    candidate = frame[y : y + height, x : x + width]
    silhouette = cv2.imread("tests/fixtures/0_1_route_mask.png", cv2.IMREAD_GRAYSCALE)[y : y + height, x : x + width]
    classifier = TemplateClassifier(TemplateLibrary.load_manifest("assets/0_1/manifest.json"))

    matches = classifier.locate(candidate, silhouette, confidence_threshold=0.6)

    assert "friendly" in {match.enemy_id for match in matches}
    assert "enemy_1002_nsabr" not in {match.enemy_id for match in matches}
