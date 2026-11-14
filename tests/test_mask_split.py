from __future__ import annotations

import json
from pathlib import Path

import cv2

from arkeye.assets import TemplateLibrary
from arkeye.board import BoardCalibration
from arkeye.detector import BattlefieldDetector, TemplateClassifier
from arkeye.routes import build_route_mask
from arkeye.tracking import TrackSnapshot


def track(track_id: int, enemy_id: str, bbox) -> TrackSnapshot:
    x, y, width, height = bbox
    return TrackSnapshot(
        track_id=track_id,
        enemy_id=enemy_id,
        confidence=0.9,
        status="confirmed",
        bbox=bbox,
        foot_point=(x + width / 2, y + height - 1),
        velocity=(0, 0),
        hits=5,
        misses=0,
        occlusion_group=None,
        history=(),
    )


def detector():
    level = json.loads(Path("tests/fixtures/level_main_00-01.json").read_text(encoding="utf-8"))
    calibration = BoardCalibration.load("configs/main_00-01.json")
    background = cv2.imread("tests/fixtures/t_08.jpg")
    roi = build_route_mask(level, calibration, background.shape, width_tiles=1.0)
    return BattlefieldDetector(
        background,
        TemplateClassifier(TemplateLibrary.load_manifest("assets/0_1/manifest.json")),
        roi_mask=roi,
        confidence_threshold=0.6,
        min_area=500,
        blur_size=5,
    )


def test_alpha_mask_splits_friend_and_slimes_into_components() -> None:
    frame = cv2.imread("tests/fixtures/t_24.jpg")
    tracks = [track(1, "friendly", (413, 335, 178, 275)), track(2, "enemy_1007_slime", (450, 450, 150, 150))]

    detections = detector().detect(frame, overlapping_tracks=tracks)

    classes = [detection.enemy_id for detection in detections]
    assert "friendly" in classes
    assert classes.count("enemy_1007_slime") >= 2
    assert "enemy_1002_nsabr" not in classes


def test_alpha_mask_splits_friend_and_soldier_into_components() -> None:
    frame = cv2.imread("tests/fixtures/t_48.jpg")
    tracks = [track(1, "friendly", (409, 335, 152, 265)), track(2, "enemy_1002_nsabr", (430, 450, 150, 150))]

    detections = detector().detect(frame, overlapping_tracks=tracks)

    classes = {detection.enemy_id for detection in detections}
    assert classes == {"friendly", "enemy_1002_nsabr"}
    assert all(detection.bbox[2] > 0 and detection.bbox[3] > 0 for detection in detections)
