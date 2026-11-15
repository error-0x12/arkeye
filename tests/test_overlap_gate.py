from __future__ import annotations

from unittest.mock import Mock

import cv2
import numpy as np

from arkeye.assets import BattleTemplate, TemplateLibrary
from arkeye.detector import BattlefieldDetector, TemplateClassifier
from arkeye.tracking import TrackSnapshot


def track_at(bbox, enemy_id="enemy_a") -> TrackSnapshot:
    x, y, width, height = bbox
    return TrackSnapshot(
        track_id=1,
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


def test_local_split_only_runs_when_existing_tracks_overlap() -> None:
    sprite = np.zeros((40, 40, 4), dtype=np.uint8)
    cv2.circle(sprite, (20, 20), 14, (20, 20, 220, 255), -1)
    template = BattleTemplate.from_bgra("enemy_a", sprite, source="game_sprite")
    classifier = TemplateClassifier(TemplateLibrary([template]))
    classifier.locate = Mock(wraps=classifier.locate)
    background = np.full((100, 140, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(background, classifier, confidence_threshold=0.99, min_area=20)
    frame = background.copy()
    height, width = template.alpha.shape
    frame[30 : 30 + height, 50 : 50 + width] = template.image
    bbox = (50, 30, width, height)

    detector.detect(frame, overlapping_tracks=[track_at(bbox)])
    assert classifier.locate.call_count == 0

    detector.detect(frame, overlapping_tracks=[track_at(bbox), track_at(bbox, "enemy_b")])
    assert classifier.locate.call_count == 1


def test_normal_match_is_smoothed_but_impossible_reacquire_is_rejected() -> None:
    from arkeye.detector import Detection
    from arkeye.slot_tracking import StageSlotTracker

    def detection(x):
        return Detection((round(x - 30), 450, 60, 100), (x, 549), "enemy_1007_slime", 0.9, "test")

    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
    )
    tracker.update((detection(1500),), 0.0)
    smoothed = tracker.update((detection(1400),), 0.1)[0]

    assert smoothed.foot_point[0] == 1465.0
    assert smoothed.foot_point[1] == 549.0

    tracker.update((), 0.2)
    tracker.update((), 0.3)
    reacquired = tracker.update((detection(1200),), 0.4)[0]
    assert reacquired.status == "lost"
    assert reacquired.foot_point[0] == 1465.0
