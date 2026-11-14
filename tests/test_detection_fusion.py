from __future__ import annotations

import cv2
import numpy as np

from arkeye.assets import BattleTemplate, TemplateLibrary
from arkeye.detector import BattlefieldDetector, Detection, TemplateClassifier
from arkeye.health_bars import HealthBarDetector
from arkeye.motion import RouteSpeedFilter
from arkeye.slot_tracking import StageSlotTracker


def test_classifier_preserves_aspect_and_bottom_alignment() -> None:
    sprite = np.zeros((20, 40, 4), dtype=np.uint8)
    sprite[:, :, :3] = (20, 80, 220)
    sprite[:, :, 3] = 255
    template = BattleTemplate.from_bgra("enemy_wide", sprite, source="game_sprite")
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    image[20:40] = template.image
    mask[20:40] = 255

    match = TemplateClassifier(TemplateLibrary([template])).classify(image, mask)

    assert match is not None
    assert match.confidence > 0.94
    assert match.color_score == 1.0
    assert match.silhouette_score == 1.0


def test_paired_friendly_bars_override_ambiguous_visual_class() -> None:
    sprite = np.zeros((48, 48, 4), dtype=np.uint8)
    cv2.circle(sprite, (24, 24), 16, (20, 20, 220, 255), -1)
    template = BattleTemplate.from_bgra("enemy_circle", sprite, source="game_sprite")
    background = np.full((120, 120, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(
        background,
        TemplateClassifier(TemplateLibrary([template])),
        allowed_enemy_ids=("enemy_circle", "friendly"),
        min_area=20,
        health_bars=HealthBarDetector(),
    )
    frame = background.copy()
    height, width = template.alpha.shape
    frame[30 : 30 + height, 20 : 20 + width] = template.image
    cv2.rectangle(frame, (20, 82), (67, 89), (200, 200, 0), -1)
    cv2.rectangle(frame, (20, 92), (67, 98), (50, 200, 50), -1)

    detections = detector.detect(frame)
    anchored = [detection for detection in detections if detection.faction_evidence == "friendly"]

    assert anchored
    assert all(detection.enemy_id == "friendly" for detection in anchored)
    assert all(detection.possible_friendly for detection in anchored)


def test_stale_motion_track_cannot_jump_beyond_recovery_radius() -> None:
    inner = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
    )
    route = np.asarray([[1700, 500], [0, 500]], dtype=np.int32)
    motion = RouteSpeedFilter(inner, (route,), frame_diagonal=2000.0, pixels_per_tile=100.0)
    first = Detection((1470, 460, 60, 80), (1500, 539), "enemy_1007_slime", 0.9, "test")
    second = Detection((1468, 460, 60, 80), (1498, 539), "enemy_1007_slime", 0.9, "test")
    distant = Detection((970, 460, 60, 80), (1000, 539), "enemy_1007_slime", 0.9, "test")
    motion.update((first,), 0.0)
    motion.update((second,), 0.02)

    snapshots = motion.update((distant,), 10.0)

    assert len(snapshots) == 1
    assert snapshots[0].status == "lost"
    assert snapshots[0].foot_point == (1498.0, 539.0)


def test_occluded_track_keeps_its_own_bbox() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        typical_bbox_areas={"enemy_1007_slime": 10_000},
    )
    first = Detection((1400, 450, 200, 100), (1500, 549), "enemy_1007_slime", 0.8, "test")
    second = Detection((1300, 450, 200, 100), (1400, 549), "enemy_1007_slime", 0.8, "test")
    tracker.update((first, second), 0.0)
    tracker.update((), 0.1)
    merged = Detection((1050, 400, 400, 200), (1250, 599), "enemy_1007_slime", 0.8, "test")

    snapshots = tracker.update((merged,), 0.2)

    occluded = [track for track in snapshots if track.status == "occluded"]
    assert len(occluded) == 2
    assert all(track.bbox in {first.bbox, second.bbox} for track in occluded)
    assert all(track.status == "occluded" for track in snapshots)


def test_ambiguous_candidate_requires_consecutive_class_votes() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1, "enemy_1002_nsabr": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        class_confirmation_hits=2,
    )

    def candidate(enemy_id: str) -> Detection:
        return Detection(
            (1450, 450, 100, 100),
            (1500, 549),
            "unknown",
            0.59,
            None,
            candidate_enemy_id=enemy_id,
            faction_evidence="enemy",
        )

    tracker.update((candidate("enemy_1007_slime"),), 0.0)
    tracker.update((candidate("enemy_1007_slime"),), 0.1)
    tracker.update((candidate("enemy_1002_nsabr"),), 0.2)
    tracker.update((candidate("enemy_1007_slime"),), 0.3)
    confirmed = tracker.update((candidate("enemy_1007_slime"),), 0.4)

    assert len(confirmed) == 1
    assert confirmed[0].enemy_id == "enemy_1007_slime"
