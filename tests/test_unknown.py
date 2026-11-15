from __future__ import annotations

from arkeye.detector import Detection
from arkeye.slot_tracking import StageSlotTracker


def detection(x: float, enemy_id="unknown") -> Detection:
    return Detection(
        bbox=(round(x - 100), 450, 200, 100),
        foot_point=(x, 549),
        enemy_id=enemy_id,
        confidence=0.0 if enemy_id == "unknown" else 0.8,
        template_name=None,
    )


def test_unknown_detection_creates_track_outside_spawn_region() -> None:
    tracker = StageSlotTracker(expected_counts={}, spawn_bbox=(1200, 350, 550, 400), despawn_x=700)

    snapshots = tracker.update((detection(700),), 0.0)

    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "unknown"
    assert snapshots[0].foot_point == (700.0, 549.0)


def test_unknown_detection_reuses_known_track_without_changing_class() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
    )
    tracker.update((detection(1500, "enemy_1007_slime"),), 0.0)

    snapshots = tracker.update((detection(1450),), 0.1)

    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "enemy_1007_slime"


def test_unknown_merged_blob_preserves_existing_slots() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        typical_bbox_areas={"enemy_1007_slime": 10_000},
        max_speed=4000.0,
    )
    tracker.update((detection(1500, "enemy_1007_slime"), detection(1400, "enemy_1007_slime")), 0.0)
    tracker.update((), 0.1)

    snapshots = tracker.update((detection(1450),), 0.2)

    assert len(snapshots) == 2
    assert {track.status for track in snapshots} == {"confirmed", "occluded"}
    assert {track.enemy_id for track in snapshots} == {"enemy_1007_slime"}


def test_stale_unknown_track_is_held_and_reacquired() -> None:
    tracker = StageSlotTracker(expected_counts={}, spawn_bbox=(1200, 350, 550, 400), despawn_x=700)
    first = tracker.update((detection(700),), 0.0)

    for index in range(500):
        snapshots = tracker.update((), (index + 1) / 30)

    assert len(snapshots) == 1
    assert snapshots[0].status == "lost"
    assert snapshots[0].track_id == first[0].track_id

    recovered = tracker.update((detection(700),), 17.0)
    assert recovered[0].track_id == first[0].track_id
    assert recovered[0].status == "confirmed"
