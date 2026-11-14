from __future__ import annotations

from dataclasses import replace

from arkeye.detector import Detection
from arkeye.slot_tracking import StageSlotTracker


def blob(x: float, y: float = 500) -> Detection:
    return Detection(
        bbox=(round(x - 100), round(y - 50), 200, 100),
        foot_point=(x, y),
        enemy_id="enemy_1007_slime",
        confidence=0.8,
        template_name="test",
    )


def test_merged_group_never_manufactures_extra_slots() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        typical_bbox_areas={"enemy_1007_slime": 10_000},
    )

    spawned = tracker.update((blob(1500),), 0.0)
    assert len(spawned) == 1
    assert spawned[0].enemy_id == "enemy_1007_slime"
    assert spawned[0].foot_point == (1500.0, 500.0)

    lost = tracker.update((), 0.1)
    assert {track.status for track in lost} == {"lost"}

    reacquired = tracker.update((blob(1200),), 0.2)
    assert len(reacquired) == 1
    assert reacquired[0].foot_point == (1200.0, 500.0)
    assert reacquired[0].status == "confirmed"


def test_speed_limit_rejects_guide_line_jump_without_creating_duplicate() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        max_speed=300.0,
    )
    tracker.update((blob(1500),), 0.0)

    rejected = tracker.update((blob(1300),), 1 / 30)

    assert len(rejected) == 1
    assert rejected[0].status == "lost"
    assert rejected[0].foot_point == (1500.0, 500.0)

    recovered = tracker.update((blob(1490),), 2 / 30)
    assert len(recovered) == 1
    assert recovered[0].status == "confirmed"
    assert recovered[0].foot_point == (1490.0, 500.0)


def test_track_records_measured_velocity() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
    )
    tracker.update((blob(1500),), 0.0)

    snapshot = tracker.update((blob(1490),), 0.1)[0]

    assert snapshot.velocity == (-100.0, 0.0)


def test_class_requires_consistent_votes_before_using_stage_budget() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        class_confirmation_hits=2,
    )

    first = tracker.update((blob(1500),), 0.0)
    assert first[0].enemy_id == "unknown"
    assert tracker._activated_counts["enemy_1007_slime"] == 0

    second = tracker.update((blob(1490),), 0.1)
    assert second[0].enemy_id == "enemy_1007_slime"
    assert tracker._activated_counts["enemy_1007_slime"] == 1
    assert tracker._activated_counts["unknown"] == 0


def test_new_spawn_does_not_reuse_old_track_that_already_left_entrance() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
    )
    tracker.update((blob(1500),), 0.0)
    timestamp = 0.1
    for x in (1400, 1300, 1200, 1100, 1000):
        tracker.update((blob(x),), timestamp)
        timestamp += 0.1
    for _ in range(12):
        tracker.update((), timestamp)
        timestamp += 1 / 30

    snapshots = tracker.update((blob(1500),), timestamp)

    assert len(snapshots) == 2
    assert len({track.track_id for track in snapshots}) == 2


def test_spawn_time_defers_class_but_preserves_unknown_position() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1000_gopro_2": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        class_confirmation_hits=2,
        earliest_spawn_times={"enemy_1000_gopro_2": 34.1},
        battle_origin_timestamp=7.77,
    )
    early = tracker.update((replace(blob(1500), enemy_id="enemy_1000_gopro_2"),), 10.0)
    assert early[0].enemy_id == "unknown"
    assert early[0].foot_point == (1500.0, 500.0)

    too_early = tracker.update((replace(blob(1490), enemy_id="enemy_1000_gopro_2"),), 20.0)
    assert too_early[0].enemy_id == "unknown"

    on_time = tracker.update((replace(blob(1480), enemy_id="enemy_1000_gopro_2"),), 40.0)
    assert on_time[0].enemy_id == "enemy_1000_gopro_2"


def test_stage_time_starts_at_battle_origin_and_respects_slowdown() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1000_gopro_2": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        earliest_spawn_times={"enemy_1000_gopro_2": 10.0},
        battle_origin_timestamp=8.0,
        spawn_time_tolerance=0.0,
    )

    tracker.update((replace(blob(1500), enemy_id="enemy_1000_gopro_2"),), 8.0)
    tracker.set_time_scale(0.2)
    early = tracker.update((replace(blob(1490), enemy_id="enemy_1000_gopro_2"),), 18.0)
    tracker.set_time_scale(1.0)
    on_time = tracker.update((replace(blob(1480), enemy_id="enemy_1000_gopro_2"),), 26.0)

    assert early[0].enemy_id == "unknown"
    assert on_time[0].enemy_id == "enemy_1000_gopro_2"
    assert tracker._stage_elapsed == 10.0


def test_possible_friendly_requires_consistent_votes() -> None:
    tracker = StageSlotTracker(
        expected_counts={},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        class_confirmation_by_id={"friendly": 2},
    )
    candidate = replace(blob(900), enemy_id="unknown", confidence=0.59, possible_friendly=True)

    assert tracker.update((candidate,), 0.0)[0].enemy_id == "unknown"
    assert tracker.update((candidate,), 0.1)[0].enemy_id == "unknown"
    assert tracker.update((candidate,), 0.2)[0].enemy_id == "friendly"
