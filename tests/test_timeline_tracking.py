from __future__ import annotations

import numpy as np
import pytest

from arkeye.board import BoardCalibration, Homography
from arkeye.detector import Detection
from arkeye.timeline import TimedSpawnEvent
from arkeye.timeline_tracking import RouteTimeline, TimelinePrimaryTracker
from arkeye.tracking import MultiObjectTracker


def calibration() -> BoardCalibration:
    identity = Homography.identity()
    return BoardCalibration(preview_size=(100, 100), map_to_preview=identity, preview_to_frame=identity)


def route() -> dict:
    return {
        "motionMode": "WALK",
        "startPosition": {"col": 2, "row": 0},
        "endPosition": {"col": 0, "row": 0},
        "checkpoints": [
            {"type": "MOVE", "position": {"col": 1, "row": 0}},
            {"type": "WAIT_FOR_SECONDS", "time": 2.0},
            {"type": "MOVE", "position": {"col": 0, "row": 0}},
        ],
    }


def spawn(enemy_id: str = "enemy_1007_slime", count: int = 1) -> TimedSpawnEvent:
    return TimedSpawnEvent(
        timestamp=0.0,
        wave_timestamp=0.0,
        enemy_key=enemy_id,
        count=count,
        interval=1.0,
        route_index=0,
        wave_index=0,
        fragment_index=0,
        action_index=0,
    )


def tracker(count: int = 1) -> TimelinePrimaryTracker:
    return TimelinePrimaryTracker(
        visual_tracker=MultiObjectTracker(min_hits=1),
        routes=(RouteTimeline.from_mapping(route(), calibration()),),
        spawn_events=(spawn(count=count),),
        enemy_move_speeds={"enemy_1007_slime": 1.0},
        typical_bbox_areas={"enemy_1007_slime": 10_000},
        pixels_per_tile=100.0,
        battle_origin_timestamp=0.0,
    )


def blob(x: float, enemy_id: str = "enemy_1007_slime") -> Detection:
    return Detection(
        bbox=(round(x - 20), -40, 40, 40),
        foot_point=(x, 0),
        enemy_id=enemy_id,
        confidence=0.9,
        template_name=None,
        possible_friendly=enemy_id == "friendly",
        faction_evidence="friendly" if enemy_id == "friendly" else "enemy",
    )


def test_route_timeline_includes_waiting_checkpoint() -> None:
    timeline = RouteTimeline.from_mapping(route(), calibration())

    assert timeline.total_duration(1.0) == pytest.approx(4.0)
    assert np.allclose(timeline.position_at(0.5, 1.0)[0], [1.5, 0])
    assert np.allclose(timeline.position_at(1.5, 1.0)[0], [1.0, 0])
    assert np.allclose(timeline.position_at(2.5, 1.0)[0], [1.0, 0])
    assert timeline.position_at(4.0, 1.0)[1]


def test_unobserved_spawn_is_explicitly_predicted_and_moves() -> None:
    timeline = tracker()

    initial = timeline.update((), 0.0)
    later = timeline.update((), 1.0)

    assert initial[0].status == "predicted"
    assert initial[0].enemy_id == "enemy_1007_slime"
    assert initial[0].foot_point == pytest.approx((2.0, 0.0))
    assert later[0].status == "predicted"
    assert later[0].foot_point == pytest.approx((1.0, 0.0))
    assert later[0].track_id == initial[0].track_id
    assert later[0].hits == 0


def test_visual_detection_confirms_but_does_not_replace_timeline_class() -> None:
    timeline = tracker()
    timeline.update((), 0.0)

    wrong = timeline.update((blob(1.0, "enemy_1002_nsabr"),), 1.0)
    assert wrong[0].status == "predicted"
    assert wrong[0].enemy_id == "enemy_1007_slime"

    confirmed = timeline.update((blob(1.2),), 2.0)
    assert confirmed[0].status == "confirmed"
    assert confirmed[0].enemy_id == "enemy_1007_slime"
    assert confirmed[0].foot_point[0] == pytest.approx(1.05)
    assert confirmed[0].hits == 1
    assert len(confirmed) == 1


def test_coast_advances_prediction_without_adding_a_miss() -> None:
    timeline = tracker()
    timeline.update((), 0.0)

    coasted = timeline.coast(0.5)

    assert coasted[0].foot_point == pytest.approx((1.5, 0.0))
    assert coasted[0].misses == 1


def test_spawn_count_creates_stable_distinct_slots() -> None:
    timeline = tracker(count=2)

    first = timeline.update((), 0.0)
    second = timeline.update((), 1.0)

    assert [track.track_id for track in first] == [1]
    assert [track.track_id for track in second] == [1, 2]
    assert second[1].status == "predicted"


def test_route_end_retires_scheduled_track() -> None:
    timeline = tracker()
    timeline.update((), 0.0)

    assert timeline.update((), 5.0)[0].status == "predicted"
    assert timeline.update((), 6.0) == ()


def test_route_visual_evidence_restores_a_blocked_slot_with_the_same_id() -> None:
    timeline = tracker()
    original_id = timeline.update((), 0.0)[0].track_id
    assert timeline.update((), 6.0) == ()

    restored = timeline.update((blob(1.0),), 7.0)

    assert len(restored) == 1
    assert restored[0].track_id == original_id
    assert restored[0].status == "confirmed"
    assert timeline._tracks[original_id].progress_delay > 0


def test_friendly_visual_track_is_retained_with_distinct_id() -> None:
    timeline = tracker()
    timeline.update((), 0.0)

    snapshots = timeline.update((blob(1.0), blob(1.0, "friendly")), 1.0)

    assert len(snapshots) == 2
    friendly = next(track for track in snapshots if track.enemy_id == "friendly")
    assert friendly.track_id == 1001
    assert snapshots[0].status == "confirmed"
