from __future__ import annotations

import numpy as np
import pytest

from arkeye.detector import Detection
from arkeye.motion import RouteSpeedFilter
from arkeye.slot_tracking import StageSlotTracker


def blob(x: float, y: float = 500) -> Detection:
    return Detection(
        bbox=(round(x - 30), round(y - 40), 60, 80),
        foot_point=(x, y),
        enemy_id="enemy_1007_slime",
        confidence=0.9,
        template_name="test",
    )


def filter_with_one_slot() -> RouteSpeedFilter:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
    )
    route = np.asarray([[1700, 500], [600, 500]], dtype=np.int32)
    return RouteSpeedFilter(tracker, (route,), frame_diagonal=2000.0, pixels_per_tile=100.0)


def test_fast_path_track_is_rejected_on_its_second_observation() -> None:
    motion = filter_with_one_slot()

    assert motion.update((blob(1500),), 0.0) == ()
    assert motion.update((blob(1300),), 0.02) == ()
    assert motion.update((blob(1100),), 0.04) == ()

    assert motion.speed_threshold == pytest.approx(250.0)
    assert motion.motion_speeds


def test_slow_path_target_is_not_rejected() -> None:
    motion = filter_with_one_slot()

    motion.update((blob(1500),), 0.0)
    snapshots = motion.update((blob(1498),), 0.02)

    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "enemy_1007_slime"


def test_speed_threshold_follows_battle_speed() -> None:
    motion = filter_with_one_slot()
    motion.set_game_speed(2)

    assert motion.speed_threshold == pytest.approx(500.0)


def test_fixed_speed_rejects_later_guide_without_duplicate_target() -> None:
    motion = filter_with_one_slot()
    motion.update((blob(1500),), 0.0)
    motion.update((blob(1300),), 0.02)
    motion.update((blob(1100),), 0.04)
    motion.update((blob(1500),), 0.06)
    normal = motion.update((blob(1495),), 0.08)
    assert len(normal) == 1

    rejected = motion.update((blob(1300),), 0.10)

    assert len(rejected) == 1
    assert rejected[0].status == "lost"
    assert rejected[0].foot_point == (1495.0, 500.0)

    motion.update((blob(1495),), 0.12)
    recovered = motion.update((blob(1490),), 0.14)
    assert len(recovered) == 1
    assert recovered[0].status == "confirmed"
    assert recovered[0].track_id == normal[0].track_id


def test_off_path_candidate_never_activates_an_output_track() -> None:
    motion = filter_with_one_slot()

    assert motion.update((blob(1500, 850),), 0.0) == ()
    assert motion.update((blob(1498, 850),), 1 / 30) == ()


def test_enemy_on_another_classs_route_is_preserved_as_unknown() -> None:
    motion = filter_with_one_slot()
    other_route = np.asarray([[1700, 850], [600, 850]], dtype=np.int32)
    motion.route_lines = (*motion.route_lines, other_route)
    motion.class_route_indices = {"enemy_1007_slime": (0,)}

    assert motion.update((blob(1500, 850),), 0.0) == ()
    snapshots = motion.update((blob(1498, 850),), 1 / 30)
    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "unknown"


def test_early_classification_is_demoted_without_losing_position() -> None:
    motion = filter_with_one_slot()
    motion.earliest_spawn_times = {"enemy_1007_slime": 30.0}
    motion.battle_origin_timestamp = 8.0

    assert motion.update((blob(1500),), 9.0) == ()
    snapshots = motion.update((blob(1498),), 9.0 + 1 / 30)

    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "unknown"
    assert snapshots[0].foot_point == (1498.0, 500.0)


def test_route_is_inactive_before_its_first_scheduled_spawn() -> None:
    motion = filter_with_one_slot()
    motion.route_earliest_times = {0: 10.0}
    motion.spawn_time_tolerance = 0.0

    assert motion.update((blob(1500),), 9.0) == ()
    assert motion.update((blob(1498),), 9.0 + 1 / 30) == ()
    assert motion.update((blob(1496),), 10.0) == ()
    snapshots = motion.update((blob(1494),), 10.0 + 1 / 30)
    assert len(snapshots) == 1


def test_deployment_slowdown_scales_stage_time_from_battle_origin() -> None:
    motion = filter_with_one_slot()
    motion.route_earliest_times = {0: 10.0}
    motion.spawn_time_tolerance = 0.0
    motion.battle_origin_timestamp = 8.0

    assert motion.update((blob(1500),), 8.0) == ()
    motion.set_time_scale(0.2)
    assert motion.update((blob(1498),), 18.0) == ()
    motion.set_time_scale(1.0)
    assert motion.update((blob(1496),), 24.0) == ()
    assert motion.update((blob(1494),), 26.0) == ()
    snapshots = motion.update((blob(1492),), 26.0 + 1 / 30)

    assert len(snapshots) == 1
    assert motion._stage_elapsed == pytest.approx(10.0 + 1 / 30)


def test_mid_route_artifact_cannot_birth_a_new_track() -> None:
    motion = filter_with_one_slot()

    assert motion.update((blob(900),), 0.0) == ()
    assert motion.update((blob(898),), 1 / 30) == ()


def test_possible_friendly_is_preserved_before_class_confirmation() -> None:
    motion = filter_with_one_slot()
    original = blob(900)
    candidate = Detection(
        original.bbox,
        original.foot_point,
        "unknown",
        0.59,
        None,
        possible_friendly=True,
    )

    assert motion.update((candidate,), 0.0) == ()
    snapshots = motion.update((candidate,), 1 / 30)
    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "unknown"


def test_stage_route_and_time_resolve_visual_ambiguity() -> None:
    motion = filter_with_one_slot()
    motion.earliest_spawn_times = {"enemy_1007_slime": 30.0}
    motion.class_route_indices = {"enemy_1007_slime": (0,)}
    motion.battle_origin_timestamp = 8.0
    original = blob(1500)
    candidate = Detection(
        original.bbox,
        original.foot_point,
        "unknown",
        0.59,
        None,
        candidate_enemy_id="enemy_1007_slime",
    )

    assert motion.update((candidate,), 9.0) == ()
    assert motion.update((candidate,), 9.0 + 1 / 30)[0].enemy_id == "unknown"
    assert motion.update((candidate,), 39.0)[0].enemy_id == "enemy_1007_slime"


def test_shared_route_does_not_resolve_visual_ambiguity() -> None:
    motion = filter_with_one_slot()
    motion.earliest_spawn_times = {"enemy_1007_slime": 30.0, "enemy_1000_gopro_2": 30.0}
    motion.class_route_indices = {"enemy_1007_slime": (0,), "enemy_1000_gopro_2": (0,)}
    motion.battle_origin_timestamp = 8.0
    original = blob(1500)
    candidate = Detection(
        original.bbox,
        original.foot_point,
        "unknown",
        0.59,
        None,
        candidate_enemy_id="enemy_1007_slime",
    )

    assert motion.update((candidate,), 39.0) == ()
    assert motion.update((candidate,), 39.0 + 1 / 30)[0].enemy_id == "unknown"
