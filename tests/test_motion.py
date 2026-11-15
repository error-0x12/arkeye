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


def filter_with_slots(count: int) -> RouteSpeedFilter:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": count},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
    )
    route = np.asarray([[1700, 500], [600, 500]], dtype=np.int32)
    return RouteSpeedFilter(tracker, (route,), frame_diagonal=2000.0, pixels_per_tile=100.0)


def test_fast_unrecognized_candidate_is_rejected_as_guide() -> None:
    motion = filter_with_one_slot()

    assert motion.update((unknown_blob(1500),), 0.0) == ()
    assert motion.update((unknown_blob(1300),), 0.02) == ()
    assert motion.update((unknown_blob(1100),), 0.04) == ()

    assert motion.speed_threshold == pytest.approx(250.0)
    assert motion.motion_speeds


def test_guide_rejection_is_recorded_with_the_candidate() -> None:
    """被指引规则丢弃的候选必须留在 trace 里，否则看起来像凭空消失。"""

    motion = filter_with_one_slot()
    frames: list[dict] = []
    motion.trace_hook = frames.append

    motion.update((unknown_blob(1500),), 0.0)
    motion.update((unknown_blob(1300),), 0.02)

    guides = [item for frame in frames for item in frame.get("guides", [])]
    assert guides, "a rejected guide candidate should be logged"
    assert guides[0]["detection"]["bbox"], "the log must point at the rejected candidate"


def test_recognized_candidate_is_never_dropped_as_a_guide() -> None:
    """已识别出具体敌人类别的候选是真单位，不能被路线指引规则吃掉。"""

    motion = filter_with_one_slot()
    frames: list[dict] = []
    motion.trace_hook = frames.append

    motion.update((blob(1500),), 1.0)
    motion.update((blob(1300),), 1.2)

    guides = [item for frame in frames for item in frame.get("guides", [])]
    new_tracks = [item for frame in frames for item in frame.get("new_tracks", [])]
    assert not guides
    assert len(new_tracks) >= 2, "it should become its own candidate track instead"


def test_lost_track_does_not_steal_a_detection_from_a_confirmed_track() -> None:
    """HELD 轨迹重新接回要付代价，新检测优先归已确认目标。"""

    motion = filter_with_slots(2)
    frames: list[dict] = []
    motion.trace_hook = frames.append

    motion.update((blob(1000), blob(1100)), 1.0)      # 两条轨迹各自建立
    motion.update((blob(1100),), 1.2)                 # 只喂右边那条 -> 左边开始丢失
    frames.clear()
    motion.update((blob(1040),), 1.4)                 # 两边都在门限内，距离左边更近

    matched = [item["track_id"] for frame in frames for item in frame.get("matched", [])]
    assert matched == [2], "the confirmed track should win over the HELD one"


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
    assert rejected[0].misses >= 1
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


def test_initial_mid_route_target_without_health_bar_is_rejected() -> None:
    motion = filter_with_one_slot()

    assert motion.update((blob(900),), 0.0) == ()
    assert motion.update((blob(898),), 1 / 30) == ()
    assert motion._tracks == {}


def test_initial_mid_route_target_with_health_bar_can_be_tracked() -> None:
    motion = filter_with_one_slot()
    candidate = Detection(
        blob(900).bbox,
        blob(900).foot_point,
        blob(900).enemy_id,
        blob(900).confidence,
        blob(900).template_name,
        faction_evidence="enemy",
        health_bar_bbox=(870, 440, 60, 8),
    )

    assert motion.update((candidate,), 0.0) == ()
    snapshots = motion.update((candidate,), 1 / 30)
    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "enemy_1007_slime"


def test_visible_on_path_target_can_birth_after_route_entry() -> None:
    motion = filter_with_one_slot()

    assert motion.update((blob(900),), 1.0) == ()
    snapshots = motion.update((blob(898),), 1.0 + 1 / 30)
    assert len(snapshots) == 1
    assert snapshots[0].foot_point == (898.0, 500.0)


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


def motion_with_spawn_record() -> RouteSpeedFilter:
    from arkeye.timeline import TimedSpawnEvent

    motion = filter_with_one_slot()
    motion.spawn_events = (
        TimedSpawnEvent(
            timestamp=30.0,
            wave_timestamp=30.0,
            enemy_key="enemy_1007_slime",
            count=1,
            interval=0.0,
            route_index=0,
            wave_index=0,
            fragment_index=0,
            action_index=0,
        ),
    )
    return motion


def unknown_blob(x: float) -> Detection:
    original = blob(x)
    return Detection(original.bbox, original.foot_point, "unknown", 0.5, None)


def test_stationary_unscheduled_candidate_is_interference_without_track_id() -> None:
    motion = motion_with_spawn_record()

    for index in range(6):
        assert motion.update((unknown_blob(900),), 1.0 + index / 30) == ()

    assert motion.tracker.snapshots() == ()
    assert motion._tracks[1].guide


def test_moving_unscheduled_unknown_can_be_tracked_after_motion_evidence() -> None:
    motion = motion_with_spawn_record()

    for index, x in enumerate((900, 898, 896)):
        assert motion.update((unknown_blob(x),), 1.0 + index / 30) == ()
    snapshots = motion.update((unknown_blob(894),), 1.1)

    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "unknown"


def test_scheduled_spawn_entry_is_not_rejected_as_stationary_interference() -> None:
    motion = motion_with_spawn_record()

    assert motion.update((unknown_blob(1700),), 30.0) == ()
    snapshots = motion.update((unknown_blob(1700),), 30.0 + 1 / 30)

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
    # 持续推进战斗时间越过最早出怪时刻，同一目标应保持轨迹并恢复类别；
    # 长时间的“消失后再出现”不再沿用旧轨迹身份（那是干扰目标的典型特征）。
    snapshots: tuple = ()
    timestamp = 9.0 + 1 / 30
    while timestamp < 39.0:
        timestamp += 1 / 30
        snapshots = motion.update((candidate,), timestamp)
    assert snapshots[0].enemy_id == "enemy_1007_slime"


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


def test_stale_motion_track_is_deleted_instead_of_vouching_forever() -> None:
    motion = filter_with_one_slot()
    motion.update((blob(1500),), 0.0)
    motion.update((blob(1498),), 1 / 30)

    # 连续超过 max_track_misses 帧没有任何匹配，内部轨迹必须被删除
    for index in range(2, 2 + motion.max_track_misses + 3):
        motion.update((), index / 30)
    assert not motion._tracks

    # 删除后同一位置再次出现的目标必须重新积累运动证据
    motion.update((blob(1496),), 1.0)
    assert len(motion._tracks) == 1
    assert next(iter(motion._tracks.values())).hits == 1
    motion.update((blob(1494),), 1.0 + 1 / 30)
    assert next(iter(motion._tracks.values())).hits == 2


def test_long_lost_track_may_not_adopt_unrelated_detection() -> None:
    motion = filter_with_one_slot()
    motion.update((blob(1500),), 0.0)
    motion.update((blob(1500),), 1 / 30)
    # 真实目标消失很久后，路径上出现了一个无血条、类别不同的候选
    stranger = Detection(
        bbox=(1530, 460, 60, 80),
        foot_point=(1560, 500),
        enemy_id="enemy_1000_gopro_2",
        confidence=0.8,
        template_name="hound",
    )
    gap = 8 / 30
    for offset in range(4):
        motion.update((), gap + offset / 30)
    snapshots = motion.update((stranger,), gap + 4 / 30)
    # 收养被拒绝：陌生人以全新轨迹身份只算 1 次命中，不能立即输出；
    # 返回的只是原 slime 轨迹的惯性预测
    assert all(snapshot.enemy_id != "enemy_1000_gopro_2" for snapshot in snapshots)
    stranger_tracks = [
        track for track in motion._tracks.values() if track.detection.enemy_id == "enemy_1000_gopro_2"
    ]
    assert len(stranger_tracks) == 1
    assert stranger_tracks[0].hits == 1


def test_class_scattering_track_is_rejected_as_interference() -> None:
    motion = filter_with_one_slot()
    identity = [(1500, "enemy_1007_slime"), (1498, "enemy_1000_gopro_2"), (1496, "enemy_1011_wizard")]

    def classed(x: float, enemy_id: str) -> Detection:
        return Detection(
            bbox=(round(x - 30), 460, 60, 80),
            foot_point=(x, 500),
            enemy_id=enemy_id,
            confidence=0.9,
            template_name="test",
        )

    snapshots: tuple = ()
    for index, (x, enemy_id) in enumerate(identity):
        snapshots = motion.update((classed(x, enemy_id),), index / 30)
    # 类别在多个互不相关的敌人之间跳变，必须按干扰丢弃
    assert snapshots == ()


def test_stationary_near_spawn_unknown_becomes_interference_after_grace() -> None:
    motion = motion_with_spawn_record()
    motion.update((unknown_blob(1700),), 30.0)
    snapshots = motion.update((unknown_blob(1700),), 30.0 + 1 / 30)
    assert len(snapshots) == 1  # 出怪口附近的未识别目标先放行

    track_id = snapshots[0].track_id
    # 但真实敌人出怪后会立即移动；长期驻场的无血条目标按干扰清除
    last: tuple = ()
    saw_guide = False
    for index in range(2, 2 + motion.STATIONARY_GRACE_MATCHES + 4):
        last = motion.update((unknown_blob(1700),), 30.0 + index / 30)
        if any(track.guide for track in motion._tracks.values()):
            saw_guide = True
    assert saw_guide, "长期驻场的无血条未识别目标必须被判为干扰"
    # 干扰被隔离后不再供给检测；保持窗内状态仍是 confirmed，但 misses 会开始累积
    assert all(snapshot.misses >= 1 for snapshot in last if snapshot.track_id == track_id)


def test_class_votes_require_strict_majority() -> None:
    motion = filter_with_one_slot()

    def classed(x: float, enemy_id: str) -> Detection:
        return Detection(
            bbox=(round(x - 30), 460, 60, 80),
            foot_point=(x, 500),
            enemy_id=enemy_id,
            confidence=0.9,
            template_name="test",
        )

    motion.update((classed(1500, "enemy_1005_yokai"),), 0.0)
    snapshots = motion.update((classed(1498, "enemy_1007_slime_2"),), 1 / 30)
    # 两类各一票、无严格多数，不允许输出
    assert snapshots == ()


# --------------------------------------------------------------- 路线软权重

def test_route_indices_stay_aligned_when_a_route_is_degenerate() -> None:
    """点数不足的路线必须保留占位，否则后面所有下标都错位。"""

    from arkeye.board import BoardCalibration, Homography
    from arkeye.routes import route_frame_points

    level = {
        "routes": [
            {"startPosition": {"col": 0, "row": 0}, "endPosition": {"col": 3, "row": 0}, "checkpoints": []},
            {"startPosition": {"col": 0, "row": 1}, "endPosition": {"col": 0, "row": 1}, "checkpoints": []},
            {"startPosition": {"col": 0, "row": 2}, "endPosition": {"col": 3, "row": 2}, "checkpoints": []},
        ]
    }
    identity = BoardCalibration(
        preview_size=(100, 100),
        map_to_preview=Homography.identity(),
        preview_to_frame=Homography.identity(),
    )
    routes = route_frame_points(level, identity)
    assert len(routes) == 3
    assert routes[0] is not None
    assert routes[1] is None
    assert routes[2] is not None


def test_off_route_candidate_is_downweighted_not_dropped() -> None:
    """内圈到外圈之间：仍然保留，只是权重下降。"""

    motion = filter_with_one_slot()
    frames: list[dict] = []
    motion.trace_hook = frames.append

    # frame_diagonal=2000 -> 内圈 90 px、外圈 240 px；偏离 150 px 落在中间带
    assert motion.update((blob(1500, 650),), 1.0) == ()
    snapshots = motion.update((blob(1498, 650),), 1.2)

    assert len(snapshots) == 1, "轻微偏离路线不应被丢掉"
    dropped = [item for frame in frames for item in frame.get("off_path_dropped", [])]
    assert not dropped


def test_far_off_route_candidate_is_dropped_with_distance() -> None:
    motion = filter_with_one_slot()
    frames: list[dict] = []
    motion.trace_hook = frames.append

    # 偏离 300 px > 外圈 240 px，且只有一个敌方候选 -> 直接丢弃
    assert motion.update((blob(1500, 800),), 1.0) == ()
    assert motion.update((blob(1498, 800),), 1.2) == ()

    dropped = [item for frame in frames for item in frame.get("off_path_dropped", [])]
    assert dropped
    assert "px" in dropped[0]["reason"]


def test_whole_frame_off_route_relaxes_instead_of_blinding() -> None:
    """整帧都对不上路线时放宽一倍继续跟，而不是全军覆没。"""

    motion = filter_with_slots(2)
    frames: list[dict] = []
    motion.trace_hook = frames.append

    motion.update((blob(1000, 800), blob(1100, 800)), 1.0)
    snapshots = motion.update((blob(1000, 800), blob(1100, 800)), 1.2)

    relaxed = [item for frame in frames for item in frame.get("path_relaxed", [])]
    assert relaxed, "整帧偏离路线时应记录一次放宽"
    assert len(snapshots) == 2


def test_candidate_before_route_wakes_up_is_dropped_with_a_readable_reason() -> None:
    motion = filter_with_one_slot()
    motion.route_earliest_times = {0: 10.0}
    motion.spawn_time_tolerance = 0.0
    frames: list[dict] = []
    motion.trace_hook = frames.append

    motion.update((blob(1500),), 9.0)

    dropped = [item for frame in frames for item in frame.get("off_path_dropped", [])]
    assert dropped
    assert "出怪时间" in dropped[0]["reason"]
