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
        max_speed=4000.0,
    )

    spawned = tracker.update((blob(1500),), 0.0)
    assert len(spawned) == 1
    assert spawned[0].enemy_id == "enemy_1007_slime"
    assert spawned[0].foot_point == (1500.0, 500.0)

    # 保持窗内不会翻 HELD，只体现为「这一帧没有匹配到」
    held = tracker.update((), 0.1)
    assert all(track.misses >= 1 for track in held)

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
    assert rejected[0].misses == 1
    assert rejected[0].foot_point == (1500.0, 500.0)

    recovered = tracker.update((blob(1490),), 2 / 30)
    assert len(recovered) == 1
    assert recovered[0].status == "confirmed"
    assert recovered[0].foot_point == (1490.0, 500.0)


def test_speed_rejected_target_does_not_create_a_duplicate_track() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        max_speed=300.0,
    )
    tracker.update((blob(1500),), 0.0)

    snapshots = tracker.update((blob(1300),), 1 / 30)

    assert len(snapshots) == 1
    assert snapshots[0].misses == 1
    assert snapshots[0].foot_point == (1500.0, 500.0)


def test_lost_track_cannot_recover_through_an_impossible_speed_jump() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        max_speed=300.0,
    )
    tracker.update((blob(1500),), 0.0)
    tracker.update((), 0.1)

    snapshots = tracker.update((blob(1300),), 0.2)
    original = next(track for track in snapshots if track.track_id == 1)

    assert original.misses >= 1
    assert original.foot_point == (1500.0, 500.0)
    assert original.velocity == (0.0, 0.0)


def test_double_speed_expands_the_association_gate() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        max_speed=300.0,
    )
    tracker.update((blob(1500),), 0.0)
    tracker.set_time_scale(2.0)

    snapshot = tracker.update((blob(1480),), 1 / 30)[0]

    assert snapshot.status == "confirmed"
    assert snapshot.foot_point == (1493.0, 500.0)


def test_measured_velocity_is_capped_by_the_physical_speed_limit() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        max_speed=10.0,
    )
    tracker.update((blob(1500),), 0.0)
    tracker.set_time_scale(2.0)

    snapshot = tracker.update((blob(1488),), 1 / 30)[0]

    assert snapshot.status == "confirmed"
    assert snapshot.velocity == (-20.0, 0.0)


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


def test_confirmed_track_keeps_its_state_through_a_short_occlusion() -> None:
    """被特效盖住几帧不该立刻翻 HELD，超过保持窗才翻。"""

    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        confirmed_grace_frames=10,
    )
    tracker.update((blob(1500),), 0.0)

    for step in range(1, 11):
        snapshots = tracker.update((), step / 30)
        assert snapshots[0].status == "confirmed", f"第 {step} 帧不该转 HELD"
        assert snapshots[0].misses == step

    late = tracker.update((), 11 / 30)
    assert late[0].status == "lost"


def test_occluded_track_also_uses_the_grace_window() -> None:
    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        confirmed_grace_frames=3,
    )
    tracker.update((blob(1500),), 0.0)
    tracker._tracks[1].status = "occluded"

    assert tracker.update((), 1 / 30)[0].status == "occluded"
    assert tracker.update((), 2 / 30)[0].status == "occluded"
    assert tracker.update((), 3 / 30)[0].status == "occluded"
    assert tracker.update((), 4 / 30)[0].status == "lost"


def test_lost_track_does_not_steal_a_detection_from_a_confirmed_track() -> None:
    """HELD 轨迹离得更近也不能抢走已确认目标的检测。"""

    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        max_speed=4000.0,
        typical_bbox_areas={"enemy_1007_slime": 20_000},
    )
    # 两条轨迹：A 在 1400，B 在 1500
    tracker.update((blob(1400), blob(1500)), 0.0)
    # 只喂 B，A 开始丢失；连续丢到超过保持窗
    for step in range(1, 12):
        tracker.update((blob(1500),), step / 30)
    # 现在 A 是 lost、B 是 confirmed，检测落在 1440（离 A 40、离 B 60）
    snapshots = tracker.update((blob(1440),), 12 / 30)
    by_id = {track.track_id: track for track in snapshots}
    assert by_id[1].misses >= 1, "HELD 的 A 不该接走这个检测"
    assert by_id[2].misses == 0, "已确认的 B 应该拿到它"
    # 已确认轨迹的框/脚点按 65/35 平滑，所以只是往检测方向移动，不会精确落在 1440
    assert 1440.0 <= by_id[2].foot_point[0] < 1500.0


def test_new_identity_rejections_are_logged_instead_of_swallowed() -> None:
    """被「位移超门限」或「附近有丢失轨迹」挡住的新身份，都必须留下可溯源理由。"""

    def run(wide: Detection) -> list[tuple]:
        events: list[tuple] = []
        tracker = StageSlotTracker(
            expected_counts={"enemy_1007_slime": 2},
            spawn_bbox=(1200, 350, 550, 400),
            despawn_x=700,
            typical_bbox_areas={"enemy_1007_slime": 20_000},
        )
        tracker.trace_hook = lambda detection, reason, track_id=None, kind=None: events.append(
            (kind, track_id, reason)
        )
        tracker.update((blob(1400),), 0.0)
        tracker.update((wide,), 1 / 30)
        return events

    # 框大小不变、只是跳了 112 px：超过当帧门限 40 px，判为不可能的一帧位移
    jumped = Detection((1400, 450, 200, 100), (1500.0, 549.0), "enemy_1007_slime", 0.8, "test")
    events = run(jumped)
    kinds = [event[0] for event in events]
    assert "speed_rejected" in kinds, kinds
    reason = next(event[2] for event in events if event[0] == "speed_rejected")
    assert "运动门限" in reason

    # 700 px 宽的大框：面积跳变不会进速度通道，但落点仍被框覆盖，属于「同位置已有丢失轨迹」
    wide = Detection((1300, 450, 700, 100), (1650.0, 549.0), "enemy_1007_slime", 0.8, "test")
    events = run(wide)
    kinds = [event[0] for event in events]
    assert "id_blocked" in kinds, kinds
    reason = next(event[2] for event in events if event[0] == "id_blocked")
    assert "丢失" in reason


def test_unknown_held_track_is_locked_after_one_second() -> None:
    """未识别轨迹丢失满 held_lock_frames 就冻结：不参与关联，也不阻挡新身份。"""

    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        typical_bbox_areas={"enemy_1007_slime": 20_000},
        held_lock_frames=30,
    )
    unknown_first = Detection(
        bbox=(1300, 450, 200, 100),
        foot_point=(1400.0, 500.0),
        enemy_id="unknown",
        confidence=0.5,
        template_name=None,
    )
    first = tracker.update((unknown_first,), 0.0)
    locked_id = first[0].track_id

    for step in range(1, tracker.held_lock_frames + 3):
        tracker.update((), step / 30)

    snapshots = tracker.update((blob(1400),), (tracker.held_lock_frames + 4) / 30)
    locked = next((s for s in snapshots if s.track_id == locked_id), None)
    assert locked is not None, "还没到删除阈值，旧轨迹应当仍然存在"
    assert locked.misses > tracker.held_lock_frames
    assert locked.foot_point == (1400.0, 500.0), "冻结后位置不能再被更新"

    fresh = [s for s in snapshots if s.track_id != locked_id and s.misses == 0]
    assert fresh, "冻结轨迹不该挡住新身份接手这个位置"


def test_known_class_held_track_keeps_a_longer_reacquisition_window() -> None:
    """已确认敌人不能一掉检就换 ID，冻结窗口要放宽。"""

    tracker = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        typical_bbox_areas={"enemy_1007_slime": 20_000},
        held_lock_frames=20,
    )
    first = tracker.update((blob(1400),), 0.0)
    track_id = first[0].track_id

    # 超过基础窗口但没到 4 倍窗口：不该冻结
    for step in range(1, tracker.held_lock_frames * 3):
        tracker.update((), step / 30)
    assert not tracker._is_locked(tracker._tracks[track_id])

    # 继续丢到 4 倍窗口之外：冻结
    for step in range(tracker.held_lock_frames * 3, tracker.held_lock_frames * 5):
        tracker.update((), step / 30)
    assert tracker._is_locked(tracker._tracks[track_id])
