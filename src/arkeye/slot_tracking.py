"""Deterministic stage-slot tracking for dense same-class enemy groups."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from arkeye.detector import Detection
from arkeye.tracking import MultiObjectTracker, TrackSnapshot, _Track

_INVALID = 1_000_000.0


class StageSlotTracker(MultiObjectTracker):
    """Assign every visible enemy to a bounded set of stage-derived slots."""

    def __init__(
        self,
        *,
        expected_counts: dict[str, int],
        spawn_bbox: tuple[int, int, int, int],
        despawn_x: float,
        despawn_misses: int = 240,
        unknown_despawn_misses: int = 15,
        spawn_class_windows: list[tuple[float, float, str]] | None = None,
        typical_bbox_areas: dict[str, float] | None = None,
        unknown_budget: int = 30,
        max_speed: float = 1200.0,
        friendly_limit: int = 1,
        class_confirmation_hits: int = 1,
        class_confirmation_by_id: dict[str, int] | None = None,
        class_spawn_regions: dict[str, tuple[int, int, int, int]] | None = None,
        earliest_spawn_times: dict[str, float] | None = None,
        battle_origin_timestamp: float = 0.0,
        spawn_time_tolerance: float = 2.0,
        spawn_score_threshold: float = 0.75,
        confirmed_grace_frames: int = 10,
        **tracker_options,
    ) -> None:
        if max_speed <= 0 or friendly_limit < 1 or class_confirmation_hits < 1:
            raise ValueError("Tracker limits must be positive")
        tracker_options.setdefault("min_hits", 1)
        tracker_options.setdefault("max_missed", 10_000)
        super().__init__(**tracker_options)
        self.expected_counts = expected_counts
        self.spawn_bbox = spawn_bbox
        self.despawn_x = despawn_x
        self.despawn_misses = despawn_misses
        self.unknown_despawn_misses = unknown_despawn_misses
        self.spawn_class_windows = spawn_class_windows or []
        self.typical_bbox_areas = typical_bbox_areas or {}
        self.unknown_budget = unknown_budget
        self.max_speed = max_speed
        self.friendly_limit = friendly_limit
        self.class_confirmation_hits = class_confirmation_hits
        self.class_confirmation_by_id = class_confirmation_by_id or {}
        self.class_spawn_regions = class_spawn_regions or {}
        self.earliest_spawn_times = earliest_spawn_times or {}
        self.battle_origin_timestamp = battle_origin_timestamp
        self.spawn_time_tolerance = spawn_time_tolerance
        self.spawn_score_threshold = spawn_score_threshold
        # 已确认/重叠支撑的轨迹刚丢失时，先保持状态这么多帧再翻成 HELD。
        # 技能特效覆盖目标通常只有几帧到小半秒，直接翻 HELD 会造成抖动。
        self.confirmed_grace_frames = max(0, confirmed_grace_frames)
        self.game_speed = 1
        self.time_scale = 1.0
        self._stage_elapsed = 0.0
        # 可选：拒绝新建轨迹时把原因告诉上层，便于逐帧溯源
        self.trace_hook = None
        self._activated_counts: dict[str, int] = defaultdict(int)
        self._class_votes: dict[int, Counter[str]] = defaultdict(Counter)
        self._created_at: dict[int, float] = {}
        self._last_update_timestamp: float | None = None

    def _class_at(self, enemy_id: str, timestamp: float) -> str:
        if enemy_id != "unknown":
            return enemy_id
        candidates = [class_id for start, end, class_id in self.spawn_class_windows if start <= timestamp < end]
        return candidates[0] if len(candidates) == 1 else enemy_id

    def _in_spawn_region(self, detection: Detection) -> bool:
        x, y, width, height = self.spawn_bbox
        foot_x, foot_y = detection.foot_point
        return x <= foot_x <= x + width and y <= foot_y <= y + height

    def _track_in_spawn_region(self, track: _Track) -> bool:
        x, y, width, height = self.spawn_bbox
        foot_x, foot_y = self._position(track)
        return x <= foot_x <= x + width and y <= foot_y <= y + height

    def _class_allowed_at(self, enemy_id: str, foot_point: tuple[float, float]) -> bool:
        region = self.class_spawn_regions.get(enemy_id)
        if region is None:
            return True
        x, y, width, height = region
        foot_x, foot_y = foot_point
        return x <= foot_x <= x + width and y <= foot_y <= y + height

    def set_game_speed(self, speed: int) -> None:
        if speed not in {1, 2}:
            raise ValueError("game_speed must be 1 or 2")
        self.game_speed = speed
        self.time_scale = float(speed)

    def set_time_scale(self, scale: float) -> None:
        if not 0.05 <= scale <= 2.0:
            raise ValueError("time_scale must be in [0.05, 2]")
        self.time_scale = scale

    def _class_allowed_time(self, enemy_id: str, timestamp: float) -> bool:
        del timestamp
        earliest = self.earliest_spawn_times.get(enemy_id)
        if earliest is None:
            return True
        return self._stage_elapsed >= earliest - self.spawn_time_tolerance

    def _spawn_stage_score(self, enemy_id: str, timestamp: float, foot_point: tuple[float, float]) -> float:
        time_score = float(self._class_allowed_time(enemy_id, timestamp))
        region_score = float(self._class_allowed_at(enemy_id, foot_point))
        if enemy_id == "friendly":
            budget_score = float(self._activated_counts[enemy_id] < self.friendly_limit)
        else:
            budget_score = float(self._activated_counts[enemy_id] < self.expected_counts.get(enemy_id, 0))
        return 0.4 * time_score + 0.3 * region_score + 0.3 * budget_score

    def _spawn_score(self, enemy_id: str, timestamp: float, foot_point: tuple[float, float], visual_confidence: float) -> float:
        return 0.501 * self._spawn_stage_score(enemy_id, timestamp, foot_point) + 0.499 * visual_confidence

    def _refuse(
        self,
        detection: Detection,
        reason: str,
        track_id: int | None = None,
        kind: str | None = None,
    ) -> None:
        """Report a refusal (or an occlusion-support assignment) to the caller."""

        if self.trace_hook is not None:
            self.trace_hook(detection, reason, track_id, kind)

    def _activate(self, detection: Detection, timestamp: float, status: str = "confirmed") -> _Track | None:
        enemy_id = self._class_at(detection.enemy_id, timestamp)
        if enemy_id not in {"unknown", "friendly"} and not self._class_allowed_time(enemy_id, timestamp):
            enemy_id = "unknown"
        if not self._class_allowed_at(enemy_id, detection.foot_point):
            self._refuse(detection, f"脚下不在 {enemy_id} 允许的出怪区域")
            return None
        if enemy_id == "friendly":
            if self._activated_counts[enemy_id] >= self.friendly_limit:
                self._refuse(detection, f"我方干员上限 {self.friendly_limit} 已满")
                return None
        elif enemy_id == "unknown":
            if self._activated_counts[enemy_id] >= self.unknown_budget:
                self._refuse(detection, f"未识别目标预算 {self.unknown_budget} 已用尽")
                return None
        elif enemy_id not in self.expected_counts:
            self._refuse(detection, f"{enemy_id} 不在本关计划出怪列表里")
            return None
        elif self._activated_counts[enemy_id] >= self.expected_counts[enemy_id]:
            self._refuse(
                detection,
                f"{enemy_id} 的计划数量 {self.expected_counts[enemy_id]} 已全部激活",
            )
            return None
        track = super()._new_track(detection, timestamp)
        confirmed_now = enemy_id == "unknown" or self.class_confirmation_by_id.get(enemy_id, self.class_confirmation_hits) <= 1
        track.enemy_id = enemy_id if confirmed_now else "unknown"
        track.status = status
        self._created_at[track.track_id] = timestamp
        self._activated_counts[track.enemy_id] += 1
        if enemy_id != "unknown":
            self._class_votes[track.track_id][enemy_id] += 1
        return track

    def _vote_class(self, track: _Track, enemy_id: str, timestamp: float, visual_confidence: float) -> None:
        if enemy_id == "unknown" or track.enemy_id != "unknown":
            return
        votes = self._class_votes[track.track_id]
        if votes and enemy_id not in votes:
            votes.clear()
        votes[enemy_id] += 1
        candidate, count = votes.most_common(1)[0]
        required_hits = self.class_confirmation_by_id.get(candidate, self.class_confirmation_hits)
        if count < required_hits or not self._class_allowed_at(candidate, self._position(track)):
            return
        if self._spawn_score(candidate, timestamp, self._position(track), visual_confidence) < self.spawn_score_threshold:
            return
        if candidate == "friendly":
            if self._activated_counts[candidate] >= self.friendly_limit:
                return
        elif self._activated_counts[candidate] >= self.expected_counts.get(candidate, 0):
            return
        self._activated_counts["unknown"] = max(0, self._activated_counts["unknown"] - 1)
        self._activated_counts[candidate] += 1
        track.enemy_id = candidate

    # 丢失轨迹重新接回检测的代价系数（乘以当帧运动门限）
    REACQUIRE_COST_WEIGHT = 0.6

    @staticmethod
    def _covers(track: _Track, detection: Detection, margin_ratio: float = 0.08) -> bool:
        """这条轨迹的落点是否真的落在该检测框内（带一点余量）。

        之前用的是「到检测框脚点 300 px 内」，对 574x156 这类大框来说，
        框外几十像素的轨迹也会被判成「被覆盖」，于是同一块前景被分给好几条轨迹，
        看起来就像目标被死死锁住。
        """

        x, y, width, height = detection.bbox
        margin_x, margin_y = width * margin_ratio, height * margin_ratio
        foot_x, foot_y = track.kalman.statePost[0, 0], track.kalman.statePost[1, 0]
        return (
            x - margin_x <= float(foot_x) <= x + width + margin_x
            and y - margin_y <= float(foot_y) <= y + height + margin_y
        )

    @staticmethod
    def _distance(track: _Track, detection: Detection) -> float:
        return float(np.linalg.norm(StageSlotTracker._position(track) - np.asarray(detection.foot_point)))

    def _correct(self, track: _Track, detection: Detection, timestamp: float, status: str = "confirmed") -> None:
        previous_position = self._position(track).copy()
        previous_bbox = track.bbox
        if status != "occluded":
            track.kalman.statePre = track.kalman.statePost.copy()
            track.kalman.correct(np.asarray(detection.foot_point, dtype=np.float32).reshape(2, 1))
            if status == "confirmed" and track.misses <= 1 and track.hits > 0:
                track.kalman.statePost[:2, 0] = previous_position * 0.65 + np.asarray(detection.foot_point) * 0.35
                track.bbox = tuple(
                    round(previous * 0.65 + current * 0.35)
                    for previous, current in zip(previous_bbox, detection.bbox, strict=True)
                )
            else:
                track.kalman.statePost[:2, 0] = detection.foot_point
                track.bbox = detection.bbox
            delta_time = max(1e-3, timestamp - track.history[-1][0]) if track.history else 1.0 / 30.0
            measured_velocity = (np.asarray(detection.foot_point) - previous_position) / delta_time
            speed = float(np.linalg.norm(measured_velocity))
            speed_limit = self.max_speed * self.time_scale
            if speed > speed_limit:
                measured_velocity *= speed_limit / speed
            track.kalman.statePost[2:4, 0] = measured_velocity.astype(np.float32)
        self._vote_class(track, detection.enemy_id, timestamp, detection.confidence)
        if detection.enemy_id == "unknown" and detection.possible_friendly:
            self._vote_class(track, "friendly", timestamp, detection.confidence)
        elif detection.enemy_id == "unknown" and detection.candidate_enemy_id not in {None, "friendly"}:
            self._vote_class(track, detection.candidate_enemy_id, timestamp, detection.confidence)
        track.confidence = detection.confidence
        if detection.template_name is not None:
            track.template_name = detection.template_name
        if detection.appearance is not None:
            track.appearance = self._copy_appearance(detection.appearance)
        track.hits += 1
        track.misses = 0
        track.status = status
        track.occlusion_group = None
        x, y = self._position(track)
        track.history.append((timestamp, float(x), float(y)))

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[TrackSnapshot, ...]:
        if self._last_update_timestamp is None:
            delta_time = 1.0 / 30.0
            self._stage_elapsed = max(0.0, timestamp - self.battle_origin_timestamp) * self.time_scale
        else:
            delta_time = max(1e-3, timestamp - self._last_update_timestamp)
            start = max(self._last_update_timestamp, self.battle_origin_timestamp)
            end = max(timestamp, self.battle_origin_timestamp)
            self._stage_elapsed += max(0.0, end - start) * self.time_scale
        self._last_update_timestamp = timestamp
        motion_gate = max(12.0, self.max_speed * delta_time * self.time_scale)
        tracks = list(self._tracks.values())
        for track in tracks:
            track.age += 1
            track.misses += 1

        matched: dict[int, Detection] = {}
        matched_detection_indices: set[int] = set()
        matched_by_detection: dict[int, list[int]] = {}
        speed_rejected_pairs: set[tuple[int, int]] = set()
        if tracks and detections:
            costs = np.full((len(tracks), len(detections)), _INVALID)
            for row, track in enumerate(tracks):
                for column, detection in enumerate(detections):
                    if track.enemy_id == "friendly" and detection.faction_evidence == "enemy":
                        continue
                    if track.enemy_id not in {"unknown", "friendly"} and detection.faction_evidence == "friendly":
                        continue
                    if track.enemy_id != "unknown" and detection.enemy_id != "unknown" and track.enemy_id != detection.enemy_id:
                        continue
                    if track.misses > 10 and self._in_spawn_region(detection) and not self._track_in_spawn_region(track):
                        continue
                    track_area = track.bbox[2] * track.bbox[3]
                    detection_area = detection.bbox[2] * detection.bbox[3]
                    if track.misses > 0 and detection_area > max(track_area * 3, track_area + 30_000):
                        continue
                    distance = self._distance(track, detection)
                    gate = 300.0
                    if distance <= gate and distance > motion_gate:
                        speed_rejected_pairs.add((track.track_id, column))
                    if distance <= gate and distance <= motion_gate:
                        # 丢失中的轨迹（含刚丢失、仍处于保持窗的）要额外付代价：
                        # 同一个检测必须优先归仍在跟的目标，HELD 不能把它抢走。
                        penalty = 0.0
                        if track.misses > 0:
                            weight = min(1.0, track.misses / max(1, self.confirmed_grace_frames))
                            penalty = weight * motion_gate * self.REACQUIRE_COST_WEIGHT
                        costs[row, column] = distance + penalty
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] >= _INVALID:
                    continue
                track = tracks[row]
                matched[track.track_id] = detections[column]
                matched_detection_indices.add(column)
                matched_by_detection.setdefault(column, []).append(track.track_id)
                self._correct(track, detections[column], timestamp)

        speed_rejected_detections = {column for _, column in speed_rejected_pairs}
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            if detection_index in speed_rejected_detections:
                # 超速的检测以前是静默跳过的，同样会让「进了正式跟踪却没有轨迹」无从解释
                nearest = min(
                    (
                        track
                        for track in self._tracks.values()
                        if (track.track_id, detection_index) in speed_rejected_pairs
                    ),
                    key=lambda track: self._distance(track, detection),
                    default=None,
                )
                if nearest is not None:
                    self._refuse(
                        detection,
                        f"与最近轨迹 #{nearest.track_id} 相距 "
                        f"{self._distance(nearest, detection):.0f} px，超过本帧运动门限 "
                        f"{motion_gate:.0f} px，判定为不可能的一帧位移，不新建身份",
                        nearest.track_id,
                        "speed_rejected",
                    )
                continue
            blocker = next(
                (
                    track
                    for track in self._tracks.values()
                    if track.misses > 0
                    and (track.track_id, detection_index) not in speed_rejected_pairs
                    and (
                        track.enemy_id == "unknown"
                        or detection.enemy_id == "unknown"
                        or track.enemy_id == detection.enemy_id
                    )
                    and self._covers(track, detection)
                    and not (
                        track.misses > 10
                        and self._in_spawn_region(detection)
                        and not self._track_in_spawn_region(track)
                    )
                ),
                None,
            )
            if blocker is not None:
                # 以前这里是静默 continue，于是「已识别候选进了正式跟踪却在输出里消失」，
                # 溯源日志里一条记录都没有。现在明确写出被谁挡住。
                self._refuse(
                    detection,
                    f"该位置已有丢失中的轨迹 #{blocker.track_id}（连续 {blocker.misses} 帧未匹配），"
                    "先不新建身份，避免同一目标换 ID 重复计数",
                    blocker.track_id,
                    "id_blocked",
                )
                continue
            self._activate(detection, timestamp)

        matched_ids = set(matched)
        for track in tracks:
            if track.track_id in matched_ids:
                continue
            # 抖动消抖：刚丢失的前 N 帧保持原状态（confirmed/occluded），
            # 期间重新匹配上就不会留下 HELD 痕迹；超过 N 帧才真正转 lost。
            if track.status in {"confirmed", "occluded"} and track.misses <= self.confirmed_grace_frames:
                pass
            else:
                track.status = "lost"
            x, y = self._position(track)
            track.history.append((timestamp, float(x), float(y)))

        # A merged blob may cover existing tracks, but never manufactures a new one from area.
        for detection_index, detection in enumerate(detections):
            assigned = [self._tracks[tid] for tid in matched_by_detection.get(detection_index, [])]
            if any(track.enemy_id == "unknown" and track.hits < self.class_confirmation_hits for track in assigned):
                continue
            typical_area = self.typical_bbox_areas.get(detection.enemy_id, 10_000.0)
            ratio = detection.bbox[2] * detection.bbox[3] / typical_area
            group_size = 1 if ratio < 1.2 else 2 if ratio < 3.0 else 3
            candidates = [
                track
                for track in self._tracks.values()
                if track.misses > 0
                and (track.track_id, detection_index) not in speed_rejected_pairs
                and self._covers(track, detection)
                and not (track.misses > 10 and self._in_spawn_region(detection) and not self._track_in_spawn_region(track))
            ]
            candidates.sort(key=lambda track: self._distance(track, detection))
            # 候选数量不得反过来把 group_size 撑大：那样任何 300 px 内的丢失轨迹
            # 都能分到别人的检测，misses 被清零，HELD 轨迹就能一直占着已确认目标的观测。
            # group_size 只由「这块前景比典型目标大多少」决定。
            if detection.enemy_id == "unknown" and not assigned and not candidates:
                group_size = 1
            # 只有明显大于单个目标的框才需要分摊；普通大小的框即便没人匹配上，
            # 也不该拿去给别人当「遮挡支撑」
            needed = (group_size - len(assigned)) if group_size > 1 else 0
            if needed <= 0:
                continue
            for track in candidates[:needed]:
                self._refuse(
                    detection,
                    "该检测的框同时覆盖这条轨迹的预测位置，按重叠支撑处理（不新建轨迹）",
                    track.track_id,
                )
                self._correct(track, detection, timestamp, status="occluded")
                needed -= 1

        return self.snapshots()
