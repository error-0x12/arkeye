"""Generic route-conforming motion profiling for spawn-guide rejection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from arkeye.detector import Detection
from arkeye.timeline import TimedSpawnEvent
from arkeye.tracking import MultiObjectTracker, TrackSnapshot

_INVALID = 1_000_000.0


@dataclass(slots=True)
class _MotionTrack:
    track_id: int
    detection: Detection
    timestamp: float
    hits: int = 1
    misses: int = 0
    guide: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)
    speeds: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    path_hits: deque[bool] = field(default_factory=lambda: deque(maxlen=5))
    classes: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    consecutive: int = 1
    stationary_hits: int = 0

    def class_history(self) -> deque[str]:
        """Class votes across recent matches (lazy for checkpoints saved before this field)."""

        history = getattr(self, "classes", None)
        if history is None:
            history = deque(maxlen=6)
            self.classes = history
        return history


class RouteSpeedFilter:
    """Reject route motion faster than regular path-following enemies."""

    MAX_ENEMY_SPEED_TILES_PER_SECOND = 2.5
    # 处于丢失（HELD）状态的轨迹重新接回检测时要额外付出的匹配代价（乘以门限）。
    # 没有这一项时纯距离代价会让 HELD 轨迹把本该属于已确认目标的新检测抢走。
    RECAPTURE_COST_PENALTY = 0.35
    # 允许出怪口附近的未识别驻场目标保留的内部匹配次数；
    # 真实敌人出怪后会立刻移动，超过该次数仍静止的无血条目标按干扰处理。
    STATIONARY_GRACE_MATCHES = 30

    def __init__(
        self,
        tracker: MultiObjectTracker,
        route_lines: tuple[np.ndarray | None, ...],
        *,
        frame_diagonal: float,
        pixels_per_tile: float,
        class_route_indices: dict[str, tuple[int, ...]] | None = None,
        earliest_spawn_times: dict[str, float] | None = None,
        route_earliest_times: dict[int, float] | None = None,
        spawn_events: tuple[TimedSpawnEvent, ...] = (),
        battle_origin_timestamp: float = 0.0,
        spawn_time_tolerance: float = 2.0,
        max_track_misses: int = 15,
    ) -> None:
        if frame_diagonal <= 0 or pixels_per_tile <= 0:
            raise ValueError("frame_diagonal and pixels_per_tile must be positive")
        if max_track_misses < 2:
            raise ValueError("max_track_misses must be at least 2")
        self.tracker = tracker
        # 位置必须与关卡 route 下标一一对应，不可用的路线保留为 None
        self.route_lines = tuple(
            np.asarray(line, dtype=np.float32) if line is not None and len(line) >= 2 else None
            for line in route_lines
        )
        self.frame_diagonal = frame_diagonal
        self.pixels_per_tile = pixels_per_tile
        self.class_route_indices = class_route_indices or {}
        self.earliest_spawn_times = earliest_spawn_times or {}
        self.route_earliest_times = route_earliest_times or {}
        self.spawn_events = spawn_events
        self.battle_origin_timestamp = battle_origin_timestamp
        self.spawn_time_tolerance = spawn_time_tolerance
        self.max_track_misses = max_track_misses
        self.game_speed = 1
        self.time_scale = 1.0
        # 内圈：完全信任；外圈：超出才算彻底偏离。两者之间连续降权。
        self.path_tolerance = frame_diagonal * 0.045
        self.path_cutoff = frame_diagonal * 0.12
        # 关联代价里「偏离路线」的分量权重
        self.PATH_COST_WEIGHT = 0.6
        self.spawn_entry_tolerance = frame_diagonal * 0.12
        self._current_timestamp = battle_origin_timestamp
        self._stage_elapsed = 0.0
        self._last_stage_timestamp: float | None = None
        self._tracks: dict[int, _MotionTrack] = {}
        self._next_id = 1
        self.trace_hook = None
        self._trace_frame: dict[str, Any] | None = None
        # 内层槽位跟踪器会因为配额/区域/出怪时间拒绝新建轨迹，这里把原因接过来
        if hasattr(self.tracker, "trace_hook"):
            self.tracker.trace_hook = self._on_slot_refusal

    def _on_slot_refusal(
        self,
        detection: Detection,
        reason: str,
        track_id: int | None = None,
        kind: str | None = None,
    ) -> None:
        """内层跟踪器上报的三种事件：配额拒绝、附近有丢失轨迹、重叠支撑补入。"""

        if self._trace_frame is None:
            return
        group = kind or ("occluded_support" if track_id is not None else "budget_rejected")
        item: dict[str, Any] = {"reason": reason, "detection": self._trace_detection(detection)}
        if track_id is not None:
            item["track_id"] = track_id
        self._trace_frame.setdefault(group, []).append(item)

    def _nearest_track(self, detection: Detection) -> dict[str, Any]:
        """最近内部轨迹的编号与距离，用来解释「这个候选本来可能是谁的」。"""

        if not self._tracks:
            return {}
        track_id, track = min(
            self._tracks.items(),
            key=lambda item: self._distance(item[1].detection, detection),
        )
        distance = self._distance(track.detection, detection)
        return {"nearest_track_id": track_id, "nearest_track_px": round(distance, 1)}

    @property
    def speed_threshold(self) -> float:
        return self.MAX_ENEMY_SPEED_TILES_PER_SECOND * self.pixels_per_tile * self.time_scale

    def set_game_speed(self, speed: int) -> None:
        if speed not in {1, 2}:
            raise ValueError("game_speed must be 1 or 2")
        self.game_speed = speed
        self.time_scale = float(speed)
        set_inner_speed = getattr(self.tracker, "set_game_speed", None)
        if callable(set_inner_speed):
            set_inner_speed(speed)

    def set_time_scale(self, scale: float) -> None:
        if not 0.05 <= scale <= 2.0:
            raise ValueError("time_scale must be in [0.05, 2]")
        self.time_scale = scale
        set_inner_scale = getattr(self.tracker, "set_time_scale", None)
        if callable(set_inner_scale):
            set_inner_scale(scale)

    def stage_elapsed_at(self, timestamp: float) -> float:
        """Project stage time without mutating the tracker's clock."""

        if self._last_stage_timestamp is None:
            return max(0.0, timestamp - self.battle_origin_timestamp) * self.time_scale
        start = max(self._last_stage_timestamp, self.battle_origin_timestamp)
        end = max(timestamp, self.battle_origin_timestamp)
        return self._stage_elapsed + max(0.0, end - start) * self.time_scale

    @property
    def motion_speeds(self) -> dict[int, float]:
        return {
            track_id: float(np.median(track.speeds))
            for track_id, track in self._tracks.items()
            if track.speeds
        }

    def _routes_for(self, enemy_id: str) -> tuple[np.ndarray, ...]:
        indices = self.class_route_indices.get(enemy_id)
        if indices is None:
            indices = range(len(self.route_lines))
        stage_time = self._stage_elapsed
        routes = []
        for index in indices:
            if not 0 <= index < len(self.route_lines):
                continue
            line = self.route_lines[index]
            if line is None:
                continue
            if stage_time < self.route_earliest_times.get(index, float("-inf")) - self.spawn_time_tolerance:
                continue
            routes.append(line)
        return tuple(routes)

    def _has_distinct_route(self, enemy_id: str, stage_time: float) -> bool:
        indices = self.class_route_indices.get(enemy_id)
        if not indices:
            return False
        own_routes = set(indices)
        for other_id, other_indices in self.class_route_indices.items():
            if other_id == enemy_id:
                continue
            earliest = self.earliest_spawn_times.get(other_id, float("inf"))
            if stage_time >= earliest - self.spawn_time_tolerance and own_routes.intersection(other_indices):
                return False
        return True

    def _near_scheduled_spawn(self, detection: Detection, enemy_id: str | None, stage_time: float) -> bool:
        foot = np.asarray(detection.foot_point, dtype=np.float32)
        tolerance = max(4.0, self.spawn_time_tolerance)
        for event in self.spawn_events:
            if enemy_id is not None and event.enemy_key != enemy_id:
                continue
            if not 0 <= event.route_index < len(self.route_lines):
                continue
            route_line = self.route_lines[event.route_index]
            if route_line is None:
                continue
            entry = route_line[0]
            if float(np.linalg.norm(foot - entry)) > self.spawn_entry_tolerance:
                continue
            for index in range(event.count):
                if abs(stage_time - (event.timestamp + index * event.interval)) <= tolerance:
                    return True
        return False

    def _path_distance(self, detection: Detection) -> float | None:
        """脚点到该类别任一活动路线的最近距离；没有可用路线时返回 None。"""

        routes = self._routes_for(detection.enemy_id)
        if not routes:
            return None
        x, y = (float(value) for value in detection.foot_point)
        return min(abs(cv2.pointPolygonTest(line, (x, y), True)) for line in routes)

    def _path_measure(self, detection: Detection, cutoff_scale: float = 1.0) -> tuple[float | None, float]:
        """返回 (距离, 权重)：内圈内满分，内圈到外圈线性降权，外圈外归零。"""

        distance = self._path_distance(detection)
        if distance is None:
            # 关卡有路线但当前一条都没启用（还没到出怪时间）-> 按偏离处理；
            # 只有整关都没有可用路线时才认为「没有信息」，不惩罚。
            has_any_route = any(line is not None for line in self.route_lines)
            return None, (0.0 if has_any_route else 1.0)
        inner = self.path_tolerance
        cutoff = self.path_cutoff * cutoff_scale
        if distance <= inner:
            return distance, 1.0
        if distance >= cutoff:
            return distance, 0.0
        return distance, 1.0 - (distance - inner) / (cutoff - inner)

    def _on_path(self, detection: Detection) -> bool:
        distance = self._path_distance(detection)
        return distance is not None and distance <= self.path_tolerance

    def _measure_paths(self, detections: tuple[Detection, ...]) -> dict[int, tuple[float | None, float]]:
        """给每个候选算路线距离与权重，并在整帧都对不上路线时整帧放宽。"""

        measures: dict[int, tuple[float | None, float]] = {}
        for detection in detections:
            if self._is_friendly(detection):
                measures[id(detection)] = (None, 1.0)
            else:
                measures[id(detection)] = self._path_measure(detection)
        enemies = [detection for detection in detections if not self._is_friendly(detection)]
        # 只统计真正量到距离的候选：路线还没启用时 distance 为 None，那不是「整帧偏离」，
        # 否则会每帧刷一条放宽日志却什么也没放宽。
        measured = [detection for detection in enemies if measures[id(detection)][0] is not None]
        if len(measured) >= 2 and all(measures[id(detection)][1] <= 0 for detection in measured):
            # 整帧全部判成偏离路线，通常是路线/标定和画面不一致（例如地形被改的关卡）。
            # 这时宁可放宽一倍继续跟，也不要整帧瞎掉。
            for detection in measured:
                measures[id(detection)] = self._path_measure(detection, cutoff_scale=2.0)
            if self._trace_frame is not None:
                self._trace_frame.setdefault("path_relaxed", []).append(
                    {
                        "reason": (
                            f"本帧 {len(measured)} 个敌方候选全部超出路线外圈 "
                            f"{self.path_cutoff:.0f} px，判定为路线/标定不匹配，整帧放宽一倍继续跟踪"
                        )
                    }
                )
        return measures

    def _near_route_entry(self, detection: Detection) -> bool:
        foot = np.asarray(detection.foot_point, dtype=np.float32)
        return any(
            float(np.linalg.norm(foot - line[0])) <= self.spawn_entry_tolerance
            for line in self._routes_for(detection.enemy_id)
        )

    def _path_ratio(self, track: _MotionTrack) -> float:
        if not track.path_hits:
            return 0.0
        return sum(track.path_hits) / len(track.path_hits)

    @staticmethod
    def _distance(first: Detection, second: Detection) -> float:
        return float(np.linalg.norm(np.asarray(first.foot_point) - np.asarray(second.foot_point)))

    @staticmethod
    def _is_friendly(detection: Detection) -> bool:
        return detection.enemy_id == "friendly" or detection.possible_friendly

    def _normalize_detections(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[Detection, ...]:
        stage_time = self._stage_elapsed
        normalized = []
        for detection in detections:
            original_id = detection.enemy_id
            candidate_id = detection.candidate_enemy_id
            if detection.enemy_id == "unknown" and candidate_id is not None:
                candidate_earliest = self.earliest_spawn_times.get(candidate_id)
                candidate_on_route = not self.class_route_indices or self._on_path(
                    replace(detection, enemy_id=candidate_id)
                )
                if (
                    candidate_earliest is not None
                    and stage_time >= candidate_earliest - self.spawn_time_tolerance
                    and candidate_on_route
                    and (
                        self._near_scheduled_spawn(detection, candidate_id, stage_time)
                        if self.spawn_events
                        else self._has_distinct_route(candidate_id, stage_time)
                    )
                ):
                    detection = replace(detection, enemy_id=candidate_id)
                    if self._trace_frame is not None:
                        self._trace_frame.setdefault("promoted", []).append(
                            {
                                "to": candidate_id,
                                "reason": f"未识别候选与 {candidate_id} 的出怪时间和路线吻合，恢复其类别",
                                "detection": self._trace_detection(detection),
                            }
                        )
            earliest = self.earliest_spawn_times.get(detection.enemy_id)
            too_early = earliest is not None and stage_time < earliest - self.spawn_time_tolerance
            wrong_route = (
                detection.enemy_id not in {"unknown", "friendly"}
                and self.class_route_indices
                and not self._on_path(detection)
                and self._on_path(replace(detection, enemy_id="unknown"))
            )
            if too_early or wrong_route:
                reason = (
                    f"出现时间 {stage_time:.1f}s 早于 {detection.enemy_id} 的最早出怪时间 {earliest:.1f}s"
                    if too_early
                    else f"位置不在 {detection.enemy_id} 的任何活动路线上"
                )
                detection = replace(detection, enemy_id="unknown", template_name=None, candidate_enemy_id=None)
                if self._trace_frame is not None and detection.enemy_id != original_id:
                    self._trace_frame.setdefault("demoted", []).append(
                        {"from": original_id, "reason": reason, "detection": self._trace_detection(detection)}
                    )
            normalized.append(detection)
        return tuple(normalized)

    @staticmethod
    def _trace_detection(detection: Detection) -> dict[str, Any]:
        x, y, width, height = detection.bbox
        return {
            "bbox": [round(x, 1), round(y, 1), round(width, 1), round(height, 1)],
            "foot": [round(detection.foot_point[0], 1), round(detection.foot_point[1], 1)],
            "class": detection.enemy_id,
            "confidence": round(detection.confidence, 3),
            "health_bar": detection.health_bar_bbox is not None,
            "candidate": detection.candidate_enemy_id,
        }

    def _adoption_allowed(self, track: _MotionTrack, detection: Detection) -> bool:
        """A long-lost track may not silently adopt an unrelated re-detection."""

        if track.misses < 4:
            return True
        previous = track.detection
        if previous.enemy_id == detection.enemy_id:
            return True
        # Gaining a health bar or a class on a previously unknown target is plausible.
        if previous.enemy_id == "unknown" and detection.health_bar_bbox is not None:
            return True
        return bool(detection.enemy_id == "unknown" and previous.health_bar_bbox is None)

    def _associate(
        self,
        detections: tuple[Detection, ...],
        timestamp: float,
        path_scores: dict[int, float] | None = None,
    ) -> dict[int, int]:
        """Assign detections to internal tracks; off-route candidates pay extra cost."""
        tracks = list(self._tracks.values())
        for track in tracks:
            track.misses += 1
        matched: dict[int, int] = {}
        if tracks and detections:
            costs = np.full((len(tracks), len(detections)), _INVALID)
            for row, track in enumerate(tracks):
                if track.guide:
                    continue
                delta_time = max(1e-3, timestamp - track.timestamp)
                # A long occlusion must not turn into an unbounded cross-board jump.
                gate = min(
                    self.frame_diagonal * 0.125,
                    max(30.0, self.speed_threshold * delta_time * 1.25 + self.path_tolerance * 0.25),
                )
                for column, detection in enumerate(detections):
                    if self._is_friendly(track.detection) != self._is_friendly(detection):
                        continue
                    distance = self._distance(track.detection, detection)
                    if distance <= gate:
                        # 本帧开头刚把 misses 加过 1，>1 才代表上一帧之前就已经丢失
                        lost_before = track.misses > 1
                        route_score = 1.0 if path_scores is None else path_scores.get(id(detection), 1.0)
                        costs[row, column] = (
                            distance
                            + (gate * self.RECAPTURE_COST_PENALTY if lost_before else 0.0)
                            + (1.0 - route_score) * gate * self.PATH_COST_WEIGHT
                        )
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] >= _INVALID:
                    continue
                track = tracks[row]
                detection = detections[column]
                if not self._adoption_allowed(track, detection):
                    if self._trace_frame is not None:
                        self._trace_frame.setdefault("adoption_rejected", []).append(
                            {
                                "track_id": track.track_id,
                                "reason": (
                                    f"轨迹已丢失 {track.misses} 帧，新检测类别 {detection.enemy_id} "
                                    f"与健康条证据和原轨迹（{track.detection.enemy_id}）不一致，"
                                    "拒绝收养，按新候选处理"
                                ),
                                "detection": self._trace_detection(detection),
                            }
                        )
                    continue
                delta_time = max(1e-3, timestamp - track.timestamp)
                displacement = np.asarray(detection.foot_point) - np.asarray(track.detection.foot_point)
                track.velocity = tuple((displacement / delta_time).tolist())
                track.speeds.append(self._distance(track.detection, detection) / delta_time)
                if track.speeds[-1] < max(8.0, self.speed_threshold * 0.03):
                    track.stationary_hits = getattr(track, "stationary_hits", 0) + 1
                else:
                    track.stationary_hits = 0
                track.path_hits.append(self._on_path(detection))
                track.class_history().append(detection.enemy_id)
                track.consecutive = getattr(track, "consecutive", 0) + 1
                track.detection = detection
                track.timestamp = timestamp
                track.hits += 1
                track.misses = 0
                matched[track.track_id] = column
                if self._trace_frame is not None:
                    self._trace_frame.setdefault("matched", []).append(
                        {
                            "track_id": track.track_id,
                            "speed_px_s": round(float(track.speeds[-1]), 1) if track.speeds else 0.0,
                            "detection": self._trace_detection(detection),
                        }
                    )

        matched_columns = set(matched.values())
        for track in tracks:
            if track.track_id not in matched:
                track.consecutive = 0
        for column, detection in enumerate(detections):
            if column in matched_columns:
                continue
            rejected_as_guide = False
            # 路线指引是图形覆盖层，只可能表现为「未识别」候选；
            # 已经识别出具体敌人类别的候选是真单位，不能被这条速度规则吃掉。
            if tracks and detection.enemy_id == "unknown" and not self._is_friendly(detection):
                nearest = min(tracks, key=lambda track: self._distance(track.detection, detection))
                rejected_distance = self._distance(nearest.detection, detection)
                if 0 < rejected_distance <= self.frame_diagonal * 0.25 and self._on_path(detection):
                    delta_time = max(1e-3, timestamp - nearest.timestamp)
                    # Keep rejected guide speeds observable without moving a real track.
                    rejected_speed = rejected_distance / delta_time
                    nearest.speeds.append(rejected_speed)
                    if rejected_speed > self.speed_threshold:
                        rejected_as_guide = True
                        if nearest.hits < 2:
                            nearest.guide = True
                        # 无论最近轨迹有多少命中，都要留下痕迹，否则这一帧会像「候选凭空消失」
                        if self._trace_frame is not None:
                            self._trace_frame.setdefault("guides", []).append(
                                {
                                    "track_id": nearest.track_id,
                                    "reason": (
                                        f"未识别候选与最近轨迹 #{nearest.track_id} 相距 {rejected_distance:.0f} px，"
                                        f"折算速度 {rejected_speed:.0f} px/s 超过上限 {self.speed_threshold:.0f} px/s，"
                                        "判定为路线指引干扰并放弃该候选"
                                    ),
                                    "detection": self._trace_detection(detection),
                                }
                            )
            if rejected_as_guide:
                continue
            near_stationary_guide = any(
                track.guide
                and track.stationary_hits >= 3
                and self._distance(track.detection, detection) <= self.frame_diagonal * 0.05
                for track in tracks
            )
            if near_stationary_guide:
                if self._trace_frame is not None:
                    self._trace_frame.setdefault("gate_rejected", []).append(
                        {
                            "reason": "旁边存在刚被判为驻场干扰的轨迹，同一静止杂波不允许换新身份重新进入",
                            "detection": self._trace_detection(detection),
                        }
                    )
                continue
            if (
                self._stage_elapsed <= 1.0 / 30.0 + 1e-6
                and not self._is_friendly(detection)
                and detection.health_bar_bbox is None
                and not self._near_route_entry(detection)
            ):
                if self._trace_frame is not None:
                    self._trace_frame.setdefault("gate_rejected", []).append(
                        {
                            "reason": "战斗刚开始且无血条、不在出怪口，拒绝凭空出现的候选",
                            **self._nearest_track(detection),
                            "detection": self._trace_detection(detection),
                        }
                    )
                continue
            track = _MotionTrack(self._next_id, detection, timestamp)
            track.path_hits.append(self._on_path(detection))
            track.class_history().append(detection.enemy_id)
            self._tracks[self._next_id] = track
            self._next_id += 1
            if self._trace_frame is not None:
                self._trace_frame.setdefault("new_tracks", []).append(
                    {"track_id": track.track_id, "detection": self._trace_detection(detection)}
                )

        return matched

    def _movement_score(self, track: _MotionTrack) -> float:
        path_score = float(self._path_ratio(track) >= 0.75)
        if not track.speeds or track.speeds[-1] <= 0:
            visual_motion_score = 1.0
        else:
            visual_motion_score = min(1.0, self.speed_threshold / track.speeds[-1])
        return 0.499 * path_score + 0.501 * visual_motion_score

    def _reject_guides(self, matched: dict[int, int]) -> None:
        for track_id in matched:
            track = self._tracks[track_id]
            excessive_speed = bool(track.speeds and track.speeds[-1] > self.speed_threshold)
            stationary_off_path = bool(
                len(track.path_hits) >= 3
                and track.speeds
                and track.speeds[-1] < max(8.0, self.speed_threshold * 0.03)
                and self._path_ratio(track) < 0.60
                and not self._is_friendly(track.detection)
            )
            stationary_unscheduled = bool(
                self.spawn_events
                and track.detection.enemy_id == "unknown"
                and track.detection.faction_evidence != "enemy"
                and track.detection.health_bar_bbox is None
                and not self._is_friendly(track.detection)
                and len(track.speeds) >= 3
                and float(np.median(list(track.speeds)[-3:])) < max(8.0, self.speed_threshold * 0.03)
                and (
                    not self._near_scheduled_spawn(
                        track.detection,
                        track.detection.candidate_enemy_id,
                        self._stage_elapsed,
                    )
                    or track.stationary_hits >= self.STATIONARY_GRACE_MATCHES
                )
            )
            reasons: list[str] = []
            if excessive_speed:
                reasons.append(
                    f"瞬时速度 {track.speeds[-1]:.0f} px/s 超过上限 {self.speed_threshold:.0f} px/s，"
                    "判定为路线指引干扰"
                )
            if stationary_off_path:
                reasons.append(
                    f"连续 {len(track.path_hits)} 帧多数不在任何敌人路线上且几乎不动，判定为驻场干扰"
                )
            if stationary_unscheduled:
                if track.stationary_hits >= self.STATIONARY_GRACE_MATCHES:
                    reasons.append(
                        f"出怪口附近驻场超过 {track.stationary_hits} 次匹配仍无血条、无类别，"
                        "真实敌人出怪后会立即移动，判定为干扰"
                    )
                else:
                    reasons.append(
                        "未识别目标，无敌方血条，不在任何出怪时间窗口内且持续驻场，判定为干扰"
                    )
            track.guide = excessive_speed or stationary_off_path or stationary_unscheduled
            if track.guide and self._trace_frame is not None:
                self._trace_frame.setdefault("guides", []).append(
                    {"track_id": track_id, "reason": "；".join(reasons)}
                )

    def _class_stable(self, track: _MotionTrack) -> bool:
        """Recognized classes must dominate a track's recognized votes.

        Interference such as UI popups keeps being re-recognized as whatever template
        happens to match, so its recognized votes scatter across unrelated enemy types.
        Unknown votes only dilute evidence and do not count against stability.
        """

        current = track.detection.enemy_id
        if current in {"unknown", "friendly"}:
            return True
        recognized = [cls for cls in track.class_history() if cls not in {"unknown", "friendly"}]
        if len(recognized) < 2:
            return True
        return recognized.count(current) * 2 > len(recognized)

    def _minimum_hits(self, track: _MotionTrack) -> int:
        if (
            self.spawn_events
            and track.detection.enemy_id == "unknown"
            and track.detection.faction_evidence != "enemy"
            and track.detection.health_bar_bbox is None
            and not self._is_friendly(track.detection)
            and (
                not self._near_scheduled_spawn(
                    track.detection,
                    track.detection.candidate_enemy_id,
                    self._stage_elapsed,
                )
                or track.stationary_hits >= self.STATIONARY_GRACE_MATCHES
            )
        ):
            return 4
        return 2

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[TrackSnapshot, ...]:
        self._current_timestamp = timestamp
        if self._last_stage_timestamp is None:
            self._stage_elapsed = max(0.0, timestamp - self.battle_origin_timestamp) * self.time_scale
        else:
            start = max(self._last_stage_timestamp, self.battle_origin_timestamp)
            end = max(timestamp, self.battle_origin_timestamp)
            self._stage_elapsed += max(0.0, end - start) * self.time_scale
        self._last_stage_timestamp = timestamp
        self._trace_frame = (
            {
                "stage_time": round(self._stage_elapsed, 3),
                "speed_threshold_px_s": round(self.speed_threshold, 1),
            }
            if self.trace_hook is not None
            else None
        )
        detections = self._normalize_detections(detections, timestamp)
        if not any(line is not None for line in self.route_lines):
            snapshots = self.tracker.update(detections, timestamp)
            self._emit_trace(None, snapshots)
            return snapshots
        measures = self._measure_paths(detections)
        on_path = []
        for detection in detections:
            distance, score = measures[id(detection)]
            if score > 0:
                on_path.append(detection)
            elif self._trace_frame is not None:
                if distance is None:
                    reason = (
                        f"{detection.enemy_id} 的路线本帧都还没到出怪时间，"
                        "该候选出现得太早，判定与路线无关"
                    )
                else:
                    reason = (
                        f"离该类别的任何活动路线 {distance:.0f} px，"
                        f"超过外圈 {self.path_cutoff:.0f} px，判定与路线无关"
                    )
                self._trace_frame.setdefault("off_path_dropped", []).append(
                    {
                        "reason": reason,
                        **self._nearest_track(detection),
                        "detection": self._trace_detection(detection),
                    }
                )
        detections = tuple(on_path)
        path_scores = {key: value[1] for key, value in measures.items()}
        stale = [tid for tid, track in self._tracks.items() if track.misses > self.max_track_misses]
        for tid in stale:
            track = self._tracks.pop(tid)
            if self._trace_frame is not None:
                self._trace_frame.setdefault("stale_removed", []).append(
                    {
                        "track_id": tid,
                        "reason": f"连续 {track.misses} 帧未匹配到任何检测，删除内部轨迹，避免给新出现的候选错误担保",
                        "last_class": track.detection.enemy_id,
                    }
                )
        matched = self._associate(detections, timestamp, path_scores)
        self._reject_guides(matched)
        accepted = [
            (column, self._tracks[track_id].detection)
            for track_id, column in matched.items()
            if not self._tracks[track_id].guide
            and self._tracks[track_id].consecutive >= self._minimum_hits(self._tracks[track_id])
            and self._class_stable(self._tracks[track_id])
        ]
        if self._trace_frame is not None:
            for track_id in matched:
                track = self._tracks[track_id]
                if track.guide or not self._class_stable(track):
                    if not track.guide:
                        votes = ", ".join(track.class_history())
                        self._trace_frame.setdefault("guides", []).append(
                            {
                                "track_id": track_id,
                                "reason": (
                                    f"近期类别投票（{votes}）与当前类别 {track.detection.enemy_id} "
                                    "不一致，判定为干扰"
                                ),
                            }
                        )
                    continue
                minimum = self._minimum_hits(track)
                if track.consecutive < minimum:
                    self._trace_frame.setdefault("pending", []).append(
                        {
                            "track_id": track_id,
                            "reason": (
                                f"已连续匹配 {track.consecutive} 次（累计 {track.hits} 次），"
                                f"需满 {minimum} 次连续运动证据才允许进入正式跟踪"
                            ),
                        }
                    )
        filtered = tuple(detection for _, detection in sorted(accepted))
        snapshots = self.tracker.update(filtered, timestamp)
        self._emit_trace(filtered, snapshots)
        return snapshots

    @staticmethod
    def _bbox_key(bbox: object) -> tuple[float, ...] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        return tuple(round(float(value)) for value in bbox[:4])

    def _emit_trace(
        self,
        accepted: tuple[Detection, ...] | None,
        snapshots: tuple[TrackSnapshot, ...],
    ) -> None:
        if self.trace_hook is None or self._trace_frame is None:
            self._trace_frame = None
            return
        frame = self._trace_frame
        self._trace_frame = None

        # 内部跟踪 ID 与 HUD 输出 ID 是两套编号，这里把输出 ID 贴到每条检测上，
        # 查看器才能按输出 ID 归属过滤步骤。
        slot_by_bbox: dict[tuple[float, ...], int] = {}
        for snapshot in snapshots:
            if snapshot.misses == 0:
                key = self._bbox_key(snapshot.bbox)
                if key is not None:
                    slot_by_bbox[key] = snapshot.track_id

        def attach_slot(item: dict[str, Any]) -> None:
            detection = item.get("detection")
            if not isinstance(detection, dict):
                return
            key = self._bbox_key(detection.get("bbox"))
            if key is not None and key in slot_by_bbox:
                item["slot_track_id"] = slot_by_bbox[key]

        for group in ("matched", "new_tracks", "guides", "off_path_dropped", "gate_rejected", "accepted"):
            for item in frame.get(group) or []:
                if isinstance(item, dict):
                    attach_slot(item)

        if accepted is not None:
            frame["accepted_count"] = len(accepted)
            frame["accepted"] = [self._trace_detection(detection) for detection in accepted]
            for item in frame["accepted"]:
                attach_slot(item)
        frame["output_track_ids"] = [snapshot.track_id for snapshot in snapshots]
        self.trace_hook(frame)

    def snapshots(self) -> tuple[TrackSnapshot, ...]:
        return self.tracker.snapshots()

    def coast(self, timestamp: float) -> tuple[TrackSnapshot, ...]:
        return self.tracker.coast(timestamp)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.game_speed = 1
        self.time_scale = 1.0
        self._stage_elapsed = 0.0
        self._last_stage_timestamp = None
        self.tracker.reset()
