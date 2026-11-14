"""Generic route-conforming motion profiling for spawn-guide rejection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

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


class RouteSpeedFilter:
    """Reject route motion faster than regular path-following enemies."""

    MAX_ENEMY_SPEED_TILES_PER_SECOND = 2.5

    def __init__(
        self,
        tracker: MultiObjectTracker,
        route_lines: tuple[np.ndarray, ...],
        *,
        frame_diagonal: float,
        pixels_per_tile: float,
        class_route_indices: dict[str, tuple[int, ...]] | None = None,
        earliest_spawn_times: dict[str, float] | None = None,
        route_earliest_times: dict[int, float] | None = None,
        spawn_events: tuple[TimedSpawnEvent, ...] = (),
        battle_origin_timestamp: float = 0.0,
        spawn_time_tolerance: float = 2.0,
    ) -> None:
        if frame_diagonal <= 0 or pixels_per_tile <= 0:
            raise ValueError("frame_diagonal and pixels_per_tile must be positive")
        self.tracker = tracker
        self.route_lines = tuple(np.asarray(line, dtype=np.float32) for line in route_lines if len(line) >= 2)
        self.frame_diagonal = frame_diagonal
        self.pixels_per_tile = pixels_per_tile
        self.class_route_indices = class_route_indices or {}
        self.earliest_spawn_times = earliest_spawn_times or {}
        self.route_earliest_times = route_earliest_times or {}
        self.spawn_events = spawn_events
        self.battle_origin_timestamp = battle_origin_timestamp
        self.spawn_time_tolerance = spawn_time_tolerance
        self.game_speed = 1
        self.time_scale = 1.0
        self.path_tolerance = frame_diagonal * 0.03
        self.spawn_entry_tolerance = frame_diagonal * 0.12
        self._current_timestamp = battle_origin_timestamp
        self._stage_elapsed = 0.0
        self._last_stage_timestamp: float | None = None
        self._tracks: dict[int, _MotionTrack] = {}
        self._next_id = 1

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
            indices = tuple(range(len(self.route_lines)))
        stage_time = self._stage_elapsed
        return tuple(
            self.route_lines[index]
            for index in indices
            if 0 <= index < len(self.route_lines)
            and stage_time >= self.route_earliest_times.get(index, float("-inf")) - self.spawn_time_tolerance
        )

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

    def _near_scheduled_spawn(self, detection: Detection, enemy_id: str, stage_time: float) -> bool:
        foot = np.asarray(detection.foot_point, dtype=np.float32)
        tolerance = max(4.0, self.spawn_time_tolerance)
        for event in self.spawn_events:
            if event.enemy_key != enemy_id or not 0 <= event.route_index < len(self.route_lines):
                continue
            entry = self.route_lines[event.route_index][0]
            if float(np.linalg.norm(foot - entry)) > self.spawn_entry_tolerance:
                continue
            for index in range(event.count):
                if abs(stage_time - (event.timestamp + index * event.interval)) <= tolerance:
                    return True
        return False

    def _on_path(self, detection: Detection) -> bool:
        routes = self._routes_for(detection.enemy_id)
        if not routes:
            return False
        x, y = (float(value) for value in detection.foot_point)
        distance = min(abs(cv2.pointPolygonTest(line, (x, y), True)) for line in routes)
        return distance <= self.path_tolerance

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
            earliest = self.earliest_spawn_times.get(detection.enemy_id)
            too_early = earliest is not None and stage_time < earliest - self.spawn_time_tolerance
            wrong_route = (
                detection.enemy_id not in {"unknown", "friendly"}
                and self.class_route_indices
                and not self._on_path(detection)
                and self._on_path(replace(detection, enemy_id="unknown"))
            )
            if too_early or wrong_route:
                detection = replace(detection, enemy_id="unknown", template_name=None, candidate_enemy_id=None)
            normalized.append(detection)
        return tuple(normalized)

    def _associate(self, detections: tuple[Detection, ...], timestamp: float) -> dict[int, int]:
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
                        costs[row, column] = distance
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] >= _INVALID:
                    continue
                track = tracks[row]
                detection = detections[column]
                delta_time = max(1e-3, timestamp - track.timestamp)
                displacement = np.asarray(detection.foot_point) - np.asarray(track.detection.foot_point)
                track.velocity = tuple((displacement / delta_time).tolist())
                track.speeds.append(self._distance(track.detection, detection) / delta_time)
                track.path_hits.append(self._on_path(detection))
                track.detection = detection
                track.timestamp = timestamp
                track.hits += 1
                track.misses = 0
                matched[track.track_id] = column

        matched_columns = set(matched.values())
        for column, detection in enumerate(detections):
            if column in matched_columns:
                continue
            if tracks:
                nearest = min(tracks, key=lambda track: self._distance(track.detection, detection))
                rejected_distance = self._distance(nearest.detection, detection)
                if 0 < rejected_distance <= self.frame_diagonal * 0.25 and self._on_path(detection):
                    delta_time = max(1e-3, timestamp - nearest.timestamp)
                    # Keep rejected guide speeds observable without moving a real track.
                    rejected_speed = rejected_distance / delta_time
                    nearest.speeds.append(rejected_speed)
                    if rejected_speed > self.speed_threshold and nearest.hits < 2:
                        nearest.guide = True
            if not self._is_friendly(detection) and not self._near_route_entry(detection):
                continue
            track = _MotionTrack(self._next_id, detection, timestamp)
            track.path_hits.append(self._on_path(detection))
            self._tracks[self._next_id] = track
            self._next_id += 1

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
            track.guide = excessive_speed or stationary_off_path

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[TrackSnapshot, ...]:
        self._current_timestamp = timestamp
        if self._last_stage_timestamp is None:
            self._stage_elapsed = max(0.0, timestamp - self.battle_origin_timestamp) * self.time_scale
        else:
            start = max(self._last_stage_timestamp, self.battle_origin_timestamp)
            end = max(timestamp, self.battle_origin_timestamp)
            self._stage_elapsed += max(0.0, end - start) * self.time_scale
        self._last_stage_timestamp = timestamp
        detections = self._normalize_detections(detections, timestamp)
        if not self.route_lines:
            return self.tracker.update(detections, timestamp)
        detections = tuple(detection for detection in detections if self._is_friendly(detection) or self._on_path(detection))
        matched = self._associate(detections, timestamp)
        self._reject_guides(matched)
        accepted = [
            (column, self._tracks[track_id].detection)
            for track_id, column in matched.items()
            if not self._tracks[track_id].guide
            and self._tracks[track_id].hits >= 2
            and (
                self._is_friendly(self._tracks[track_id].detection)
                or self._on_path(self._tracks[track_id].detection)
            )
        ]
        filtered = tuple(detection for _, detection in sorted(accepted))
        return self.tracker.update(filtered, timestamp)

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
