"""Timeline-primary enemy tracking with bounded visual corrections.

Every scheduled spawn owns a stable track ID. Public route checkpoints and
enemy movement speeds drive position; image detections can only confirm a slot
and apply a small, decaying correction. Unobserved enemies are explicitly
labelled ``predicted`` rather than presented as visually confirmed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from arkeye.board import BoardCalibration
from arkeye.detector import BoundingBox, Detection
from arkeye.timeline import TimedSpawnEvent
from arkeye.tracking import MultiObjectTracker, TrackSnapshot

_INVALID = 1_000_000.0


@dataclass(frozen=True, slots=True)
class _RouteStep:
    start: tuple[float, float]
    end: tuple[float, float]
    distance: float = 0.0
    wait: float = 0.0

    def duration(self, speed: float) -> float:
        return self.wait if self.wait > 0 else self.distance / speed


@dataclass(frozen=True, slots=True)
class RouteTimeline:
    """Projected route geometry including waiting checkpoints."""

    steps: tuple[_RouteStep, ...]
    start: tuple[float, float]
    end: tuple[float, float]

    @classmethod
    def from_mapping(cls, route: Mapping[str, Any], calibration: BoardCalibration) -> RouteTimeline:
        def point(value: Mapping[str, Any]) -> tuple[float, float]:
            return float(value["col"]), float(value["row"])

        def frame_point(value: tuple[float, float]) -> tuple[float, float]:
            projected = calibration.map_points_to_frame([value])[0]
            return float(projected[0]), float(projected[1])

        current_map = point(route["startPosition"])
        start = frame_point(current_map)
        current_frame = start
        steps: list[_RouteStep] = []

        def move(destination: Mapping[str, Any]) -> None:
            nonlocal current_map, current_frame
            target_map = point(destination)
            target_frame = frame_point(target_map)
            distance = float(np.hypot(target_map[0] - current_map[0], target_map[1] - current_map[1]))
            steps.append(_RouteStep(current_frame, target_frame, distance=distance))
            current_map, current_frame = target_map, target_frame

        for checkpoint in route.get("checkpoints") or []:
            if checkpoint.get("type") == "MOVE":
                move(checkpoint["position"])
            elif checkpoint.get("type") == "WAIT_FOR_SECONDS":
                duration = max(0.0, float(checkpoint.get("time") or 0.0))
                if duration:
                    steps.append(_RouteStep(current_frame, current_frame, wait=duration))
        move(route["endPosition"])
        return cls(tuple(steps), start, current_frame)

    def position_at(self, elapsed: float, speed: float) -> tuple[np.ndarray, bool]:
        if speed <= 0:
            raise ValueError("Enemy movement speed must be positive")
        remaining = max(0.0, elapsed)
        for step in self.steps:
            duration = step.duration(speed)
            if duration > 0 and remaining < duration:
                ratio = 0.0 if step.wait > 0 else remaining / duration
                start = np.asarray(step.start, dtype=np.float64)
                end = np.asarray(step.end, dtype=np.float64)
                return start + (end - start) * ratio, False
            remaining -= duration
        return np.asarray(self.end, dtype=np.float64), True

    def project_elapsed(self, point: np.ndarray, speed: float, preferred_elapsed: float) -> tuple[float, float]:
        """Estimate route progress from an image point, breaking loops by expected progress."""

        best: tuple[float, float, float] | None = None
        base_elapsed = 0.0
        for step in self.steps:
            duration = step.duration(speed)
            if step.wait > 0:
                candidate = base_elapsed + step.wait * 0.5
                distance = float(np.linalg.norm(point - np.asarray(step.start)))
            else:
                start = np.asarray(step.start, dtype=np.float64)
                delta = np.asarray(step.end, dtype=np.float64) - start
                length_squared = float(np.dot(delta, delta))
                ratio = float(np.clip(np.dot(point - start, delta) / length_squared, 0.0, 1.0)) if length_squared else 0.0
                candidate = base_elapsed + duration * ratio
                distance = float(np.linalg.norm(point - (start + delta * ratio)))
            score = distance + 0.05 * abs(candidate - preferred_elapsed)
            if best is None or score < best[0]:
                best = score, candidate, distance
            base_elapsed += duration
        return (best[1], best[2]) if best is not None else (0.0, float("inf"))

    def total_duration(self, speed: float) -> float:
        return sum(step.duration(speed) for step in self.steps)


@dataclass(frozen=True, slots=True)
class _SpawnSlot:
    track_id: int
    enemy_id: str
    route_index: int
    spawn_time: float


@dataclass(slots=True)
class _TimelineTrack:
    slot: _SpawnSlot
    bbox_size: tuple[int, int]
    position: np.ndarray
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    progress_delay: float = 0.0
    bbox: BoundingBox = (0, 0, 1, 1)
    velocity: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.35
    status: Literal["confirmed", "predicted"] = "predicted"
    hits: int = 0
    misses: int = 0
    active: bool = False
    finished: bool = False
    last_timestamp: float | None = None
    history: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=90))


def load_enemy_move_speeds(database: Mapping[str, Any], *, level: int = 0) -> dict[str, float]:
    """Read public per-enemy tile/second speeds without hard-coded class values."""

    speeds: dict[str, float] = {}
    for entry in database.get("enemies") or []:
        key = str(entry.get("Key") or "")
        values = entry.get("Value") or []
        selected = next((value for value in values if int(value.get("level") or 0) == level), values[0] if values else None)
        if not key or selected is None:
            continue
        attribute = selected["enemyData"]["attributes"]["moveSpeed"]
        speed = float(attribute.get("m_value") or 0.0)
        if speed > 0:
            speeds[key] = speed
    return speeds


class TimelinePrimaryTracker(MultiObjectTracker):
    """Drive enemies from stage data while retaining visual friendly tracking."""

    def __init__(
        self,
        *,
        visual_tracker: MultiObjectTracker,
        routes: tuple[RouteTimeline | None, ...],
        spawn_events: tuple[TimedSpawnEvent, ...],
        enemy_move_speeds: Mapping[str, float],
        typical_bbox_areas: Mapping[str, float],
        pixels_per_tile: float,
        battle_origin_timestamp: float,
        friendly_id_offset: int = 1000,
        end_grace: float = 1.0,
    ) -> None:
        if pixels_per_tile <= 0:
            raise ValueError("pixels_per_tile must be positive")
        self.visual_tracker = visual_tracker
        self.routes = routes
        self.enemy_move_speeds = dict(enemy_move_speeds)
        self.typical_bbox_areas = dict(typical_bbox_areas)
        self.pixels_per_tile = pixels_per_tile
        self.battle_origin_timestamp = battle_origin_timestamp
        self.friendly_id_offset = friendly_id_offset
        self.end_grace = end_grace
        self.association_gate = pixels_per_tile * 1.5
        self.max_visual_correction = pixels_per_tile * 0.65
        self.game_speed = 1
        self.time_scale = 1.0
        self._stage_elapsed = 0.0
        self._last_stage_timestamp: float | None = None
        self._slots = self._expand_slots(spawn_events)
        self._tracks = self._make_tracks()
        self._friendly_snapshots: tuple[TrackSnapshot, ...] = ()

    @staticmethod
    def _expand_slots(spawn_events: tuple[TimedSpawnEvent, ...]) -> tuple[_SpawnSlot, ...]:
        slots = []
        for event in spawn_events:
            for index in range(event.count):
                slots.append(
                    _SpawnSlot(
                        track_id=len(slots) + 1,
                        enemy_id=event.enemy_key,
                        route_index=event.route_index,
                        spawn_time=event.timestamp + index * event.interval,
                    )
                )
        return tuple(slots)

    def _make_tracks(self) -> dict[int, _TimelineTrack]:
        tracks = {}
        for slot in self._slots:
            if slot.enemy_id not in self.enemy_move_speeds:
                raise ValueError(f"Missing movement speed for {slot.enemy_id}")
            if not 0 <= slot.route_index < len(self.routes) or self.routes[slot.route_index] is None:
                raise ValueError(f"Missing route {slot.route_index} for {slot.enemy_id}")
            area = self.typical_bbox_areas.get(slot.enemy_id, 12_000.0)
            width = max(24, round(area**0.5 * 0.85))
            height = max(32, round(area**0.5 * 1.15))
            start = np.asarray(self.routes[slot.route_index].start, dtype=np.float64)  # type: ignore[union-attr]
            track = _TimelineTrack(slot, (width, height), start)
            track.bbox = self._bbox_at(track, start)
            tracks[slot.track_id] = track
        return tracks

    @staticmethod
    def _bbox_at(track: _TimelineTrack, position: np.ndarray) -> BoundingBox:
        width, height = track.bbox_size
        return round(float(position[0]) - width / 2), round(float(position[1]) - height + 1), width, height

    def set_game_speed(self, speed: int) -> None:
        if speed not in {1, 2}:
            raise ValueError("game_speed must be 1 or 2")
        self.game_speed = speed
        self.time_scale = float(speed)
        setter = getattr(self.visual_tracker, "set_game_speed", None)
        if callable(setter):
            setter(speed)

    def set_time_scale(self, scale: float) -> None:
        if not 0.05 <= scale <= 2.0:
            raise ValueError("time_scale must be in [0.05, 2]")
        self.time_scale = scale
        setter = getattr(self.visual_tracker, "set_time_scale", None)
        if callable(setter):
            setter(scale)

    def _advance_stage_time(self, timestamp: float) -> None:
        if self._last_stage_timestamp is None:
            self._stage_elapsed = max(0.0, timestamp - self.battle_origin_timestamp) * self.time_scale
        else:
            start = max(self._last_stage_timestamp, self.battle_origin_timestamp)
            end = max(timestamp, self.battle_origin_timestamp)
            self._stage_elapsed += max(0.0, end - start) * self.time_scale
        self._last_stage_timestamp = timestamp

    @staticmethod
    def _eligible(detection: Detection, enemy_id: str) -> bool:
        if detection.faction_evidence == "friendly" or detection.possible_friendly:
            return False
        if detection.enemy_id not in {"unknown", enemy_id}:
            return False
        return detection.enemy_id != "unknown" or detection.candidate_enemy_id in {None, enemy_id}

    def _prepare_positions(
        self,
        timestamp: float,
        *,
        mark_missing: bool,
    ) -> dict[int, np.ndarray]:
        route_positions: dict[int, np.ndarray] = {}
        for track in self._tracks.values():
            if track.finished:
                continue
            slot = track.slot
            if not track.active:
                if self._stage_elapsed + 1e-9 < slot.spawn_time:
                    continue
                track.active = True
                track.last_timestamp = timestamp
            route = self.routes[slot.route_index]
            speed = self.enemy_move_speeds[slot.enemy_id]
            assert route is not None
            raw_elapsed = max(0.0, self._stage_elapsed - slot.spawn_time)
            elapsed = max(0.0, raw_elapsed - track.progress_delay)
            route_position, reached_end = route.position_at(elapsed, speed)
            if reached_end and elapsed > route.total_duration(speed) + self.end_grace:
                track.finished = True
                continue
            previous = track.position.copy()
            if mark_missing:
                track.status = "predicted"
                track.confidence = 0.35
                track.misses += 1
                track.offset *= 0.90
            track.position = route_position + track.offset
            delta_time = max(1e-3, timestamp - track.last_timestamp) if track.last_timestamp is not None else 1.0 / 30.0
            track.velocity = tuple(((track.position - previous) / delta_time).tolist())
            track.bbox = self._bbox_at(track, track.position)
            track.last_timestamp = timestamp
            route_positions[slot.track_id] = route_position
        return route_positions

    def _append_history(self, track: _TimelineTrack, timestamp: float) -> None:
        track.history.append((timestamp, float(track.position[0]), float(track.position[1])))

    @staticmethod
    def _snapshot(track: _TimelineTrack) -> TrackSnapshot:
        return TrackSnapshot(
            track_id=track.slot.track_id,
            enemy_id=track.slot.enemy_id,
            confidence=track.confidence,
            status=track.status,
            bbox=track.bbox,
            foot_point=(float(track.position[0]), float(track.position[1])),
            velocity=track.velocity,
            hits=track.hits,
            misses=track.misses,
            occlusion_group=None,
            template_name=None,
            history=tuple(track.history),
        )

    def _set_friendlies(self, snapshots: tuple[TrackSnapshot, ...]) -> None:
        self._friendly_snapshots = tuple(
            replace(track, track_id=self.friendly_id_offset + track.track_id)
            for track in snapshots
            if track.enemy_id == "friendly"
        )

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[TrackSnapshot, ...]:
        self._advance_stage_time(timestamp)
        friendly_detections = tuple(
            detection for detection in detections if detection.enemy_id == "friendly" or detection.possible_friendly
        )
        self._set_friendlies(self.visual_tracker.update(friendly_detections, timestamp))
        route_positions = self._prepare_positions(timestamp, mark_missing=True)
        active = [self._tracks[track_id] for track_id in route_positions]
        candidates = [*active, *(track for track in self._tracks.values() if track.finished)]
        progress_estimates: dict[tuple[int, int], tuple[float, float]] = {}
        path_gate = self.pixels_per_tile * 0.55
        if candidates and detections:
            costs = np.full((len(candidates), len(detections)), _INVALID, dtype=np.float64)
            for row, track in enumerate(candidates):
                slot = track.slot
                route = self.routes[slot.route_index]
                speed = self.enemy_move_speeds[slot.enemy_id]
                assert route is not None
                raw_elapsed = max(0.0, self._stage_elapsed - slot.spawn_time)
                preferred_elapsed = max(0.0, raw_elapsed - track.progress_delay)
                route_position = route_positions.get(slot.track_id, np.asarray(route.end, dtype=np.float64))
                for column, detection in enumerate(detections):
                    if not self._eligible(detection, slot.enemy_id):
                        continue
                    point = np.asarray(detection.foot_point, dtype=np.float64)
                    distance = float(np.linalg.norm(route_position - point))
                    if not track.finished and distance <= self.association_gate:
                        costs[row, column] = distance / self.association_gate + 0.05 * (1.0 - detection.confidence)
                        continue
                    projected_elapsed, path_distance = route.project_elapsed(point, speed, preferred_elapsed)
                    if path_distance <= path_gate and projected_elapsed <= raw_elapsed + 2.0:
                        progress_estimates[(slot.track_id, column)] = (projected_elapsed, path_distance)
                        dormant_penalty = 0.20 if track.finished else 0.0
                        costs[row, column] = 1.10 + dormant_penalty + path_distance / path_gate + 0.05 * (1.0 - detection.confidence)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] >= _INVALID:
                    continue
                track = candidates[row]
                detection = detections[column]
                slot = track.slot
                route = self.routes[slot.route_index]
                speed = self.enemy_move_speeds[slot.enemy_id]
                assert route is not None
                raw_elapsed = max(0.0, self._stage_elapsed - slot.spawn_time)
                preferred_elapsed = max(0.0, raw_elapsed - track.progress_delay)
                point = np.asarray(detection.foot_point, dtype=np.float64)
                projected_elapsed, path_distance = progress_estimates.get(
                    (slot.track_id, column), route.project_elapsed(point, speed, preferred_elapsed)
                )
                if path_distance <= path_gate:
                    desired_delay = float(np.clip(raw_elapsed - projected_elapsed, -2.0, raw_elapsed))
                    weight = 0.65 if costs[row, column] >= 1.10 else 0.35
                    track.progress_delay += weight * (desired_delay - track.progress_delay)
                elapsed = max(0.0, raw_elapsed - track.progress_delay)
                route_position, _ = route.position_at(elapsed, speed)
                measured_offset = point - route_position
                correction = 0.75 * track.offset + 0.25 * measured_offset
                length = float(np.linalg.norm(correction))
                if length > self.max_visual_correction:
                    correction *= self.max_visual_correction / length
                previous = track.position.copy()
                track.offset = correction
                track.position = route_position + correction
                current_area = track.bbox[2] * track.bbox[3]
                detection_area = detection.bbox[2] * detection.bbox[3]
                if detection_area <= max(current_area * 3, current_area + 30_000):
                    track.bbox = detection.bbox
                    track.bbox_size = (detection.bbox[2], detection.bbox[3])
                else:
                    track.bbox = self._bbox_at(track, track.position)
                track.active = True
                track.finished = False
                track.status = "confirmed"
                track.confidence = max(0.65, detection.confidence)
                track.hits += 1
                track.misses = 0
                track.velocity = tuple(((track.position - previous) / 0.03333333333333333).tolist())
                route_positions[slot.track_id] = route_position

        for track_id in route_positions:
            track = self._tracks[track_id]
            self._append_history(track, timestamp)
        return self.snapshots()

    def coast(self, timestamp: float) -> tuple[TrackSnapshot, ...]:
        self._advance_stage_time(timestamp)
        self._set_friendlies(self.visual_tracker.coast(timestamp))
        route_positions = self._prepare_positions(timestamp, mark_missing=False)
        for track_id in route_positions:
            self._append_history(self._tracks[track_id], timestamp)
        return self.snapshots()

    def snapshots(self) -> tuple[TrackSnapshot, ...]:
        scheduled = [
            self._snapshot(track)
            for track in self._tracks.values()
            if track.active and not track.finished
        ]
        return tuple(sorted((*scheduled, *self._friendly_snapshots), key=lambda track: track.track_id))

    def _routes_for(self, enemy_id: str) -> tuple[np.ndarray, ...]:
        routes_for = getattr(self.visual_tracker, "_routes_for", None)
        return routes_for(enemy_id) if callable(routes_for) else ()

    def reset(self) -> None:
        self.visual_tracker.reset()
        self._tracks = self._make_tracks()
        self._friendly_snapshots = ()
        self.game_speed = 1
        self.time_scale = 1.0
        self._stage_elapsed = 0.0
        self._last_stage_timestamp = None
