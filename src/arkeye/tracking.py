"""Constant-velocity multi-object tracking with explicit occlusion handling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from arkeye.detector import BoundingBox, Detection

TrackStatus = Literal["tentative", "confirmed", "occluded", "lost", "predicted"]
_INVALID_COST = 1_000_000.0


@dataclass(slots=True)
class _Track:
    track_id: int
    kalman: cv2.KalmanFilter
    bbox: BoundingBox
    enemy_id: str
    confidence: float
    appearance: np.ndarray | None
    status: TrackStatus
    hits: int
    misses: int
    age: int
    occlusion_group: int | None
    template_name: str | None = None
    history: deque[tuple[float, float, float]] = field(default_factory=deque)


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    track_id: int
    enemy_id: str
    confidence: float
    status: TrackStatus
    bbox: BoundingBox
    foot_point: tuple[float, float]
    velocity: tuple[float, float]
    hits: int
    misses: int
    occlusion_group: int | None
    template_name: str | None = None
    history: tuple[tuple[float, float, float], ...] = ()


class MultiObjectTracker:
    def __init__(
        self,
        *,
        distance_gate: float = 80.0,
        min_hits: int = 2,
        max_missed: int = 30,
        history_length: int = 90,
        appearance_weight: float = 0.75,
    ) -> None:
        if distance_gate <= 0 or min_hits < 1 or max_missed < 0 or history_length < 1:
            raise ValueError("Invalid tracker limits")
        self.distance_gate = distance_gate
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.history_length = history_length
        self.appearance_weight = appearance_weight
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_timestamp: float | None = None

    def _new_track(self, detection: Detection, timestamp: float) -> _Track:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kalman.processNoiseCov = np.diag([0.2, 0.2, 2.0, 2.0]).astype(np.float32)
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        x, y = detection.foot_point
        kalman.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        track = _Track(
            track_id=self._next_id,
            kalman=kalman,
            bbox=detection.bbox,
            enemy_id=detection.enemy_id,
            confidence=detection.confidence,
            appearance=self._copy_appearance(detection.appearance),
            status="confirmed" if self.min_hits == 1 else "tentative",
            hits=1,
            misses=0,
            age=1,
            occlusion_group=None,
            template_name=detection.template_name,
            history=deque(maxlen=self.history_length),
        )
        track.history.append((timestamp, x, y))
        self._next_id += 1
        self._tracks[track.track_id] = track
        return track

    @staticmethod
    def _copy_appearance(appearance: np.ndarray | None) -> np.ndarray | None:
        return np.asarray(appearance, dtype=np.float32).reshape(-1, 1).copy() if appearance is not None else None

    @staticmethod
    def _position(track: _Track) -> np.ndarray:
        return track.kalman.statePost[:2, 0].astype(np.float64)

    @staticmethod
    def _predict(track: _Track, delta_time: float) -> np.ndarray:
        track.kalman.transitionMatrix = np.array(
            [[1, 0, delta_time, 0], [0, 1, 0, delta_time], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        return track.kalman.predict()[:2, 0].astype(np.float64)

    @staticmethod
    def _predicted_bbox(track: _Track, position: np.ndarray) -> BoundingBox:
        _, _, width, height = track.bbox
        x = round(position[0] - width / 2.0)
        y = round(position[1] - height + 1.0)
        return x, y, width, height

    @staticmethod
    def _iou(first: BoundingBox, second: BoundingBox) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _appearance_distance(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
        if first is None or second is None or first.shape != second.shape:
            return None
        return float(cv2.compareHist(first.astype(np.float32), second.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))

    def _association_cost(self, track: _Track, detection: Detection, position: np.ndarray) -> float:
        if track.enemy_id != "unknown" and detection.enemy_id != "unknown" and track.enemy_id != detection.enemy_id:
            return _INVALID_COST
        distance = float(np.linalg.norm(position - np.asarray(detection.foot_point)))
        gate = self.distance_gate
        if track.status in {"occluded", "lost"}:
            gate *= min(2.0, 1.5 + track.misses * 0.05)
        if distance > gate:
            return _INVALID_COST
        overlap = self._iou(self._predicted_bbox(track, position), detection.bbox)
        cost = distance / gate + 0.5 * (1.0 - overlap)
        appearance_distance = self._appearance_distance(track.appearance, detection.appearance)
        if appearance_distance is not None:
            cost += self.appearance_weight * appearance_distance
        return cost

    def _inside_detection(self, position: np.ndarray, detection: Detection) -> bool:
        x, y, width, height = detection.bbox
        margin_x = max(2.0, width * 0.05)
        margin_y = max(2.0, height * 0.05)
        return x - margin_x <= position[0] <= x + width + margin_x and y - margin_y <= position[1] <= y + height + margin_y

    def _find_merged_detections(
        self,
        detections: tuple[Detection, ...],
        predictions: dict[int, np.ndarray],
    ) -> dict[int, int]:
        """Return track_id -> detection index for blobs supporting multiple tracks."""

        merged: dict[int, int] = {}
        for detection_index, detection in enumerate(detections):
            supporters: list[int] = []
            for track_id, position in predictions.items():
                track = self._tracks[track_id]
                if track.status == "tentative":
                    continue
                distance = float(np.linalg.norm(position - np.asarray(detection.foot_point)))
                if distance <= self.distance_gate * 1.5 and self._inside_detection(position, detection):
                    supporters.append(track_id)
            if len(supporters) > 1:
                for track_id in supporters:
                    merged[track_id] = detection_index
        return merged

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[TrackSnapshot, ...]:
        if self._last_timestamp is None:
            delta_time = 1.0 / 30.0
        else:
            delta_time = float(np.clip(timestamp - self._last_timestamp, 1e-3, 0.2))
        self._last_timestamp = timestamp

        predictions: dict[int, np.ndarray] = {}
        for track in self._tracks.values():
            track.age += 1
            predictions[track.track_id] = self._predict(track, delta_time)

        occluded = self._find_merged_detections(detections, predictions)
        merged_detection_indices = set(occluded.values())
        track_ids = list(self._tracks)
        matched_tracks: dict[int, Detection] = {}
        matched_detections: set[int] = set()

        if track_ids and detections:
            costs = np.full((len(track_ids), len(detections)), _INVALID_COST, dtype=np.float64)
            for row, track_id in enumerate(track_ids):
                if track_id in occluded:
                    continue
                track = self._tracks[track_id]
                for column, detection in enumerate(detections):
                    if column not in merged_detection_indices:
                        costs[row, column] = self._association_cost(track, detection, predictions[track_id])
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] >= _INVALID_COST:
                    continue
                track_id = track_ids[row]
                matched_tracks[track_id] = detections[column]
                matched_detections.add(column)

        for track_id, detection in matched_tracks.items():
            track = self._tracks[track_id]
            measurement = np.asarray(detection.foot_point, dtype=np.float32).reshape(2, 1)
            track.kalman.correct(measurement)
            track.bbox = detection.bbox
            if detection.enemy_id != "unknown":
                track.enemy_id = detection.enemy_id
            track.confidence = detection.confidence
            if detection.template_name is not None:
                track.template_name = detection.template_name
            if detection.appearance is not None:
                appearance = self._copy_appearance(detection.appearance)
                if track.appearance is None:
                    track.appearance = appearance
                else:
                    track.appearance = 0.8 * track.appearance + 0.2 * appearance
            track.hits += 1
            track.misses = 0
            track.status = "confirmed" if track.hits >= self.min_hits else "tentative"
            track.occlusion_group = None
            x, y = self._position(track)
            track.history.append((timestamp, float(x), float(y)))

        for track_id, track in self._tracks.items():
            if track_id in matched_tracks:
                continue
            track.misses += 1
            if track_id in occluded:
                track.status = "occluded"
                track.occlusion_group = occluded[track_id]
            else:
                track.status = "lost"
                track.occlusion_group = None
            x, y = predictions[track_id]
            track.history.append((timestamp, float(x), float(y)))

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections and detection_index not in merged_detection_indices:
                self._new_track(detection, timestamp)

        return self.snapshots()

    def snapshots(self) -> tuple[TrackSnapshot, ...]:
        snapshots: list[TrackSnapshot] = []
        for track in self._tracks.values():
            position = self._position(track)
            velocity = track.kalman.statePost[2:4, 0]
            bbox = track.bbox if track.misses == 0 else self._predicted_bbox(track, position)
            snapshots.append(
                TrackSnapshot(
                    track_id=track.track_id,
                    enemy_id=track.enemy_id,
                    confidence=track.confidence,
                    status=track.status,
                    bbox=bbox,
                    foot_point=(float(position[0]), float(position[1])),
                    velocity=(float(velocity[0]), float(velocity[1])),
                    hits=track.hits,
                    misses=track.misses,
                    occlusion_group=track.occlusion_group,
                    template_name=track.template_name,
                    history=tuple(track.history),
                )
            )
        return tuple(sorted(snapshots, key=lambda snapshot: snapshot.track_id))

    def coast(self, timestamp: float) -> tuple[TrackSnapshot, ...]:
        """Advance display history without treating an analysis skip as a miss."""

        self._last_timestamp = timestamp
        for track in self._tracks.values():
            x, y = self._position(track)
            track.history.append((timestamp, float(x), float(y)))
        return self.snapshots()

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_timestamp = None
