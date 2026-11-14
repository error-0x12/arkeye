"""Short-lived Lucas-Kanade optical-flow assistance for existing tracks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from arkeye.detector import BoundingBox, Detection
from arkeye.tracking import TrackSnapshot


@dataclass(slots=True)
class _Observation:
    gray: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    tracks: dict[int, TrackSnapshot]


class LKFlowAssistant:
    """Predict short gaps from masked local features, never from bare candidates."""

    def __init__(
        self,
        *,
        min_inliers: int = 4,
        max_missing: int = 5,
        max_flow_only: int = 3,
        forward_backward_threshold: float = 1.5,
    ) -> None:
        if min_inliers < 2 or max_missing < 0 or max_flow_only < 1 or forward_backward_threshold <= 0:
            raise ValueError("Invalid optical-flow limits")
        self.min_inliers = min_inliers
        self.max_missing = max_missing
        self.max_flow_only = max_flow_only
        self.forward_backward_threshold = forward_backward_threshold
        self._observation: _Observation | None = None
        self._flow_only: dict[int, int] = {}

    @staticmethod
    def _shift_bbox(bbox: BoundingBox, delta: NDArray[np.float64]) -> BoundingBox:
        x, y, width, height = bbox
        return round(x + delta[0]), round(y + delta[1]), width, height

    def _eligible(self, track: TrackSnapshot) -> bool:
        return (
            track.status in {"confirmed", "occluded"}
            and track.hits >= 3
            and track.misses <= self.max_missing
            and self._flow_only.get(track.track_id, 0) < self.max_flow_only
        )

    def _predict_track(
        self,
        current_gray: NDArray[np.uint8],
        previous: _Observation,
        track: TrackSnapshot,
    ) -> Detection | None:
        frame_height, frame_width = previous.gray.shape
        x, y, width, height = track.bbox
        left, top = max(0, x), max(0, y)
        right, bottom = min(frame_width, x + width), min(frame_height, y + height)
        if right - left < 8 or bottom - top < 8:
            return None
        local_mask = previous.mask[top:bottom, left:right]
        if cv2.countNonZero(local_mask) < self.min_inliers * 4:
            return None
        previous_gray = previous.gray[top:bottom, left:right]
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=40,
            qualityLevel=0.01,
            minDistance=4,
            mask=local_mask,
        )
        if points is None or len(points) < self.min_inliers:
            return None
        points = points.astype(np.float32) + np.asarray([left, top], dtype=np.float32)
        parameters = {"winSize": (21, 21), "maxLevel": 3}
        forward, status, error = cv2.calcOpticalFlowPyrLK(previous.gray, current_gray, points, None, **parameters)
        if forward is None:
            return None
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous.gray,
            forward,
            None,
            **parameters,
        )
        if backward is None:
            return None
        status_flat = status.reshape(-1).astype(bool)
        backward_flat = backward_status.reshape(-1).astype(bool)
        forward_backward_error = np.linalg.norm(points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
        error_flat = error.reshape(-1)
        valid = (
            status_flat
            & backward_flat
            & (error_flat <= 35.0)
            & (forward_backward_error <= self.forward_backward_threshold)
        )
        if int(valid.sum()) < self.min_inliers:
            return None
        valid_points = forward.reshape(-1, 2)[valid].astype(np.float64)
        origin_points = points.reshape(-1, 2)[valid].astype(np.float64)
        displacements = valid_points - origin_points
        median = np.median(displacements, axis=0)
        displacement_norm = float(np.linalg.norm(median))
        consistency_limit = max(6.0, displacement_norm * 0.75 + 3.0)
        consistent = np.linalg.norm(displacements - median, axis=1) <= consistency_limit
        if int(consistent.sum()) < self.min_inliers:
            return None
        median = np.median(displacements[consistent], axis=0)
        bbox = self._shift_bbox(track.bbox, median)
        confidence = min(0.59, track.confidence * 0.8)
        return Detection(
            bbox=bbox,
            foot_point=(track.foot_point[0] + float(median[0]), track.foot_point[1] + float(median[1])),
            enemy_id=track.enemy_id,
            confidence=confidence,
            template_name=track.template_name,
            possible_friendly=track.enemy_id == "friendly",
            candidate_enemy_id=track.enemy_id if track.enemy_id != "unknown" else None,
            faction_evidence="friendly" if track.enemy_id == "friendly" else "enemy" if track.enemy_id != "unknown" else None,
        )

    def predict(self, frame: NDArray[np.uint8]) -> tuple[Detection, ...]:
        if self._observation is None:
            return ()
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        predictions = []
        for track_id, track in self._observation.tracks.items():
            if not self._eligible(track):
                continue
            prediction = self._predict_track(current_gray, self._observation, track)
            if prediction is None:
                continue
            predictions.append(prediction)
            self._flow_only[track_id] = self._flow_only.get(track_id, 0) + 1
        return tuple(predictions)

    def observe(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None,
        tracks: Sequence[TrackSnapshot],
        visual_detections: Sequence[Detection],
    ) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mask is None and self._observation is not None:
            mask = self._observation.mask
        if mask is None:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
        elif mask.shape != gray.shape:
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        current_tracks = {track.track_id: track for track in tracks}
        active_ids = set(current_tracks)
        self._flow_only = {track_id: count for track_id, count in self._flow_only.items() if track_id in active_ids}
        for track_id, track in current_tracks.items():
            for detection in visual_detections:
                distance = float(np.linalg.norm(np.asarray(track.foot_point) - np.asarray(detection.foot_point)))
                if distance <= 60.0:
                    self._flow_only[track_id] = 0
                    break
        self._observation = _Observation(gray=gray, mask=mask.copy(), tracks=current_tracks)
