"""Optional MOSSE/KCF assistance for existing tracks (OpenCV contrib)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from arkeye.detector import Detection
from arkeye.tracking import TrackSnapshot


@dataclass(slots=True)
class _Observation:
    trackers: dict[int, Any]
    tracks: dict[int, TrackSnapshot]


class LegacyTrackerAssistant:
    """Run short-lived MOSSE or KCF predictions without creating new identities."""

    def __init__(self, mode: Literal["mosse", "kcf"], *, max_missing: int = 5, max_flow_only: int = 3) -> None:
        if mode not in {"mosse", "kcf"} or max_missing < 0 or max_flow_only < 1:
            raise ValueError("Invalid legacy tracker configuration")
        self.mode = mode
        self.max_missing = max_missing
        self.max_flow_only = max_flow_only
        self._observation: _Observation | None = None
        self._flow_only: dict[int, int] = {}
        factory_name = f"Tracker{mode.upper()}_create"
        self._factory = getattr(cv2, factory_name, None)
        if self._factory is None and hasattr(cv2, "legacy"):
            self._factory = getattr(cv2.legacy, factory_name, None)
        if self._factory is None:
            raise RuntimeError(f"OpenCV contrib is required for {factory_name}")

    def _eligible(self, track: TrackSnapshot) -> bool:
        return (
            track.status in {"confirmed", "occluded"}
            and track.hits >= 3
            and track.misses <= self.max_missing
            and self._flow_only.get(track.track_id, 0) < self.max_flow_only
        )

    def predict(self, frame: NDArray[np.uint8]) -> tuple[Detection, ...]:
        if self._observation is None:
            return ()
        predictions = []
        frame_height, frame_width = frame.shape[:2]
        for track_id, tracker in self._observation.trackers.items():
            track = self._observation.tracks[track_id]
            if not self._eligible(track):
                continue
            ok, values = tracker.update(frame)
            if not ok or len(values) != 4:
                continue
            x, y, width, height = (float(value) for value in values)
            if not np.isfinite([x, y, width, height]).all() or width < 8 or height < 8:
                continue
            old_x, old_y, old_width, old_height = track.bbox
            old_center = np.asarray([old_x + old_width / 2, old_y + old_height / 2])
            new_center = np.asarray([x + width / 2, y + height / 2])
            displacement = new_center - old_center
            if float(np.linalg.norm(displacement)) > max(100.0, np.hypot(old_width, old_height) * 1.5):
                continue
            if x + width < 0 or y + height < 0 or x >= frame_width or y >= frame_height:
                continue
            bbox = (round(x), round(y), round(width), round(height))
            predictions.append(
                Detection(
                    bbox=bbox,
                    foot_point=(
                        track.foot_point[0] + float(displacement[0]),
                        track.foot_point[1] + float(displacement[1]),
                    ),
                    enemy_id=track.enemy_id,
                    confidence=min(0.59, track.confidence * 0.8),
                    template_name=track.template_name,
                    possible_friendly=track.enemy_id == "friendly",
                    candidate_enemy_id=track.enemy_id if track.enemy_id != "unknown" else None,
                    faction_evidence="friendly" if track.enemy_id == "friendly" else "enemy" if track.enemy_id != "unknown" else None,
                )
            )
            self._flow_only[track_id] = self._flow_only.get(track_id, 0) + 1
        return tuple(predictions)

    def observe(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8] | None,
        tracks: Sequence[TrackSnapshot],
        visual_detections: Sequence[Detection],
    ) -> None:
        del mask  # MOSSE/KCF use the full bbox; the mask is only needed by LK.
        frame_height, frame_width = frame.shape[:2]
        current_tracks = {track.track_id: track for track in tracks}
        active_ids = set(current_tracks)
        self._flow_only = {track_id: count for track_id, count in self._flow_only.items() if track_id in active_ids}
        trackers = {}
        for track_id, track in current_tracks.items():
            x, y, width, height = track.bbox
            left, top = max(0, x), max(0, y)
            right, bottom = min(frame_width, x + width), min(frame_height, y + height)
            if right - left < 8 or bottom - top < 8:
                continue
            tracker = self._factory()
            tracker.init(frame, (left, top, right - left, bottom - top))
            trackers[track_id] = tracker
            for detection in visual_detections:
                distance = float(np.linalg.norm(np.asarray(track.foot_point) - np.asarray(detection.foot_point)))
                if distance <= 60.0:
                    self._flow_only[track_id] = 0
                    break
        self._observation = _Observation(trackers=trackers, tracks=current_tracks)
