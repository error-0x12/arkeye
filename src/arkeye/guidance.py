"""Stage-guided spawn and despawn constraints built on visual tracking."""

from __future__ import annotations

import numpy as np

from arkeye.detector import Detection
from arkeye.tracking import MultiObjectTracker, TrackSnapshot, _Track


class SpawnBudgetTracker(MultiObjectTracker):
    """Reject mid-map spawns and enforce per-stage spawn budgets.

    Spawn constraints only reject impossible new tracks. They never relabel an
    existing visual detection or manufacture an expected enemy.
    """

    def __init__(
        self,
        *,
        expected_counts: dict[str, int],
        spawn_bbox: tuple[int, int, int, int],
        despawn_x: float,
        despawn_misses: int = 45,
        friendly_limit: int = 1,
        spawn_schedule: list[tuple[float, int]] | None = None,
        spawn_class_windows: list[tuple[float, float, str]] | None = None,
        **tracker_options,
    ) -> None:
        tracker_options.setdefault("max_missed", 10_000)
        super().__init__(**tracker_options)
        self.expected_counts = expected_counts
        self.spawn_bbox = spawn_bbox
        self.despawn_x = despawn_x
        self.despawn_misses = despawn_misses
        self.friendly_limit = friendly_limit
        self.spawn_schedule = sorted(spawn_schedule or [])
        self.spawn_class_windows = spawn_class_windows or []
        self._created_at: dict[int, float] = {}
        self._created_enemies = 0
        self._created_by_class: dict[str, int] = {}
        self._created_friendly = 0
        self._enemy_track_ids: set[int] = set()
        self._friendly_track_ids: set[int] = set()
        self._motion_validated_ids: set[int] = set()

    def _new_track(self, detection: Detection, timestamp: float) -> _Track | None:
        foot_x, foot_y = detection.foot_point
        is_left_candidate = foot_x <= self.despawn_x
        if is_left_candidate:
            if detection.enemy_id not in {"friendly", "unknown"}:
                return None
            if any(self._position(track)[0] <= self.despawn_x for track in self._tracks.values()):
                return None
            if detection.enemy_id == "friendly" and self._created_friendly >= self.friendly_limit:
                return None
        else:
            x, y, width, height = self.spawn_bbox
            left, top, right, bottom = x, y, x + width, y + height
            if not (left <= foot_x <= right and top <= foot_y <= bottom):
                return None
            if self.spawn_schedule:
                allowed = max((count for time, count in self.spawn_schedule if time <= timestamp), default=0)
                active = sum(track.enemy_id != "friendly" and track.misses < 30 for track in self._tracks.values())
                if active >= allowed:
                    return None
            if detection.enemy_id in self.expected_counts:
                if self._created_by_class.get(detection.enemy_id, 0) >= self.expected_counts[detection.enemy_id]:
                    return None
            elif self._created_enemies >= sum(self.expected_counts.values()) + 2:
                return None

        track = super()._new_track(detection, timestamp)
        self._created_at[track.track_id] = timestamp
        if is_left_candidate or detection.enemy_id == "friendly":
            self._created_friendly += int(detection.enemy_id == "friendly")
            self._friendly_track_ids.add(track.track_id)
        else:
            self._created_enemies += 1
            self._created_by_class[detection.enemy_id] = self._created_by_class.get(detection.enemy_id, 0) + 1
            self._enemy_track_ids.add(track.track_id)
        return track

    def _predict(self, track: _Track, delta_time: float):
        # Keep the last measurement while preserving Kalman correct() semantics.
        track.kalman.statePre = track.kalman.statePost.copy()
        return self._position(track)

    def _reacquire_unmatched(self, detections: tuple[Detection, ...], timestamp: float) -> None:
        """Attach uncovered visual blobs to distinct lost tracks, never new mid-map IDs."""

        for detection in detections:
            if detection.enemy_id == "unknown":
                continue
            detection_position = np.asarray(detection.foot_point)

            def distance(track: _Track, position=detection_position) -> float:
                return float(np.linalg.norm(self._position(track) - position))

            if any(track.misses == 0 and distance(track) < 100 for track in self._tracks.values()):
                continue
            candidates = [
                track
                for track in self._tracks.values()
                if track.misses > 0
                and distance(track) <= 800
                and (
                    track.enemy_id == detection.enemy_id
                    or (track.enemy_id == "unknown" and detection.enemy_id != "friendly")
                )
            ]
            if not candidates:
                continue
            candidates.sort(key=distance)
            typical_area = float(np.median([track.bbox[2] * track.bbox[3] for track in candidates]))
            ratio = detection.bbox[2] * detection.bbox[3] / typical_area
            group_size = 1 if ratio < 1.2 else 2 if ratio < 2.5 else 3
            for track in candidates[:group_size]:
                track.kalman.correct(np.asarray(detection.foot_point, dtype=np.float32).reshape(2, 1))
                track.bbox = detection.bbox
                track.enemy_id = detection.enemy_id
                track.confidence = detection.confidence
                if detection.appearance is not None:
                    track.appearance = self._copy_appearance(detection.appearance)
                track.hits += 1
                track.misses = 0
                track.status = "confirmed"
                track.occlusion_group = None
                x, y = self._position(track)
                track.history.append((timestamp, float(x), float(y)))

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[TrackSnapshot, ...]:
        previous_classes = {track_id: track.enemy_id for track_id, track in self._tracks.items()}
        previous_positions = {track_id: self._position(track).copy() for track_id, track in self._tracks.items()}
        super().update(detections, timestamp)
        self._reacquire_unmatched(detections, timestamp)
        for track_id, track in self._tracks.items():
            if track_id in self._enemy_track_ids and track.enemy_id == "friendly":
                track.enemy_id = previous_classes.get(track_id, "unknown")
            if track_id in self._friendly_track_ids and track.enemy_id != "friendly":
                track.enemy_id = previous_classes.get(track_id, "friendly")
            if track_id in self._enemy_track_ids and track.enemy_id == "unknown":
                created_at = self._created_at[track_id]
                for start, end, enemy_id in self.spawn_class_windows:
                    if start <= created_at < end:
                        track.enemy_id = enemy_id
                        break
            if track_id in previous_positions and track.misses > 0:
                x, y = previous_positions[track_id]
                track.kalman.statePost[:2, 0] = [x, y]
                track.history[-1] = (timestamp, float(x), float(y))
            if track.enemy_id == "friendly":
                track.kalman.statePost[2:4, 0] = 0.0
            else:
                track.kalman.statePost[2, 0] = max(-350.0, min(50.0, float(track.kalman.statePost[2, 0])))
                track.kalman.statePost[3, 0] = max(-200.0, min(200.0, float(track.kalman.statePost[3, 0])))
            track.kalman.statePost[0, 0] = max(0.0, min(1920.0, float(track.kalman.statePost[0, 0])))
            track.kalman.statePost[1, 0] = max(0.0, min(1080.0, float(track.kalman.statePost[1, 0])))
        return self.snapshots()
