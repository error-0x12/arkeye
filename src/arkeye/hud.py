"""HUD rendering and machine-readable track output."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Self, TextIO

import cv2
import numpy as np
from numpy.typing import NDArray

from arkeye.tracking import TrackSnapshot


class HUDRenderer:
    def __init__(self, *, history_length: int = 30, operator_names: dict[str, dict[str, str]] | None = None) -> None:
        self.history_length = history_length
        self.operator_names = operator_names or {}

    @staticmethod
    def _color(track_id: int) -> tuple[int, int, int]:
        return 80 + (track_id * 67) % 176, 80 + (track_id * 113) % 176, 80 + (track_id * 173) % 176

    @staticmethod
    def _label(image: NDArray[np.uint8], text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
        x = max(0, min(x, image.shape[1] - width - 4))
        y = max(height + 4, min(y, image.shape[0] - baseline - 2))
        cv2.rectangle(image, (x, y - height - 4), (x + width + 4, y + baseline + 2), (20, 20, 20), -1)
        cv2.putText(image, text, (x + 2, y - 1), font, scale, color, 1, cv2.LINE_AA)

    def render(
        self,
        frame: NDArray[np.uint8],
        tracks: Sequence[TrackSnapshot],
        *,
        stage_name: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> NDArray[np.uint8]:
        output = frame.copy()
        for track in tracks:
            color = self._color(track.track_id)
            history = track.history[-self.history_length :]
            if len(history) > 1:
                points = np.asarray([(round(x), round(y)) for _, x, y in history], dtype=np.int32)
                cv2.polylines(output, [points], False, color, 1, cv2.LINE_AA)

            x, y, width, height = track.bbox
            foot = tuple(round(value) for value in track.foot_point)
            display_id = track.enemy_id
            operator = self.operator_names.get(track.template_name or "")
            if track.enemy_id == "friendly" and operator and track.confidence >= 0.65:
                display_id = operator.get("display", operator["name"])
            label = f"#{track.track_id} {display_id} {track.confidence:.2f} {track.status}"
            if track.status == "predicted":
                color = (220, 190, 60)
                cv2.rectangle(output, (x, y), (x + width, y + height), color, 1)
                cv2.circle(output, foot, 3, color, -1)
                self._label(output, label.replace(track.status, "TIMELINE"), (x, y - 4), color)
                continue
            if track.status == "occluded":
                radius = min(80, max(8, round(max(width, height) * 0.6)))
                cv2.circle(output, foot, radius, (0, 210, 255), 2, cv2.LINE_AA)
                cv2.circle(output, foot, 3, (0, 210, 255), -1)
                self._label(output, label, (foot[0] + radius, foot[1]), (0, 210, 255))
                continue
            if track.status == "lost":
                radius = min(40, max(8, round(max(width, height) * 0.35)))
                points = np.asarray(
                    [(foot[0], foot[1] - radius), (foot[0] + radius, foot[1]), (foot[0], foot[1] + radius), (foot[0] - radius, foot[1])],
                    dtype=np.int32,
                )
                cv2.polylines(output, [points], True, (180, 120, 220), 1, cv2.LINE_AA)
                cv2.circle(output, foot, 3, (180, 120, 220), -1)
                self._label(output, label.replace(track.status, "HELD"), (foot[0] + radius + 4, foot[1]), (180, 120, 220))
                continue

            box_color = (160, 160, 160) if track.status == "tentative" else color
            cv2.rectangle(output, (x, y), (x + width, y + height), box_color, 2)
            cv2.circle(output, foot, 3, box_color, -1)
            self._label(output, label, (x, y - 4), box_color)

        header = stage_name or "ArkEye"
        if metrics:
            header += " | " + " | ".join(f"{key}={value}" for key, value in metrics.items())
        self._label(output, header, (4, 20), (230, 230, 230))
        return output


class TrackJsonlWriter:
    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", encoding="utf-8", newline="\n")

    def write_frame(
        self,
        timestamp: float,
        frame_index: int,
        tracks: Sequence[TrackSnapshot],
        *,
        map_points: Sequence[Sequence[float] | None] | None = None,
    ) -> None:
        if map_points is not None and len(map_points) != len(tracks):
            raise ValueError("map_points must have one entry per track")
        for index, track in enumerate(tracks):
            record = {
                "timestamp": float(timestamp),
                "frame_index": int(frame_index),
                "track_id": track.track_id,
                "enemy_id": track.enemy_id,
                "status": track.status,
                "confidence": float(track.confidence),
                "pixel": {
                    "bbox": list(track.bbox),
                    "foot_point": list(track.foot_point),
                },
                "velocity": list(track.velocity),
                "map_point": list(map_points[index]) if map_points and map_points[index] is not None else None,
            }
            self._file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
