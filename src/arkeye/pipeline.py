"""Synchronous offline/live-independent battlefield processing pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from arkeye.board import BoardCalibration
from arkeye.capture import Frame
from arkeye.detector import BattlefieldDetector, Detection
from arkeye.hud import HUDRenderer, TrackJsonlWriter
from arkeye.optical_flow import LKFlowAssistant
from arkeye.speed import detect_battle_speed, detect_deployment_slowdown
from arkeye.tracking import MultiObjectTracker, TrackSnapshot


@dataclass(frozen=True, slots=True)
class PipelineResult:
    frame: Frame
    detections: tuple[Detection, ...]
    tracks: tuple[TrackSnapshot, ...]
    annotated: NDArray[np.uint8]
    debug_images: tuple[tuple[str, NDArray[np.uint8]], ...] = ()


def merge_visual_and_flow(
    visual: Sequence[Detection],
    flow: Sequence[Detection],
) -> tuple[Detection, ...]:
    """Keep visual evidence first and add only uncovered flow predictions."""

    merged = list(visual)
    for prediction in flow:
        covered = False
        for detection in visual:
            if (
                prediction.enemy_id != "unknown"
                and detection.enemy_id != "unknown"
                and prediction.enemy_id != detection.enemy_id
            ):
                continue
            distance = float(np.linalg.norm(np.asarray(prediction.foot_point) - np.asarray(detection.foot_point)))
            x, y, width, height = detection.bbox
            foot_x, foot_y = prediction.foot_point
            inside = x - 10 <= foot_x <= x + width + 10 and y - 10 <= foot_y <= y + height + 10
            if distance <= 50.0 or inside:
                covered = True
                break
        if not covered:
            merged.append(prediction)
    return tuple(merged)


class BattlefieldPipeline:
    def __init__(
        self,
        detector: BattlefieldDetector,
        tracker: MultiObjectTracker,
        *,
        hud: HUDRenderer | None = None,
        writer: TrackJsonlWriter | None = None,
        calibration: BoardCalibration | None = None,
        stage_name: str | None = None,
        flow_assistant: LKFlowAssistant | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.hud = hud or HUDRenderer()
        self.flow_assistant = flow_assistant
        self.writer = writer
        self.calibration = calibration
        self.stage_name = stage_name
        self.game_speed = 1
        self.time_scale = 1.0

    def process(self, frame: Frame, *, detect: bool = True) -> PipelineResult:
        self.game_speed = detect_battle_speed(frame.data) or self.game_speed
        self.time_scale = 0.2 if detect_deployment_slowdown(frame.data) else float(self.game_speed)
        set_game_speed = getattr(self.tracker, "set_game_speed", None)
        if callable(set_game_speed):
            set_game_speed(self.game_speed)
        set_time_scale = getattr(self.tracker, "set_time_scale", None)
        if callable(set_time_scale):
            set_time_scale(self.time_scale)
        visual_detections = self.detector.detect(frame.data, overlapping_tracks=self.tracker.snapshots()) if detect else ()
        flow_detections = self.flow_assistant.predict(frame.data) if self.flow_assistant is not None else ()
        detections = merge_visual_and_flow(visual_detections, flow_detections)
        tracks = self.tracker.update(detections, frame.timestamp) if detect or detections else self.tracker.coast(frame.timestamp)
        if self.flow_assistant is not None:
            mask = self.detector.foreground.last_mask if detect else None
            self.flow_assistant.observe(frame.data, mask, tracks, visual_detections)
        map_points = None
        if self.calibration is not None and tracks:
            map_points = list(self.calibration.frame_points_to_map([track.foot_point for track in tracks]))
        if self.writer is not None and (detect or detections):
            self.writer.write_frame(frame.timestamp, frame.index, tracks, map_points=map_points)
        metrics: dict[str, Any] = {
            "detections": len(visual_detections) if detect else "skip",
            "flow": len(flow_detections),
            "tracks": len(tracks),
            "frame": frame.index,
            "speed": f"{self.game_speed}X",
        }
        annotated = self.hud.render(frame.data, tracks, stage_name=self.stage_name, metrics=metrics)
        return PipelineResult(
            frame=frame,
            detections=detections,
            tracks=tracks,
            annotated=annotated,
            debug_images=tuple(self.detector.debug_images),
        )
