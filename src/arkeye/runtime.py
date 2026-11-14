"""Shared construction of the validated 0-1 recognition pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from arkeye.assets import TemplateLibrary
from arkeye.board import BoardCalibration
from arkeye.capture import Frame, VideoSource
from arkeye.detector import BattlefieldDetector, TemplateClassifier
from arkeye.gate import detect_pause_button
from arkeye.health_bars import HealthBarDetector
from arkeye.hud import HUDRenderer, TrackJsonlWriter
from arkeye.motion import RouteSpeedFilter
from arkeye.operators import load_operator_metadata
from arkeye.optical_flow import LKFlowAssistant
from arkeye.pipeline import BattlefieldPipeline
from arkeye.routes import build_deployable_mask, build_route_mask, build_ui_mask, route_frame_points
from arkeye.slot_tracking import StageSlotTracker
from arkeye.timeline import build_stage_timeline
from arkeye.timeline_tracking import RouteTimeline, TimelinePrimaryTracker, load_enemy_move_speeds


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    pipeline: BattlefieldPipeline
    level: Mapping[str, Any]
    route_lines: tuple[np.ndarray, ...]
    calibration: BoardCalibration


def read_video_background(path: str, timestamp: float = 8.0) -> np.ndarray:
    with VideoSource(path) as source:
        for frame in source:
            if frame.timestamp >= timestamp:
                return frame.data.copy()
    raise ValueError(f"Video ends before the empty-board timestamp {timestamp}")


def read_stable_video_background_frame(
    path: str,
    start_timestamp: float = 8.0,
    end_timestamp: float = 20.0,
    sample_interval: float = 0.5,
) -> Frame:
    """Choose a low-motion battle frame after the intro instead of assuming a fixed empty-board time."""

    if start_timestamp < 0 or end_timestamp <= start_timestamp or sample_interval <= 0:
        raise ValueError("Invalid stable-background time range")
    samples: list[tuple[Frame, np.ndarray, bool]] = []
    next_sample = start_timestamp
    with VideoSource(path) as source:
        for frame in source:
            if frame.timestamp < start_timestamp:
                continue
            if frame.timestamp > end_timestamp:
                break
            if frame.timestamp + 1e-6 < next_sample:
                continue
            gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
            thumbnail = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
            samples.append((frame, thumbnail, detect_pause_button(frame.data)))
            next_sample = start_timestamp + len(samples) * sample_interval
    if not samples:
        raise ValueError(f"Video has no background candidates after {start_timestamp}")

    scored = []
    for index, (frame, gray, pause_visible) in enumerate(samples):
        if not pause_visible:
            continue
        differences = []
        if index > 0:
            differences.append(float(np.mean(cv2.absdiff(gray, samples[index - 1][1]))))
        if index + 1 < len(samples):
            differences.append(float(np.mean(cv2.absdiff(gray, samples[index + 1][1]))))
        if differences:
            scored.append((float(np.mean(differences)), frame.timestamp, index))
    selected = min(scored)[2] if scored else 0
    return samples[selected][0]


def read_stable_video_background(path: str, start_timestamp: float = 8.0) -> np.ndarray:
    return read_stable_video_background_frame(path, start_timestamp).data.copy()


def build_0_1_pipeline(
    background: np.ndarray,
    *,
    writer: TrackJsonlWriter | None = None,
    analysis_scale: float = 1.0,
    battle_origin_timestamp: float = 0.0,
    use_optical_flow: bool = False,
    use_health_bars: bool = False,
    timeline_primary: bool = False,
) -> RuntimeComponents:
    if timeline_primary:
        raise ValueError("Timeline-primary mode is currently validated only for 1-8")
    level = json.loads(Path(".cache/levels/obt/main/level_main_00-01.json").read_text(encoding="utf-8"))
    calibration = BoardCalibration.load("configs/main_00-01.json")
    roi_mask = build_route_mask(level, calibration, background.shape, width_tiles=1.0)
    route_lines = route_frame_points(level, calibration)
    origin, x_unit, y_unit = calibration.map_points_to_frame([[0, 0], [1, 0], [0, 1]])
    pixels_per_tile = max(float(np.linalg.norm(x_unit - origin)), float(np.linalg.norm(y_unit - origin)))
    detector = BattlefieldDetector(
        background,
        TemplateClassifier(TemplateLibrary.load_manifest("assets/0_1/manifest.json")),
        roi_mask=roi_mask,
        allowed_enemy_ids=("enemy_1002_nsabr", "enemy_1007_slime", "friendly"),
        confidence_threshold=0.60,
        difference_threshold=35,
        min_area=500,
        blur_size=5,
        analysis_scale=analysis_scale,
        health_bars=HealthBarDetector() if use_health_bars else None,
    )
    slots = StageSlotTracker(
        expected_counts={"enemy_1007_slime": 10, "enemy_1002_nsabr": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        despawn_misses=240,
        spawn_class_windows=[(0.0, 39.5, "enemy_1007_slime"), (39.5, 60.0, "enemy_1002_nsabr")],
        typical_bbox_areas={"enemy_1007_slime": 10_000, "enemy_1002_nsabr": 15_000, "friendly": 20_000},
    )
    tracker = RouteSpeedFilter(
        slots,
        route_lines,
        frame_diagonal=float(np.hypot(background.shape[1], background.shape[0])),
        pixels_per_tile=pixels_per_tile,
    )
    pipeline = BattlefieldPipeline(
        detector,
        tracker,
        writer=writer,
        calibration=calibration,
        stage_name="0-1",
        flow_assistant=LKFlowAssistant() if use_optical_flow else None,
    )
    return RuntimeComponents(
        pipeline=pipeline,
        level=level,
        route_lines=route_lines,
        calibration=calibration,
    )


def build_1_8_pipeline(
    background: np.ndarray,
    *,
    writer: TrackJsonlWriter | None = None,
    analysis_scale: float = 1.0,
    battle_origin_timestamp: float = 0.0,
    use_optical_flow: bool = False,
    use_health_bars: bool = False,
    timeline_primary: bool = False,
) -> RuntimeComponents:
    level = json.loads(Path(".cache/levels/obt/main/level_main_01-08.json").read_text(encoding="utf-8"))
    timeline = build_stage_timeline(level)
    calibration = BoardCalibration.load("configs/main_01-08.json")
    roi_mask = build_route_mask(level, calibration, background.shape, width_tiles=1.0)
    route_lines = route_frame_points(level, calibration)
    origin, x_unit, y_unit = calibration.map_points_to_frame([[0, 0], [1, 0], [0, 1]])
    pixels_per_tile = max(float(np.linalg.norm(x_unit - origin)), float(np.linalg.norm(y_unit - origin)))
    enemy_counts = timeline.expected_counts
    ui_config = json.loads(Path("configs/main_01-08_ui_mask.json").read_text(encoding="utf-8"))
    detector = BattlefieldDetector(
        background,
        TemplateClassifier(TemplateLibrary.load_manifest("assets/1_8/manifest.json")),
        roi_mask=roi_mask,
        deployable_mask=build_deployable_mask(level, calibration, background.shape),
        ui_mask=build_ui_mask(background.shape, ui_config),
        allowed_enemy_ids=(*enemy_counts, "friendly"),
        confidence_threshold=0.65,
        class_confidence_thresholds={"friendly": 0.82},
        difference_threshold=25,
        min_area=800,
        blur_size=5,
        analysis_scale=analysis_scale,
        health_bars=HealthBarDetector() if use_health_bars else None,
    )
    slots = StageSlotTracker(
        expected_counts=enemy_counts,
        spawn_bbox=(1400, 150, 450, 650),
        despawn_x=350,
        despawn_misses=45,
        typical_bbox_areas={
            "enemy_1007_slime": 10_000,
            "enemy_1007_slime_2": 12_000,
            "enemy_1000_gopro_2": 20_000,
            "enemy_1002_nsabr": 16_000,
            "enemy_1502_crowns": 18_000,
            "enemy_1005_yokai": 14_000,
            "enemy_1011_wizard": 18_000,
            "enemy_1029_shdsbr": 18_000,
            "friendly": 25_000,
        },
        friendly_limit=8,
        class_confirmation_hits=2,
        class_confirmation_by_id={"friendly": 5},
        earliest_spawn_times=timeline.earliest_spawn_times,
        battle_origin_timestamp=battle_origin_timestamp,
    )
    tracker = RouteSpeedFilter(
        slots,
        route_lines,
        frame_diagonal=float(np.hypot(background.shape[1], background.shape[0])),
        pixels_per_tile=pixels_per_tile,
        class_route_indices=timeline.class_route_indices,
        earliest_spawn_times=timeline.earliest_spawn_times,
        route_earliest_times=timeline.route_earliest_times,
        spawn_events=timeline.events,
        battle_origin_timestamp=battle_origin_timestamp,
    )
    if timeline_primary:
        database = json.loads(Path(".cache/levels/enemydata/enemy_database.json").read_text(encoding="utf-8"))
        projected_routes = tuple(
            RouteTimeline.from_mapping(route, calibration) if route.get("motionMode") in {"WALK", "FLY"} else None
            for route in level.get("routes") or []
        )
        tracker = TimelinePrimaryTracker(
            visual_tracker=tracker,
            routes=projected_routes,
            spawn_events=timeline.events,
            enemy_move_speeds=load_enemy_move_speeds(database),
            typical_bbox_areas=slots.typical_bbox_areas,
            pixels_per_tile=pixels_per_tile,
            battle_origin_timestamp=battle_origin_timestamp,
        )
    pipeline = BattlefieldPipeline(
        detector,
        tracker,
        hud=HUDRenderer(operator_names=load_operator_metadata("assets/1_8/operators.json")),
        writer=writer,
        calibration=calibration,
        stage_name="1-8",
        flow_assistant=LKFlowAssistant() if use_optical_flow else None,
    )
    return RuntimeComponents(
        pipeline=pipeline,
        level=level,
        route_lines=route_lines,
        calibration=calibration,
    )


def build_stage_pipeline(
    stage_id: str,
    background: np.ndarray,
    *,
    writer: TrackJsonlWriter | None = None,
    analysis_scale: float = 1.0,
    battle_origin_timestamp: float = 0.0,
    use_optical_flow: bool = False,
    use_health_bars: bool = False,
    timeline_primary: bool = False,
) -> RuntimeComponents:
    builders = {
        "main_00-01": build_0_1_pipeline,
        "main_01-08": build_1_8_pipeline,
    }
    try:
        builder = builders[stage_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported stage: {stage_id}") from exc
    return builder(
        background,
        writer=writer,
        analysis_scale=analysis_scale,
        battle_origin_timestamp=battle_origin_timestamp,
        use_optical_flow=use_optical_flow,
        use_health_bars=use_health_bars,
        timeline_primary=timeline_primary,
    )
