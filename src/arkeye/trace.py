"""Per-frame, human-readable Chinese trace logging for offline source tracing.

Writes one JSON line per processed frame to a JSONL file. Each line records the
full decision chain of a frame: raw detector candidates, weighted-fusion class
scores, route/speed filter decisions (including every dropped candidate and its
Chinese reason), and the final track snapshots.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self, TextIO

import numpy as np

from arkeye.detector import Detection
from arkeye.tracking import TrackSnapshot

ReasonHook = Callable[[Mapping[str, Any]], None]


def _jsonable(value: Any) -> Any:
    """把 numpy 标量/数组转成原生类型。

    trace 里混进 np.bool_ / np.float32 会让 json.dumps 直接抛 TypeError，
    整段渲染会在写日志时崩掉；这里统一兜住。
    """

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def summarize_detection(detection: Detection) -> dict[str, Any]:
    x, y, width, height = detection.bbox
    return {
        "bbox": [round(x, 1), round(y, 1), round(width, 1), round(height, 1)],
        "foot": [round(detection.foot_point[0], 1), round(detection.foot_point[1], 1)],
        "class": detection.enemy_id,
        "confidence": round(detection.confidence, 3),
        "health_bar": detection.health_bar_bbox is not None,
        "faction": detection.faction_evidence,
        "possible_friendly": detection.possible_friendly,
        "candidate": detection.candidate_enemy_id,
        "template": detection.template_name,
    }


class TraceLogger:
    """Collect per-frame pipeline decisions and append them as Chinese JSONL."""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float = 30.0,
        stage_name: str | None = None,
        enemy_names: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.stage_name = stage_name
        self.enemy_names = dict(enemy_names or {})
        self._fh: TextIO = self.path.open("a", encoding="utf-8")
        self._frame: dict[str, Any] | None = None
        self.frames_written = 0

    def display_name(self, enemy_id: str | None) -> str:
        """Best-effort Chinese display name; falls back to the raw id."""

        if enemy_id is None:
            return "未知"
        if enemy_id == "unknown":
            return "未识别"
        if enemy_id == "friendly":
            return "我方干员"
        name = self.enemy_names.get(enemy_id)
        return f"{name}（{enemy_id}）" if name else enemy_id

    def attach(self, pipeline: Any) -> None:
        """Wire hooks into a BattlefieldPipeline and its tracker/fusion."""

        pipeline.trace_logger = self
        tracker = pipeline.tracker
        visual = getattr(tracker, "visual_tracker", None)
        for candidate in (tracker, visual):
            if hasattr(candidate, "trace_hook"):
                candidate.trace_hook = self.log_filter_decisions
        fusion = pipeline.weighted_fusion
        if fusion is not None:
            fusion.trace_hook = self.log_fusion

    # ---- collection API used by BattlefieldPipeline ---------------------

    def begin_frame(
        self,
        frame_index: int,
        timestamp: float,
        *,
        stage_time: float,
        game_speed: int,
        time_scale: float,
        detect: bool,
    ) -> None:
        self._frame = {
            "frame_index": int(frame_index),
            "timestamp": round(float(timestamp), 3),
            "stage_time": round(float(stage_time), 3),
            "game_speed": int(game_speed),
            "time_scale": round(float(time_scale), 3),
            "detect": bool(detect),
            "stage": self.stage_name,
        }

    def log_detections(self, detections: Sequence[Detection]) -> None:
        if self._frame is None:
            return
        self._frame["raw_detections"] = [summarize_detection(d) for d in detections]

    def log_fusion(self, details: Mapping[str, Any]) -> None:
        if self._frame is None:
            return
        self._frame.setdefault("fusion", {})[str(details["kind"])] = details["items"]

    def log_filter_decisions(self, decisions: Mapping[str, Any]) -> None:
        if self._frame is None:
            return
        self._frame["filter"] = dict(decisions)

    def log_tracks(self, tracks: Sequence[TrackSnapshot], map_points: Sequence[Any] | None) -> None:
        if self._frame is None:
            return
        records = []
        for index, track in enumerate(tracks):
            x, y, width, height = track.bbox
            speed = float((track.velocity[0] ** 2 + track.velocity[1] ** 2) ** 0.5)
            record: dict[str, Any] = {
                "track_id": track.track_id,
                "class": track.enemy_id,
                "class_display": self.display_name(track.enemy_id),
                "status": str(track.status),
                "confidence": round(track.confidence, 3),
                "bbox": [round(x, 1), round(y, 1), round(width, 1), round(height, 1)],
                "foot": [round(track.foot_point[0], 1), round(track.foot_point[1], 1)],
                "speed_px_s": round(speed, 1),
                "hits": track.hits,
                "misses": track.misses,
            }
            if map_points is not None and index < len(map_points):
                point = map_points[index]
                record["map_point"] = [round(float(point[0]), 2), round(float(point[1]), 2)]
            records.append(record)
        self._frame["tracks"] = records

    def end_frame(self) -> None:
        if self._frame is None:
            return
        frame = self._frame
        self._frame = None
        raw = len(frame.get("raw_detections") or [])
        fused = frame.get("fusion", {}).get("fused")
        fused_count = len(fused) if fused is not None else None
        tracks = frame.get("tracks") or []
        filter_info = frame.get("filter") or {}
        dropped = len(filter_info.get("off_path_dropped") or []) + len(filter_info.get("gate_rejected") or [])
        parts = [f"第 {frame['frame_index']} 帧"]
        if frame["detect"]:
            parts.append(f"原始候选 {raw} 个")
            if fused_count is not None:
                parts.append(f"融合后 {fused_count} 个")
            if dropped:
                parts.append(f"过滤器丢弃 {dropped} 个")
        else:
            parts.append("本帧未做检测，仅跟踪预测")
        parts.append(f"最终目标 {len(tracks)} 个")
        frame["summary"] = "，".join(parts) + "。"
        self._fh.write(json.dumps(_jsonable(frame), ensure_ascii=False) + "\n")
        self.frames_written += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_enemy_names(database_path: str | Path) -> dict[str, str]:
    """Extract enemy_id -> Chinese name from the Arknights enemy database."""

    data = json.loads(Path(database_path).read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    for entry in data.get("enemies", []):
        key = entry.get("Key")
        values = entry.get("Value") or []
        for value in values:
            enemy_data = value.get("enemyData") or {}
            name_field = enemy_data.get("name") or {}
            name = name_field.get("m_value")
            if key and name:
                names[key] = str(name)
                break
    return names
