"""Gate recognition on a resolved loading stage and the battle pause button."""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from arkeye.stage_data import StageInfo
from arkeye.stage_ocr import RapidOCREngine, resolve_stage_from_tokens


class GateState(StrEnum):
    WAITING_STAGE = "waiting_stage"
    WAITING_BATTLE = "waiting_battle"
    RUNNING = "running"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class GateUpdate:
    state: GateState
    stage: StageInfo | None
    ocr_confidence: float | None
    pause_detected: bool
    error: str | None = None


def detect_pause_button(frame: np.ndarray) -> bool:
    height, width = frame.shape[:2]
    left, right = round(width * 1740 / 1920), width
    top, bottom = round(height * 15 / 1080), round(height * 115 / 1080)
    crop = frame[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray > 100).astype(np.uint8) * 255
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scale_x, scale_y = width / 1920, height / 1080
    boxes = []
    for x, y, box_width, box_height, area in stats[1:]:
        if (
            5 * scale_x <= box_width <= 35 * scale_x
            and 20 * scale_y <= box_height <= 70 * scale_y
            and area >= 50 * scale_x * scale_y
        ):
            boxes.append((x, y, box_width, box_height))
    boxes.sort()
    for first, second in itertools.pairwise(boxes):
        x1, y1, w1, h1 = first
        x2, y2, _, h2 = second
        center_distance = (x2 + second[2] / 2) - (x1 + w1 / 2)
        if abs(y1 - y2) <= 10 * scale_y and 0.6 <= h1 / h2 <= 1.4 and 15 * scale_x <= center_distance <= 60 * scale_x:
            return True
    return False


class BattleGate:
    def __init__(
        self,
        stage_table,
        *,
        ocr_engine: RapidOCREngine | None = None,
        supported_stage_ids: frozenset[str] = frozenset({"main_00-01", "main_01-08"}),
        ocr_interval: float = 0.75,
        asynchronous: bool = False,
    ) -> None:
        self.stage_table = stage_table
        self.ocr_engine = ocr_engine or RapidOCREngine()
        self.supported_stage_ids = supported_stage_ids
        self.ocr_interval = ocr_interval
        self.asynchronous = asynchronous
        self._executor = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self._ocr_future = None
        self.state = GateState.WAITING_STAGE
        self.stage: StageInfo | None = None
        self.ocr_confidence: float | None = None
        self.pause_detected = False
        self.error: str | None = None
        self._last_ocr = float("-inf")
        self._pause_streak = 0

    def _read_match(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        crop = frame[round(height * 0.25) : round(height * 0.75), round(width * 0.25) : round(width * 0.75)]
        crop = cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)
        try:
            return resolve_stage_from_tokens(self.stage_table, self.ocr_engine.read(crop))
        except Exception:  # noqa: BLE001 - an empty OCR result is normal while loading
            return None

    def _accept_match(self, match) -> None:
        if match is None or not match.name_confirmed:
            return
        if match.info.stage_id not in self.supported_stage_ids:
            self.state = GateState.UNSUPPORTED
            self.error = f"Unsupported stage: {match.info.stage_id} ({match.info.code} {match.info.name})"
        else:
            self.stage = match.info
            self.ocr_confidence = match.confidence
            self.state = GateState.WAITING_BATTLE

    def update(self, frame: np.ndarray) -> GateUpdate:
        if self.state in {GateState.RUNNING, GateState.UNSUPPORTED}:
            return self._snapshot()
        now = time.monotonic()
        if self.state == GateState.WAITING_STAGE and now - self._last_ocr >= self.ocr_interval:
            self._last_ocr = now
            if self.asynchronous:
                if self._ocr_future is None:
                    self._ocr_future = self._executor.submit(self._read_match, frame.copy())
            else:
                self._accept_match(self._read_match(frame))
        if self.asynchronous and self._ocr_future is not None and self._ocr_future.done():
            try:
                self._accept_match(self._ocr_future.result())
            except Exception as exc:  # noqa: BLE001 - keep video running and expose OCR failures
                self.error = f"OCR failed: {exc}"
            self._ocr_future = None
        if self.state == GateState.WAITING_BATTLE:
            detected = detect_pause_button(frame)
            self._pause_streak = self._pause_streak + 1 if detected else 0
            self.pause_detected = detected
            if self._pause_streak >= 2:
                self.state = GateState.RUNNING
        return self._snapshot()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _snapshot(self) -> GateUpdate:
        return GateUpdate(
            state=self.state,
            stage=self.stage,
            ocr_confidence=self.ocr_confidence,
            pause_detected=self.pause_detected,
            error=self.error,
        )
