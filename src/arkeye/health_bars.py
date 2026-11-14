"""Faction evidence from horizontal health and skill-energy bars."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

BoundingBox = tuple[int, int, int, int]


BarKind = Literal["enemy", "friendly", "device"]


@dataclass(frozen=True, slots=True)
class HealthBarEvidence:
    kind: BarKind
    bbox: BoundingBox
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return x + width / 2.0, y + height / 2.0


@dataclass(frozen=True, slots=True)
class _Bar:
    color: str
    bbox: BoundingBox

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bbox
        return x + width / 2.0, y + height / 2.0


class HealthBarDetector:
    """Detect red enemy bars and paired friendly health/energy bars."""

    def __init__(self, *, max_bar_width: int = 180) -> None:
        self.max_bar_width = max_bar_width

    @staticmethod
    def _color_masks(frame: NDArray[np.uint8]) -> dict[str, NDArray[np.uint8]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 120, 110), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 110), (179, 255, 255))
        return {
            "red": red,
            "blue": cv2.inRange(hsv, (85, 55, 110), (105, 255, 255)),
            "green": cv2.inRange(hsv, (35, 55, 90), (84, 255, 255)),
            "yellow": cv2.inRange(hsv, (11, 80, 110), (34, 255, 255)),
        }

    def _bars(self, frame: NDArray[np.uint8]) -> list[_Bar]:
        bars = []
        for color, mask in self._color_masks(frame).items():
            horizontal_open = 9 if color == "red" else 3
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_open, 2)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, count):
                x, y, width, height, area = (int(value) for value in stats[label])
                minimum_width = 40 if color == "red" else 12
                if not minimum_width <= width <= self.max_bar_width or not 2 <= height <= 14:
                    continue
                if width / height < 4.0 or area / (width * height) < 0.40:
                    continue
                bars.append(_Bar(color=color, bbox=(x, y, width, height)))
        return bars

    @staticmethod
    def _paired(health: _Bar, energy: _Bar) -> bool:
        hx, hy, hw, hh = health.bbox
        ex, ey, ew, eh = energy.bbox
        overlap = max(0, min(hx + hw, ex + ew) - max(hx, ex))
        if overlap / min(hw, ew) < 0.55 or min(hw, ew) / max(hw, ew) < 0.20:
            return False
        return abs((hy + hh / 2) - (ey + eh / 2)) <= 14.0

    @staticmethod
    def _union(first: BoundingBox, second: BoundingBox) -> BoundingBox:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left, top = min(ax, bx), min(ay, by)
        right, bottom = max(ax + aw, bx + bw), max(ay + ah, by + bh)
        return left, top, right - left, bottom - top

    def detect(self, frame: NDArray[np.uint8]) -> tuple[HealthBarEvidence, ...]:
        bars = self._bars(frame)
        evidence = [HealthBarEvidence("enemy", bar.bbox, 0.80) for bar in bars if bar.color == "red"]
        green_bars = [bar for bar in bars if bar.color == "green"]
        paired_green: set[int] = set()
        health_bars = [bar for bar in bars if bar.color in {"blue", "yellow"}]
        for health in health_bars:
            candidates = [
                (index, energy)
                for index, energy in enumerate(green_bars)
                if index not in paired_green and self._paired(health, energy)
            ]
            if not candidates:
                continue
            index, energy = min(
                candidates,
                key=lambda item: abs(health.center[1] - item[1].center[1]) + abs(health.center[0] - item[1].center[0]) * 0.1,
            )
            paired_green.add(index)
            evidence.append(HealthBarEvidence("friendly", self._union(health.bbox, energy.bbox), 0.95))
        for index, energy in enumerate(green_bars):
            if index not in paired_green:
                evidence.append(HealthBarEvidence("device", energy.bbox, 0.75))
        return tuple(evidence)

    @staticmethod
    def match(
        foot_point: tuple[float, float],
        bbox: BoundingBox,
        evidence: Sequence[HealthBarEvidence],
    ) -> HealthBarEvidence | None:
        """Associate a unit with a nearby bar below its feet, never across the board."""

        foot_x, foot_y = foot_point
        _, _, width, _ = bbox
        horizontal_limit = max(70.0, width * 0.8)
        candidates = []
        for item in evidence:
            bar_x, bar_y = item.center
            horizontal = abs(bar_x - foot_x)
            vertical = bar_y - foot_y
            if horizontal <= horizontal_limit and -45.0 <= vertical <= 65.0:
                distance = horizontal + abs(vertical) * 0.5 - item.confidence * 30.0
                candidates.append((distance, item))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None
