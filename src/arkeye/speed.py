"""Battle speed-button recognition independent from OCR."""

from __future__ import annotations

import cv2
import numpy as np


def detect_battle_speed(frame: np.ndarray) -> int | None:
    """Return 1 or 2 from the top-right speed button, if it is visible."""

    height, width = frame.shape[:2]
    left, right = round(width * 1540 / 1920), round(width * 1720 / 1920)
    top, bottom = round(height * 15 / 1080), round(height * 115 / 1080)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray > 100).astype(np.uint8) * 255
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scale_x, scale_y = width / 1920, height / 1080
    glyphs = []
    for x, y, glyph_width, glyph_height, area in stats[1:]:
        if (
            5 * scale_x <= glyph_width <= 50 * scale_x
            and 20 * scale_y <= glyph_height <= 60 * scale_y
            and area >= 40 * scale_x * scale_y
            and y <= 70 * scale_y
        ):
            glyphs.append((x, y, glyph_width, glyph_height))
    if len(glyphs) < 2:
        return None
    glyphs.sort()
    _, _, digit_width, digit_height = glyphs[0]
    return 1 if digit_width / digit_height < 0.70 else 2


def detect_deployment_slowdown(frame: np.ndarray) -> bool:
    """Detect the large green deploy-range overlay that triggers bullet time."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 70, 80), (85, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return bool(cv2.countNonZero(mask) > frame.shape[0] * frame.shape[1] * 0.02)
