"""Detect in-game UI panels (the enemy codex popup and friends) before they become units.

Tapping an enemy in Arknights opens a codex panel: a wide, bordered, mostly flat
card that appears over the battlefield. Background subtraction sees it as a large
foreground blob, and if it is handed to the classifier it can be tracked as a
unit for as long as it stays open.

The panel is recognised in two stages:

1. **Shape** — a candidate that is much wider than it is tall, sits above the
   cards area, and is mostly a flat light-coloured field with straight edges.
   This is cheap and runs on every frame.
2. **OCR** — the region is only accepted as a panel when OCR actually finds text
   in it. Map scenery never contains text, so this rules out wide light-coloured
   terrain without paying the OCR cost on ordinary frames.

OCR only runs on candidates that survive the shape gate, and the verdict is
cached for a few frames so an open panel costs one OCR call instead of one per
frame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from arkeye.stage_ocr import OCRToken

Reader = Callable[[np.ndarray], Sequence[OCRToken]]


@dataclass(frozen=True)
class PanelVerdict:
    """Outcome for one candidate region."""

    is_panel: bool
    reason: str
    text: str = ""


@dataclass(frozen=True)
class PanelShape:
    """Cheap geometry/colour statistics for a candidate."""

    width: int
    height: int
    aspect: float
    flat_ratio: float
    light_ratio: float


class UIPanelDetector:
    """Recognise wide UI panels and decide whether they must be ignored."""

    def __init__(
        self,
        *,
        reader: Reader | None = None,
        min_width: int = 380,
        max_aspect: float = 6.5,
        min_aspect: float = 1.6,
        min_flat_ratio: float = 0.0,
        min_light_ratio: float = 0.0,
        min_letters: int = 6,
        cache_frames: int = 12,
        top_limit: float = 0.45,
    ) -> None:
        self.reader = reader
        self.min_width = min_width
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_flat_ratio = min_flat_ratio
        self.min_light_ratio = min_light_ratio
        self.min_letters = min_letters
        self.cache_frames = cache_frames
        self.top_limit = top_limit
        self._cache: dict[tuple[int, int, int, int], PanelVerdict] = {}
        self._frame_index = 0
        self.ocr_calls = 0
        self.last_reason = ""

    # -- geometry -------------------------------------------------------

    @staticmethod
    def describe(image: np.ndarray) -> PanelShape:
        """Measure flatness and brightness of a candidate crop."""

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 平坦度：先轻度模糊再算梯度，压缩贴图噪点，保留真正的直边/文字
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        flat_ratio = 1.0 - min(1.0, edge_ratio * 6.0)
        light_ratio = float(np.count_nonzero(gray >= 90)) / float(gray.size)
        return PanelShape(
            width=width,
            height=height,
            aspect=width / max(1, height),
            flat_ratio=flat_ratio,
            light_ratio=light_ratio,
        )

    def shape_only(self, crop: np.ndarray, bbox: tuple[int, int, int, int], frame_height: int) -> PanelVerdict:
        """Cheap gate: would this region be worth running OCR on?"""

        _, top, _, _ = bbox
        shape = self.describe(crop)
        if shape.width < self.min_width:
            return PanelVerdict(False, f"宽度 {shape.width} px 不足 {self.min_width}")
        if not self.min_aspect <= shape.aspect <= self.max_aspect:
            return PanelVerdict(False, f"宽高比 {shape.aspect:.1f} 不在 {self.min_aspect}–{self.max_aspect}")
        if top > frame_height * self.top_limit:
            return PanelVerdict(False, "位置偏下，不像面板")
        if shape.flat_ratio < self.min_flat_ratio:
            return PanelVerdict(False, f"内部不够平坦（{shape.flat_ratio:.2f}）")
        if shape.light_ratio < self.min_light_ratio:
            return PanelVerdict(False, f"亮度不足（{shape.light_ratio:.2f}）")
        return PanelVerdict(True, "形态符合面板")

    # -- OCR ------------------------------------------------------------

    def classify(
        self,
        crop: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_height: int,
        *,
        frame_index: int | None = None,
    ) -> PanelVerdict:
        """Shape gate, then OCR confirmation. Each quantised region is judged once."""

        if frame_index is not None:
            self._frame_index = frame_index
        key = self._cache_key(bbox)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        verdict = self.shape_only(crop, bbox, frame_height)
        if not verdict.is_panel:
            self._cache[key] = verdict
            return verdict

        text = self._read_text(crop) if self.reader is not None else ""
        self.ocr_calls += 1
        is_panel = len(text) >= self.min_letters
        result = PanelVerdict(
            is_panel,
            f"OCR 读到 {len(text)} 个字（{text[:14]}）" if is_panel else f"OCR 没读到文字（{len(text)} 字）",
            text,
        )
        self._cache[key] = result
        return result

    @staticmethod
    def _cache_key(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Quantise to 64 px: a panel drifts a little, one verdict per area is enough."""

        x, y, width, height = bbox
        return (x // 64, y // 64, width // 64, height // 64)

    def _read_text(self, crop: np.ndarray) -> str:
        if self.reader is None:
            return ""
        # OCR 对太小/太大的输入都不稳，缩放到高约 160 px 再读
        height, width = crop.shape[:2]
        scale = 160.0 / max(1, height)
        if 0.4 < scale < 2.5:
            crop = cv2.resize(crop, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
        try:
            tokens = self.reader(crop)
        except Exception:  # noqa: BLE001 - OCR 失败不应该让整帧崩掉
            return ""
        return "".join(token.text for token in tokens if token.text)


def confirmed_reason(is_panel: bool, text: str) -> str:
    if is_panel:
        return f"面板已确认（缓存，OCR 读到 {len(text)} 个字）"
    return f"面板未确认（缓存，OCR {len(text)} 个字）"
