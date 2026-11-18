from __future__ import annotations

import numpy as np
import pytest

from arkeye.stage_ocr import OCRToken
from arkeye.ui_panels import UIPanelDetector


def _card(width: int = 600, height: int = 180) -> np.ndarray:
    """深色卡片 + 浅色文字块，模拟游戏内图鉴面板。"""

    crop = np.full((height, width, 3), 24, dtype=np.uint8)
    for row in range(0, height, 18):
        crop[row + 4 : row + 12, 40 : width - 40] = 235
    return crop


def _reader_ok(_image: np.ndarray) -> list[OCRToken]:
    return [OCRToken(text="弑君者整合运动干部", confidence=0.9, box=None)]


def _reader_empty(_image: np.ndarray) -> list[OCRToken]:
    return []


def test_wide_card_with_text_is_a_panel() -> None:
    detector = UIPanelDetector(reader=_reader_ok)

    verdict = detector.classify(_card(), (1250, 160, 600, 180), 1080, frame_index=1)

    assert verdict.is_panel
    assert "OCR" in verdict.reason
    assert detector.ocr_calls == 1


def test_wide_card_without_text_is_not_a_panel() -> None:
    """地图上也有很宽很亮的东西，OCR 读不到字就不能当面板处理。"""

    detector = UIPanelDetector(reader=_reader_empty)

    verdict = detector.classify(_card(), (200, 300, 600, 180), 1080, frame_index=1)

    assert not verdict.is_panel
    assert detector.ocr_calls == 1


def test_narrow_candidate_never_reaches_ocr() -> None:
    detector = UIPanelDetector(reader=_reader_ok)

    verdict = detector.classify(_card(width=200, height=120), (300, 300, 200, 120), 1080, frame_index=1)

    assert not verdict.is_panel
    assert detector.ocr_calls == 0, "普通尺寸候选不该触发 OCR"


def test_tall_candidate_is_rejected_by_aspect() -> None:
    detector = UIPanelDetector(reader=_reader_ok)

    verdict = detector.classify(_card(width=420, height=500), (300, 100, 420, 500), 1080, frame_index=1)

    assert not verdict.is_panel
    assert "宽高比" in verdict.reason
    assert detector.ocr_calls == 0


def test_panel_below_the_top_band_is_rejected() -> None:
    detector = UIPanelDetector(reader=_reader_ok)

    verdict = detector.classify(_card(), (600, 700, 600, 180), 1080, frame_index=1)

    assert not verdict.is_panel
    assert "偏下" in verdict.reason


def test_same_region_is_only_ocrd_once() -> None:
    """OCR 一次 0.6 秒，同一块区域反复出现时不能反复跑。"""

    calls = {"n": 0}

    def counting(_image: np.ndarray) -> list[OCRToken]:
        calls["n"] += 1
        return [OCRToken(text="弑君者整合运动干部", confidence=0.9, box=None)]

    detector = UIPanelDetector(reader=counting)
    bbox = (1250, 160, 600, 180)

    for frame in range(1, 40, 3):
        verdict = detector.classify(_card(), bbox, 1080, frame_index=frame)
        assert verdict.is_panel

    assert calls["n"] == 1, f"同一区域被 OCR 了 {calls['n']} 次"


def test_ocr_failure_is_not_fatal() -> None:
    def boom(_image: np.ndarray) -> list[OCRToken]:
        raise RuntimeError("ocr exploded")

    detector = UIPanelDetector(reader=boom)

    verdict = detector.classify(_card(), (1250, 160, 600, 180), 1080, frame_index=1)

    assert not verdict.is_panel
    assert "OCR" in verdict.reason


@pytest.mark.parametrize("bbox", [(960, 64, 384, 128), (1024, 128, 640, 192)])
def test_quantised_cache_key_is_stable(bbox: tuple[int, int, int, int]) -> None:
    key = UIPanelDetector._cache_key(bbox)
    # 面板会轻微漂移；量化到 64 px 后小幅移动应当命中同一块缓存
    shifted = (bbox[0] + 8, bbox[1] + 8, bbox[2], bbox[3])

    assert key == UIPanelDetector._cache_key(shifted)
