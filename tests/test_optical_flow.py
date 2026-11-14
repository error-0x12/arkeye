from __future__ import annotations

import numpy as np
import pytest

from arkeye.detector import Detection
from arkeye.optical_flow import LKFlowAssistant
from arkeye.pipeline import merge_visual_and_flow
from arkeye.tracking import TrackSnapshot


def snapshot(*, status: str = "confirmed", misses: int = 0) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=7,
        enemy_id="enemy_test",
        confidence=0.9,
        status=status,
        bbox=(20, 20, 64, 64),
        foot_point=(52.0, 83.0),
        velocity=(0.0, 0.0),
        hits=5,
        misses=misses,
        occlusion_group=None,
    )


def textured_frame(shift_x: int = 0, shift_y: int = 0) -> tuple[np.ndarray, np.ndarray]:
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    texture = rng.integers(40, 240, (64, 64, 3), dtype=np.uint8)
    frame[20 + shift_y : 84 + shift_y, 20 + shift_x : 84 + shift_x] = texture
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[20 + shift_y : 84 + shift_y, 20 + shift_x : 84 + shift_x] = 255
    return frame, mask


def test_lk_predicts_existing_track_translation() -> None:
    first, mask = textured_frame()
    second, _ = textured_frame(4, 2)
    assistant = LKFlowAssistant()
    assistant.observe(first, mask, [snapshot()], [])

    predictions = assistant.predict(second)

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.enemy_id == "enemy_test"
    assert prediction.foot_point == pytest.approx((56.0, 85.0), abs=1.0)
    assert prediction.bbox[:2] == pytest.approx((24, 22), abs=1.0)


def test_lk_never_predicts_without_a_previous_observation() -> None:
    frame, _ = textured_frame()
    assert LKFlowAssistant().predict(frame) == ()


def test_lk_does_not_resurrect_long_lost_track() -> None:
    first, mask = textured_frame()
    second, _ = textured_frame(4, 2)
    assistant = LKFlowAssistant()
    assistant.observe(first, mask, [snapshot(status="lost", misses=6)], [])
    assert assistant.predict(second) == ()


def test_lk_rejects_low_texture_region() -> None:
    first = np.zeros((120, 120, 3), dtype=np.uint8)
    second = first.copy()
    mask = np.full(first.shape[:2], 255, dtype=np.uint8)
    assistant = LKFlowAssistant()
    assistant.observe(first, mask, [snapshot()], [])
    assert assistant.predict(second) == ()


def test_lk_flow_only_assistance_is_bounded() -> None:
    first, mask = textured_frame()
    second, _ = textured_frame(4, 2)
    assistant = LKFlowAssistant(max_flow_only=2)
    assistant.observe(first, mask, [snapshot()], [])

    assert len(assistant.predict(second)) == 1
    assert len(assistant.predict(second)) == 1
    assert assistant.predict(second) == ()


def test_visual_detection_has_priority_over_flow() -> None:
    visual = Detection((20, 20, 64, 64), (52.0, 83.0), "enemy_test", 0.9, "visual")
    covered_flow = Detection((24, 22, 64, 64), (56.0, 85.0), "enemy_test", 0.59, "flow")
    separate_flow = Detection((100, 20, 64, 64), (132.0, 83.0), "enemy_test", 0.59, "flow")

    merged = merge_visual_and_flow([visual], [covered_flow, separate_flow])

    assert len(merged) == 2
    assert merged[0] is visual
    assert merged[1] is separate_flow
