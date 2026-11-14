from __future__ import annotations

import cv2
import numpy as np
import pytest

from arkeye.legacy_tracking import LegacyTrackerAssistant
from arkeye.tracking import TrackSnapshot


def available(mode: str) -> bool:
    name = f"Tracker{mode.upper()}_create"
    return hasattr(cv2, name) or (hasattr(cv2, "legacy") and hasattr(cv2.legacy, name))


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


def textured_frame(shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    texture = rng.integers(40, 240, (64, 64, 3), dtype=np.uint8)
    frame[20 + shift_y : 84 + shift_y, 20 + shift_x : 84 + shift_x] = texture
    return frame


@pytest.mark.parametrize("mode", ["mosse", "kcf"])
def test_legacy_tracker_predicts_translation(mode: str) -> None:
    if not available(mode):
        pytest.skip(f"OpenCV contrib {mode} is unavailable")
    assistant = LegacyTrackerAssistant(mode)
    assistant.observe(textured_frame(), None, [snapshot()], [])

    predictions = assistant.predict(textured_frame(4, 2))

    assert len(predictions) == 1
    assert predictions[0].enemy_id == "enemy_test"
    assert predictions[0].foot_point == pytest.approx((56.0, 85.0), abs=2.0)


@pytest.mark.parametrize("mode", ["mosse", "kcf"])
def test_legacy_tracker_does_not_resurrect_long_lost(mode: str) -> None:
    if not available(mode):
        pytest.skip(f"OpenCV contrib {mode} is unavailable")
    assistant = LegacyTrackerAssistant(mode)
    assistant.observe(textured_frame(), None, [snapshot(status="lost", misses=6)], [])
    assert assistant.predict(textured_frame(4, 2)) == ()


def test_legacy_flow_only_assistance_is_bounded() -> None:
    if not available("mosse"):
        pytest.skip("OpenCV contrib MOSSE is unavailable")
    assistant = LegacyTrackerAssistant("mosse", max_flow_only=2)
    assistant.observe(textured_frame(), None, [snapshot()], [])

    assert len(assistant.predict(textured_frame(4, 2))) == 1
    assert len(assistant.predict(textured_frame(4, 2))) == 1
    assert assistant.predict(textured_frame(4, 2)) == ()
