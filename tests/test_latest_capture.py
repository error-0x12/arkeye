from __future__ import annotations

import time
from fractions import Fraction

import numpy as np

from arkeye.capture import Frame, LatestFrameSource


class FakeSource:
    frame_rate = Fraction(30, 1)

    def __init__(self) -> None:
        self._index = 0
        self.closed = False

    def read(self) -> Frame | None:
        if self._index >= 3 or self.closed:
            return None
        time.sleep(0.001)
        frame = Frame(np.zeros((2, 2, 3), dtype=np.uint8), self._index / 30, self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        self.closed = True


def test_latest_frame_source_discards_backlog_and_preserves_last_frame() -> None:
    fake = FakeSource()
    with LatestFrameSource(fake) as source:
        time.sleep(0.05)
        frame = source.read()
        assert frame is not None
        assert frame.index == 2
        assert source.dropped_frames >= 1
        assert source.read() is None
        assert source.frame_rate == Fraction(30, 1)

    assert fake.closed


class StoppableSource:
    def __init__(self) -> None:
        self.stopped = False
        self.events: list[str] = []

    def read(self) -> Frame | None:
        while not self.stopped:
            time.sleep(0.001)
        return None

    def stop(self) -> None:
        self.events.append("stop")
        self.stopped = True

    def close(self) -> None:
        self.events.append("close")


def test_two_phase_close_stops_worker_before_releasing_decoder() -> None:
    fake = StoppableSource()
    source = LatestFrameSource(fake)
    time.sleep(0.01)

    source.close()

    assert fake.events == ["stop", "close"]
