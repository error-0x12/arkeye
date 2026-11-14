"""Frame sources for repeatable offline processing and later live capture."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import Protocol, Self

import av
import numpy as np
from numpy.typing import NDArray


class CaptureError(RuntimeError):
    """Raised when a frame source cannot be opened or decoded."""


@dataclass(frozen=True, slots=True)
class Frame:
    data: NDArray[np.uint8]
    timestamp: float
    index: int


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...

    def close(self) -> None: ...


class VideoSource:
    """Decode a video file to BGR frames with source timestamps."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path)
        self._index = 0
        try:
            self._container = av.open(str(self.path), mode="r")
            self._stream = self._container.streams.video[0]
        except (av.FFmpegError, OSError, IndexError) as exc:
            raise CaptureError(f"Unable to open video: {self.path}") from exc
        self._stream.thread_type = "AUTO"
        self._frames = self._container.decode(self._stream)

    @property
    def frame_rate(self) -> Fraction | None:
        rate = self._stream.average_rate
        return rate if rate and rate.denominator else None

    def read(self) -> Frame | None:
        try:
            packet_frame = next(self._frames)
        except StopIteration:
            return None
        except av.FFmpegError as exc:
            raise CaptureError(f"Unable to decode video: {self.path}") from exc

        if packet_frame.pts is not None and packet_frame.time_base is not None:
            timestamp = float(packet_frame.pts * packet_frame.time_base)
        elif self.frame_rate:
            timestamp = self._index / float(self.frame_rate)
        else:
            timestamp = float(self._index)
        frame = Frame(
            data=packet_frame.to_ndarray(format="bgr24"),
            timestamp=timestamp,
            index=self._index,
        )
        self._index += 1
        return frame

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class LatestFrameSource:
    """Read a source on a background thread and retain only the newest frame."""

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self._condition = threading.Condition()
        self._latest: Frame | None = None
        self._finished = False
        self._closed = False
        self._error: CaptureError | None = None
        self._dropped_frames = 0
        self._thread = threading.Thread(target=self._run, name="arkeye-capture", daemon=True)
        self._thread.start()

    @property
    def frame_rate(self) -> Fraction | None:
        return getattr(self._source, "frame_rate", None)

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def started_at(self) -> float:
        return getattr(self._source, "started_at", time.monotonic())

    def _run(self) -> None:
        try:
            while True:
                frame = self._source.read()
                if frame is None:
                    break
                with self._condition:
                    if self._latest is not None:
                        self._dropped_frames += 1
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as exc:  # noqa: BLE001 - worker errors must reach the reader instead of dying silently
            self._error = exc if isinstance(exc, CaptureError) else CaptureError(f"Capture thread failed: {exc}")
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()

    def read(self) -> Frame | None:
        with self._condition:
            while self._latest is None and not self._finished:
                self._condition.wait()
            if self._latest is not None:
                frame = self._latest
                self._latest = None
                return frame
            if self._error is not None:
                raise self._error
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stop = getattr(self._source, "stop", None)
        if callable(stop):
            stop()
        else:
            self._source.close()
        with self._condition:
            self._finished = True
            self._condition.notify_all()
        self._thread.join(timeout=5)
        if callable(stop):
            self._source.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
