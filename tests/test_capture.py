from __future__ import annotations

import av
import numpy as np
import pytest

from arkeye.capture import CaptureError, VideoSource


def create_test_video(path, frame_count: int = 3) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            pixels = np.full((48, 64, 3), (20 + index * 10, 80, 160), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_source_decodes_bgr_frames_and_timestamps(tmp_path) -> None:
    path = tmp_path / "sample.mp4"
    create_test_video(path)

    with VideoSource(path) as source:
        frames = list(source)

    assert len(frames) == 3
    assert [frame.index for frame in frames] == [0, 1, 2]
    assert frames[0].data.shape == (48, 64, 3)
    assert frames[0].data.dtype == np.uint8
    assert frames[0].timestamp == pytest.approx(0.0)
    assert frames[1].timestamp == pytest.approx(1 / 30)
    assert np.all(frames[0].data[:, :, 2] > frames[0].data[:, :, 0])


def test_video_source_reports_missing_file(tmp_path) -> None:
    with pytest.raises(CaptureError, match="Unable to open video"):
        VideoSource(tmp_path / "missing.mp4")
