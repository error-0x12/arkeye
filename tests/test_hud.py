from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from arkeye.hud import HUDRenderer, TrackJsonlWriter
from arkeye.tracking import TrackSnapshot


def snapshot(status: str = "confirmed") -> TrackSnapshot:
    return TrackSnapshot(
        track_id=7,
        enemy_id="enemy_a",
        confidence=0.9,
        status=status,
        bbox=(20, 30, 40, 50),
        foot_point=(40.0, 79.0),
        velocity=(2.0, 0.0),
        hits=3,
        misses=0,
        occlusion_group=None,
        history=((0.0, 30.0, 79.0), (0.1, 40.0, 79.0)),
    )


def test_hud_draws_without_mutating_source() -> None:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    renderer = HUDRenderer()

    output = renderer.render(frame, [snapshot(), snapshot("occluded")], metrics={"fps": 30})

    assert output.shape == frame.shape
    assert np.any(output != frame)
    assert not np.any(frame)


def test_lost_tracks_use_held_marker() -> None:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    ghosts = [snapshot("lost"), replace(snapshot("lost"), enemy_id="unknown")]

    output = HUDRenderer().render(frame, ghosts)

    assert np.any(np.all(output == (180, 120, 220), axis=-1))


def test_jsonl_writer_outputs_navigable_track_records(tmp_path) -> None:
    path = tmp_path / "tracks.jsonl"
    with TrackJsonlWriter(path) as writer:
        writer.write_frame(1.25, 12, [snapshot()], map_points=[[3.5, 4.0]])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["timestamp"] == 1.25
    assert record["frame_index"] == 12
    assert record["track_id"] == 7
    assert record["pixel"]["foot_point"] == [40.0, 79.0]
    assert record["map_point"] == [3.5, 4.0]
