from __future__ import annotations

import json

import cv2
import numpy as np

from arkeye.assets import BattleTemplate, TemplateLibrary
from arkeye.capture import Frame
from arkeye.detector import BattlefieldDetector, TemplateClassifier
from arkeye.hud import TrackJsonlWriter
from arkeye.pipeline import BattlefieldPipeline
from arkeye.tracking import MultiObjectTracker


def test_offline_pipeline_connects_detection_tracking_hud_and_jsonl(tmp_path) -> None:
    sprite = np.zeros((40, 40, 4), dtype=np.uint8)
    cv2.circle(sprite, (20, 20), 14, (20, 20, 220, 255), -1)
    template = BattleTemplate.from_bgra("enemy_a", sprite, source="game_sprite")
    background = np.full((100, 140, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(background, TemplateClassifier(TemplateLibrary([template])), min_area=20)
    output_path = tmp_path / "tracks.jsonl"

    with TrackJsonlWriter(output_path) as writer:
        pipeline = BattlefieldPipeline(detector, MultiObjectTracker(min_hits=1), writer=writer)
        empty_result = pipeline.process(Frame(background.copy(), 0.0, 0))
        frame = background.copy()
        height, width = template.alpha.shape
        frame[30 : 30 + height, 50 : 50 + width] = template.image
        result = pipeline.process(Frame(frame, 1 / 30, 1))

    assert empty_result.tracks == ()
    assert len(result.detections) == 1
    assert len(result.tracks) == 1
    assert result.tracks[0].enemy_id == "enemy_a"
    assert result.annotated.shape == frame.shape
    assert np.any(result.annotated != frame)
    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["track_id"] == result.tracks[0].track_id
    assert record["enemy_id"] == "enemy_a"


def test_analysis_skip_preserves_confirmed_status() -> None:
    sprite = np.zeros((40, 40, 4), dtype=np.uint8)
    cv2.circle(sprite, (20, 20), 14, (20, 20, 220, 255), -1)
    template = BattleTemplate.from_bgra("enemy_a", sprite, source="game_sprite")
    background = np.full((100, 140, 3), 100, dtype=np.uint8)
    detector = BattlefieldDetector(background, TemplateClassifier(TemplateLibrary([template])), min_area=20)
    pipeline = BattlefieldPipeline(detector, MultiObjectTracker(min_hits=1))
    frame = background.copy()
    height, width = template.alpha.shape
    frame[30 : 30 + height, 50 : 50 + width] = template.image
    confirmed = pipeline.process(Frame(frame, 0.0, 0)).tracks[0]

    skipped = pipeline.process(Frame(background.copy(), 1 / 30, 1), detect=False).tracks[0]
    skipped_again = pipeline.process(Frame(background.copy(), 2 / 30, 2), detect=False).tracks[0]

    assert confirmed.status == "confirmed"
    assert skipped.status == "confirmed"
    assert skipped_again.status == "confirmed"
    assert skipped_again.misses == confirmed.misses
