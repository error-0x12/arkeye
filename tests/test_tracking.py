from __future__ import annotations

import numpy as np

from arkeye.detector import Detection
from arkeye.tracking import MultiObjectTracker


def detection(
    x: float,
    y: float,
    *,
    enemy_id: str = "enemy_a",
    appearance: list[float] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> Detection:
    if bbox is None:
        bbox = (round(x - 10), round(y - 19), 20, 20)
    histogram = np.asarray(appearance, dtype=np.float32).reshape(-1, 1) if appearance is not None else None
    return Detection(
        bbox=bbox,
        foot_point=(x, y),
        enemy_id=enemy_id,
        confidence=0.9 if enemy_id != "unknown" else 0.0,
        template_name="test" if enemy_id != "unknown" else None,
        appearance=histogram,
    )


def test_new_track_requires_consecutive_detections() -> None:
    tracker = MultiObjectTracker(min_hits=2)

    first = tracker.update((detection(50, 50),), 0.0)
    assert len(first) == 1
    assert first[0].status == "tentative"

    assert tracker.update((), 1 / 30)[0].status == "lost"
    restarted = tracker.update((detection(50, 50),), 2 / 30)
    assert restarted[0].track_id == first[0].track_id
    assert restarted[0].status == "confirmed"


def test_lost_track_reacquires_same_id() -> None:
    tracker = MultiObjectTracker(min_hits=1, max_missed=3)
    tracker.update((detection(50, 50),), 0.0)
    confirmed = tracker.update((detection(55, 50),), 1 / 30)
    track_id = confirmed[0].track_id

    lost = tracker.update((), 2 / 30)
    assert lost[0].status == "lost"
    assert lost[0].misses == 1

    reacquired = tracker.update((detection(62, 50),), 3 / 30)
    assert reacquired[0].track_id == track_id
    assert reacquired[0].status == "confirmed"
    assert reacquired[0].misses == 0


def test_merged_blob_preserves_ids_and_split_reassociates_tracks() -> None:
    tracker = MultiObjectTracker(min_hits=1, max_missed=5, distance_gate=60)
    tracker.update(
        (detection(40, 50, appearance=[1, 0]), detection(80, 50, appearance=[0, 1])),
        0.0,
    )
    initial = tracker.update(
        (detection(45, 50, appearance=[1, 0]), detection(75, 50, appearance=[0, 1])),
        1 / 30,
    )
    left_id = min(initial, key=lambda track: track.foot_point[0]).track_id
    right_id = max(initial, key=lambda track: track.foot_point[0]).track_id

    merged = detection(
        65,
        55,
        enemy_id="unknown",
        bbox=(42, 38, 48, 24),
    )
    during_occlusion = tracker.update((merged,), 2 / 30)

    assert len(during_occlusion) == 2
    assert {track.status for track in during_occlusion} == {"occluded"}
    assert {track.track_id for track in during_occlusion} == {left_id, right_id}
    assert all(track.enemy_id == "enemy_a" for track in during_occlusion)

    separated = tracker.update(
        (
            detection(50, 50, appearance=[1, 0]),
            detection(85, 50, appearance=[0, 1]),
        ),
        3 / 30,
    )

    assert len(separated) == 2
    assert {track.status for track in separated} == {"confirmed"}
    left = next(track for track in separated if track.track_id == left_id)
    right = next(track for track in separated if track.track_id == right_id)
    assert left.foot_point[0] < right.foot_point[0]
    assert left.foot_point[0] < 60
    assert right.foot_point[0] > 70


def test_known_different_classes_do_not_share_a_track() -> None:
    tracker = MultiObjectTracker(min_hits=1)
    tracker.update((detection(50, 50, enemy_id="enemy_a"),), 0.0)

    snapshots = tracker.update((detection(52, 50, enemy_id="enemy_b"),), 1 / 30)

    assert len(snapshots) == 2
    assert {track.enemy_id for track in snapshots} == {"enemy_a", "enemy_b"}


def test_history_contains_measured_and_predicted_positions() -> None:
    tracker = MultiObjectTracker(min_hits=1)
    tracker.update((detection(50, 50),), 0.0)
    tracker.update((detection(55, 50),), 1 / 30)

    snapshot = tracker.update((), 2 / 30)[0]

    assert len(snapshot.history) == 3
    assert snapshot.history[-1][1] > snapshot.history[-2][1]
