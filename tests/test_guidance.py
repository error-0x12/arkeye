from __future__ import annotations

from arkeye.detector import Detection
from arkeye.guidance import SpawnBudgetTracker


def detection(x: float, y: float = 500, enemy_id: str = "enemy_1007_slime") -> Detection:
    return Detection(
        bbox=(round(x - 30), round(y - 40), 60, 80),
        foot_point=(x, y),
        enemy_id=enemy_id,
        confidence=0.9 if enemy_id != "unknown" else 0.0,
        template_name=None,
    )


def tracker(**overrides) -> SpawnBudgetTracker:
    options = {
        "expected_counts": {"enemy_1007_slime": 1},
        "spawn_bbox": (1200, 350, 550, 400),
        "despawn_x": 700,
        "min_hits": 1,
    }
    options.update(overrides)
    return SpawnBudgetTracker(**options)


def test_spawn_schedule_rejects_extra_concurrent_enemies() -> None:
    tracker = SpawnBudgetTracker(
        expected_counts={"enemy_1007_slime": 2},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        min_hits=1,
        spawn_schedule=[(8.0, 1)],
    )

    first = tracker.update((detection(1500),), 8.0)
    second = tracker.update((detection(1400),), 8.2)

    assert len(first) == 1
    assert len(second) == 1


def test_unknown_enemy_uses_unambiguous_creation_window() -> None:
    tracker = SpawnBudgetTracker(
        expected_counts={"enemy_1007_slime": 1},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        min_hits=1,
        spawn_class_windows=[(0.0, 10.0, "enemy_1007_slime")],
    )

    snapshots = tracker.update((detection(1500, enemy_id="unknown"),), 1.0)

    assert snapshots[0].enemy_id == "enemy_1007_slime"


def test_left_unknown_cannot_create_duplicate_friendly_tracks() -> None:
    tracker = SpawnBudgetTracker(
        expected_counts={},
        spawn_bbox=(1200, 350, 550, 400),
        despawn_x=700,
        min_hits=1,
    )
    tracker.update((detection(400, enemy_id="friendly"),), 0.0)

    snapshots = tracker.update((detection(420, enemy_id="unknown"),), 0.1)

    assert len(snapshots) == 1
    assert snapshots[0].enemy_id == "friendly"
