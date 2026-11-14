from __future__ import annotations

import json
from pathlib import Path

import pytest

from arkeye.timeline import build_stage_timeline


def test_1_8_timeline_matches_public_spawn_schedule() -> None:
    level = json.loads(Path("tests/fixtures/level_main_01-08.json").read_text(encoding="utf-8"))

    timeline = build_stage_timeline(level)

    assert timeline.total_count == 46
    first = timeline.events[0]
    assert first.timestamp == pytest.approx(3.0)
    assert first.enemy_key == "enemy_1007_slime"
    assert first.count == 1
    assert first.route_index == 1
    earliest = timeline.earliest_spawn_times
    assert earliest["enemy_1000_gopro_2"] == pytest.approx(34.1, abs=0.01)
    assert earliest["enemy_1002_nsabr"] == pytest.approx(46.1, abs=0.01)
    assert earliest["enemy_1502_crowns"] == pytest.approx(63.1, abs=0.01)
    assert earliest["enemy_1005_yokai"] == pytest.approx(69.1, abs=0.01)
    assert earliest["enemy_1011_wizard"] == pytest.approx(99.1, abs=0.01)
    assert earliest["enemy_1029_shdsbr"] == pytest.approx(124.1, abs=0.01)


def test_wave_offset_uses_longest_fragment_action() -> None:
    level = {
        "waves": [
            {
                "preDelay": 2,
                "postDelay": 1,
                "fragments": [
                    {
                        "preDelay": 3,
                        "actions": [
                            {"actionType": "SPAWN", "key": "enemy_a", "count": 3, "preDelay": 4, "interval": 2, "routeIndex": 1},
                            {"actionType": "DISPLAY_ENEMY_INFO", "key": "enemy_b", "count": 1, "preDelay": 20, "interval": 1},
                        ],
                    }
                ],
            },
            {
                "fragments": [
                    {
                        "actions": [
                            {"actionType": "SPAWN", "key": "enemy_b", "count": 2, "preDelay": 1, "interval": 3, "routeIndex": 2}
                        ]
                    }
                ]
            },
        ]
    }

    events = build_stage_timeline(level).events

    assert [event.enemy_key for event in events] == ["enemy_a", "enemy_b"]
    assert events[0].timestamp == pytest.approx(9.0)
    assert events[0].count == 3
    assert events[1].timestamp == pytest.approx(32.0)
    assert events[1].wave_timestamp == pytest.approx(1.0)
