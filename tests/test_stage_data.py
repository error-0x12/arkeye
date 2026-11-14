from __future__ import annotations

import pytest

from arkeye.stage_data import (
    StageDataError,
    expand_spawn_events,
    level_enemy_ids,
    resolve_stage,
)


@pytest.fixture
def stage_table() -> dict:
    return {
        "stages": {
            "main_16-01": {
                "stageId": "main_16-01",
                "levelId": "obt/main/level_main_16-01",
                "zoneId": "act3mainss_zone1",
                "code": "16-2",
                "name": "晨曦遥远",
                "difficulty": "NORMAL",
            },
            "other_01": {
                "stageId": "other_01",
                "levelId": "obt/other/level_other_01",
                "zoneId": "other_zone",
                "code": "XX-1",
                "name": "普通",
                "difficulty": "NORMAL",
            },
            "other_01#f#": {
                "stageId": "other_01#f#",
                "levelId": "obt/other/level_other_01",
                "zoneId": "other_zone",
                "code": "XX-1",
                "name": "突袭",
                "difficulty": "FOUR_STAR",
            },
        }
    }


def test_resolve_16_2_from_table_instead_of_numeric_encoding(stage_table: dict) -> None:
    stage = resolve_stage(stage_table, "16-2", name="晨曦遥远", difficulty="普通")

    assert stage.stage_id == "main_16-01"
    assert stage.level_id == "obt/main/level_main_16-01"
    assert stage.zone_id == "act3mainss_zone1"


def test_resolve_normalizes_ocr_dashes_spaces_and_leading_zeroes(stage_table: dict) -> None:
    assert resolve_stage(stage_table, " 16—02 ").stage_id == "main_16-01"


def test_resolve_uses_difficulty_to_disambiguate(stage_table: dict) -> None:
    assert resolve_stage(stage_table, "XX-1", difficulty="NORMAL").stage_id == "other_01"
    assert resolve_stage(stage_table, "XX-1", difficulty="突袭").stage_id == "other_01#f#"


def test_resolve_rejects_ambiguous_or_mismatched_results(stage_table: dict) -> None:
    with pytest.raises(StageDataError, match="ambiguous"):
        resolve_stage(stage_table, "XX-1")
    with pytest.raises(StageDataError, match="does not match name"):
        resolve_stage(stage_table, "16-2", name="错误名称")
    with pytest.raises(StageDataError, match="No stage matches"):
        resolve_stage(stage_table, "16-3")


def test_expand_spawn_events_preserves_relative_timing_and_order() -> None:
    level = {
        "waves": [
            {
                "preDelay": 2,
                "fragments": [
                    {
                        "preDelay": 3,
                        "actions": [
                            {
                                "actionType": "SPAWN",
                                "key": "enemy_a",
                                "count": 2,
                                "interval": 8,
                                "preDelay": 5,
                                "routeIndex": 4,
                            },
                            {"actionType": "ACTIVATE", "key": "ignored"},
                        ],
                    }
                ],
            }
        ]
    }

    events = expand_spawn_events(level)

    assert len(events) == 1
    event = events[0]
    assert event.enemy_key == "enemy_a"
    assert event.count == 2
    assert event.interval == 8
    assert event.route_index == 4
    assert event.wave_pre_delay == 2
    assert event.fragment_pre_delay == 3
    assert event.action_pre_delay == 5


def test_level_enemy_ids_combines_references_and_spawn_actions() -> None:
    level = {
        "enemyDbRefs": [{"id": "enemy_b", "useDb": True}],
        "waves": [
            {
                "fragments": [
                    {"actions": [{"actionType": "SPAWN", "key": "enemy_a", "count": 1}]},
                ]
            }
        ],
    }

    assert level_enemy_ids(level) == ("enemy_a", "enemy_b")
