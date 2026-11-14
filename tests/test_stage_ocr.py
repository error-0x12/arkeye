from __future__ import annotations

import pytest

from arkeye.stage_data import StageDataError
from arkeye.stage_ocr import OCRToken, extract_stage_codes, resolve_stage_from_tokens


@pytest.fixture
def stage_table() -> dict:
    return {
        "stages": {
            "main_16-01": {
                "stageId": "main_16-01",
                "levelId": "Obt/Main/level_main_16-01",
                "zoneId": "act3mainss_zone1",
                "code": "16-2",
                "name": "晨曦遥远",
                "difficulty": "NORMAL",
            },
            "other_01": {
                "stageId": "other_01",
                "levelId": "Obt/Other/level_other_01",
                "zoneId": "other_zone",
                "code": "XX-1",
                "name": "另一关",
                "difficulty": "NORMAL",
            },
        }
    }


def test_extract_stage_codes_normalizes_dashes_and_ignores_plain_numbers() -> None:
    assert extract_stage_codes("关卡 16—02，消耗 21") == ("16-02",)
    assert extract_stage_codes("TR-1 与 PR-A-1") == ("TR-1", "PR-A-1")
    assert extract_stage_codes("理智 21") == ()


def test_ocr_resolves_16_2_and_name(stage_table: dict) -> None:
    match = resolve_stage_from_tokens(
        stage_table,
        [OCRToken("16—02", 0.92), OCRToken("晨曦遥远", 0.88)],
    )

    assert match.info.stage_id == "main_16-01"
    assert match.name_confirmed
    assert match.confidence == 0.92


def test_ocr_uses_name_to_disambiguate_multiple_valid_codes(stage_table: dict) -> None:
    tokens = [OCRToken("16-2", 0.9), OCRToken("XX-1", 0.8)]
    with pytest.raises(StageDataError, match="ambiguous"):
        resolve_stage_from_tokens(stage_table, tokens)

    match = resolve_stage_from_tokens(stage_table, [*tokens, OCRToken("晨曦遥远", 0.7)])
    assert match.info.stage_id == "main_16-01"


def test_ocr_rejects_unknown_or_low_confidence_codes(stage_table: dict) -> None:
    with pytest.raises(StageDataError, match="did not produce"):
        resolve_stage_from_tokens(stage_table, [OCRToken("99-99", 0.99)])
    with pytest.raises(StageDataError, match="did not produce"):
        resolve_stage_from_tokens(stage_table, [OCRToken("16-2", 0.2)])
