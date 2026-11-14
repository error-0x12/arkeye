from __future__ import annotations

import json
import urllib.error

import pytest

from arkeye.stage_data import StageDataError, StageRepository


def test_level_download_falls_back_to_cached_json(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "levels" / "obt" / "main" / "level_main_16-01.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"options": {"maxLifePoint": 3}}), encoding="utf-8")

    def fail_download(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_download)
    repository = StageRepository(tmp_path)

    level = repository.load_level("OBT/MAIN/LEVEL_MAIN_16-01")

    assert level["options"]["maxLifePoint"] == 3


def test_missing_cache_reports_download_failure(tmp_path, monkeypatch) -> None:
    def fail_download(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_download)
    repository = StageRepository(tmp_path)

    with pytest.raises(StageDataError, match="no cached copy"):
        repository.load_stage_table()


@pytest.mark.parametrize("level_id", ["../secret", "/absolute/path", "obt/main/level.json"])
def test_level_id_cannot_escape_or_change_cache_layout(tmp_path, level_id: str) -> None:
    repository = StageRepository(tmp_path)

    with pytest.raises(StageDataError, match="Invalid level ID"):
        repository.load_level(level_id)


def test_map_preview_rejects_unsafe_stage_id(tmp_path) -> None:
    repository = StageRepository(tmp_path)

    with pytest.raises(StageDataError, match="Invalid stage ID"):
        repository.download_map_preview("../main_16-01")
