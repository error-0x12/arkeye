"""Resolve stages and load structured Arknights level data.

A stage's display code is not a reversible encoding of its stage ID. Always
resolve the code through stage_table.json instead of constructing IDs from it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

# jsDelivr serves the same files from Kengxxiao/ArknightsGameData and avoids
# raw.githubusercontent.com TLS revocation-check failures on some Windows hosts.
GAME_DATA_BASE = "https://cdn.jsdelivr.net/gh/Kengxxiao/ArknightsGameData@master/zh_CN/gamedata"
THERESA_STATIC_BASE = "https://static.theresa.wiki"


class StageDataError(RuntimeError):
    """Raised when stage data cannot be resolved or loaded."""


@dataclass(frozen=True, slots=True)
class StageInfo:
    stage_id: str
    level_id: str
    zone_id: str
    code: str
    name: str
    difficulty: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StageInfo:
        try:
            return cls(
                stage_id=str(value["stageId"]),
                level_id=str(value["levelId"]),
                zone_id=str(value["zoneId"]),
                code=str(value["code"]),
                name=str(value["name"]),
                difficulty=str(value.get("difficulty", "NONE")),
            )
        except KeyError as exc:
            raise StageDataError(f"Stage entry is missing required field: {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class SpawnEvent:
    """One spawn action, preserving the relative delays from the level JSON.

    Fragment scheduling has additional runtime semantics, so these fields are
    deliberately not collapsed into an absolute timestamp.
    """

    order: int
    wave_index: int
    fragment_index: int
    action_index: int
    enemy_key: str
    count: int
    interval: float
    route_index: int
    wave_pre_delay: float
    fragment_pre_delay: float
    action_pre_delay: float


@dataclass(frozen=True, slots=True)
class StageDefinition:
    info: StageInfo
    level: Mapping[str, Any]
    spawn_events: tuple[SpawnEvent, ...]
    enemy_ids: tuple[str, ...]


def normalize_stage_code(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).upper().strip()
    value = re.sub(r"[\u2010-\u2015\u2212\u30fc\uff0d]", "-", value)
    return re.sub(r"\s+", "", value)


def _stage_code_key(value: str) -> str:
    """Allow OCR to add leading zeroes without changing the stage-table ID."""

    return re.sub(r"\d+", lambda match: str(int(match.group())), normalize_stage_code(value))


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _normalize_difficulty(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    aliases = {
        "普通": "NORMAL",
        "NORMAL": "NORMAL",
        "突袭": "FOUR_STAR",
        "FOUR_STAR": "FOUR_STAR",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise StageDataError(f"Unsupported difficulty filter: {value}") from exc


def resolve_stage(
    stage_table: Mapping[str, Any],
    code: str,
    *,
    name: str | None = None,
    difficulty: str | None = None,
) -> StageInfo:
    """Resolve a display code to a unique stage using table content."""

    raw_stages = stage_table.get("stages")
    if not isinstance(raw_stages, Mapping):
        raise StageDataError("Invalid stage table: missing 'stages' mapping")

    stages = [StageInfo.from_mapping(value) for value in raw_stages.values()]
    normalized_code = normalize_stage_code(code)
    candidates = [stage for stage in stages if normalize_stage_code(stage.code) == normalized_code]
    if not candidates:
        code_key = _stage_code_key(code)
        candidates = [stage for stage in stages if _stage_code_key(stage.code) == code_key]
    if not candidates:
        raise StageDataError(f"No stage matches code {code!r}")

    if name:
        normalized_name = _normalize_name(name)
        candidates = [stage for stage in candidates if _normalize_name(stage.name) == normalized_name]
        if not candidates:
            raise StageDataError(f"Stage {code!r} does not match name {name!r}")

    if difficulty:
        normalized_difficulty = _normalize_difficulty(difficulty)
        candidates = [stage for stage in candidates if stage.difficulty == normalized_difficulty]
        if not candidates:
            raise StageDataError(f"Stage {code!r} has no difficulty {difficulty!r}")

    if len(candidates) != 1:
        choices = ", ".join(f"{stage.stage_id} ({stage.name}, {stage.difficulty})" for stage in candidates)
        raise StageDataError(f"Stage {code!r} is ambiguous; provide name or difficulty: {choices}")
    return candidates[0]


def expand_spawn_events(level: Mapping[str, Any]) -> tuple[SpawnEvent, ...]:
    """Flatten spawn actions in scheduler order without inventing timestamps."""

    events: list[SpawnEvent] = []
    waves = level.get("waves") or []
    for wave_index, wave in enumerate(waves):
        wave_pre_delay = float(wave.get("preDelay") or 0.0)
        for fragment_index, fragment in enumerate(wave.get("fragments") or []):
            fragment_pre_delay = float(fragment.get("preDelay") or 0.0)
            for action_index, action in enumerate(fragment.get("actions") or []):
                if action.get("actionType") != "SPAWN":
                    continue
                events.append(
                    SpawnEvent(
                        order=len(events),
                        wave_index=wave_index,
                        fragment_index=fragment_index,
                        action_index=action_index,
                        enemy_key=str(action["key"]),
                        count=int(action.get("count") or 0),
                        interval=float(action.get("interval") or 0.0),
                        route_index=int(action.get("routeIndex") or 0),
                        wave_pre_delay=wave_pre_delay,
                        fragment_pre_delay=fragment_pre_delay,
                        action_pre_delay=float(action.get("preDelay") or 0.0),
                    )
                )
    return tuple(events)


def level_enemy_ids(level: Mapping[str, Any]) -> tuple[str, ...]:
    enemy_ids = {
        str(reference["id"])
        for reference in level.get("enemyDbRefs") or []
        if isinstance(reference, Mapping) and reference.get("id")
    }
    enemy_ids.update(event.enemy_key for event in expand_spawn_events(level))
    return tuple(sorted(enemy_ids))


class StageRepository:
    """Download immutable-looking source files once and reuse them offline."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        game_data_base: str = GAME_DATA_BASE,
        static_base: str = THERESA_STATIC_BASE,
        timeout: float = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.game_data_base = game_data_base.rstrip("/")
        self.static_base = static_base.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                temporary.write(data)
                temporary_name = temporary.name
            os.replace(temporary_name, path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _get_bytes(self, url: str, cache_path: Path) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "ArkEye/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if cache_path.is_file():
                return cache_path.read_bytes()
            raise StageDataError(f"Unable to download {url}; no cached copy exists") from exc

        self._atomic_write(cache_path, data)
        metadata = {
            "source": url,
            "downloaded_at": datetime.now(UTC).isoformat(),
        }
        self._atomic_write(cache_path.with_suffix(cache_path.suffix + ".meta.json"), json.dumps(metadata, indent=2).encode())
        return data

    def _get_json(self, relative_url: str, cache_path: Path, *, base_url: str | None = None) -> Mapping[str, Any]:
        url = f"{(base_url or self.game_data_base).rstrip('/')}/{relative_url}"
        try:
            value = json.loads(self._get_bytes(url, cache_path).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StageDataError(f"Invalid JSON downloaded from {url}") from exc
        if not isinstance(value, Mapping):
            raise StageDataError(f"Expected a JSON object from {url}")
        return value

    def load_stage_table(self) -> Mapping[str, Any]:
        return self._get_json("excel/stage_table.json", self.cache_dir / "excel" / "stage_table.json")

    def load_enemy_database(self) -> Mapping[str, Any]:
        relative_url = "levels/enemydata/enemy_database.json"
        return self._get_json(relative_url, self.cache_dir / "levels" / "enemydata" / "enemy_database.json")

    def load_level(self, level_id: str) -> Mapping[str, Any]:
        relative = PurePosixPath(level_id.lower())
        if relative.is_absolute() or ".." in relative.parts or relative.suffix:
            raise StageDataError(f"Invalid level ID: {level_id!r}")
        relative_url = f"levels/{relative}.json"
        cache_path = self.cache_dir / "levels" / Path(*relative.parts).with_suffix(".json")
        return self._get_json(relative_url, cache_path)

    def load_public_level(self, level_id: str) -> Mapping[str, Any]:
        """Load the PRTS.Map public level JSON, with the game-data mirror as fallback."""

        relative = PurePosixPath(level_id.lower())
        if relative.is_absolute() or ".." in relative.parts or relative.suffix:
            raise StageDataError(f"Invalid level ID: {level_id!r}")
        relative_url = f"levels/{relative}.json"
        cache_path = self.cache_dir / "map_site" / Path(*relative.parts).with_suffix(".json")
        try:
            return self._get_json(relative_url, cache_path, base_url="https://map.ark-nights.com/data")
        except StageDataError:
            return self.load_level(level_id)

    def download_map_preview(self, stage_id: str, *, width: int = 1080, quality: int = 75) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_#-]+", stage_id):
            raise StageDataError(f"Invalid stage ID for map preview: {stage_id!r}")
        normal_stage_id = stage_id.replace("__f__", "").replace("#f#", "")
        url = f"{self.static_base}/api/v0/AK/CN/Android/mappreview/{normal_stage_id}/{width}/{quality}"
        cache_path = self.cache_dir / "mappreview" / f"{normal_stage_id}_{width}_{quality}.webp"
        self._get_bytes(url, cache_path)
        return cache_path

    def load_stage(self, info: StageInfo) -> StageDefinition:
        level = self.load_level(info.level_id)
        return StageDefinition(
            info=info,
            level=level,
            spawn_events=expand_spawn_events(level),
            enemy_ids=level_enemy_ids(level),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve and cache Arknights stage data")
    parser.add_argument("--stage", required=True, help="Displayed stage code, for example 0-1")
    parser.add_argument("--name", help="Displayed stage name used to disambiguate OCR results")
    parser.add_argument("--difficulty", help="NORMAL/普通 or FOUR_STAR/突袭")
    parser.add_argument("--cache-dir", default=".cache", help="Local source-data cache directory")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository = StageRepository(args.cache_dir)
    info = resolve_stage(repository.load_stage_table(), args.stage, name=args.name, difficulty=args.difficulty)
    definition = repository.load_stage(info)
    preview_path = repository.download_map_preview(info.stage_id)
    repository.load_enemy_database()
    summary = {
        "stage": {
            "stage_id": info.stage_id,
            "level_id": info.level_id,
            "zone_id": info.zone_id,
            "code": info.code,
            "name": info.name,
            "difficulty": info.difficulty,
        },
        "spawn_action_count": len(definition.spawn_events),
        "spawned_enemy_count": sum(event.count for event in definition.spawn_events),
        "enemy_ids": definition.enemy_ids,
        "map_preview": str(preview_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
