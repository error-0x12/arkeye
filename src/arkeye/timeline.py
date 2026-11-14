"""Expand public Arknights level scheduler data into spawn timestamps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TimedSpawnEvent:
    timestamp: float
    wave_timestamp: float
    enemy_key: str
    count: int
    interval: float
    route_index: int
    wave_index: int
    fragment_index: int
    action_index: int


@dataclass(frozen=True, slots=True)
class StageTimeline:
    events: tuple[TimedSpawnEvent, ...]

    @property
    def total_count(self) -> int:
        return sum(event.count for event in self.events)

    @property
    def earliest_spawn_times(self) -> dict[str, float]:
        earliest: dict[str, float] = {}
        for event in self.events:
            earliest[event.enemy_key] = min(earliest.get(event.enemy_key, float("inf")), event.timestamp)
        return earliest

    @property
    def expected_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.enemy_key] = counts.get(event.enemy_key, 0) + event.count
        return counts

    @property
    def class_route_indices(self) -> dict[str, tuple[int, ...]]:
        routes: dict[str, set[int]] = {}
        for event in self.events:
            routes.setdefault(event.enemy_key, set()).add(event.route_index)
        return {enemy_id: tuple(sorted(indices)) for enemy_id, indices in routes.items()}

    @property
    def route_earliest_times(self) -> dict[int, float]:
        earliest: dict[int, float] = {}
        for event in self.events:
            earliest[event.route_index] = min(earliest.get(event.route_index, float("inf")), event.timestamp)
        return earliest


def build_stage_timeline(level: Mapping[str, Any], *, wave_interval_fallback: float = 5.0) -> StageTimeline:
    """Mirror the public PRTS.Map scheduler expansion without UI assumptions."""

    events: list[TimedSpawnEvent] = []
    wave_offset = 0.0
    for wave_index, wave in enumerate(level.get("waves") or []):
        wave_time = float(wave.get("preDelay") or 0.0)
        for fragment_index, fragment in enumerate(wave.get("fragments") or []):
            wave_time += float(fragment.get("preDelay") or 0.0)
            fragment_start = wave_time
            fragment_end = fragment_start
            actions = sorted(fragment.get("actions") or [], key=lambda action: float(action.get("preDelay") or 0.0))
            for action_index, action in enumerate(actions):
                pre_delay = float(action.get("preDelay") or 0.0)
                count = int(action.get("count") or 0)
                interval = float(action.get("interval") or 0.0)
                fragment_end = max(fragment_end, fragment_start + pre_delay + (count - 1) * interval)
                wave_time = fragment_start + pre_delay
                if action.get("actionType") == "SPAWN":
                    events.append(
                        TimedSpawnEvent(
                            timestamp=wave_offset + wave_time,
                            wave_timestamp=wave_time,
                            enemy_key=str(action["key"]),
                            count=count,
                            interval=interval,
                            route_index=int(action.get("routeIndex") or 0),
                            wave_index=wave_index,
                            fragment_index=fragment_index,
                            action_index=action_index,
                        )
                    )
                wave_time += (count - 1) * interval
            wave_time = fragment_end
        wave_time += float(wave.get("postDelay") or 0.0)
        wave_offset += wave_time + wave_interval_fallback
    return StageTimeline(tuple(events))
