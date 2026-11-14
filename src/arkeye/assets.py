"""Battle-sprite templates with alpha masks; handbook icons are not accepted."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

BATTLE_TEMPLATE_SOURCES = frozenset({"game_sprite", "battle_capture"})


class AssetError(ValueError):
    """Raised when a battle template is invalid or has an unsuitable source."""


@dataclass(frozen=True, slots=True)
class BattleTemplate:
    enemy_id: str
    name: str
    image: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    source: str

    @classmethod
    def from_bgra(
        cls,
        enemy_id: str,
        image: NDArray[np.uint8],
        *,
        source: str,
        name: str | None = None,
    ) -> BattleTemplate:
        if source not in BATTLE_TEMPLATE_SOURCES:
            raise AssetError(f"Template source {source!r} is not a battle sprite or battle capture")
        if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
            raise AssetError("A battle template must be a uint8 BGRA image")
        alpha = image[:, :, 3]
        if not np.any(alpha > 10):
            raise AssetError("A battle template must contain non-transparent pixels")
        points = cv2.findNonZero((alpha > 10).astype(np.uint8))
        if points is None:
            raise AssetError("Unable to locate template alpha bounds")
        x, y, width, height = cv2.boundingRect(points)
        cropped = image[y : y + height, x : x + width]
        return cls(
            enemy_id=enemy_id,
            name=name or enemy_id,
            image=np.ascontiguousarray(cropped[:, :, :3]),
            alpha=np.ascontiguousarray(cropped[:, :, 3]),
            source=source,
        )

    def resized(self, width: int, height: int) -> BattleTemplate:
        if width <= 0 or height <= 0:
            raise AssetError("Template dimensions must be positive")
        return BattleTemplate(
            enemy_id=self.enemy_id,
            name=self.name,
            image=cv2.resize(self.image, (width, height), interpolation=cv2.INTER_AREA),
            alpha=cv2.resize(self.alpha, (width, height), interpolation=cv2.INTER_NEAREST),
            source=self.source,
        )


class TemplateLibrary:
    def __init__(self, templates: Iterable[BattleTemplate]) -> None:
        self.templates = tuple(templates)

    def for_enemies(self, enemy_ids: Iterable[str] | None) -> tuple[BattleTemplate, ...]:
        if enemy_ids is None:
            return self.templates
        allowed = set(enemy_ids)
        return tuple(template for template in self.templates if template.enemy_id in allowed)

    @classmethod
    def load_manifest(cls, path: str | PathLike[str]) -> TemplateLibrary:
        manifest_path = Path(path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetError(f"Unable to load template manifest: {manifest_path}") from exc
        entries = manifest.get("templates") if isinstance(manifest, dict) else None
        if not isinstance(entries, list):
            raise AssetError("Template manifest must contain a 'templates' list")

        templates: list[BattleTemplate] = []
        for entry in entries:
            try:
                enemy_id = str(entry["enemy_id"])
                source = str(entry["source"])
                relative_path = Path(entry["path"])
                name = str(entry.get("name") or relative_path.stem)
            except (KeyError, TypeError) as exc:
                raise AssetError("Each template requires enemy_id, source, and path") from exc
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise AssetError("Template paths must stay inside the manifest directory")
            image_path = manifest_path.parent / relative_path
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise AssetError(f"Unable to read battle template: {image_path}")
            templates.append(BattleTemplate.from_bgra(enemy_id, image, source=source, name=name))
        return cls(templates)
