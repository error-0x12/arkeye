from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from arkeye.assets import AssetError, BattleTemplate, TemplateLibrary


def test_template_library_loads_only_battle_assets(tmp_path) -> None:
    image = np.zeros((20, 30, 4), dtype=np.uint8)
    image[4:16, 8:24] = (20, 40, 200, 255)
    image_path = tmp_path / "enemy.png"
    cv2.imwrite(str(image_path), image)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "templates": [
                    {"enemy_id": "enemy_a", "source": "game_sprite", "path": "enemy.png", "name": "walk_01"}
                ]
            }
        ),
        encoding="utf-8",
    )

    library = TemplateLibrary.load_manifest(manifest_path)

    assert len(library.templates) == 1
    template = library.templates[0]
    assert template.enemy_id == "enemy_a"
    assert template.name == "walk_01"
    assert template.image.shape == (12, 16, 3)
    assert template.alpha.shape == (12, 16)
    assert library.for_enemies(["other"]) == ()


def test_handbook_icons_are_rejected() -> None:
    image = np.zeros((10, 10, 4), dtype=np.uint8)
    image[:, :, 3] = 255

    with pytest.raises(AssetError, match="not a battle sprite"):
        BattleTemplate.from_bgra("enemy_a", image, source="handbook_icon")


def test_manifest_cannot_escape_asset_directory(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"templates": [{"enemy_id": "enemy_a", "source": "game_sprite", "path": "../x.png"}]}),
        encoding="utf-8",
    )

    with pytest.raises(AssetError, match="inside the manifest"):
        TemplateLibrary.load_manifest(manifest_path)
