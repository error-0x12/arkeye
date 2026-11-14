"""Operator metadata used for friendly-unit naming and terrain checks."""

from __future__ import annotations

import json
from pathlib import Path


def load_operator_metadata(path: str) -> dict[str, dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
