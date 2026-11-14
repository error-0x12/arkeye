"""OCR pre-battle stage codes and resolve them through stage_table.json."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arkeye.stage_data import StageDataError, StageInfo, StageRepository, normalize_stage_code, resolve_stage

_STAGE_CODE_PATTERN = re.compile(r"(?<![A-Z0-9])[A-Z0-9]+(?:-[A-Z0-9]+)+(?![A-Z0-9])")


@dataclass(frozen=True, slots=True)
class OCRToken:
    text: str
    confidence: float
    box: Any = None


@dataclass(frozen=True, slots=True)
class StageOCRMatch:
    info: StageInfo
    confidence: float
    name_confirmed: bool
    source_text: str


def extract_stage_codes(text: str) -> tuple[str, ...]:
    normalized = normalize_stage_code(text)
    return tuple(dict.fromkeys(match.group() for match in _STAGE_CODE_PATTERN.finditer(normalized)))


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def resolve_stage_from_tokens(
    stage_table: Mapping[str, Any],
    tokens: Sequence[OCRToken],
    *,
    difficulty: str = "NORMAL",
    min_confidence: float = 0.5,
) -> StageOCRMatch:
    candidates: dict[str, StageOCRMatch] = {}
    for token in tokens:
        if token.confidence < min_confidence:
            continue
        for code in extract_stage_codes(token.text):
            try:
                info = resolve_stage(stage_table, code, difficulty=difficulty)
            except StageDataError:
                continue
            name_confirmed = any(_normalized_text(info.name) in _normalized_text(other.text) for other in tokens)
            match = StageOCRMatch(
                info=info,
                confidence=token.confidence,
                name_confirmed=name_confirmed,
                source_text=token.text,
            )
            previous = candidates.get(info.stage_id)
            if previous is None or match.confidence > previous.confidence:
                candidates[info.stage_id] = match

    if not candidates:
        raise StageDataError("OCR did not produce a valid stage code")
    matches = list(candidates.values())
    named_matches = [match for match in matches if match.name_confirmed]
    if len(named_matches) == 1:
        return named_matches[0]
    if len(matches) == 1:
        return matches[0]
    choices = ", ".join(match.info.stage_id for match in matches)
    raise StageDataError(f"OCR stage is ambiguous; a matching stage name is required: {choices}")


class RapidOCREngine:
    def __init__(self) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise StageDataError("OCR support requires the 'ocr' project extra") from exc
        self._engine = RapidOCR()

    def read(self, image: np.ndarray) -> tuple[OCRToken, ...]:
        result = self._engine(image)
        if result is None or result.txts is None:
            return ()
        texts = list(result.txts)
        scores = list(result.scores) if result.scores is not None else [0.0] * len(texts)
        boxes = list(result.boxes) if result.boxes is not None else [None] * len(texts)
        return tuple(
            OCRToken(text=str(text), confidence=float(score), box=box)
            for text, score, box in zip(texts, scores, boxes, strict=False)
        )

    def read_stage(
        self,
        image: np.ndarray,
        stage_table: Mapping[str, Any],
        *,
        difficulty: str = "NORMAL",
    ) -> StageOCRMatch:
        return resolve_stage_from_tokens(stage_table, self.read(image), difficulty=difficulty)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve an Arknights stage from a pre-battle screenshot")
    parser.add_argument("--image", required=True, help="Pre-battle screenshot")
    parser.add_argument("--difficulty", default="NORMAL", help="NORMAL/普通 or FOUR_STAR/突袭")
    parser.add_argument("--cache-dir", default=".cache", help="Stage-data cache directory")
    args = parser.parse_args(argv)

    import cv2

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise StageDataError(f"Unable to read OCR image: {args.image}")
    repository = StageRepository(args.cache_dir)
    match = RapidOCREngine().read_stage(image, repository.load_stage_table(), difficulty=args.difficulty)
    print(
        json.dumps(
            {
                "stage_id": match.info.stage_id,
                "level_id": match.info.level_id,
                "zone_id": match.info.zone_id,
                "code": match.info.code,
                "name": match.info.name,
                "ocr_confidence": match.confidence,
                "name_confirmed": match.name_confirmed,
                "source_text": match.source_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
