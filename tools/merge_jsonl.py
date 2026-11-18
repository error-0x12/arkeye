"""Merge per-chunk JSONL outputs into one file, dropping duplicate records.

Chunked rendering writes one ``tracks.jsonl`` / ``trace.jsonl`` per segment and
the segments are concatenated at the end. A plain ``cat`` duplicated whole
frames whenever two segments overlapped (for example after re-rendering a range),
and duplicated frames make the viewer and any per-frame statistic disagree.

    python tools/merge_jsonl.py output/hud_v5/part_*/trace.jsonl --output output/1_8_hud/trace.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _record_key(record: dict, is_tracks: bool) -> object:
    frame = record.get("frame_index")
    if is_tracks:
        return frame, record.get("track_id")
    return frame


def merge(paths: list[Path], output: Path) -> dict:
    records: dict[object, dict] = {}
    total = 0
    duplicates = 0
    is_tracks: bool | None = None

    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if is_tracks is None:
                    is_tracks = "track_id" in record and "detect" not in record
                key = _record_key(record, bool(is_tracks))
                if key in records:
                    duplicates += 1
                    continue
                records[key] = record

    ordered = sorted(
        records.values(),
        key=lambda record: (record.get("frame_index") or 0, record.get("track_id") or 0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    frames = {record.get("frame_index") for record in records.values()}
    summary = {
        "segments": len(paths),
        "input_records": total,
        "kept_records": len(records),
        "duplicates_dropped": duplicates,
        "distinct_frames": len(frames),
        "kind": "tracks" if is_tracks else "trace",
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSONL segments without duplicating frames")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(sorted(args.inputs), args.output)


if __name__ == "__main__":
    main()
