# ArkEye Recognition

Experimental, local-first computer-vision core for Arknights battlefield recognition and multi-object tracking.

ArkEye combines background subtraction, template classification, health-bar evidence, public stage data, route constraints, and persistent track IDs. It does not read or modify the game process.

> **Project status:** research prototype. The included 1-8 experiments are not a benchmark or a promise of production accuracy. Offline video latency is not end-to-end real-time latency.

## Features

- Foreground detection with route, deployable-tile, and UI masks.
- Template classification with explicit `unknown` support.
- Friendly/enemy evidence from health bars.
- Route- and spawn-aware tracking with stable IDs and held tracks.
- Soft route scoring: distance from a route lowers a candidate's association
  weight instead of discarding it, and a frame where every candidate misses the
  route is relaxed rather than dropped, so a stage whose terrain or calibration
  drifts does not go blind. Route indices stay aligned with the level file even
  when a route has fewer than two points.
- Public stage-code resolution and cached stage-data downloads.
- Optional timeline-primary tracking for 1-8:
  - the public spawn schedule owns enemy identity, class, route, and ID;
  - image evidence confirms visibility and applies bounded position corrections;
  - unsupported positions are labelled `predicted`, never visually confirmed;
  - friendly units still use the visual tracking pipeline.

## Requirements

- Python 3.11 or newer
- OpenCV, NumPy, SciPy, and PyAV (installed automatically)

## Install from a source checkout

```bash
python -m venv .venv
# Activate .venv using your platform's standard command.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install OCR support only if you need automatic stage-code recognition:

```bash
python -m pip install -e ".[ocr]"
```

## Fetch public stage data

Run from the repository root:

```bash
python -m arkeye.stage_data --stage 1-8
```

This resolves the display code through the public stage table and downloads the level, enemy database, and map preview into `.cache/`. Cached copies are reused when the network is unavailable.

The runtime builders expect the repository's `assets/`, `configs/`, and `.cache/` directories relative to the current working directory. Applications embedding the core can also construct detectors and trackers directly with their own resources.

## Use the recognition core

```python
from arkeye.runtime import build_stage_pipeline, read_stable_video_background

background = read_stable_video_background("gameplay.mp4", start_timestamp=8.0)
components = build_stage_pipeline(
    "main_01-08",
    background,
    analysis_scale=0.5,
    battle_origin_timestamp=10.066666666666666,
    use_health_bars=True,
    timeline_primary=False,
)
pipeline = components.pipeline
```

Feed timestamped `arkeye.capture.Frame` objects to `pipeline.process(...)`. Each result contains visual detections, track snapshots, an annotated frame, and optional debug images.

Set `timeline_primary=True` only for the currently supported 1-8 configuration. Timeline output is schedule-driven inference, not proof that an enemy is alive or visible.

## Validate

```bash
python -m pytest
python -m ruff check src tests
```

Tests use bundled offline fixtures and do not require the original recordings.

## Project layout

```text
src/arkeye/       Recognition, tracking, HUD, and public-data core
tests/            Unit and regression tests
tests/fixtures/   Offline test data
assets/           Recognition templates and operator metadata
configs/          Board calibration and UI masks
tools/            Offline trace viewer and JSONL helpers
```

## Sample output

Rendered results are too large for git, so they live on the release page:

<https://github.com/error-0x12/arkeye/releases/tag/delivery-1-8-v14>

- `hud.mp4` — 1-8 gameplay with the recognition overlay
- `chain.mp4` — the same run as a four-panel chain view
- `trace.jsonl` — per-frame decision log (2498 frames, 86 tracks)
- `tracks.jsonl` — per-frame track snapshots
- `summary.json` — aggregate counts for that run

To look at a trace locally, serve it and open the printed URL:

```bash
python tools/trace_viewer/serve.py trace.jsonl --video hud.mp4
```

`--video` makes the page draw real frames instead of a coordinate schematic; the
mp4 is handed out at `/video.mp4` with Range support so seeking works.

## Third-party data and assets

Public game data comes from [Kengxxiao/ArknightsGameData](https://github.com/Kengxxiao/ArknightsGameData), [PRTS.Map](https://map.ark-nights.com/), and the sources recorded in `stage_data.py`.

Game imagery and character/enemy templates are included only for development and regression testing. They are not relicensed under MIT; see `NOTICE.md`.

## License

The original source code is licensed under the [MIT License](LICENSE).

ArkEye is an independent project and is not affiliated with or endorsed by the Arknights rights holders.
