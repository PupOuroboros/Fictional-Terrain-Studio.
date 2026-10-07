# Fictional Terrain Studio

> **Status: active development / pre-release.** Fictional Terrain Studio is not finished yet. The terrain engine and major workflows are functional and tested, but the UI, packaging, real-world terrain acquisition, and broader user testing are still in progress. Expect breaking changes before a stable 1.0 release.

Fictional Terrain Studio (FTS) is an open-source, engine-first terrain authoring toolkit focused on **Cities: Skylines II**. It combines procedural geography, semantic spline-based landforms, hydrology QA, donor-DEM composition, GIS interchange, CS2-specific export validation, and focused non-destructive terrain editing.

FTS is designed around one core idea: **the editable geographic project is the source of truth; the final heightmap is a render product.**

FTS is not affiliated with or endorsed by Colossal Order or Paradox Interactive.

## Current release: v0.17

v0.17 adds **Editing Performance & Focused Local Terrain Tools**:

- measurement-to-edit workflow for road, rail, river, and arbitrary terrain profiles;
- non-destructive **Grade Repair** and **Corridor Smooth** project objects;
- undoable, disableable, serializable focused terrain corrections;
- cached-base incremental previews for local-edit-only changes;
- exact equivalence checks between incremental and normal full previews;
- optional Qwen/Ollama integration that is **disabled and hidden by default**.

On the included Upper Mississippi reference project, the v0.17 regression suite currently passes **100 tests**. A representative local-edit preview was about **10× faster** than the full preview in the development environment while producing **0.0 m maximum terrain difference** for the same project state.

## What works today

### Terrain authoring

- deterministic terrain recipes and seeds;
- multiscale fictional relief;
- semantic river valleys, floodplains, channels, terraces, bluffs, moraines, ridges, lakes, coasts, and harbors;
- arbitrary editable splines with insert/move/delete control points;
- reusable donor-DEM and semantic landform composition;
- first-class water-source planning objects;
- focused non-destructive local correction objects.

### Hydrology and QA

- Priority-Flood depression conditioning for diagnostics;
- D8 flow routing and accumulation;
- HAND/floodplain diagnostics;
- river connectivity and outlet checks;
- longitudinal river profile QA;
- buildability/slope analysis using the current project design profile;
- post-water hydrology checks.

### CS2 production

For the standard vanilla profile, FTS currently uses:

- **14.336 km × 14.336 km** playable terrain;
- **57.344 km × 57.344 km** World Map;
- **4096 × 4096** 16-bit grayscale playable heightmap;
- **4096 × 4096** World Map with the playable terrain encoded in the central **1024 × 1024** region;
- automated export validation and CS2 import guidance.

### GIS interchange

- Float32 GeoTIFF export/import;
- semantic GeoJSON export/import with stable FTS object IDs and provenance;
- fictional local-metre engineering grid or real projected CRS workflows;
- QGIS round-trip support while preserving editable FTS semantic objects;
- protection against silently overwriting locked hydrography.

### Desktop/editor workflow

- project tree and inspector;
- 2D terrain viewport;
- interactive spline control-point editing;
- undo/redo;
- preview and final-bake jobs;
- multiple QA/analysis views;
- terrain measurements and transects;
- autosave/recovery and staged final output.

## Not finished yet

FTS should be considered **pre-release software**. Important remaining work includes:

- broader Windows testing and a polished end-user installer;
- real-world location/DEM acquisition and map selector (planned for v0.18);
- more extensive testing across unrelated real-world and fictional maps;
- UI/UX redesign after the core feature set stabilizes;
- further performance work and memory optimization;
- more mature GIS topology/format support where it provides concrete value;
- optional hydrology cross-validation/backends;
- documentation cleanup as older milestone notes are consolidated.

The current UI is intentionally an engineering shell. Functionality and architecture are being stabilized before a dedicated visual/UI redesign.

## Installation from source

FTS currently targets **Python 3.11+**.

```bash
python -m pip install -e .
```

For development/tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

For GIS interchange:

```bash
python -m pip install -e ".[gis]"
```

Optional local AI/API integrations are deliberately separate:

```bash
python -m pip install -e ".[ai]"
```

A normal FTS installation does **not** require Ollama, Qwen, FastAPI, or MCP.

## Launching the desktop editor

After installation:

```bash
fts-gui presets/upper_mississippi_v17_local_editing.ftsproject.json
```

Or directly from source:

```bash
PYTHONPATH=src python -m fictional_terrain_studio.desktop \
  presets/upper_mississippi_v17_local_editing.ftsproject.json
```

## CLI examples

Generate a terrain recipe:

```bash
fts generate presets/upper_mississippi.yaml --out out/upper_mississippi
```

Render an editable project preview:

```bash
fts project-render presets/upper_mississippi_v17_local_editing.ftsproject.json \
  --mode preview --out out/preview
```

Render a final production output:

```bash
fts project-render presets/upper_mississippi_v17_local_editing.ftsproject.json \
  --mode final --out out/final
```

## Optional local AI integration

FTS includes an optional structured agent/API layer for local model workflows. It is **off by default** and not required for terrain generation or editing.

When enabled, supported surfaces include:

- local Ollama/Qwen tool calling;
- local REST API;
- optional MCP adapter;
- explicit read-only/edit/final-bake permission boundaries;
- structured tool-call auditing.

AI configuration is stored per installation, not inside terrain project files, so projects remain portable.

## Project philosophy

FTS aims to stay focused rather than become an all-purpose GIS/3D application.

- The terrain engine is independent of the GUI.
- Semantic geography remains editable instead of being immediately flattened into a raster.
- Hydrologic conditioning is a QA signal, not a substitute for good terrain design.
- Local corrections should preserve macro-landforms rather than flattening them.
- GIS/QGIS interoperability is preferred over rebuilding every GIS capability inside FTS.
- Optional integrations remain optional.
- Features need to justify their complexity.

## Documentation

Useful starting points:

- [`ROADMAP.md`](ROADMAP.md) — development roadmap;
- [`LOCAL_EDITING.md`](LOCAL_EDITING.md) — v0.17 focused correction workflow;
- [`MEASUREMENTS.md`](MEASUREMENTS.md) — profiles, transects, and grade QA;
- [`GIS_INTERCHANGE.md`](GIS_INTERCHANGE.md) — QGIS/GeoTIFF/GeoJSON workflow;
- [`CS2_PRODUCTION.md`](CS2_PRODUCTION.md) — CS2 production/export workflow;
- [`HYDROLOGY.md`](HYDROLOGY.md) — hydrology implementation and QA;
- [`SPLINES.md`](SPLINES.md) — semantic spline authoring;
- [`BIBLE_PROFILE.md`](BIBLE_PROFILE.md) — machine-readable design constraints;
- [`AGENT_INTEGRATION.md`](AGENT_INTEGRATION.md) — optional API/Qwen/MCP architecture.

Older `V0_*` documents are retained as development history and regression references.

## Contributing

Contributions, bug reports, test maps, and design discussion are welcome while the project is evolving. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

Because the project is not yet stable, larger architectural changes should be discussed in an issue first.

## License

Fictional Terrain Studio is released under the **MIT License**. See [`LICENSE`](LICENSE).
