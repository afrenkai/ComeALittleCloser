# TODO

## Git preparation

- [x] Preserve the existing GitHub initial commit and use local `master`.
- [x] Connect `git@github.com:afrenkai/ComeALittleCloser.git`.
- [x] Resolve query genres without sticky filters; use shared eligible candidates for spatial ranking.
- [x] Show raw model genre-tag agreement separately from distance and overlap.
- [x] Remove unused map data and styles; extract shared station markup and genre normalization.
- [x] Make batch scripts independent of the author's checkout path.
- [x] Add CPU/Ampere extras, default to CPU, and require `--ampere` for GPU inference commands.
- [x] Add GitHub checks for H3, GeoJSON, fake query vectors, and a tiny local CPU transformer.
- [x] Complete final local checks, package inspection, and browser smoke test: 28 tests, Ruff, Black, JavaScript/shell syntax, Lean, and package builds pass. The real E5 browser query shows genre scope, raw tag agreement, and spatial error; startup remains empty.

## Research

- [ ] Acquire and audit station audio.
- [ ] Compare metadata, audio-inferred, and human genre labels.
- [ ] Audit station coordinates and station-family membership.
- [ ] Add buffered geographic splits and a names-only ablation.
- [x] Add layerwise hidden-state probes. Validation runs for 19 models (layers-v4 to v6); held-out test run once under layers-v7 on 2026-11-04 (`experiments/LAYERS_V7_CLAIMS.md`, `docs/FREEZE.md`).
- [x] Check unstable Qwen3-4B-Embedding middle layers in the `latlon` probe: one activation outlier ("Carcassonne, France"); see `experiments/LAYERS_RUNS.md`, layers-v5.
- [x] Within-country neighbour metrics for cosine retrieval (Spearman ρ, kNN gain); frozen and tested once under reflected-v1 (`experiments/REFLECTED_V1_RESULTS.md`).
- [ ] Remaining retrieval metrics (overlap@10, excess distance, random baseline on station origins) and the layer-by-region map view (see `docs/PROGRESS.md`).
- [x] Coordinate-free fix (instruction + layer fusion + distance probe); frozen and tested once under fix-v1 (`experiments/FIX_V1_RESULTS.md`).
- [x] Fix baselines on validation: similarity instruction, factorial ablation, Feuerstein et al. 2026 axes (`experiments/FIX_RUNS.md`, fix-baselines-v1).
- [x] Fresh held-out places: GeoNames towns of 5,000-14,999 people; G1-G3 tested once under fresh-v1 (`experiments/FRESH_V1_RESULTS.md`).
- [x] Five more retrieval models, a 1,000-4,999 town band and model size (models-v1, size-v1), tested once (`experiments/MODELS_V1_RESULTS.md`).
- [x] Station search task (search-v1, search-v2), tested once; see the same file.
- [ ] Fix or flag the 247 catalog stations whose coordinates lie nearest a place in another country (`located_mask` in `evaluation/search_items.py`).
- [ ] Evaluate H3 candidate recall before using it as a retrieval baseline.
- [ ] Consider geographic adaptation after the audited controls.

Completed text experiments and their provenance are recorded in `experiments/LOCATION_RUNS.md`; explorer history is in `experiments/EXPLORER_RUNS.md`. Turing's web service and tunnel stay disabled. Its Codex SSH control connection remains enabled.
