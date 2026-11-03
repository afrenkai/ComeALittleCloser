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
- [ ] Add layerwise hidden-state probes. (Dev run complete, validation results only, recorded in `experiments/LAYERS_RUNS.md`; `final` mode pending.)
- [ ] Evaluate H3 candidate recall before using it as a retrieval baseline.
- [ ] Consider geographic adaptation after the audited controls.

Completed text experiments and their provenance are recorded in `experiments/LOCATION_RUNS.md`; explorer history is in `experiments/EXPLORER_RUNS.md`. Turing's web service and tunnel stay disabled. Its Codex SSH control connection remains enabled.
