# Layerwise Geographic Probes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, per layer, whether base and retrieval-tuned encoders linearly encode place coordinates beyond a country-centroid control, on GeoNames cities and radio stations.

**Architecture:** A GPU extraction job writes per-layer pooled hidden states for every (model, dataset, condition, pooling) to float16 arrays with provenance manifests. A CPU probe job reuses the existing ridge, angular-error, block-mean, and bootstrap code to score every layer on validation (`dev`) or, once, on test (`final`). A report module draws the three spec figures.

**Tech Stack:** Python 3.12, uv, numpy, transformers (via the `ampere` extra), pydantic, jaxtyping + beartype, matplotlib (`research` extra), Slurm on BC Andromeda.

**Spec:** `docs/superpowers/specs/2026-11-02-layerwise-probes-design.md`

## Global Constraints

- AGENTS.md is binding: `uv run` for everything; no `_`-prefixed names, `_` throwaways, `zip`, `getattr`, `Any`, `__all__`, or inline imports; type every boundary, with `@jaxtyped(typechecker=beartype)` on numerical functions; schemas and type aliases in the domain's schema module.
- Base branch `master`; work on `research/layer-probes`.
- GPU code runs only under `UV_PROJECT_ENVIRONMENT=.venv-ampere ... --no-group cpu --extra ampere` and an explicit `--ampere` flag; CPU is the default.
- Jobs are submitted after `source ops/bc.env`, with GPU jobs via `bc_sbatch_gpu` or `bc_sbatch_gtml`. Every job emails `$USER@bc.edu` through `bc.env`.
- Model loading in jobs is offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) from the pinned revisions in the spec's family table.
- Identical input text across all models: no prompts or instructions.
- `dev` mode never reads test indices; `final` mode refuses to run twice for the same output directory.
- Raw data and feature caches stay out of Git; `data/manifest.json` records every acquired file's checksum.
- Verify each task with `uv run --locked ruff check src tests`, `uv run --locked black --check src tests`, and the task's focused tests.

## Review Focus

- A station metadata document longer than `max_length` tokens should be truncated and counted in `truncated_items`, not crash or be dropped. Test in Task 4.
- Hidden-state values that overflow float16 (Qwen massive activations) must raise rather than store `inf`. Test in Task 4.
- Re-running extraction on an existing feature file with a different revision or item order must be rejected, not overwritten or mixed. Test in Task 4.
- Places whose country has no training items (all its blocks in validation/test) must fall back to the training global centroid in the country control, exactly as `country_predictions` already does. Test in Task 5.
- A tokenizer without a pad token (some decoder bases) should pad with its EOS token, and last-token pooling must still choose the final real token under left or right padding. Test in Task 3.

---

### Task 1: Shared probe settings and item-level splits

**Files:**
- Modify: `src/come_a_little_bit_closer/evaluation/location_schemas.py` (add `ProbeSettings`; `LocationPlan` subclasses it)
- Modify: `src/come_a_little_bit_closer/evaluation/probes.py` (`fit_ridge`, `bootstrap_interval` take `ProbeSettings`)
- Modify: `src/come_a_little_bit_closer/evaluation/splits.py`
- Modify: `src/come_a_little_bit_closer/evaluation/location_run.py:138` and `tests/test_location.py:62,99` (new `spatial_split` call shape)
- Test: `tests/test_location.py`

**Interfaces:**
- Produces: `class ProbeSettings(BaseModel)` with `split_seeds: list[int]`, `ridge_alphas: list[float]`, `block_degrees`, `train_fraction`, `validation_fraction`, `bootstrap_replicates`, `bootstrap_seed`, `geometry`. These fields move out of `LocationPlan` with their current defaults and validators, so `LocationPlan(ProbeSettings)` parses existing JSON unchanged.
- Produces: `spatial_split(item_ids: list[str], coordinates: list[Coordinates], settings: ProbeSettings, seed: int, source_sha256: str, plan_sha256: str) -> SplitManifest`
- Produces: `random_split(item_ids: list[str], coordinates: list[Coordinates], settings: ProbeSettings, seed: int, source_sha256: str, plan_sha256: str) -> SplitManifest`. Items are permuted with `np.random.default_rng(seed)` and cut at `train_fraction` and `train_fraction + validation_fraction`; `block_id` is still `spatial_block(...)` so aggregation is unchanged.
- `fit_ridge(..., plan: ProbeSettings)` and `bootstrap_interval(values, plan: ProbeSettings)` keep their parameter names, so call sites that pass a `LocationPlan` are untouched.

- [ ] **Step 1: Write the failing test** `test_random_split_is_deterministic_and_disjoint` in `tests/test_location.py`: 100 fixture items, two calls with seed 1729 are equal; train, validation, and test indices partition `range(100)` with sizes 60, 20, 20; a different seed gives a different assignment; each assignment's `block_id == spatial_block(lat, lon, 10.0)`.
- [ ] **Step 2: Run** `uv run --locked pytest -q tests/test_location.py::test_random_split_is_deterministic_and_disjoint`. Expected: FAIL, `ImportError: random_split`.
- [ ] **Step 3: Implement** `ProbeSettings`, the `LocationPlan` subclassing, the new `spatial_split` signature, and `random_split`. Update the location call sites to pass `[s.station_id for s in stations]` and `[s.coordinates for s in stations]`.
- [ ] **Step 4: Run** `uv run --locked pytest -q tests/test_location.py`. Expected: all pass, including the existing split and ridge tests unchanged in their assertions.
- [ ] **Step 5: Regression check:** before Step 3, copy the completed `outputs/location-v2/` to `outputs/location-v2-before-refactor/`. After Step 4, rerun `sbatch ops/location-evaluate.sbatch` (same plan, so the plan checksum matches the prepared artifacts) and `diff -r` the probe and split files against the copy. Expected: no differences other than timestamps.
- [ ] **Step 6: Commit** `refactor: share probe settings and add item-level random split`.

### Task 2: GeoNames places

**Files:**
- Create: `src/come_a_little_bit_closer/data/cities.py`
- Modify: `src/come_a_little_bit_closer/data/schemas.py` (add `Place`)
- Modify: `data/manifest.json` (add `countryInfo.txt` with URL `https://download.geonames.org/export/dump/countryInfo.txt`, sha256, and row count)
- Test: `tests/test_cities.py`

**Interfaces:**
- Produces: `class Place(BaseModel)` with `place_id: str` (GeoNames ID), `name: str`, `country_code: str`, `country_name: str`, `population: int`, `coordinates: Coordinates`.
- Produces: `load_places(archive_path: Path, country_info_path: Path) -> list[Place]`, sorted by `place_id` as an integer. It reads `cities15000.txt` from the archive with the GeoNames column order (0 id, 1 name, 4 latitude, 5 longitude, 8 country code, 14 population). Country names come from `countryInfo.txt` (skip `#` lines; column 0 ISO code, column 4 name), and an unknown code maps to an empty name.
- Produces: `place_text(place: Place, condition: Literal["name", "name-country"]) -> str` returning `Lyon` or `Lyon, France`, or just the name when the country name is empty.

- [ ] **Step 1: Write the failing tests** `test_load_places_reads_geonames_columns` (writes a two-row zip and a three-line country file to `tmp_path`, including a `#` comment; asserts IDs, names, coordinates, population, and `country_name == "France"`) and `test_place_text_conditions` (`"Lyon"`, `"Lyon, France"`, and an empty-country fallback to `"Lyon"`).
- [ ] **Step 2: Run** `uv run --locked pytest -q tests/test_cities.py`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement** `Place`, `load_places`, and `place_text`, using `zipfile` and `csv` with tab delimiter and `quoting=csv.QUOTE_NONE`.
- [ ] **Step 4: Download** `countryInfo.txt` to `data/raw/` and add its manifest entry with `sha256sum` output. Then run `uv run --locked python -c` with `load_places` on the real files. Expected: 34,196 places, with every place's country code present in `countryInfo.txt`.
- [ ] **Step 5: Run** the tests again. Expected: PASS. **Commit** `feat: load GeoNames cities as places`.

### Task 3: Layer model registry and hidden-state pooling

**Files:**
- Create: `src/come_a_little_bit_closer/models/layer_catalog.py`
- Create: `src/come_a_little_bit_closer/models/hidden_states.py`
- Modify: `src/come_a_little_bit_closer/models/schemas.py`
- Test: `tests/test_hidden_states.py` (CPU, offline, tiny random BERT built like `tests/test_cpu_model.py`)

**Interfaces:**
- Produces: `class LayerModelSpec(BaseModel)` with `model_id: str`, `family: str`, `stage: str`, `stage_order: int`, `checkpoint: str`, `revision: str`, `architecture: Literal["decoder", "encoder"]`, `random_init: bool = False`. Every pinned snapshot, including `bge-m3-retromae`, ships its own tokenizer.
- Produces: `layer_model_specs() -> list[LayerModelSpec]` holding the ten spec rows plus `qwen3-0.6b-random` (checkpoint and revision of Qwen3-0.6B-Base, `random_init=True`, stage `random`, order 0) and `xlmr-base-random` (likewise from xlm-roberta-base). IDs: `qwen3-0.6b-base`, `qwen3-0.6b-embedding`, `qwen3-4b-base`, `qwen3-4b-embedding`, `xlmr-base`, `e5-base`, `xlmr-large`, `bge-m3-retromae`, `bge-m3-unsupervised`, `bge-m3`. Base stages get order 1, intermediate stages 2–3, and retrieval the last order in its family.
- Produces: `find_layer_model(model_id: str) -> LayerModelSpec`.
- Produces type aliases in `models/schemas.py`: `type HiddenStates = Float32[torch.Tensor, "batch tokens hidden"]`, `type TokenMask = Bool[torch.Tensor, "batch tokens"]`, `type PooledStates = Float32[torch.Tensor, "batch hidden"]`.
- Produces in `hidden_states.py`: `mean_pool(states: HiddenStates, mask: TokenMask) -> PooledStates`, where a row with an all-false mask raises `ValueError`; `last_token_pool(states: HiddenStates, attention_mask: TokenMask) -> PooledStates`, which selects the highest index whose attention mask is true, so it works under either padding side; and `class LayerExtractor` with `__init__(self, spec: LayerModelSpec, device: str, max_length: int)` and `extract(self, texts: list[str], batch_size: int) -> dict[str, Float16[np.ndarray, "layers items hidden"]]`, keyed by `"mean"` and, for decoders, `"last"`.
- Extractor rules: load with `AutoModel.from_pretrained(checkpoint, revision=..., torch_dtype=torch.float32)`, or with `AutoModel.from_config(AutoConfig.from_pretrained(...))` after `torch.manual_seed(1729)` when `random_init`. Call it with `output_hidden_states=True` and keep every entry, i.e. `num_hidden_layers + 1`. Tokenize with `padding=True, truncation=True, max_length=max_length, return_special_tokens_mask=True`. The mean mask is the attention mask AND NOT the special-tokens mask. When `tokenizer.pad_token is None`, set it to `eos_token`. Process texts sorted by length and restore input order. Before the float16 cast, raise `ValueError` if `abs(value) > np.finfo(np.float16).max` anywhere.

- [ ] **Step 1: Write the failing tests:**
  - `test_mean_pool_ignores_padding_and_special_tokens`: hand-built 2×4×2 tensor and mask; assert exact means.
  - `test_last_token_pool_left_and_right_padding`: the same sequence padded left and right returns the same row.
  - `test_layer_extractor_tiny_bert_shapes_and_order`: a tiny BERT saved to `tmp_path` as a local checkpoint spec with 2 layers and hidden size 16; `extract(["rock", "jazz station near rock"], 1)` gives `mean.shape == (3, 2, 16)`, dtype float16, and the same rows as `batch_size=2`.
  - `test_registry_pins_every_revision`: every spec has a 40-character hex revision; IDs are unique; every family has exactly one stage with the highest `stage_order` and `stage == "retrieval"`.
- [ ] **Step 2: Run** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked pytest -q tests/test_hidden_states.py`. Expected: FAIL, imports.
- [ ] **Step 3: Implement** the schemas, the registry, the pooling functions, and `LayerExtractor`. `LayerModelSpec` accepts a local path as `checkpoint` for the test fixture.
- [ ] **Step 4: Run** the tests. Expected: PASS. Also add `tests/test_hidden_states.py` beside `tests/test_cpu_model.py` in the offline CI job in `.github/workflows/checks.yml`.
- [ ] **Step 5: Commit** `feat: add layer model registry and hidden-state pooling`.

### Task 4: Layer plan, feature artifacts, and the extraction job

**Files:**
- Create: `src/come_a_little_bit_closer/evaluation/layer_schemas.py`
- Create: `src/come_a_little_bit_closer/evaluation/layer_extract.py`
- Create: `experiments/layers-v1.json`
- Create: `ops/layers-extract.sbatch`
- Test: `tests/test_layer_extract.py`

**Interfaces:**
- Produces: `type LayerDataset = Literal["cities", "stations"]`, `type TextCondition = Literal["name", "name-country", "metadata"]`, `type Pooling = Literal["mean", "last"]`, `type SplitKind = Literal["spatial-block", "random-item"]`, `type EvaluationMode = Literal["dev", "final"]`.
- Produces: `class LayerPlan(ProbeSettings)` with `experiment_id: str`, `model_ids: list[str]`, `datasets: dict[LayerDataset, list[TextCondition]]`, `split_kinds: list[SplitKind]`, `max_length: int = 64`, `batch_size: int = 256`, `feature_directory: Path`, `output_directory: Path`, `final_cells: list[FinalCell] = []`. `FinalCell` has `model_id`, `dataset`, `condition`, `pooling`, and `layer`.
- Produces: `class FeatureManifest(BaseModel)` with `model_id`, `checkpoint`, `revision`, `random_init`, `dataset`, `condition`, `pooling`, `source_sha256`, `item_ids: list[str]`, `layers: int`, `hidden: int`, `dtype: str`, `max_length`, `truncated_items: int`, `transformers_version`, `created_at`.
- Produces: `feature_path(plan: LayerPlan, model_id: str, dataset: LayerDataset, condition: TextCondition, pooling: Pooling) -> Path`, which returns `{feature_directory}/{model_id}/{dataset}-{condition}-{pooling}.npy`, with the manifest beside it as `.json`.
- Produces: `dataset_items(dataset: LayerDataset, condition: TextCondition) -> tuple[list[str], list[str], list[Coordinates], list[str], str]`, returning item IDs, texts, coordinates, country codes, and source sha256. Cities use `load_places` and `place_text`. Stations use the catalog and `location_prepare.metadata_document`, and accept only the `metadata` condition.
- Produces: `extract_model(plan: LayerPlan, model_id: str, device: str) -> list[Path]`. It skips a file whose manifest equals what would be written, excluding `created_at`. It raises `ValueError("Existing feature file does not match this run.")` on any other mismatch. It writes atomically through a temporary file plus `os.replace`.
- `main()` arguments: `--config`, `--model`, `--ampere`.
- `experiments/layers-v1.json`: all 12 model IDs; `datasets = {"cities": ["name", "name-country"], "stations": ["metadata"]}`; both split kinds; `split_seeds [1729, 2718, 3141]`; `ridge_alphas [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]`; `block_degrees 10`; fractions 0.6/0.2; `bootstrap_replicates 1000`; `bootstrap_seed 1618`; `feature_directory "data/processed/layers-v1/features"`; `output_directory "outputs/layers-v1"`; empty `final_cells`.
- `ops/layers-extract.sbatch`: modeled on `ops/location-prepare.sbatch`, with `--mem=64G`, `--time=02:00:00`, offline flags, and the model ID taken as `$1`.

- [ ] **Step 1: Write the failing tests:**
  - `test_extract_rejects_mismatched_existing_features`: build a manifest, then attempt a run whose `revision` differs; expect `ValueError`.
  - `test_float16_overflow_raises`: call the extractor's cast helper (`checked_float16(values: Float32[np.ndarray, "..."]) -> Float16[np.ndarray, "..."]`, defined in `hidden_states.py`) on an array containing `70000.0`; expect `ValueError`.
  - `test_long_text_is_truncated_and_counted`: with the tiny BERT fixture and `max_length=8`, a 40-token text yields a feature row and `truncated_items == 1` in the manifest.
  - `test_layer_plan_parses`: `LayerPlan.model_validate_json(Path("experiments/layers-v1.json").read_bytes())`, with every `model_id` found by `find_layer_model`.
- [ ] **Step 2: Run** `uv run --locked pytest -q tests/test_layer_extract.py`. Expected: FAIL.
- [ ] **Step 3: Implement** the schemas, `checked_float16`, `feature_path`, `dataset_items`, `extract_model`, the plan file, and the sbatch script.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: GPU smoke:** `source ops/bc.env && bc_sbatch_gtml ops/layers-extract.sbatch xlmr-base`. Expected: COMPLETED, with 3 `.npy` files under `data/processed/layers-v1/features/xlmr-base/` shaped `(13, 34196, 768)`, `(13, 34196, 768)`, and `(13, 11833, 768)`, and `truncated_items` recorded.
- [ ] **Step 6: Commit** `feat: extract per-layer pooled features with provenance`.

### Task 5: Layer probe runner

**Files:**
- Create: `src/come_a_little_bit_closer/evaluation/layer_run.py`
- Modify: `src/come_a_little_bit_closer/evaluation/layer_schemas.py` (add `LayerProbeScore`)
- Create: `ops/layers-probe.sbatch`
- Test: `tests/test_layer_run.py`

**Interfaces:**
- Produces: `class LayerProbeScore(BaseModel)` with `model_id`, `family`, `stage`, `stage_order`, `dataset`, `condition`, `pooling`, `layer`, `relative_depth` (layer / (layers − 1)), `split_kind`, `seed`, `mode`, `control: Literal["probe", "permuted-labels", "country-centroid"]`, `selected_alpha: float | None`, `evaluated_items`, `evaluated_blocks`, `block_mean_km`, `block_mean_ci_low`, `block_mean_ci_high`, `gain_over_country_km`, `gain_ci_low`, and `gain_ci_high`.
- Produces: `class LayerScoreValues(BaseModel)` holding the metric fields of `LayerProbeScore` (`selected_alpha` through `gain_ci_high`); `LayerProbeScore` is built as `LayerProbeScore(**identity_fields, **values.model_dump())`.
- Produces: `score_layer(features: ProbeFeatures, targets: ProbeTargets, country_codes: list[str], manifest: SplitManifest, plan: LayerPlan, mode: EvaluationMode, control: Literal["probe", "permuted-labels"]) -> LayerScoreValues`, and `score_country_control(targets, country_codes, manifest, plan, mode) -> LayerScoreValues`, written once per (dataset, condition, split kind, seed). It fits on train, selects alpha on validation, and scores validation in `dev` and test in `final`. `permuted-labels` permutes training targets with `np.random.default_rng(manifest.seed)`. The country control uses `country_predictions` on train. Gains are block means of `country_errors - probe_errors`, with `bootstrap_interval`.
- Produces: `run_model(plan: LayerPlan, model_id: str, mode: EvaluationMode) -> Path`. It writes `{output_directory}/{mode}/{model_id}.jsonl` with one `LayerProbeScore` per line. `dev` covers every layer; `final` covers only `plan.final_cells` and raises if the output file exists or `final_cells` is empty. It upcasts features to float64 one layer at a time.
- `main()` arguments: `--config`, `--model`, `--mode` (default `dev`).
- `ops/layers-probe.sbatch`: modeled on `ops/location-evaluate.sbatch`, CPU, `--cpus-per-task=16`, `--mem=64G`, `--time=06:00:00`, with `$1` the model ID and `$2` the mode.

- [ ] **Step 1: Write the failing tests:**
  - `test_dev_mode_never_reads_test_targets`: features equal the targets; set test-split targets to NaN; `score_layer(..., mode="dev")` returns finite scores and `evaluated_items` equals the validation count.
  - `test_coordinate_features_beat_country_control`: on 400 synthetic places spread over at least 8 countries, features = targets + N(0, 0.01) noise gives a `gain_over_country_km` > 0 and a `permuted-labels` gain < the `probe` gain.
  - `test_country_without_training_items_uses_global_fallback`: a country whose items all fall in validation gets the training global centroid as its control prediction.
  - `test_final_mode_refuses_rerun`: a second `run_model(..., "final")` on the same directory raises `FileExistsError`.
- [ ] **Step 2: Run** `uv run --locked pytest -q tests/test_layer_run.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `LayerProbeScore`, `score_layer`, `run_model`, `main`, and the sbatch script.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Smoke:** `source ops/bc.env && sbatch ops/layers-probe.sbatch xlmr-base dev` on the Task 4 features. Expected: COMPLETED, with `outputs/layers-v1/dev/xlmr-base.jsonl` holding 13 layers × 3 (dataset, condition) × 2 split kinds × 3 seeds × 2 layer controls + 3 × 2 × 3 country-control rows = 486 lines.
- [ ] **Step 6: Commit** `feat: score layerwise probes against the country control`.

### Task 6: Report and figures

**Files:**
- Create: `src/come_a_little_bit_closer/evaluation/layer_report.py`
- Test: none beyond a run on real outputs. Matplotlib output is verified by viewing the files.

**Interfaces:**
- Consumes: `outputs/layers-v1/{mode}/*.jsonl` (`LayerProbeScore`), plus feature and split files for the distortion map.
- Produces: `main()` with `--config` and `--mode`, writing to `{output_directory}/{mode}/report/`:
  - `where-geography-dies.png`: one panel per family; x is `relative_depth`, y is the seed-mean `gain_over_country_km` for `control == "probe"`, `split_kind == "spatial-block"`, `dataset == "cities"`, `condition == "name-country"`, `pooling == "mean"`. One line per stage in `stage_order`; a shaded CI band from the seed-mean bounds; the random-init stage dashed; a horizontal zero line labeled "country centroid".
  - `distortion-{family}.png`: base and retrieval stages side by side at each stage's best `dev` layer, chosen by maximal mean gain. Draws arrows from true to predicted coordinates for evaluated items on an equirectangular frame, with all 34,196 city positions as a faint background point layer instead of a basemap.
  - `stage-trajectory.png`: xlmr-large best-layer gain by `stage_order`, with CI.
  - `summary.md`: a best-layer table per (model, dataset, condition, split kind), and the model and revision list.
- Run under `uv run --locked --extra research`.

- [ ] **Step 1: Implement** `layer_report.py`, following the matplotlib conventions in `location_report.py` (Agg backend, explicit figure sizes, saved at dpi 200).
- [ ] **Step 2: Run** `MPLBACKEND=Agg uv run --locked --extra research -m come_a_little_bit_closer.evaluation.layer_report --config experiments/layers-v1.json --mode dev` on the xlmr-base smoke output. Expected: 4 files are written; inspect each PNG with the Read tool for legible axes and labels.
- [ ] **Step 3: Commit** `feat: draw layerwise probe report`.

### Task 7: Full dev run and record

**Files:**
- Create: `experiments/LAYERS_RUNS.md`
- Modify: `TODO.md` (layerwise probes: dev complete, final pending)

- [ ] **Step 1: Extract** all 12 models: one `bc_sbatch_gpu` or `bc_sbatch_gtml ops/layers-extract.sbatch <model_id>` per model, with `qwen3-4b-*` on a vr80g or vr144g node if memory requires. Expected: every job COMPLETED; `du -sh data/processed/layers-v1` under 150 GB.
- [ ] **Step 2: Probe** every model in dev with `sbatch --dependency=afterok:<extract job> ops/layers-probe.sbatch <model_id> dev`. Expected: all COMPLETED.
- [ ] **Step 3: Report** in dev mode (Task 6 command) and view all figures.
- [ ] **Step 4: Record** in `experiments/LAYERS_RUNS.md` every job ID, role, outcome, elapsed time, and node, including failures, plus the plan sha256, the source checksums, and the dev-mode headline numbers labeled as validation results. Do not run `final`.
- [ ] **Step 5: Commit** `docs: record layerwise probe dev run`.
