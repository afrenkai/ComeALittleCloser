# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The style guide and project contract in `AGENTS.md` (imported above) are binding: `uv` for everything, no `_`-prefixed names or `_` throwaways, no `zip`/`getattr`/`Any`/`__all__`/inline imports, typed boundaries with `jaxtyping` + `beartype`, schemas in each domain's `schemas.py`, `master` as base branch.

## Commands

Run everything from the repo root. Plain `uv sync` / `uv run` selects the CPU environment (`cpu` is a default dependency group; the `cpu` and `ampere` extras conflict).

```bash
uv sync --locked
uv run --locked ruff check src tests
uv run --locked black --check src tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked pytest -q
uv run --locked pytest -q tests/test_web.py::test_name      # single test
node --check src/come_a_little_bit_closer/web/static/explorer.js
uv build --no-sources
```

CI (`.github/workflows/checks.yml`) runs the suite without `tests/test_cpu_model.py`, then runs that file separately in an offline job (it builds a tiny random BERT locally; no weight downloads).

Build the catalog and run the explorer locally (requires `data/raw/radio-browser-geolocated.json`):

```bash
uv run --locked -m come_a_little_bit_closer.storage.catalog
uv run --locked -m come_a_little_bit_closer.models.batch --model qwen3   # one embedding cache
uv run --locked -m come_a_little_bit_closer.models.inspect
uv run --locked uvicorn come_a_little_bit_closer.web.app:create_app --factory --host 127.0.0.1 --port 8765
```

GPU inference needs a separate env, the `ampere` extra, and the explicit `--ampere` flag:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ampere uv run --locked --no-group cpu --extra ampere -m come_a_little_bit_closer.models.batch --model qwen3 --ampere
```

Experiment report: `uv run --locked --extra research -m come_a_little_bit_closer.evaluation.location_report --config experiments/location-v2.json`

Layer probes on BC Andromeda (from the repo root, after `source ops/bc.env`; one job per model):

```bash
sbatch -p gtml -A gtml --gres=gpu:l40s:1 ops/layers-extract.sbatch <model_id> experiments/layers-v4-extract-batch8.json
sbatch -p gtml -A gtml --gres=gpu:l4:1 --cpus-per-task=12 --mem=48G --time=24:00:00 ops/layers-probe.sbatch <model_id> dev experiments/layers-v4.json
# report: run layer_report --config experiments/layers-v4.json --mode dev inside a Slurm job (see experiments/LAYERS_RUNS.md)
```

Lean: from `formal/`, `lake build ComeCloser` (fresh checkout: `lake update && lake exe cache get` first).
Paper: from `paper/`, `latexmk -pdf -outdir=build proposal_extended.tex` (must stay eight pages including references).

## Architecture

Package `src/come_a_little_bit_closer/` is organized by domain; data flows left to right:

`data` (Radio Browser JSON → `Station`, MusicBrainz genre matching) → `storage` (DuckDB catalog at `data/processed/stations.duckdb`) → `models` (sentence-transformers / CLAP text encoders → unit-vector embedding artifacts under `data/processed/embeddings`) → `geometry` (Haversine, cosine) / `projection` (PCA, spherical k-means) → `evaluation` (frozen location experiments) and `web` (FastAPI explorer).

Cross-cutting invariants worth knowing before editing:

- **Provenance is enforced at load time.** `storage.catalog.initialize_catalog` rejects a source snapshot whose SHA-256 differs from the catalog's recorded one. Embedding artifacts (`models/artifacts.py`) carry station IDs, source checksum, checkpoint revision, document format, tokenizer limit, and encoder settings; incompatible ones are rejected. A new data snapshot means rebuilding catalog and embeddings together, never mixing with frozen caches.
- **Model registry:** checkpoints and revisions are pinned in `models/catalog.py`; `models/factory.create_encoder` dispatches on `spec.backend` (`clap-text` vs generic `TextEncoder`). Query-time encoders reuse the cache's revision and native prompts.
- **Web retrieval (`web/retrieval.Retriever`)**: genre is detected lexically per query (longest match wins), producing a shared genre-eligible candidate pool. Haversine and full-dimensional cosine rankings are computed on that same pool; "unfiltered genre match" is measured separately on the model's raw top-k. PCA/cluster frame is computed once over the full corpus and fixed across queries; cosine never uses projected coordinates. Settings come from `web/schemas.WebSettings` (env prefix `CLOSER_`, nested delimiter `__`).
- **Frontend**: Jinja templates + HTMX + Canvas/D3 in `web/static/explorer.js`; vendored assets are checksummed in `web/static/vendor/ASSETS.md`. The UI behavior contract is `web/UI_CONTRACT.md` — read it before changing frontend behavior.
- **Experiments are frozen.** `experiments/location-v2.json` (primary) and `location-v1.json` pin models, genres, origins, splits, and checksums. Results/provenance are recorded in `experiments/LOCATION_RUNS.md` and `LOCATION_FINDINGS.md`; the web genre matcher postdates them and does not replace their candidate pools. Pending work is in `TODO.md` (audio, human labels, buffered splits) — don't describe these as done. The layer-probe held-out test ran once under `layers-v7.json` (results: `experiments/LAYERS_V7_CLAIMS.md`); see `docs/PROGRESS.md`.
- **Remote compute**: `ops/*.sbatch` are Slurm jobs, originally for Turing (embedding, location prepare → evaluate with `afterok`), now also run on BC Andromeda via `ops/bc.env` (see the AGENTS.md rules and `ops/README.md`). SSH with forwarding disabled, and `ops/web.sbatch` must stay disabled.
- **Layer-wise probes** (`evaluation/layer_*`, `models/layer_*`, `models/hidden_states.py`): `models/layer_catalog.py` pins every checkpoint as a `LayerModelSpec` with family and training stage (base → retrieval pairs; optional `start_token`, `end_token`, `attention_implementation` per model). `evaluation/layer_extract.py` writes per-layer pooled float16 features with provenance manifests; `evaluation/layer_run.py` fits ridge probes (`sphere`, `latlon`, `country-residual`) per layer against a training-only country-centroid control on spatial-block and random-item splits, in `dev` (validation) or `final` (test, run once) mode; `evaluation/layer_report.py` makes the figures and `summary.md` (dev only). Plans are `experiments/layers-v*.json`; the newest are `layers-v5.json` (18 models; adds per-block scores and per-place predictions) and `layers-v6.json` (adds the `xlmr-xl` size control); `evaluation/layer_popularity.py` splits gain by city population. Results and job history: `experiments/LAYERS_RUNS.md`.
- `formal/` Lean definitions mirror geometry, ranking, objectives, projection, and the location experiment; they don't verify numerics or learned encoders.
