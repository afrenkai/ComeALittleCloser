# Come A Little Bit Closer

A radio explorer and research codebase comparing geographic and embedding neighborhoods. FastAPI and DuckDB serve a full-screen HTMX/Canvas globe and a three-dimensional PCA view. The five text baselines are Qwen3, BGE-M3, multilingual E5, EmbeddingGemma 2, and CLAP's text tower.

**Team progress (2026-11-04):** see [docs/PROGRESS.md](docs/PROGRESS.md) for a plain-language summary of the layer-wise probe, reflected-geography and fix results (each frozen and tested once on held-out cities), the fix's baselines, the fresh-towns test (frozen and tested once on 32,148 new towns), figures, and next steps.

## Run locally

Run commands from the project root with Python 3.12 and uv. Plain `uv sync` and `uv run` select the CPU environment. The `cpu` and `ampere` extras are mutually exclusive; the default `cpu` dependency group activates the CPU extra.

Restore the frozen Radio Browser file to `data/raw/radio-browser-geolocated.json` to use existing experiment artifacts. For a new catalog, download a fresh snapshot:

```bash
mkdir -p data/raw
curl --fail --location --output data/raw/radio-browser-geolocated.json 'https://de1.api.radio-browser.info/json/stations/search?has_geo_info=true&hidebroken=true&limit=100000&order=name'
uv sync --locked
uv run --locked -m come_a_little_bit_closer.storage.catalog
uv run --locked uvicorn come_a_little_bit_closer.web.app:create_app --factory --host 127.0.0.1 --port 8765
```

Open [localhost:8765](http://127.0.0.1:8765/). The initial globe has no query, origin, or selected model. Enter query text, select a model, and click the globe to set an origin. Station selection starts playback; search preserves the player and selected view. Both views support drag, wheel/trackpad/pinch zoom, and keyboard controls. Embedding view supports Shift/right-drag pan.

A fresh download will differ from the recorded snapshot. Rebuild its catalog and embeddings together; do not mix new data with frozen experiment caches. Raw snapshots, databases, vectors, and results are excluded from Git.

## Genre and distance

For `nearest rock station`, the query matcher resolves `rock` against the recorded MusicBrainz vocabulary. Both spatial rankings use the same genre-eligible stations: exact catalog tags after case, spacing, hyphen, and ampersand normalization. Longer genre names take precedence over their substrings. Multiple detected genres require an explicit station tag; a contradictory tag is rejected. The detected genre is shown in the result and recomputed for each query.

The main lists compare Haversine and full-dimensional cosine on that shared candidate pool. Mean distance, excess mean distance, and top-k overlap measure spatial ranking. **Unfiltered genre match** separately measures the fraction of the model's raw top-k stations with matching catalog tags; its raw neighbors are available under **Before genre filtering**. This is a tag agreement measure, not an audio or human relevance judgment.

An unrecognized genre is shown as **All stations · no genre detected**. A recognized genre with no candidates produces empty spatial rankings. Matching is lexical; it does not infer taxonomy parents, synonyms, negation, or complex intent. The coordinate control uses unit-coordinate vectors rather than a learned encoder.

PCA and deterministic spherical k-means use the full station corpus. The projection frame and cluster colors remain fixed across queries. Cosine retrieval uses original vectors, not projected distances. The globe always displays the full catalog independently of genre eligibility. Natural Earth geometry is unmodified; map colors are schematic.

## Embeddings and hardware

The CPU environment supports all five text encoders. Build one cache with:

```bash
uv run --locked -m come_a_little_bit_closer.models.batch --model qwen3
uv run --locked -m come_a_little_bit_closer.models.inspect
```

GPU use requires the Ampere dependency selection and the explicit `--ampere` command flag:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ampere uv run --locked --no-group cpu --extra ampere -m come_a_little_bit_closer.models.batch --model qwen3 --ampere
```

The Ampere extra selects CUDA 12.9 wheels on Linux, matching the other projects. The CPU path selects CPU-only wheels. GPU jobs use a separate `.venv-ampere` so synchronization cannot replace the running CPU application's dependencies. Environment synchronization remains enabled on every run. Cached checkpoints are pinned in `models/catalog.py`; query encoders reuse the cache's revision and native prompts. Gemma checkpoint access may require the model's access grant.

Artifacts include station IDs, unit vectors, source checksum, checkpoint revision, document format, tokenizer limit, and encoder settings. Qwen/BGE use 1,024 dimensions, E5/Gemma 768, and CLAP 512. Loading rejects incompatible provenance. CLAP audio embeddings, audio-derived genre labels, and human annotation remain pending.

## Experiments

The primary frozen plan is [location-v2](experiments/location-v2.json): five models, 45 genres, twelve origins, and three geographic-block splits. The original six-tag pilot is retained.

- [Protocol](experiments/LOCATION_PROTOCOL.md): probes, controls, counterfactuals, and evaluation.
- [Expanded plan notes](experiments/LOCATION_V2.md): genre coverage and native-query amendment.
- [Findings](experiments/LOCATION_FINDINGS.md) and [run record](experiments/LOCATION_RUNS.md).
- [Explorer development record](experiments/EXPLORER_RUNS.md).
- [Remaining work](TODO.md).

Frozen final-embedding probes, country/global centroid controls, permutations, random features, origin counterfactuals, tokenizer audits, and retrieval evaluation are complete. Audio, human labels, buffered/family splits, and internal-layer probes are pending. Linear recovery does not establish causal use. The new web genre matcher and hardware defaults postdate the recorded experiment; they do not replace its frozen candidate pools or results.

Regenerate the report after restoring its frozen inputs and output files:

```bash
uv run --locked --extra research -m come_a_little_bit_closer.evaluation.location_report --config experiments/location-v2.json
```

Acquisition metadata is in `data/manifest.json`; exact experiment checksums remain in the frozen plans. The live API is not an archive. [Batch execution](ops/README.md) describes the compute-only remote boundary and Ampere commands.

## Development

```bash
uv run --locked ruff check src tests
uv run --locked black --check src tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked pytest -q
node --check src/come_a_little_bit_closer/web/static/explorer.js
uv build --no-sources
```

GitHub checks use CPU workers. Spatial checks validate H3 round trips, antimeridian candidates, exact-distance reranking on a tiny fixture, and both GeoJSON files' checksums, coordinate ranges, and ring closure. The model check constructs a one-layer, 16-dimensional random BERT locally and exercises the production tokenizer, encoder, normalization, and artifact round trip without downloading weights. It is an infrastructure fixture, not a semantic baseline.

The repository uses `master` as its base branch. Create feature branches with `git checkout -b feature-name`. API documentation is served at `/docs`; the frontend contract is in `web/UI_CONTRACT.md`. Vendored sources, licenses, and checksums are in `web/static/vendor/ASSETS.md`.

## Paper and formalization

`paper/final.tex` is the full paper draft (sections in `abstract.tex` through `discussion.tex`); `paper/proposal_extended.tex` is the eight-page TMLR proposal; `paper/basic_math.tex` records the mathematical setup. Compile from `paper/` with `pdflatex -interaction=nonstopmode -output-directory=build final.tex`, then `bibtex build/final`, then `pdflatex` twice (latexmk is not installed on BC Andromeda; locally, `latexmk -pdf -outdir=build final.tex` also works).

The Lean 4.19.0 / Mathlib v4.19.0 project formalizes geometry, ranking, losses, projection, eligibility, and evaluation definitions. From `formal/`, run `lake build ComeCloser`; a fresh checkout first needs `lake update` and `lake exe cache get`. These proofs do not verify the numerical solver, learned encoders, or H3 implementation.
