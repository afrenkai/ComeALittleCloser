# Layerwise geographic probes: base versus retrieval-tuned encoders

Date: 2026-11-02. Status: approved in conversation; pending written-spec review.

## Question

Retrieval fine-tuning starts from a base language model. Does it keep, amplify, or discard the geographic information that the base model's intermediate layers carry? For every layer of each base model and of the retrieval model fine-tuned from it, we measure how well a linear probe recovers a place's coordinates and whether the probe beats a training-only country-centroid control.

The earlier location experiment (`experiments/LOCATION_FINDINGS.md`) found that final-layer metadata-only probes do not beat the country control. This experiment asks where in the network that changes, and whether retrieval training causes it.

## Model families

Each family is ordered by training stage. All checkpoints are pinned to an immutable revision.

| Family | Stage | Checkpoint | Revision |
| --- | --- | --- | --- |
| qwen3-0.6b | base | Qwen/Qwen3-0.6B-Base | da87bfb608c14b7cf20ba1ce41287e8de496c0cd |
| qwen3-0.6b | retrieval | Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |
| qwen3-4b | base | Qwen/Qwen3-4B-Base | 906bfd4b4dc7f14ee4320094d8b41684abff8539 |
| qwen3-4b | retrieval | Qwen/Qwen3-Embedding-4B | 5cf2132abc99cad020ac570b19d031efec650f2b |
| xlmr-base | base | FacebookAI/xlm-roberta-base | e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 |
| xlmr-base | retrieval | intfloat/multilingual-e5-base | d128750597153bb5987e10b1c3493a34e5a4502a |
| xlmr-large | base | FacebookAI/xlm-roberta-large | c23d21b0620b635a76227c604d44e43a9f0ee389 |
| xlmr-large | long-context pretraining | BAAI/bge-m3-retromae | 95c4f81ef8f7911dcac8e384532489c32c8dea64 |
| xlmr-large | unsupervised contrastive | BAAI/bge-m3-unsupervised | 46f03bc86361cf88102b0b517b36c8259f2946b1 |
| xlmr-large | retrieval | BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 |

Lineage sources: the Qwen3-Embedding model card lists Qwen3-0.6B-Base as its base model, the Qwen3-Embedding-4B card lists Qwen3-4B-Base, the multilingual-E5 card states initialization from xlm-roberta-base, and the BGE-M3 card describes the XLM-R-large → RetroMAE → unsupervised → fine-tuned sequence. EmbeddingGemma 2 is excluded: it has no public base checkpoint with a stated lineage.

Two random-initialization controls use the qwen3-0.6b and xlmr-base architectures with weights drawn from their configs' default initializers at a fixed seed. They separate learned geography from what tokenization and architecture alone allow a probe to decode.

## Datasets

1. **Cities**: GeoNames `cities15000`, already recorded in `data/manifest.json`. Country names come from GeoNames `countryInfo.txt`, which is added to the manifest with its checksum. Two text conditions:
   - `name`: the city name, e.g. `Lyon`.
   - `name-country`: `Lyon, France`.
2. **Stations**: the 11,833-station catalog, using the existing coordinate-free `metadata_document` (name, country code, tags).

Inputs are identical across every model: no retrieval instructions or prompts, so base and tuned models see the same tokens. Native retrieval prompts can be added later as a separate condition.

## Features

For each model, dataset, condition, and layer (embedding output plus every transformer block), the extractor stores:

- `mean`: the attention-masked mean of hidden states over input tokens, excluding special tokens.
- `last`: the final non-padding token state, for decoder models only.

Features are stored as float16 arrays shaped `[layer, item, hidden]`, one file per (model, dataset, condition, pooling) under `data/processed/layers-v1/features/`, with a manifest that records the checkpoint, revision, source checksum, item IDs, text condition, pooling, dtype, and the transformers version. Probes upcast to float64. The largest family (Qwen3-4B, 37 layers × 34k cities × 2,560 dimensions) needs about 6.5 GB per file; the full grid fits comfortably under 100 GB on `/projects`.

## Probes and controls

Probes reuse the existing ridge implementation: 3D unit-vector targets, the plan's alpha grid selected on validation blocks, angular error in kilometres, and block means.

Two split kinds, each with three seeds:

- `spatial-block`: the existing 10° occupied-block split (60/20/20). It measures generalization to unseen regions.
- `random-item`: an item-level random split (60/20/20). Countries appear in training, so the country control is strong; this measures within-country detail. Errors are still averaged within the same 10° cells before bootstrapping, so dense regions do not dominate either split kind.

For each (model, dataset, condition, pooling, layer, split, seed), the run reports the probe's block-mean error and its paired gain over the training-only country centroid, with bootstrap intervals over blocks. Controls per cell: shuffled training targets, and the random-initialization models.

**Dev and final modes.** `dev` scores validation blocks and is used for every exploratory figure and choice, including layer selection. `final` scores test blocks for the cells frozen in `experiments/LAYERS_RUNS.md` and runs once. Every run, including failed ones, is recorded there.

## Code layout

New and changed modules, following the existing domain layout:

- `data/cities.py` and `data/schemas.py`: GeoNames city and country loading into a `Place` record (ID, name, country code, country name, coordinates, population).
- `models/hidden_states.py` and `models/schemas.py`: a layer feature extractor built on `transformers.AutoModel` with `output_hidden_states=True`, plus the feature manifest and `LayerModelSpec` registry entries (family, stage, checkpoint, revision, architecture kind).
- `evaluation/layer_schemas.py`: `LayerPlan` (families, datasets, conditions, poolings, split kinds, seeds, alphas, bootstrap settings, directories) and `LayerProbeScore`.
- `evaluation/layer_run.py`: loads features, builds splits, fits and scores probes, and writes JSON lines of scores.
- `evaluation/layer_report.py`: figures and a Markdown summary.
- Targeted refactor: `fit_ridge`, `bootstrap_interval`, and `score_probe` take a small `ProbeSettings` schema (alphas, bootstrap seed and replicates, Earth radius) instead of the whole `LocationPlan`, and `score_probe` takes the split to score. `spatial_split` takes coordinates and IDs instead of `Station` records. `LocationPlan` exposes its `ProbeSettings`, so the location experiment's behavior and outputs are unchanged.
- `experiments/layers-v1.json`: the frozen plan.
- `ops/layers-extract.sbatch` (GPU, one model per array task) and `ops/layers-probe.sbatch` (CPU, after extraction).

## Figures

1. **Where geography dies**: probe gain over the country control against relative layer depth, one panel per family, with stages overlaid, the zero line as the country control, and the random-init band.
2. **Distortion map**: for the best validation layer of each family's base and retrieval stages, arrows from each test city's true position to its predicted position on an equirectangular world frame.
3. **Training-stage trajectory** (xlmr-large): the best-layer gain across the four BGE-M3 stages.

## Testing

Focused tests only: masked mean pooling ignores padding and special tokens, last-token pooling picks the final real token under left and right padding, the `random-item` split is disjoint and deterministic, `ProbeSettings` reproduces the existing location probe scores on a fixed fixture, and the feature manifest rejects mismatched revisions or item IDs. The tiny random BERT used by `tests/test_cpu_model.py` serves as the offline extractor fixture.

## Out of scope

Native retrieval prompts, nonlinear probes, causal interventions, interactive explorer integration of layer results, and Lean formalization beyond existing definitions. The metrics reuse the geometry and probe definitions already mirrored in `formal/`.
