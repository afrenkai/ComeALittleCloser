# Layerwise probe dev run record

All runs used BC Andromeda (Slurm), partition `gtml` unless noted. `gtml` rejects CPU-only submissions, so probe jobs, which need only CPUs, each held one L4 GPU (`--gres=gpu:l4:1 --cpus-per-task=12`). Two jobs were submitted to `short` and cancelled while pending (more than 10 minutes). Only `dev` mode has been run; **`final` mode has not been run**, and no held-out test results exist.

Plan: `experiments/layers-v1.json`. Two plan hashes apply, both verified from git (`git show cc497ec~1:experiments/layers-v1.json | sha256sum` and `git show cc497ec:experiments/layers-v1.json | sha256sum`):

- `c83faa375b7dca66eb6ec3ce20f016ada6fa52bc8042faa372fdae52852cabc9` (batch_size 256, before commit `cc497ec` at 2026-11-02 19:06:42 -0500): extraction jobs 3290605 (xlmr-base), 3290645 (xlmr-base-random), 3290646 (e5-base), 3290647 (xlmr-large), 3290648 (bge-m3-retromae), 3290649 (bge-m3-unsupervised), 3290650 (bge-m3), plus the failed qwen3-0.6b jobs 3290642-3290644.
- `cce8a14c7f24354ae6181d83faea11f75b63a8f817b49ebbba9ab61b1653a025` (batch_size 32, commit `cc497ec` onward): extraction jobs 3290735-3290739 and every dev probe job (3290714, 3290748, 3290754-3290764).

Batch size is not recorded in the feature manifests and does not change the features (padding is masked). All extraction jobs ran on one L40S GPU (node grs001, partition `gtml`).

Provenance gap and fix: the dev probe jobs above predate plan and split records. Plan copies (`outputs/layers-v1/<mode>/<model>.plan.json`) and split manifests (`outputs/layers-v1/<mode>/splits/<dataset>-<split_kind>-<seed>.json`) are written by runs after commit `d654c2a`. Dev probes for all 12 models were re-run at commit `03d5281` under plan `cce8a14c…a025` on gtml (L4 node grs002, 12 CPUs each): jobs 3291121 (xlmr-base), 3291122 (qwen3-0.6b-base), 3291123 (qwen3-0.6b-embedding), 3291124 (qwen3-0.6b-random), 3291125 (xlmr-base-random), 3291126 (e5-base), 3291127 (xlmr-large), 3291128 (bge-m3-retromae), 3291129 (bge-m3-unsupervised), 3291130 (bge-m3), 3291131 (qwen3-4b-embedding), 3291132 (qwen3-4b-base), all COMPLETED, writing 12 plan copies and 12 split manifests. Eleven JSONL files are byte-identical to the earlier runs. `xlmr-base.jsonl` differs only in floating-point rounding (largest relative difference 2e-12, in 330 of 486 rows; every non-float field, including selected alphas, is identical): its earlier run 3290748 used 16 CPUs, the rerun used 12, and BLAS reduction order depends on thread count. Headline numbers are unchanged at the reported precision.

## Jobs

Model-to-job assignments for the 3290642-3290650, 3290735-3290739, and 3290754-3290764 ranges follow submission order recorded by `sacct` (the sbatch argument for the probe jobs; submission order for the extraction ranges).

| Job | Role | Outcome | Elapsed | Node |
| --- | --- | --- | --- | --- |
| 3290570 | xlmr-base extraction smoke, max_length 64 (superseded; features deleted) | Completed | 0:34 | grs001 |
| 3290605 | xlmr-base extraction, max_length 256, batch_size 256, L40S | Completed | 0:35 | grs001 |
| 3290642 | qwen3-0.6b-base extraction, batch_size 256, L40S | Failed (CUDA OOM on station documents after cities files completed) | 1:40 | grs001 |
| 3290643 | qwen3-0.6b-embedding extraction, batch_size 256, L40S | Failed (same OOM) | 1:40 | grs001 |
| 3290644 | qwen3-0.6b-random extraction, batch_size 256, L40S | Failed (same OOM) | 1:37 | grs001 |
| 3290645 | xlmr-base-random extraction, batch_size 256, L40S | Completed | 0:34 | grs001 |
| 3290646 | e5-base extraction, batch_size 256, L40S | Completed | 0:31 | grs001 |
| 3290647 | xlmr-large extraction, batch_size 256, L40S | Completed | 0:55 | grs001 |
| 3290648 | bge-m3-retromae extraction, batch_size 256, L40S | Completed | 0:57 | grs001 |
| 3290649 | bge-m3-unsupervised extraction, batch_size 256, L40S | Completed | 0:57 | grs001 |
| 3290650 | bge-m3 extraction, batch_size 256, L40S | Completed | 0:56 | grs001 |
| 3290651 | qwen3-4b-base extraction on `short` | Cancelled while pending (moved to gtml) | 0:00 | none assigned |
| 3290652 | qwen3-4b-embedding extraction on `short` | Cancelled while pending (moved to gtml) | 0:00 | none assigned |
| 3290735 | qwen3-0.6b-base extraction, batch_size 32, L40S | Completed | 0:48 | grs001 |
| 3290736 | qwen3-0.6b-embedding extraction, batch_size 32, L40S | Completed | 0:47 | grs001 |
| 3290737 | qwen3-0.6b-random extraction, batch_size 32, L40S | Completed | 0:51 | grs001 |
| 3290738 | qwen3-4b-base extraction, batch_size 32, L40S | Completed | 7:54 | grs001 |
| 3290739 | qwen3-4b-embedding extraction, batch_size 32, L40S | Completed | 7:40 | grs001 |
| 3290679 | xlmr-base dev probe on `short` | Cancelled while pending (more than 10 minutes) | 0:00 | none assigned |
| 3290714 | xlmr-base dev probe smoke | Completed (superseded by 3290748) | 2:27 | grs002 |
| 3290748 | xlmr-base dev probe rerun after Task 5 fix | Completed; output byte-identical to 3290714 | 2:34 | grs002 |
| 3290754 | qwen3-0.6b-base dev probe | Completed | 14:11 | grs002 |
| 3290755 | qwen3-0.6b-embedding dev probe | Completed | 14:21 | grs002 |
| 3290756 | qwen3-0.6b-random dev probe | Completed | 14:11 | grs002 |
| 3290757 | xlmr-base-random dev probe | Completed | 2:40 | grs002 |
| 3290758 | e5-base dev probe | Completed | 2:42 | grs002 |
| 3290759 | xlmr-large dev probe | Completed | 6:15 | grs002 |
| 3290760 | bge-m3-retromae dev probe | Completed | 6:37 | grs002 |
| 3290761 | bge-m3-unsupervised dev probe | Completed | 6:41 | grs002 |
| 3290762 | bge-m3 dev probe | Completed | 6:16 | grs002 |
| 3290763 | qwen3-4b-embedding dev probe | Completed | 58:25 | grs002 |
| 3290764 | qwen3-4b-base dev probe (dependency on extraction) | Completed | 57:57 | grs002 |

Extraction used one L40S GPU on grs001 per job; the report ran on the login node (about 1.5 minutes, including the qwen3-4b distortion maps that load roughly 3 GB float16 features).

## Source checksums

- Radio Browser geolocated stations (`data/raw/radio-browser-geolocated.json`): `708d3ab081396dd674e0f8f91cfc7801f96882cf38db27ae6f6d3205891b62b0` (matches the stations catalog checksum; recorded as `source_sha256` in the `stations-metadata-mean` feature manifests).
- GeoNames `cities15000.zip`: `c7b7482b115463874798824f76381f84103233c0a9db9d43efe8501719200dc9`.
- GeoNames `countryInfo.txt`: `79f8f59ea01cbb40438bf1856dd6c2ff201f03f8fc72d0530b2646aaf603ef04`.
- The cities feature manifests record a single `source_sha256` of `3ad1ce0e98eabfd21e7815bf7523100b3b052a356020923eb8e87d04aa6e7956` (identical across the xlmr-base cities manifests); it binds the cities provenance to both GeoNames sources.

## What the run contains

Twelve models in five families: qwen3-0.6b (base, retrieval-tuned embedding, random init), qwen3-4b (base, embedding), xlmr-base (base, random init, e5-base), xlmr-large (base, bge-m3-retromae, bge-m3-unsupervised, bge-m3). Per-layer pooled features were extracted for cities (name; name-country) and stations (metadata) and stored under `data/processed/layers-v1/features/` (104 GB, gitignored). Dev probe outputs are `outputs/layers-v1/dev/<model>.jsonl` (gitignored); rows per file: qwen3-0.6b-* 2,106; qwen3-4b-* 2,682; xlmr-large and bge-m3-* 918; xlmr-base, xlmr-base-random, e5-base 486. Each file has probe and permuted-labels rows per layer and pooling, and country-centroid control rows, for two split kinds (spatial-block and random-item) and three seeds. The dev report is in `outputs/layers-v1/dev/report/`: `summary.md`, `where-geography-dies.png` (spatial-block), `where-geography-dies-random-item.png`, `stage-trajectory.png`, and `distortion-{qwen3-0.6b,qwen3-4b,xlmr-base,xlmr-large}.png`. All figures were viewed and are legible.

## Environment notes

Run with `source ops/bc.env` and `uv run --locked --extra research`. Qwen extraction at batch_size 256 ran out of GPU memory on station documents; the plan was re-frozen at batch_size 32 and the three 0.6b jobs plus both 4b jobs were rerun. The first xlmr-base smoke extraction at max_length 64 was superseded and its features deleted.

## Dev-mode headline numbers

**These are validation (dev) results. They are exploratory, not test results, unadjusted for multiple comparisons, and the best layer is selected on the same validation items, so best-layer gains are optimistic. Ridge alpha is also selected on the same validation items that dev mode scores (the country control has no tuned parameter), so dev gains are optimistic beyond layer selection.** `final` mode has not been run.

Cities, name-country text, mean pooling, probe. Gain is seed-mean km over the country-centroid control (positive means lower error than the centroid); brackets are seed-mean 95% block-bootstrap CI bounds. "Best layer" maximizes seed-mean gain. Permuted-labels gain is at the best layer (a reference; more negative means worse than the centroid). Random-init models (`*-random`) are the random-init references for their families. The spatial-block split holds out 10-degree regions; random-item holds out random cities. In this table the centroid gain is 0 by definition.

| model | split | best layer (mean pooling) | best gain km [CI] | final layer | final gain km [CI] | permuted-labels gain at best layer km |
|---|---|---|---|---|---|---|
| qwen3-0.6b-random | spatial-block | 0 | -1,206 [-1,756, -670] | 28 | -4,188 [-5,163, -3,248] | -4,950 |
| qwen3-0.6b-random | random-item | 0 | -1,441 [-1,790, -1,158] | 28 | -5,188 [-5,649, -4,722] | -6,094 |
| qwen3-0.6b-base | spatial-block | 15 | 128 [-470, 803] | 28 | -993 [-1,714, -255] | -4,943 |
| qwen3-0.6b-base | random-item | 19 | -831 [-1,071, -593] | 28 | -1,500 [-1,771, -1,245] | -6,107 |
| qwen3-0.6b-embedding | spatial-block | 11 | 144 [-403, 788] | 28 | -511 [-1,074, 80] | -4,951 |
| qwen3-0.6b-embedding | random-item | 19 | -772 [-987, -557] | 28 | -981 [-1,241, -746] | -6,108 |
| qwen3-4b-base | spatial-block | 20 | 841 [75, 1,724] | 36 | 319 [-476, 1,149] | -4,950 |
| qwen3-4b-base | random-item | 24 | -316 [-555, -55] | 36 | -525 [-784, -241] | -6,108 |
| qwen3-4b-embedding | spatial-block | 15 | 793 [-27, 1,723] | 36 | 131 [-693, 1,014] | -4,922 |
| qwen3-4b-embedding | random-item | 14 | -304 [-511, -66] | 36 | -522 [-762, -266] | -6,100 |
| xlmr-base-random | spatial-block | 0 | -1,524 [-2,070, -1,026] | 12 | -1,823 [-2,443, -1,223] | -4,942 |
| xlmr-base-random | random-item | 1 | -1,826 [-2,177, -1,522] | 12 | -2,054 [-2,436, -1,723] | -6,096 |
| xlmr-base | spatial-block | 0 | -957 [-1,487, -464] | 12 | -1,216 [-2,015, -506] | -4,947 |
| xlmr-base | random-item | 1 | -1,387 [-1,734, -1,106] | 12 | -1,527 [-1,877, -1,231] | -6,096 |
| e5-base | spatial-block | 12 | -124 [-624, 440] | 12 | -124 [-624, 440] | -4,931 |
| e5-base | random-item | 11 | -741 [-1,023, -488] | 12 | -756 [-1,048, -509] | -6,096 |
| xlmr-large | spatial-block | 1 | -744 [-1,264, -261] | 24 | -1,070 [-1,738, -427] | -4,950 |
| xlmr-large | random-item | 1 | -1,187 [-1,525, -912] | 24 | -1,365 [-1,703, -1,051] | -6,097 |
| bge-m3-retromae | spatial-block | 24 | -696 [-1,310, -117] | 24 | -696 [-1,310, -117] | -4,951 |
| bge-m3-retromae | random-item | 24 | -993 [-1,299, -750] | 24 | -993 [-1,299, -750] | -6,092 |
| bge-m3-unsupervised | spatial-block | 24 | 5 [-467, 587] | 24 | 5 [-467, 587] | -4,942 |
| bge-m3-unsupervised | random-item | 23 | -547 [-800, -311] | 24 | -569 [-852, -301] | -6,096 |
| bge-m3 | spatial-block | 22 | -194 [-732, 387] | 24 | -204 [-711, 393] | -4,929 |
| bge-m3 | random-item | 22 | -682 [-951, -447] | 24 | -763 [-1,076, -500] | -6,097 |

Observations limited to what the table shows: on the random-item split every best-layer gain is negative, with CIs excluding zero. On the spatial-block split best-layer gains are positive for qwen3-0.6b-base (CI crosses zero), qwen3-0.6b-embedding (crosses zero), qwen3-4b-base (CI above zero), qwen3-4b-embedding (CI crosses zero), and bge-m3-unsupervised (about 5 km, CI crosses zero); all others are negative. Final-layer gains are lower than best-layer gains for every row where the two layers differ. Cities name-only and stations rows are in `outputs/layers-v1/dev/report/summary.md` (best layer over pooling and layer) and in the JSONL files. The best layer in `summary.md` is chosen over both pooling and layer, so it can differ from the mean-pooling rows above.

## Before running final

1. Freeze the final cells (model, dataset, condition, pooling, layer) in both `experiments/layers-v1.json` `final_cells` and this file.
2. Record the resulting plan SHA-256 here.
3. Run `final` once per model (`ops/layers-probe.sbatch <model> final`) and record the job IDs here.

The run-once guard is per output file: do not delete or redirect outputs to rerun. A differing existing split file also blocks a final run. The report refuses `--mode final`; final results are reported explicitly from the frozen cells.

## layers-v2 (probe credibility)

Plan: `experiments/layers-v2.json`. It is `layers-v1.json` plus two extra probe targets, `latlon` (ridge onto latitude and longitude in degrees) and `country-residual` (ridge onto the offset from the training-only country centroid), and writes to `outputs/layers-v2`. Its purpose is to separate "the models lack geography" from "the sphere probe is weak" on the v1 dev results, and to report R-squared for latitude and longitude so results can be compared with Gurnee & Tegmark. It reuses the layers-v1 features (`data/processed/layers-v1/features`); nothing is re-extracted. Dev mode only.

Plan SHA-256: `1402669d36140ade8e0cccf1864001a30ba585c7e54648cc3e7cab7c9a47215c`. Dev probes ran on gtml (L4 node grs002, 12 CPUs each), all COMPLETED: 3291657 (qwen3-4b-base, 1:29:12), 3291658 (qwen3-4b-embedding, 1:30:01), 3291659 (qwen3-0.6b-base), 3291660 (qwen3-0.6b-embedding), 3291661 (qwen3-0.6b-random), 3291662 (xlmr-large), 3291663 (bge-m3-retromae), 3291664 (bge-m3-unsupervised), 3291665 (bge-m3), 3291666 (xlmr-base), 3291667 (xlmr-base-random), 3291668 (e5-base). Report job 3291864. The layers-v3 runs below reproduce all twelve layers-v2 JSONL files byte for byte, so layers-v2 results are reported once, in the layers-v3 section.

Caveats for reading layers-v2: the country-centroid control is unchanged from v1 (training-only centroid per country, global fallback). On spatial-block splits, when a country's main block is held out, its training centroid can come from a handful of distant stations (for example the Netherlands from a few overseas stations), so the control is weak there, especially for stations (country-centroid R-squared near 0.05 for latitude), and probe gains over it partly reflect fixing broken centroids. `summary.md` reports the share of evaluated items whose country has fewer than 10 training items. `latlon` R-squared is computed from raw ridge outputs in degrees; all other rows convert decoded unit vectors to latitude and longitude. `country-residual` is given the true country label, which is absent from `name` text. The `country-residual` ridge has no target intercept, so its high-alpha limit is the country centroid.

## layers-v3 (scale: Qwen3-8B pair)

Plan: `experiments/layers-v3.json`. It is `layers-v2.json` plus the Qwen3-8B base/retrieval pair, and writes to `outputs/layers-v3`. Its purpose is to test whether within-country geography keeps growing with scale across the Qwen3 0.6B, 4B and 8B families. It runs the full 14-model set on the layers-v1 features directory (`data/processed/layers-v1/features`); only the two new models need extraction. Dev mode only.

Pinned revisions and lineage:

- `qwen3-8b-base`: `Qwen/Qwen3-8B-Base` at `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` (stage base).
- `qwen3-8b-embedding`: `Qwen/Qwen3-Embedding-8B` at `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` (stage retrieval); its model card lists `Qwen/Qwen3-8B-Base` as `base_model`.

Plan SHA-256: `386088a69671eb413b7f5ef9ba42ca52f37254ba27e64059eda3cc345c7c37e5`. The 8B features were extracted with `experiments/layers-v3-extract-batch8.json` (SHA-256 `110f368b14eef10d76263c5aa3230a4b84e804d0e0a48766aae48f73634c471f`), which differs from layers-v3 only in `batch_size` 8; batch size should change features only through floating-point batching effects, and probe runs read the layers-v3 plan. The 8B features take 39 GB per model.

| Job | Role | Outcome | Elapsed | Node |
| --- | --- | --- | --- | --- |
| 3291703 | download Qwen3-8B base and embedding checkpoints (`short`) | Completed | 4:07 | c041 |
| 3291875 | qwen3-8b-base extraction, batch_size 32, L40S | Failed (CUDA OOM, 44 GB GPU) | 12:42 | grs001 |
| 3291876 | qwen3-8b-embedding extraction, batch_size 32, L40S | Failed (same OOM) | 12:23 | grs001 |
| 3291879-3291890 | 12 dev probes for the existing models (first submission) | Cancelled while pending: a `scontrol update` of mail settings on the queued jobs left them in BadConstraints | 0:00 | none assigned |
| 3291911, 3291912 | 8B extraction on `short` with vr80g/vr144g | Cancelled while pending (moved to batch 8 on gtml) | 0:00 | none assigned |
| 3291914 | qwen3-8b-base extraction, batch_size 8, L40S | Completed | 6:16 | grs001 |
| 3291915 | qwen3-8b-embedding extraction, batch_size 8, L40S | Completed | 6:04 | grs001 |
| 3291929 | qwen3-8b-base dev probe | Completed | 8:48:30 | grs002 |
| 3291930 | qwen3-8b-embedding dev probe | Completed | 8:48:30 | grs002 |
| 3295572-3295583 | resubmitted dev probes: qwen3-4b-base, qwen3-4b-embedding, qwen3-0.6b-base, qwen3-0.6b-embedding, qwen3-0.6b-random, xlmr-large, bge-m3-retromae, bge-m3-unsupervised, bge-m3, xlmr-base, xlmr-base-random, e5-base (in job-ID order), `--mem=32G` | All completed | 3:36-1:28:25 | grs002 |
| 3297183 | dev report (`layer_report --config experiments/layers-v3.json --mode dev`) | Completed | 0:47 | grs002 |

Rows per file: qwen3-4b-* and qwen3-8b-* 5,346; qwen3-0.6b-* 4,194; xlmr-large and bge-m3-* 1,818; xlmr-base, xlmr-base-random, e5-base 954. The report is in `outputs/layers-v3/dev/report/` (`summary.md`, `where-geography-dies{,-random-item,-residual}.png`, `r2-by-depth.png`, `stage-trajectory.png`, `distortion-*.png`).

### Dev-mode headline: country-residual gain

**Validation (dev) results, seed means, best layer and ridge alpha chosen on the same validation items, so gains are optimistic. `final` mode has not been run.** Cities, name-country text, `country-residual` probe; gain is km over the training-only country centroid (best layer over pooling and layer), brackets are seed-mean 95% block-bootstrap CI bounds.

| family | base, random-item | retrieval, random-item | base, spatial-block | retrieval, spatial-block |
|---|---|---|---|---|
| qwen3-0.6b | 12 [-6, 31] | 24 [1, 48] | 48 [8, 94] | 48 [2, 100] |
| qwen3-4b | 120 [68, 178] | 152 [90, 223] | 158 [73, 259] | 192 [83, 321] |
| qwen3-8b | 175 [110, 254] | 201 [128, 285] | 206 [97, 333] | 226 [102, 380] |
| xlmr-base (e5-base) | 1 [-2, 4] | 3 [-13, 21] | 19 [-4, 43] | 31 [-2, 72] |
| xlmr-large (bge-m3) | 1 [-3, 5] | 11 [-8, 32] | 8 [-19, 37] | 21 [-9, 54] |

Random-init references: qwen3-0.6b-random 1 [-1, 2] and 4 [-4, 12]; xlmr-base-random 1 [-2, 4] and 1 [-3, 5] (random-item, spatial-block).

Observations limited to what the tables show: within the Qwen3 decoders, residual gain grows with size (0.6B, 4B, 8B) on both splits, and in each size the retrieval model's point estimate is at or above its base model's, with overlapping CIs. Encoder families stay within about 30 km of the centroid with CIs that include or nearly include zero. The same ordering holds on the `name` condition (the probe is given the true country, which is absent from the text): qwen3-8b-embedding 181 [116, 256] and qwen3-8b-base 88 [39, 145] on random-item. On stations metadata (random-item) the 8B pair gives 170 [83, 264] (base) and 179 [82, 287] (retrieval). The `sphere` probe now beats the country centroid for the 4B and 8B retrieval models with last-token pooling (their native embedding vector) on random-item: qwen3-4b-embedding 231 [30, 472], qwen3-8b-embedding 311 [89, 565]; every mean-pooled random-item `sphere` row is still negative. Latlon R-squared rises with size (cities, name, random-item, latitude: 0.452, 0.625, 0.671 for the 0.6B, 4B, 8B base models). On name-country text it is close to the country-centroid reference (random-item latitude 0.953): 0.957 and 0.959 for the 4B and 8B retrieval models, with longitude below the reference (0.972 and 0.975 against 0.981).

## layers-v4 (second and third decoder families: Mistral-7B and Gemma-2-9B pairs)

Plan: `experiments/layers-v4.json` (SHA-256 `f6fdcc8aca625c18556350762c54815efa208a92f42d4132cef7596e20c49559`), which is `layers-v3.json` plus two base/retrieval pairs, written to `outputs/layers-v4`. New models were extracted with `experiments/layers-v4-extract-batch8.json` (SHA-256 `094dceac8276a15f771fea15b44cbc2a09eb33506365388930ac651a4602ce5f`; differs only in `batch_size` 8). All 18 models are probed under the v4 plan so every dev result traces to one plan. Code at commit `0c87d8b` adds the per-model `start_token` (Mistral `<s>`, Gemma `<bos>`, matching each model's own tokenizer) and `attention_implementation` (`eager` for Gemma-2, whose default SDPA path skips attention-logit soft-capping); both are unset for the earlier models, whose features and outputs are unchanged.

Pinned revisions and lineage (from the model cards):

- `mistral-7b-base`: `mistralai/Mistral-7B-v0.1` at `27d67f1b5f57dc0953326b2601d68371d40ea8da`.
- `e5-mistral`: `intfloat/e5-mistral-7b-instruct` at `d62cc4417be0e3289c03dd459d1ef2d137e2554e`; card: "initialized from Mistral-7B-v0.1".
- `gemma2-9b-base`: `google/gemma-2-9b` at `33c193028431c2fde6c6e51f29e6f17b60cbfac6` (gated; license accepted on 2026-11-03).
- `bge-multilingual-gemma2`: `BAAI/bge-multilingual-gemma2` at `8365500e19bc804136f4ad12c2494430239a2d72`; card: trained "based on google/gemma-2-9b".

Jobs (submitted 2026-11-03):

| Job | Role | Outcome | Elapsed | Node |
| --- | --- | --- | --- | --- |
| 3297324 | download Mistral pair (`short`) | Completed | 7:57 | c052 |
| 3297447 | download Gemma pair (`short`) | Completed | 33:20 | c066 |
| 3297458 | mistral-7b-base extraction, batch 8, L40S | Completed | 18:25 | grs001 |
| 3297459 | e5-mistral extraction, batch 8, L40S | Completed | 17:45 | grs001 |
| 3297460 | gemma2-9b-base extraction, batch 8, L40S | Completed | 21:37 | grs001 |
| 3297461 | bge-multilingual-gemma2 extraction, batch 8, L40S | Completed | 20:43 | grs001 |
| 3297462-3297475 | dev probes for the 14 earlier models (qwen3-8b-base, qwen3-8b-embedding, qwen3-4b-base, qwen3-4b-embedding, qwen3-0.6b-base, qwen3-0.6b-embedding, qwen3-0.6b-random, xlmr-large, bge-m3-retromae, bge-m3-unsupervised, bge-m3, xlmr-base, xlmr-base-random, e5-base, in job-ID order) | All completed (8B pair 7:54 and 8:11; others 0:04 to 1:30). All 14 JSONL files are byte-identical to layers-v3 | | grs002 |
| 3297476 | mistral-7b-base dev probe | Completed | 7:38:41 | grs002 |
| 3297477 | e5-mistral dev probe | Completed | 7:37:47 | grs002 |
| 3297478 | gemma2-9b-base dev probe (pending on Resources until 17:28) | Completed | 3:59:49 | grs002 |
| 3297479 | bge-multilingual-gemma2 dev probe | Completed | 4:13:11 | grs002 |
| 3306020 | dev report (`layer_report --config experiments/layers-v4.json --mode dev`), 8 CPUs, 48 GB | Completed | 1:11 | grs002 |

Extracted features: about 41 GB per Mistral model and 47 GB per Gemma model. The report (all 18 models present) is in `outputs/layers-v4/dev/report/`; the figures and `summary.md` are copied to `docs/figures/layers-v4/`.

### Dev-mode results for the new pairs

Country-residual gain over the country centroid (km, best layer by gain, seed mean, [95% CI]; validation):

| Model | cities name-country, random-item | cities name-country, spatial-block | stations, random-item | stations, spatial-block |
| --- | --- | --- | --- | --- |
| mistral-7b-base | 248 [170, 334] | 270 [131, 437] | 155 [68, 253] | 296 [78, 579] |
| e5-mistral | 304 [218, 400] | 290 [158, 445] | 249 [125, 404] | 448 [139, 865] |
| gemma2-9b-base | 303 [214, 405] | 321 [170, 500] | 169 [85, 261] | 304 [57, 638] |
| bge-multilingual-gemma2 | 285 [203, 375] | 307 [169, 467] | 185 [80, 311] | 330 [70, 680] |

Observations limited to the tables: both new families show within-country geography larger than Qwen3-8B (175 base, 201 retrieval on cities name-country random-item). The retrieval model is above its base in every Mistral cell and in the Gemma stations cells, and 14-18 km below its base in the Gemma cities name-country cells; every base/retrieval CI pair overlaps, so the result is "retrieval training keeps it" for both families, as for Qwen3. Absolute `sphere` probes (best layer over pooling, cities name-country, random-item) are above the country centroid for all four, with the CI crossing zero only for mistral-7b-base: mistral-7b-base 155 [-37, 397], e5-mistral 423 [201, 689], gemma2-9b-base 349 [127, 607], bge-multilingual-gemma2 459 [246, 713]. On stations random-item, no `sphere` probe of the four beats the centroid (e5-mistral -24 [-280, 271], bge-multilingual-gemma2 -169 [-445, 115]; both bases below with CIs under zero). Latitude/longitude R-squared for both families sits near the country-centroid line at every layer, without the Qwen3-4B-Embedding instability.

## layers-v5 (per-block scores and per-place predictions)

Plan: `experiments/layers-v5.json` (SHA-256 `d3e625af25fb9821f435dcad06fff931359162fd7724a5139e651d723462f1a5`). It is `layers-v4.json` with output `outputs/layers-v5` and 164 `prediction_cells`. Code at commits `7cd4cf5` and `f65bc6f` (branch `research/per-place-predictions`) makes every run also write:

- `outputs/<plan>/<mode>/blocks/<model>.jsonl`: for every scored row (probe and permuted-labels controls), each evaluated 10-degree block's item count, mean probe error (km) and mean country-centroid error (km). These feed the layer-by-region view (V2) and the decodable-vs-reflected correlation.
- `outputs/<plan>/<mode>/predictions/<model>/<dataset>-<condition>-<pooling>-<layer>.npz`: per-place predicted coordinates, raw ridge outputs, country-centroid prediction, errors and block IDs for every split, seed and probe target (probe control only). Dev mode writes these for `prediction_cells`; final mode writes them for every frozen final cell, so the one test run supports per-region analysis without a rerun.

The main `<model>.jsonl` is unchanged: the check job below reproduced `layers-v4` byte for byte. Prediction cells were chosen from layers-v4 dev results (validation only): for each model, dataset, condition and pooling, the last layer and the layer with the highest seed-mean `country-residual` gain on the random-item split.

Jobs (submitted 2026-11-03):

| Job | Role | Outcome | Elapsed | Node |
| --- | --- | --- | --- | --- |
| 3306536 | xlmr-base-random check at `7cd4cf5`, L4 | Completed; `xlmr-base-random.jsonl` byte-identical to layers-v4; blocks 10 MB, predictions 22 MB | 3:37 | grs002 |
| 3306557-3306574 | all 18 models, `--gres=gpu:l4:1` | Cancelled while pending, resubmitted below to use both gtml nodes | | |
| 3306575-3306592 | all 18 models at `f65bc6f`, `--gres=gpu:1` (qwen3-8b-base, qwen3-8b-embedding, mistral-7b-base, e5-mistral, gemma2-9b-base, bge-multilingual-gemma2, qwen3-4b-base, qwen3-4b-embedding, qwen3-0.6b-base, qwen3-0.6b-embedding, qwen3-0.6b-random, xlmr-large, bge-m3-retromae, bge-m3-unsupervised, bge-m3, xlmr-base, e5-base, xlmr-base-random, in job-ID order) | Completed 2026-11-04. All 18 `<model>.jsonl` byte-identical to layers-v4. Blocks 691 MB total, 164 prediction files 911 MB | 0:04-8:08 (qwen3-8b pair longest) | grs001, grs002 |
| 3306597 | latlon diagnosis (script path not visible from compute nodes) | Cancelled before start | | |
| 3306600 | latlon diagnosis for qwen3-4b-embedding (`outputs/diagnostics/diagnose_latlon.py`, `short`) | Completed | 1:26 | c020 |

Diagnosis of the unstable Qwen3-4B-Embedding latlon R-squared (validation, cities name-country, mean pooling, random-item): the collapse occurs only for seed 1729, whose validation set contains "Carcassonne, France". That item's mean-pooled hidden state has norm 776 at layer 6 against a median of 32 (an activation outlier), and the ridge maps it to raw outputs of about (-1866, 2863) degrees. One point drives the R-squared: at layer 6, R-squared is -0.14 / 0.66 (lat / lon) with it and 0.84 / 0.90 without it; at layer 20, 0.13 / 0.80 with it and 0.90 / 0.93 without it. Seed 2718 has no such item and is stable (0.84-0.90 / 0.90-0.93). Kilometre metrics clip latitude and wrap longitude, so the item adds one bounded error and block-mean km is barely affected. The largest remaining raw errors are Pacific islands near the antimeridian (e.g. Pago Pago, Papeete, Nasinu) whose longitude is predicted tens to 200 degrees off, the expected weakness of a raw-degree longitude target. Implication: raw-output R-squared is not robust to single activation outliers; how R-squared is computed for the test run is a decision for the freeze list.
| 3311634 | population analysis, first version as a throwaway script (`short`) | Completed; superseded by 3311647 (same numbers up to the bootstrap seed) | 0:14 | |
| 3311647 | `evaluation.layer_popularity --config experiments/layers-v5.json`, both splits (`short`) | Completed; `outputs/layers-v5/dev/popularity/{random-item,spatial-block}-country-residual.jsonl` | 0:45 | c036 |

Popularity analysis (validation, `country-residual`, cities `name-country`, mean pooling, prediction cells; seeds pooled; the CI is a block bootstrap over the 10-degree blocks within each quintile). The GeoNames population quintiles on random-item have edges of about 20k, 30k, 48k and 92k. A few entries record population 0, so Q1 runs 0-20k. Gain over the country centroid, best layer, Q1 (smallest) vs Q5 (largest):

| Model | random-item Q1 | random-item Q5 | spatial-block Q1 | spatial-block Q5 |
| --- | --- | --- | --- | --- |
| qwen3-4b-base | 60 [13, 114] | 181 [127, 245] | 103 [46, 167] | 166 [86, 251] |
| qwen3-8b-base | 120 [56, 199] | 237 [173, 308] | 136 [66, 210] | 213 [123, 321] |
| mistral-7b-base | 170 [94, 258] | 307 [230, 392] | 177 [90, 281] | 267 [161, 384] |
| gemma2-9b-base | 221 [140, 312] | 350 [265, 449] | 208 [124, 304] | 312 [216, 432] |
| qwen3-8b-embedding | 128 [66, 203] | 266 [199, 341] | 138 [74, 205] | 261 [147, 402] |
| e5-mistral | 155 [83, 242] | 319 [241, 405] | 120 [49, 208] | 277 [163, 407] |
| bge-multilingual-gemma2 | 188 [116, 274] | 347 [263, 442] | 185 [104, 273] | 290 [181, 421] |
| qwen3-0.6b-base | 8 [-9, 27] | 36 [8, 64] | 14 [-6, 37] | 46 [10, 86] |
| qwen3-0.6b-embedding | 2 [-14, 19] | 27 [2, 53] | 12 [-7, 32] | 35 [1, 72] |
| xlmr-large | 1 [-4, 6] | 1 [-3, 6] | -2 [-14, 9] | 11 [-1, 25] |

Every decoder from 4B up (base and retrieval) has a gain CI above 0 even for the smallest fifth of cities, and the largest fifth gains about 1.5-3 times more. The within-country signal is not limited to famous places, but popularity modulates it. The 0.6B decoders show gain only for the largest cities, consistent with knowing only well-known places. The encoders show none in any quintile at their best layer. At the last layer a few encoder Q5 intervals on spatial-block sit just above 0 (xlmr-large 17 [2, 34], bge-m3 13 [3, 25]); these are among many uncorrected comparisons. Caveat: quintiles differ in geography as well as fame (large cities concentrate in some countries), so this is descriptive, not causal.


## layers-v6 (encoder size control: XLM-R XL)

Plan: `experiments/layers-v6.json` (SHA-256 `2eabc1f1e33c0938de8756b0d43f4b359688278afe6cc48ba0d7e9505911880a`), which is `layers-v5.json` plus `xlmr-xl` (`facebook/xlm-roberta-xl` at `aa5d120255845efeebc9b7f42822a1dd0f9ece9d`, 3.5B parameters), written to `outputs/layers-v6`. Extraction plan `experiments/layers-v6-extract-batch8.json` (SHA-256 `3a706b6732e5cb209176d1e0a8a7dfbd5dd1c450ec2b2efd9da96cd1ab8f8145`; differs only in `experiment_id` and `batch_size` 8). Code at commit `7c9aa27`.

Purpose: every encoder in layers-v4 has at most 0.6B parameters and every decoder showing within-country geography has at least 4B, so "encoders show none" is confounded with size. XLM-R XL is a base-only control (no retrieval model is trained from it), registered as its own family; the registry test now allows single-stage base-only families. Only `xlmr-xl` needs to run under v6; the other 18 models are identical to layers-v5.

| Job | Role | Outcome | Elapsed | Node |
| --- | --- | --- | --- | --- |
| 3306604 | download `facebook/xlm-roberta-xl` safetensors (`short`) | Completed | 8:13 | c002 |
| 3306616 | xlmr-xl extraction, batch 8, `short` GPU (after 3306604) | Completed | 6:56 | g016 |
| 3306617 | xlmr-xl dev probe, gtml (after 3306616) | Completed; 2682 rows | 1:00:34 | grs002 |

Result (validation, `country-residual`, cities `name-country`, mean pooling, best layer by seed-mean gain; the CI range spans the three seeds' block-bootstrap CIs). Corrected 2026-11-04: the first version of this table averaged mean and last-token pooling for the two Qwen rows, which understated Qwen3-4B (74 km). The frozen cells were unaffected; they were chosen per pooling.

| Model | Params | cities name-country, random-item | cities name-country, spatial-block |
| --- | --- | --- | --- |
| xlmr-large | 0.56B | 1.2 km [-4.2, 11.1] (layer 3) | 8.2 km [-23.1, 57.0] (layer 15) |
| bge-m3 | 0.56B | 10.6 km [-10.2, 36.3] (layer 16) | 21.4 km [-11.1, 81.4] (layer 18) |
| xlmr-xl | 3.5B | 5.8 km [-8.1, 21.6] (layer 32) | 18.4 km [-13.4, 51.3] (layer 21) |
| qwen3-0.6b-base | 0.6B | 11.8 km [-9.7, 38.5] (layer 24) | 47.5 km [0.2, 159.4] (layer 28) |
| qwen3-4b-base | 4B | 119.6 km [61.8, 204.5] (layer 34) | 157.9 km [62.7, 315.5] (layer 35) |

A 3.5B masked-language-model encoder shows no within-country gain, the same as the 0.56B encoders, while a 4B decoder does. On validation, size alone does not explain the encoder result. The claim to test is "MLM encoders up to 3.5B show none", not "small models show none". Qwen3-0.6B shows little (12 km on random-item, CI spanning 0; 48 km on spatial-block), so among decoders size still matters. Because every best layer is chosen on validation, these gains are optimistic; the test run decides.

## layers-v7 (held-out test run)

Plan: `experiments/layers-v7.json` (SHA-256 `81c161719c1a6e0a37ee9b555356a7c53bec0b14b2e3e38c3c44ba6c1766d0e1`), written to `outputs/layers-v7`. It is `layers-v6.json` with 169 `final_cells`, all other settings identical (models, splits, seeds, alphas, bootstrap, features). The cells are the approved freeze list in `docs/FREEZE.md`: per model, dataset, condition and pooling, the best validation layer (highest seed-mean `country-residual` gain, random-item, layers-v4/v6 dev) and the last layer. That gives 164 cells for the 18 layers-v5 models and 5 for `xlmr-xl`. All 169 were checked against the feature manifests before freezing. The claims and pass criteria (H1-H6) were fixed in `docs/FREEZE.md` before any test data was scored. `outputs/layers-v7` did not exist when the plan was committed.

Final mode fits each probe on the training split, selects alpha on validation, and scores the test split once per model; the runner refuses a second run and a differing split file. Do not edit this plan; any change needs a new version and is reported as post hoc.

| Job | Role | Outcome | Elapsed | Node |
| --- | --- | --- | --- | --- |
| 3312021-3312039 | final (test) run, one job per model at `4a1ae04`, gtml `--gres=gpu:1` (qwen3-0.6b-base, qwen3-0.6b-embedding, qwen3-4b-base, qwen3-4b-embedding, qwen3-8b-base, qwen3-8b-embedding, mistral-7b-base, e5-mistral, gemma2-9b-base, bge-multilingual-gemma2, xlmr-base, e5-base, xlmr-large, bge-m3-retromae, bge-m3-unsupervised, bge-m3, qwen3-0.6b-random, xlmr-base-random, xlmr-xl, in job-ID order) | Submitted 2026-11-04 | | |
| 3312049 | dry run of `evaluation.layer_claims` on validation outputs (fixture in login-node `/tmp`, not visible to compute nodes) | Cancelled | | |
| 3312050, 3312166 | dry run of `evaluation.layer_claims` on validation outputs (layers-v5/v6 dev linked under the v7 cells; `outputs/diagnostics/claims-dry/`) | Completed; report runs end to end. 3312166 adds the both-splits H2 direction summary | 0:55 | |

Validation dry run of the frozen scoring (no test data read). On validation, H1, H2 and H6 pass. H4 fails for qwen3-4b-embedding only (seed lower bounds -1, -55, -10 km). H3 fails for xlmr-base and e5-base on spatial-block (one seed each above 0 with gain over 20 km: 21 [1, 41] and 32 [5, 58]). Spatial-block holds out whole 10-degree blocks, so the training-only country centroid is biased for held-out regions and even weak probes can beat it. The frozen criteria are not changed; the test decides.
