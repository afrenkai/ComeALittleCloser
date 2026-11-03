# Layerwise probe dev run record

All runs used BC Andromeda (Slurm), partition `gtml` unless noted. `gtml` rejects CPU-only submissions, so probe jobs, which need only CPUs, each held one L4 GPU (`--gres=gpu:l4:1 --cpus-per-task=12`). Two jobs were submitted to `short` and cancelled while pending (more than 10 minutes). Only `dev` mode has been run; **`final` mode has not been run**, and no held-out test results exist.

Plan: `experiments/layers-v1.json`. Two plan hashes apply, both verified from git (`git show f01ee44~1:experiments/layers-v1.json | sha256sum` and `git show f01ee44:experiments/layers-v1.json | sha256sum`):

- `c83faa375b7dca66eb6ec3ce20f016ada6fa52bc8042faa372fdae52852cabc9` (batch_size 256, before commit `f01ee44` at 2026-11-02 19:06:42 -0500): extraction jobs 3290605 (xlmr-base), 3290645 (xlmr-base-random), 3290646 (e5-base), 3290647 (xlmr-large), 3290648 (bge-m3-retromae), 3290649 (bge-m3-unsupervised), 3290650 (bge-m3), plus the failed qwen3-0.6b jobs 3290642-3290644.
- `cce8a14c7f24354ae6181d83faea11f75b63a8f817b49ebbba9ab61b1653a025` (batch_size 32, commit `f01ee44` onward): extraction jobs 3290735-3290739 and every dev probe job (3290714, 3290748, 3290754-3290764).

Batch size is not recorded in the feature manifests and does not change the features (padding is masked). All extraction jobs ran on one L40S GPU (node grs001, partition `gtml`).

Provenance gap and fix: the dev probe jobs above predate plan and split records. Plan copies (`outputs/layers-v1/<mode>/<model>.plan.json`) and split manifests (`outputs/layers-v1/<mode>/splits/<dataset>-<split_kind>-<seed>.json`) are written by runs after commit `241b625`; the controller will append the rerun job IDs here.

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
