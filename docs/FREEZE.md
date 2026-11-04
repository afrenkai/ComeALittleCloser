# Test-run freeze list (approved 2026-11-04)

**Status: approved by Tian-hao Zhang on 2026-11-04 with every recommendation below: cell rule as written, R² as option (c), `xlmr-xl` included, H4 pooling stated as a limitation, H6 added. Frozen in `experiments/layers-v7.json` (169 `final_cells`, SHA-256 `81c161719c1a6e0a37ee9b555356a7c53bec0b14b2e3e38c3c44ba6c1766d0e1`).** The held-out test split had not been touched when this was approved. Nothing on this list changes after the final run.

## Why freeze first

Every number so far is validation. The best layer was picked on the same validation places it was scored on, so those numbers are optimistic. The test run fixes this: layers chosen on validation are scored once on unseen places. If we pick layers or settings after seeing test numbers, the test is spent and the numbers are optimistic again.

## Proposed cells (rule, not hand-picked)

For every model, dataset (cities `name`, cities `name-country`, stations `metadata`) and pooling (mean; also last-token for decoders), freeze two layers:

1. **The best validation layer:** the highest seed-mean `country-residual` gain on the random-item split (layers-v4 dev). This tests whether the validation optimum holds up.
2. **The last layer:** where the retrieval models' output vector comes from.

This is the same rule used for `prediction_cells` in `experiments/layers-v5.json`: 164 cells for the 18 models. Applied to `xlmr-xl` (layers-v6 dev), it gives 5 more: cities `name` layer 36 (best and last coincide), cities `name-country` layers 32 and 36, stations `metadata` layers 3 and 36, all mean pooling. That makes 169 cells in total. Each cell is scored on all three targets (`sphere`, `latlon`, `country-residual`), both splits and three seeds. Final mode also writes per-block scores and per-place predictions for every cell, so the per-region views (V2, V3), the decodable-vs-reflected correlation and the popularity analysis can all be run on test data without a second run.

The full per-model list is in `experiments/layers-v5.json` under `prediction_cells`. For encoders with no signal, the "best" layer is effectively arbitrary (often layer 0 or 1). That is harmless: their claim is "near zero" at every frozen layer.

## Claims the paper would make, and what counts as support

Each headline claim names its cell and a pass criterion before we see the test.

| # | Claim | Cells | Supported if (test split) |
|---|---|---|---|
| H1 | Decoders from 4B up hold within-country geography | best layer, `country-residual`, cities `name-country`, mean pooling; Qwen3-4B/8B, Mistral, Gemma (base and retrieval) | Gain over the country centroid has a 95% CI above 0 on both splits |
| H2 | Retrieval training keeps it | same cells, base vs. retrieval pairs | Retrieval gain CI above 0 in every pair. "Improves" or "removes" is claimed only if the **paired** base-minus-retrieval block-bootstrap CI excludes 0 (computed from the per-block test scores) |
| H3 | MLM encoders up to 3.5B show none, so the encoder result is not just size | encoders incl. `xlmr-xl`, same cells | Gain CI includes 0 or is under 20 km for XLM-R base/large/XL and their retrieval models. On validation XL already shows none (5.8 km, CIs span 0; layers-v6). If XL shows gain on test, the claim falls back to "encoders up to 0.56B" and size stays a confound |
| H4 | The retrieval models' own output vector carries geography beyond the country | last layer, `sphere`, cities `name-country`, random-item; last-token pooling for decoder retrieval models | Gain over the country centroid has a CI above 0 |
| H5 | Lat/lon R² mostly reflects knowing the country | best layer, `latlon`; country-centroid and random-init rows | Country-centroid R² and random-init R² reported beside each model's R² |
| H6 | The within-country signal is not limited to famous places | H1 cells, cities only; `evaluation.layer_popularity --mode final` | For every H1 model, the gain CI for the smallest population quintile is above 0. On validation this already holds for all eight 4B+ decoder models on both splits; the largest quintile gains about 1.5-3 times more (`experiments/LAYERS_RUNS.md`, layers-v5) |

Everything else in the test output (other conditions, stations, other layers) is reported as secondary and labelled exploratory.

## Decisions (all approved as recommended)

1. **The cell rule above:** best validation layer plus last layer, for all models. Alternative: fewer cells, headline models only. Fewer cells mean fewer comparisons but less coverage.
2. **How R² is computed.** The raw-output R² that Gurnee & Tegmark use is fragile: in Qwen3-4B-Embedding one city ("Carcassonne, France", an activation outlier) sends R² from 0.84 to -0.14 (see `experiments/LAYERS_RUNS.md`, layers-v5). Options:
   - (a) raw only;
   - (b) R² after clipping latitude to ±90 and longitude to ±180;
   - (c) both, with (b) as primary.

   Recommended: (c). This only changes the analysis of saved outputs, not the run.
3. **Include `xlmr-xl`?** Recommended yes. Its validation run finished (no within-country gain at any layer, see `experiments/LAYERS_RUNS.md`, layers-v6), so its 5 cells are ready to freeze. Because its runs live under layers-v6, the final run uses one plan (`layers-v7`) holding all 19 models, so every model shares one split file.
4. **Pooling caveat for H4:** BGE-M3 searches with its CLS token, but we extract only mean pooling for encoders. Either state this as a limitation, or add CLS extraction for encoders before freezing. Recommended: state the limitation.
5. **Add H6 (popularity)?** Recommended yes. It answers the obvious reviewer question ("it only knows big cities"), needs no new run (final mode writes the predictions it reads), and is fixed in advance like the others.

## Procedure after approval

1. Write `experiments/layers-v7.json` with the approved `final_cells`. Commit it, and record its SHA-256 and the commit in `experiments/LAYERS_RUNS.md`.
2. Run `--mode final` once per model (Slurm, one job each). The runner refuses a second run, and refuses a split file that differs from the first model's.
3. Compute H1–H5 exactly as written above, then the per-region and popularity analyses.
