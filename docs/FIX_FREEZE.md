# Fix freeze list (approved 2026-11-04)

**Status: approved by Tian-hao Zhang on 2026-11-04** ("go ahead with the freeze and run the fix on test"). Written before the module's first development run and before any test output of this analysis exists. Exploration is in `experiments/FIX_RUNS.md`.

## The fix being tested

A coordinate-free combination applied to frozen retrieval models:

1. **Task instruction.** Every text is encoded with the model's native instruction format and "Given the name of a place, retrieve places that are geographically closest to it". This was the first wording tried; two alternatives were explored and not adopted.
2. **Layer fusion.** Add, with weight 1, the cosine of an intermediate layer, centred on training cities. The layer is chosen on 8,000 training cities by within-country kNN gain.
3. **Distance probe.** Subtract 0.1 × the normalised squared distance in a rank-32 linear space trained on training cities with a triplet loss (3,000 steps).

All settings are fixed in `experiments/fix-v1.json` and were chosen during exploration on validation cities and stations. Nothing is tuned on test data.

## What is run

`experiments/fix-v1.json` equals `experiments/fix-v1-dev.json` except `layers_plan` (`experiments/layers-v7.json`), `experiment_id` and `output_directory` (`outputs/fix-v1`). Command, once:

`uv run --locked -m come_a_little_bit_closer.evaluation.geo_fix --config experiments/fix-v1.json --mode final`

- **Cities:** the test cities of the layers random-item and spatial-block splits, seeds 1729, 2718 and 3141. Training cities are used for fitting.
- **Semantic cost:** the STS-Benchmark test split (1,379 pairs; `mteb/stsbenchmark-sts` at revision `96943a16ea6a35129e253c659081cb59daf81b30`; SHA-256 in the plan). Spearman correlation of the score with human similarity.
- **Stations:** all 11,833, for transfer and genre agreement. They were used during selection, so they are reported but not used for claims.

The test cities were already used for the probe claims (H1-H6) and the reflected-geography claims (R1-R4). These are new claims about the same held-out cities, fixed before this analysis reads them.

## Claims

| # | Claim | Supported if |
| --- | --- | --- |
| F1 (primary) | The fix makes cosine-style retrieval reflect within-country geography better | For e5-mistral and bge-multilingual-gemma2: the full fix minus plain cosine, city within-country kNN gain, paired by country, has a 95% bootstrap CI above 0 in all 12 runs (2 models × 2 splits × 3 seeds) |
| F2 (primary) | without a meaningful cost to general semantic similarity | For the same two models: STS-Benchmark test Spearman of the full fix is at least plain cosine's minus 0.03, in all 12 runs |
| F3 (descriptive) | Components, baselines, and the third model | Report instruction alone, fusion plus probe alone, the country-code baseline, the geocoder oracle, Qwen3-Embedding-8B, station transfer and genre agreement, by split |

The 0.03 tolerance on STS Spearman was set before the module's first STS measurement on the dev split.
