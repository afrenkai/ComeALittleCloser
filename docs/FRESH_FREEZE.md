# Fresh-towns freeze list (approved 2026-11-04)

**Status: approved by Tian-hao Zhang on 2026-11-04** ("approved"). Written before any town features existed and before the check part was scored; approved while the check jobs were still queued. The test part runs once, after the check part passes its pipeline checks. Runs are logged in `experiments/FRESH_RUNS.md`.

## Why

The original test cities have backed three frozen claim sets (layers-v7, reflected-v1, fix-v1) and are no longer a blind check. The fix-baselines-v1 findings (geographic wording, Feuerstein axes) are validation-only. These claims test them on places no analysis has used.

## Population

GeoNames `cities5000` (downloaded 2026-11-04; SHA-256 `38c3d7ee…65b79f`). We keep towns that meet all of these:

- they are absent from our `cities15000` snapshot (2026-11-02);
- they have 5,000-14,999 inhabitants;
- they have a country name;
- their "Name, Country" text is unique and differs from every `cities15000` text.

That leaves 32,148 towns (`town_items`, checksum `56cf653f…8d8442`, in `experiments/fresh-v1.json`).

This is a **different population** from the original cities: smaller and less well known. The claims are about generalization to these towns, not about cities of 15,000 or more people. Results are about places GeoNames lists, and its coverage varies by country.

## Partition

A fixed random permutation (seed 20261104) splits the towns:

- **Check part (20%, 6,430 towns):** pipeline checks only.
- **Test part (80%, 25,718 towns):** scored once.

## What is fixed

Nothing is fit, tuned or selected on towns. Every component is fit on the original training cities exactly as in fix-v1 and fix-baselines-v1:

- the fusion layer and the distance probe, per split and seed;
- the Feuerstein axes;
- the instruction wordings (`experiments/fix-v1.json`, `experiments/fix-generic-v1-dev.json`).

Scoring uses the same settings: within-country kNN gain, k = 10, countries with at least 30 scored towns, and the country block bootstrap of `layers-v7`.

- **Random-item:** all test towns, scored with the components fit on each seed's random-item training cities. The towns are the same for all three seeds; the fitted components differ.
- **Spatial-block:** only test towns outside that seed's training blocks (10° blocks with no training city), scored with that split's components.

Command, once:

`uv run --locked -m come_a_little_bit_closer.evaluation.fresh_places --config experiments/fresh-v1.json --part test`

The module refuses a second test run and refuses to run the test before the check part exists.

## Claims

Each claim covers e5-mistral and bge-multilingual-gemma2 × 2 splits × 3 seeds = 12 runs. Differences are in city within-country kNN gain, paired by country.

| # | Claim | Supported if |
| --- | --- | --- |
| G1 (primary) | The fix still beats plain cosine on fresh towns | Full fix (instruction + fusion + probe) minus plain cosine has a 95% CI above 0 in all 12 runs |
| G2 (primary) | The geographic wording, not using an instruction, drives the gain | Full fix minus the same pipeline with the similarity instruction ("Retrieve semantically similar text.") has a 95% CI above 0 in all 12 runs |
| G3 (primary) | The fix beats Feuerstein et al.'s seed-pair axes | Full fix minus within-country seed-pair axes blended with cosine (w = 0.2, the strongest axis variant on validation) has a 95% CI above 0 in all 12 runs |
| G4 (descriptive) | Components, baselines, third model | Report every method's mean kNN gain and the per-run differences for fusion and the probe. Also report Qwen3-Embedding-8B, the global axes, w = 1.0 and the geocoder oracle |

No STS claim: the STS-Benchmark test split was used once for F2 (fix-v1), and towns do not change STS. Option (a), chosen 2026-11-04.

## Expectations stated in advance

Absolute kNN gains should be lower than on large cities, because models know less about small towns. On validation cities, the mean differences behind G1-G3 were +0.06 to +0.14 for the two claim models. A failure of any claim will be reported as a failure, not re-run with other settings.
