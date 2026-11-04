# Reflected-geography freeze list: DRAFT for review (2026-11-04)

**Status: draft, not frozen, test cities untouched by this analysis.** Exploration so far used only validation cities (`experiments/REFLECTED_RUNS.md`). This draft proposes which claims to test once on the held-out test cities, with pass criteria fixed in advance, as was done for the probe claims (`docs/FREEZE.md`).

## What would be run

Plan `experiments/reflected-v1.json`. It equals `experiments/reflected-v1-dev.json` except:

- `layers_plan` is `experiments/layers-v7.json`, so it reads the frozen test-run predictions and the test rows;
- the output is `outputs/reflected-v1`;
- it adds `xlmr-xl` (mean pooling; no pair).

Command: `uv run --locked -m come_a_little_bit_closer.evaluation.layer_reflected --config experiments/reflected-v1.json --mode final`, once.

The analysis uses the test cities of the layers random-item and spatial-block splits. Centring and the nearest-neighbour database use training cities only. Nothing is fitted on test cities. The probe positions are the layers-v7 test predictions, made from probes fit on training cities. The test cities were already used once, for the probe claims H1-H6. These are different claims about the same held-out cities, fixed before this analysis has seen them, and the paper should say so.

## Proposed claims

| # | Claim | Supported if (test cities, random-item) | Validation status |
| --- | --- | --- | --- |
| R1 (primary) | Retrieval models' output vectors reflect within-country geography in cosine similarity more than their base models' output vectors | For each 4B+ pair (Qwen3-4B, Qwen3-8B, Mistral, Gemma), at both poolings, with raw and centred cosine: retrieval-minus-base kNN gain, paired by country, has a CI above 0 on all three seeds (48 intervals) | 48 of 48 |
| R2 (descriptive) | It is partial | Report each retrieval model's cosine kNN gain against the true-distance ceiling | Best: e5-mistral 0.24-0.27 of 0.73-0.75 |
| R3 (secondary) | Retrieval training shifts within-country geography from linearly decodable toward nearest-neighbour accessible | Crossover interaction, at both poolings: point estimate above 0 in all 24 runs, and CI above 0 in at least 12 of 24 | 24 of 24 positive; 17 of 24 CI above 0 |
| R4 (descriptive) | Nearest neighbours cannot extrapolate | Spatial-block: report probe vs nearest-neighbour gains | Probe ahead in all runs |

R1 compares last-layer vectors only. The depth profile (`REFLECTED_RUNS.md`, finding 12) shows that with mean pooling, base models' intermediate layers reflect geography about as well. The paper must therefore say "output vectors", not "retrieval training creates it". A depth claim is not proposed for the test; the depth profile stays validation-only and descriptive.

The negative result on cheap fixes is reported as exploratory, not as a frozen claim. The city-trained probe fix on stations is dominated by a country-code baseline (`REFLECTED_RUNS.md`, finding 6). The stations were already used in that exploration.

## Decisions needed

1. **Run this at all?** Recommended yes. R1 is the paper's clearest new finding, and validation alone will not satisfy reviewers.
2. **R3's criterion.** The "at least 12 of 24" rule is a count chosen after seeing validation. The stricter alternative, all 24 intervals above 0, already fails on validation. Option: report R3 descriptively instead of as a pass/fail claim. Recommended: descriptive, with R1 the only primary claim.
3. **Include xlmr-xl?** It has no retrieval pair, so it only adds a row to the descriptive table. Recommended yes.
