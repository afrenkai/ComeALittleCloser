# Model size within one family: freeze list (size-v1)

**Status: approved 2026-11-05.** Tian-hao Zhang approved this experiment and delegated its claims to Claude ("Yes, decide and run"). This file was written and committed before any feature or score of Qwen3-Embedding-0.6B under the fix existed. Its layer-probe features from `layers-v*` exist, but they measure coordinate probes, not the fix. Runs are logged in `experiments/MODELS_RUNS.md`.

## Why

In models-v1 the fix helped harrier-27b most and harrier-0.6b least. Two sizes of one family suggest, but do not show, that the fix needs a large enough model. Qwen3-Embedding has three sizes trained the same way: 0.6B, 4B and 8B. The 4B and 8B models have already been tested on all three sets. This experiment adds the 0.6B model.

## Model

| Model | Checkpoint @ revision | Pooling | Instruction format |
| --- | --- | --- | --- |
| qwen3-0.6b-embedding | Qwen/Qwen3-Embedding-0.6B (pinned in `models/layer_catalog.py`) | last token | `Instruct: {instruction}\nQuery:{text}`, as Qwen3-Embedding-4B and 8B |

Parity with Sentence Transformers (cosine ≥ 0.9999 on 60 texts) runs before extraction.

## Fixed settings

- The fix and every setting come from `experiments/fix-v1.json` via `fix-v3.json` and `fix-generic-v3.json`. These are copies of the v2 plans with only the model changed.
- The scored sets are the same as in models-v1, all reused, with the same partition seed and the same 25,000 small-town cap:
  - original test cities;
  - the fresh-v1 test part of the towns;
  - the models-v1 test part of the small towns.

## Claims

Each claim uses the city within-country kNN gain, paired by country, with a 95% block-bootstrap interval. It needs the interval above 0 in all 18 runs: 3 sets × 2 splits × 3 seeds.

| # | Claim | Supported if |
| --- | --- | --- |
| S1 | The frozen fix beats plain cosine for qwen3-0.6b-embedding | Instruction + fusion + probe minus cosine, interval above 0 in all 18 runs |
| S2 | The geographic wording drives its gain | Fix minus the same pipeline with "Retrieve semantically similar text.", interval above 0 in all 18 runs |
| S3 (descriptive, predicted in advance) | The fix's gain grows with size within the family | Mean fix gain (fix minus cosine) per set: 0.6B < 4B < 8B on all three sets |

The 4B and 8B means in S3 come from runs already recorded:

| Set | qwen3-4b-embedding | qwen3-8b-embedding |
| --- | --- | --- |
| Cities | `experiments/MODELS_V1_RESULTS.md` (+0.067) | `experiments/FIX_V1_RESULTS.md` |
| Towns | models-v1 (+0.032) | `experiments/FRESH_V1_RESULTS.md` |
| Small towns | models-v1 (+0.018) | models-v1 (+0.050) |

S3 has no interval, because the models are compared across separate runs. We report it as a prediction that held or did not.

## What runs

- **Parity and extraction:** one GPU job.
- **Check part:** `fresh_places --config experiments/size-v1.json --part check`. Pipeline checks only; no setting changes after it except bug fixes, which are logged.
- **Test, once:** `fresh_places --config experiments/size-v1.json --part test`.

Every outcome will be reported.
