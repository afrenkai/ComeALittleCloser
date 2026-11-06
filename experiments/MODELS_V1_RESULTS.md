# models-v1, size-v1, search-v1 and search-v2 results (tests, each run once on 2026-11-05/06)

Claims: `docs/MODELS_FREEZE.md`, approved by delegation before any new-model score existed. The addendum on mislocated stations was written after the search check part and before any test run. Job log: `experiments/MODELS_RUNS.md`. Raw scores: `outputs/models-v1/test/scores.jsonl`, `outputs/search-v1/test/scores.jsonl` and `outputs/search-v1-located/test/scores.jsonl`.

- **Test sets:**
  - cities: the original test split, reused;
  - towns: the fresh-v1 test part, reused;
  - small towns: the first 25,000 of a fresh 1,000-4,999 test part.
- **Components** are fit on the original training cities.
- **Differences** are paired by country, with 95% block-bootstrap intervals over countries.
- **Runs:** each claim needs the interval above 0 in every run, and a model has 3 sets × 2 splits × 3 seeds = 18 runs.

## Verdicts: places (job 3405093, 2:39:18)

| Claim | Criterion | Result | Verdict |
| --- | --- | --- | --- |
| P1 harrier-27b | Fix minus cosine, all 18 | 18 of 18; +0.148 to +0.224, lowest bound +0.102 | **Supported** |
| P1 octen-8b | Fix minus cosine, all 18 | 18 of 18; +0.033 to +0.124, lowest bound +0.014 | **Supported** |
| P1 nemotron-8b | Fusion + probe minus cosine (no instruction), all 18 | 18 of 18; +0.027 to +0.062, lowest bound +0.015 | **Supported** |
| P1 qwen3-4b-embedding | Fix minus cosine, all 18 | 16 of 18; +0.010 to +0.087, all positive. The two misses are small towns: random-item seed 3141 (+0.012 [−0.003, +0.026]) and spatial-block seed 2718 (+0.010 [−0.007, +0.027]) | **Not supported** |
| P1 harrier-0.6b | Fix minus cosine, all 18 | 6 of 18 (all six are cities); −0.004 to +0.033 | **Not supported** |
| P2 harrier-27b | Fix minus similarity-instruction pipeline, all 18 | 18 of 18; +0.134 to +0.174, lowest bound +0.098 | **Supported** |
| P2 octen-8b | Same | 18 of 18; +0.033 to +0.117, lowest bound +0.014 | **Supported** |
| P2 qwen3-4b-embedding | Same | 18 of 18; +0.016 to +0.087, lowest bound +0.001 | **Supported** |
| P2 harrier-0.6b | Same | 13 of 18; +0.002 to +0.054 | **Not supported** |
| P3 | Fix minus cosine on small towns, e5-mistral and bge-gemma2, all 12 | 12 of 12; e5 +0.063 to +0.087, gemma +0.065 to +0.095, lowest bound +0.039 | **Supported** |

## P4: population trend (descriptive)

Within-country kNN gain, mean over both splits and three seeds. The fix is instruction + fusion + probe; for Nemotron it is fusion + probe.

| Model | Cities, cosine → fix | Towns (5k-15k) | Small towns (1k-5k) | Oracle (w = 0.2), small towns |
| --- | --- | --- | --- | --- |
| harrier-27b | 0.278 → 0.484 | 0.217 → 0.397 | 0.169 → 0.330 | 0.209 |
| nemotron-8b | 0.236 → 0.287 | 0.173 → 0.214 | 0.140 → 0.170 | 0.179 |
| octen-8b | 0.135 → 0.246 | 0.078 → 0.153 | 0.069 → 0.110 | 0.103 |
| qwen3-4b-embedding | 0.110 → 0.177 | 0.070 → 0.102 | 0.064 → 0.082 | 0.132 |
| harrier-0.6b | 0.073 → 0.101 | 0.068 → 0.069 | 0.072 → 0.077 | 0.122 |
| e5-mistral | — | — | 0.146 → 0.229 | 0.197 |
| bge-multilingual-gemma2 | — | — | 0.146 → 0.227 | 0.183 |
| qwen3-8b-embedding | — | — | 0.069 → 0.119 | 0.133 |

- **Smaller places, less geography.** Cosine geography falls from cities to small towns in every model except harrier-0.6b, which is flat and low. The fix's gain shrinks with it but stays positive in every model and set.
- **Model size.** harrier-27b gains the most of any model at every size band, and harrier-0.6b the least. The two share a family and training recipe. On towns, harrier-0.6b gains nothing (+0.001).
- **The geocoder oracle** (cosine blended with true coordinates, w = 0.2) is beaten in every set and split by:
  - harrier-27b, by +0.112 to +0.177;
  - octen-8b, by +0.007 to +0.081;
  - e5-mistral and bge-gemma2 on small towns, by +0.024 to +0.048.

  Nemotron is within −0.013 to +0.022 of it. qwen3-4b, harrier-0.6b and qwen3-8b stay below it.
- **Nemotron with a non-native instruction.** Adding "query: {instruction}\n" raises Nemotron further. Instruction + fusion + probe reaches 0.344-0.348 on cities, 0.226-0.258 on towns and 0.168-0.194 on small towns. The similarity instruction gives 0.137-0.277. Nemotron was never trained with instructions, so this is descriptive only.
- **Layer fusion** helps in every run for:
  - harrier-27b, 18 of 18 (+0.012 to +0.034);
  - e5-mistral and bge-gemma2, 6 of 6 each.

  It is mixed for octen-8b (15 of 18), qwen3-4b (10 of 18), nemotron-8b (9 of 18) and qwen3-8b on small towns (2 of 6). It never helps harrier-0.6b (0 of 18).
- **The distance probe adds little**, as on fresh-v1 towns. Its effect ranges from −0.004 to +0.014, and its interval is above 0 in at most 10 of 18 runs per model. bge-gemma2 is the exception, at 5 of 6.

## Verdicts: station search (job 3407636, 1:24:48-1:26:30)

450 test queries in 17 countries, with origins from the fresh-v1 towns test part. The fix is the geography instruction on both sides plus fusion and probe. Standard use puts each model's web-search instruction on the query and leaves documents plain.

| Claim | Result (fix minus standard use) | Verdict |
| --- | --- | --- |
| P5 harrier-27b | 6 of 6; +0.128 to +0.160, lowest bound +0.079 | **Supported** |
| P5 bge-multilingual-gemma2 | 6 of 6; +0.061 to +0.113, lowest bound +0.001 | **Supported** |
| P5 e5-mistral | 5 of 6; +0.108 to +0.191. Random-item seed 3141 is +0.108 [−0.013, +0.230] | **Not supported** |
| P5 octen-8b | 2 of 6; −0.092 to +0.147 | **Not supported** |
| P5 harrier-0.6b | 2 of 6; −0.028 to +0.035 | **Not supported** |
| P5 qwen3-4b-embedding | 1 of 6; −0.114 to +0.101 | **Not supported** |

Qwen3-Embedding-8B (descriptive) is 2 of 6, from −0.080 to +0.130.

**Mean search gain** (random-item / spatial-block):

| Model | Plain cosine | Standard use | Geography instruction | Fix | Ceiling (true distance) |
| --- | --- | --- | --- | --- | --- |
| harrier-27b | 0.128 / 0.148 | 0.197 / 0.192 | 0.283 / 0.280 | 0.331 / 0.335 | 0.561 / 0.572 |
| e5-mistral | 0.173 / 0.155 | 0.148 / 0.128 | 0.228 / 0.227 | 0.268 / 0.281 | |
| bge-multilingual-gemma2 | 0.133 / 0.104 | 0.183 / 0.190 | 0.249 / 0.255 | 0.262 / 0.270 | |
| nemotron-8b (native prefixes) | 0.163 / 0.154 | — | 0.257 / 0.259 | 0.271 / 0.247 | |
| qwen3-8b-embedding | −0.026 / 0.052 | 0.124 / 0.118 | 0.021 / 0.152 | 0.047 / 0.174 | |
| octen-8b | 0.096 / 0.134 | 0.128 / 0.119 | 0.080 / 0.179 | 0.047 / 0.172 | |
| harrier-0.6b | 0.112 / 0.100 | 0.126 / 0.119 | 0.078 / 0.115 | 0.110 / 0.141 | |
| qwen3-4b-embedding | −0.069 / 0.014 | 0.109 / 0.087 | −0.022 / 0.100 | −0.001 / 0.101 | |

- **The failures come from one data error.** The Qwen3 family and Octen have random-item intervals reaching −0.45 to −0.51. In each case the Netherlands drives it. Two Dutch pop stations in the catalog are geolocated in Victoria, Seychelles. With the geography instruction on documents, these models rank them in the top 10 for Dutch pop queries. The Netherlands' gain then falls below −1.8 against a pool mean of about 100 km.
- **The search gain has no lower bound.** It is 1 − top-10 distance / pool distance, so one such country can outweigh the other 16.

**Descriptive sensitivity: located pools** (`search-v1-located`, post hoc, addendum in `docs/MODELS_FREEZE.md`). The same test queries are rerun with the 55 genre stations whose nearest place lies in another country removed from the pools.

| Model | Fix minus standard, intervals above 0 | Range | Lowest bound |
| --- | --- | --- | --- |
| harrier-27b | 6 of 6 | +0.139 to +0.175 | +0.086 |
| e5-mistral | 6 of 6 | +0.097 to +0.114 | +0.067 |
| octen-8b | 6 of 6 | +0.075 to +0.123 | +0.033 |
| qwen3-8b-embedding | 6 of 6 | +0.058 to +0.124 | +0.008 |
| bge-multilingual-gemma2 | 5 of 6 | +0.061 to +0.104 | −0.001 |
| qwen3-4b-embedding | 5 of 6 | +0.048 to +0.099 | −0.012 |
| harrier-0.6b | 5 of 6 | +0.029 to +0.037 | −0.001 |

With located pools the fix beats standard use in every run's mean. Fusion and probe add to the geography instruction in 6 of 6 runs for e5-mistral, qwen3-8b, octen-8b and harrier-27b. This was chosen after the check part, so it is repeated as a frozen test on fresh origins in search-v2 (`docs/SEARCH_V2_FREEZE.md`).

## Verdicts: model size within Qwen3-Embedding (size-v1, job 3408121, 0:12:09)

Claims are in `docs/SIZE_FREEZE.md`, frozen before any Qwen3-Embedding-0.6B score under the fix. It uses the same three test sets and settings as models-v1.

| Claim | Result | Verdict |
| --- | --- | --- |
| S1: the fix beats cosine for qwen3-0.6b-embedding (18 runs) | 5 of 18; −0.021 to +0.037 | **Not supported** |
| S2: geographic beats similarity instruction (18 runs) | 2 of 18; −0.027 to +0.029 | **Not supported** |
| S3: fix gain 0.6B < 4B < 8B on every set (descriptive, predicted) | Holds on all three sets (table below) | **Held** |

Each cell gives the mean within-country kNN gain as cosine → fix, with the fix's gain in brackets:

| Set | 0.6B | 4B | 8B |
| --- | --- | --- | --- |
| Cities | 0.023 → 0.051 (+0.028) | 0.110 → 0.177 (+0.067) | 0.138 → 0.241 (+0.104) |
| Towns | 0.037 → 0.037 (−0.000) | 0.070 → 0.102 (+0.032) | 0.083 → 0.153 (+0.070) |
| Small towns | 0.042 → 0.038 (−0.004) | 0.064 → 0.082 (+0.018) | 0.069 → 0.119 (+0.050) |

The 8B cities and towns numbers come from `FIX_V1_RESULTS.md` and `FRESH_V1_RESULTS.md`; all other cells are from models-v1 and size-v1.

- **Qwen3-0.6B has almost no within-country geography to start with.** Its cosine kNN gain is 0.02-0.04, against a geocoder oracle of 0.105-0.112. This matches its last-token cosine in `REFLECTED_V1_RESULTS.md`.
- **The geography instruction does not help it either.** It responds to the instruction no differently from the similarity instruction (S2).
- **Two families now show the same pattern.** Within harrier and within Qwen3-Embedding, the fix's gain grows with model size, and at 0.6B it is near zero beyond the original cities. The fix relies on a model that can follow a geographic instruction.

## Verdicts: station search on fresh origins with clean pools (search-v2, job 3409349, 1:17:53)

Claims are in `docs/SEARCH_V2_FREEZE.md`, frozen after search-v1 and before any search-v2 feature existed. The test has 1,105 queries in 17 countries, with origins from the capped small-town (1,000-4,999) test part and up to 25 per genre and country. The 247 mislocated stations are removed from the pools. Spatial-block runs score the queries of held-out blocks, which covers 12 countries.

| Claim | Result (fix minus standard use) | Verdict |
| --- | --- | --- |
| P6 harrier-27b | 6 of 6; +0.091 to +0.142, lowest bound +0.053 | **Supported** |
| P6 e5-mistral | 6 of 6; +0.037 to +0.078, lowest bound +0.010 | **Supported** |
| P6 harrier-0.6b | 4 of 6; +0.006 to +0.045 | **Not supported** |
| P6 octen-8b | 4 of 6; +0.033 to +0.051 | **Not supported** |
| P6 qwen3-4b-embedding | 3 of 6; +0.038 to +0.067 | **Not supported** |
| P6 bge-multilingual-gemma2 | 1 of 6; +0.026 to +0.068 | **Not supported** |

Qwen3-Embedding-8B (descriptive) is 1 of 6, from +0.027 to +0.068.

**Mean search gain** (random-item / spatial-block):

| Model | Plain cosine | Standard use | Geography instruction | Fix | Ceiling (true distance) |
| --- | --- | --- | --- | --- | --- |
| harrier-27b | 0.094 / 0.096 | 0.110 / 0.109 | 0.164 / 0.167 | 0.218 / 0.223 | 0.503 / 0.513 |
| nemotron-8b (native prefixes) | 0.065 / 0.061 | — | 0.174 / 0.186 | 0.196 / 0.203 | |
| e5-mistral | 0.086 / 0.088 | 0.102 / 0.111 | 0.125 / 0.133 | 0.177 / 0.171 | |
| bge-multilingual-gemma2 | 0.055 / 0.050 | 0.089 / 0.103 | 0.117 / 0.134 | 0.140 / 0.147 | |
| octen-8b | 0.054 / 0.066 | 0.055 / 0.067 | 0.066 / 0.080 | 0.092 / 0.109 | |
| qwen3-8b-embedding | 0.044 / 0.038 | 0.056 / 0.056 | 0.052 / 0.078 | 0.085 / 0.103 | |
| qwen3-4b-embedding | 0.016 / 0.003 | 0.017 / 0.012 | 0.028 / 0.047 | 0.060 / 0.067 | |
| harrier-0.6b | 0.038 / 0.034 | 0.041 / 0.033 | 0.053 / 0.049 | 0.059 / 0.055 | |

- **The fix beats standard use in every run's mean for every model**, by +0.006 to +0.142. With clean pools, the lowest country gain under the fix is −0.03 to −0.28 per model, against below −1.8 for the Netherlands in search-v1.
- **The strict all-6 rule passes only for e5-mistral and harrier-27b.** Small-town origins carry less geography (as in P4), so the differences are smaller than in search-v1. With 17 countries (12 in spatial-block) the intervals stay wide. bge-gemma2 passed P5 but misses P6; e5-mistral missed P5 by one run but passes P6.
- **harrier-27b doubles standard use** (0.110 → 0.218), reaching about 43% of the true-distance ceiling, up from 22%.
- **Fusion and probe add to the geography instruction** in 6 of 6 runs for e5-mistral, harrier-27b, octen-8b and qwen3-8b, 5 for qwen3-4b, 4 for nemotron-8b, 1 for bge-gemma2 and 0 for harrier-0.6b. The geography instruction alone beats standard use in 6 of 6 runs only for harrier-27b.

**Search overall.** The fix helps station search on average for every model and every run once the catalog's geolocation errors are removed. It passes a strict per-run interval test for the largest model (harrier-27b, in both search tests) and for e5-mistral (search-v2). For the others the effect is positive but too small relative to the country-level noise to clear the all-runs rule.
