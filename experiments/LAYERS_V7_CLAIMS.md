# Frozen claims, layers-v7 final

Gain over the country centroid in km, one `gain [95% CI]` per seed (seeds in plan order). Criteria: `docs/FREEZE.md`.

## H1: decoders from 4B up hold within-country geography

| Model | Layer | spatial-block | random-item | Verdict |
| --- | --- | --- | --- | --- |
| qwen3-4b-base | 34 | 201 [134, 281]; 259 [81, 525]; 116 [40, 206] | 146 [85, 217]; 118 [67, 174]; 123 [64, 196] | supported |
| qwen3-4b-embedding | 33 | 165 [103, 234]; 277 [81, 560]; 137 [45, 241] | 150 [91, 220]; 131 [79, 186]; 127 [77, 190] | supported |
| qwen3-8b-base | 33 | 267 [152, 410]; 305 [94, 599]; 162 [45, 306] | 207 [133, 295]; 188 [125, 255]; 184 [120, 267] | supported |
| qwen3-8b-embedding | 33 | 249 [125, 393]; 335 [99, 679]; 199 [68, 366] | 214 [150, 289]; 207 [142, 279]; 205 [130, 299] | supported |
| mistral-7b-base | 27 | 301 [168, 452]; 306 [88, 610]; 210 [78, 366] | 257 [178, 351]; 274 [183, 368]; 245 [174, 331] | supported |
| e5-mistral | 25 | 306 [181, 446]; 297 [115, 531]; 243 [89, 440] | 263 [185, 359]; 259 [173, 348]; 239 [166, 327] | supported |
| gemma2-9b-base | 35 | 375 [219, 562]; 375 [146, 693]; 259 [127, 423] | 325 [235, 426]; 305 [219, 403]; 296 [214, 402] | supported |
| bge-multilingual-gemma2 | 33 | 338 [190, 520]; 367 [148, 663]; 277 [132, 457] | 304 [218, 400]; 274 [193, 361]; 272 [193, 372] | supported |

H1 overall: supported.

## H2: retrieval training keeps it (paired retrieval minus base, per block)

| Pair | Split | Retrieval gain CIs above 0 | Paired difference per seed | Direction |
| --- | --- | --- | --- | --- |
| qwen3-4b-base → qwen3-4b-embedding (H2) | spatial-block | True | -36 [-101, 20]; 18 [-20, 59]; 21 [-5, 50] | no claim |
| qwen3-4b-base → qwen3-4b-embedding (H2) | random-item | True | 4 [-17, 25]; 13 [-10, 34]; 4 [-18, 26] | no claim |
| qwen3-8b-base → qwen3-8b-embedding (H2) | spatial-block | True | -18 [-91, 44]; 30 [-44, 111]; 37 [-10, 87] | no claim |
| qwen3-8b-base → qwen3-8b-embedding (H2) | random-item | True | 7 [-16, 27]; 19 [-1, 40]; 21 [-2, 45] | no claim |
| mistral-7b-base → e5-mistral (H2) | spatial-block | True | 5 [-47, 61]; -9 [-91, 46]; 34 [-21, 103] | no claim |
| mistral-7b-base → e5-mistral (H2) | random-item | True | 6 [-11, 23]; -15 [-42, 14]; -7 [-27, 15] | no claim |
| gemma2-9b-base → bge-multilingual-gemma2 (H2) | spatial-block | True | -37 [-81, 2]; -9 [-64, 39]; 18 [-24, 67] | no claim |
| gemma2-9b-base → bge-multilingual-gemma2 (H2) | random-item | True | -21 [-41, -2]; -31 [-52, -11]; -25 [-43, -6] | removes some |
| qwen3-0.6b-base → qwen3-0.6b-embedding (secondary) | spatial-block | False | -36 [-61, -13]; 10 [-26, 43]; 6 [-3, 16] | no claim |
| qwen3-0.6b-base → qwen3-0.6b-embedding (secondary) | random-item | False | -11 [-21, -2]; -0 [-14, 12]; -1 [-8, 7] | no claim |
| xlmr-base → e5-base (secondary) | spatial-block | False | 9 [-18, 37]; 38 [-8, 95]; 20 [-4, 45] | no claim |
| xlmr-base → e5-base (secondary) | random-item | False | 3 [-11, 19]; 5 [-5, 17]; 6 [-10, 23] | no claim |
| xlmr-large → bge-m3 (secondary) | spatial-block | False | 19 [-1, 40]; 40 [-20, 113]; 18 [-4, 43] | no claim |
| xlmr-large → bge-m3 (secondary) | random-item | False | 2 [-13, 20]; -2 [-14, 11]; 4 [-16, 28] | no claim |

Direction per pair (claimed only if both splits agree):

- qwen3-4b-base → qwen3-4b-embedding: no claim
- qwen3-8b-base → qwen3-8b-embedding: no claim
- mistral-7b-base → e5-mistral: no claim
- gemma2-9b-base → bge-multilingual-gemma2: no claim
- qwen3-0.6b-base → qwen3-0.6b-embedding: no claim
- xlmr-base → e5-base: no claim
- xlmr-large → bge-m3: no claim

H2 overall (retrieval keeps the gain in every H2 pair): supported.

## H3: MLM encoders up to 3.5B show none

| Model | Layer | spatial-block | random-item | Verdict |
| --- | --- | --- | --- | --- |
| xlmr-base | 12 | 11 [-10, 32]; 12 [-13, 44]; 0 [-14, 16] | 0 [-0, 0]; -5 [-15, 4]; 0 [-0, 0] | supported |
| e5-base | 8 | 20 [-11, 57]; 50 [-3, 119]; 20 [-4, 48] | 3 [-11, 19]; 0 [-15, 15]; 6 [-10, 23] | supported |
| xlmr-large | 3 | 6 [-17, 30]; 0 [-0, 0]; -12 [-30, 6] | 1 [-2, 4]; -0 [-8, 7]; -0 [-1, 1] | supported |
| bge-m3-retromae | 16 | -0 [-1, 0]; 20 [-13, 55]; -2 [-4, 0] | 0 [-0, 0]; -4 [-15, 8]; 3 [-9, 17] | supported |
| bge-m3-unsupervised | 16 | 35 [5, 66]; 31 [-45, 122]; 6 [-19, 31] | 5 [-11, 25]; 4 [-13, 20]; 12 [-11, 37] | NOT supported |
| bge-m3 | 16 | 25 [-5, 56]; 41 [-20, 113]; 6 [-19, 32] | 3 [-13, 22]; -2 [-17, 14]; 4 [-16, 27] | supported |
| xlmr-xl | 32 | 20 [4, 39]; 31 [-7, 78]; 0 [-12, 14] | 8 [-2, 19]; -1 [-13, 10]; 12 [-1, 25] | NOT supported |

H3 overall: NOT supported.

## H4: the retrieval output vector carries geography beyond the country

Last layer, `sphere` target, random-item.

| Model | Pooling | Layer | Gain per seed | Verdict |
| --- | --- | --- | --- | --- |
| qwen3-4b-embedding (H4) | last | 36 | 252 [46, 527]; 286 [49, 560]; 193 [-55, 476] | NOT supported |
| qwen3-8b-embedding (H4) | last | 36 | 332 [111, 603]; 379 [145, 667]; 354 [90, 659] | supported |
| e5-mistral (H4) | last | 32 | 381 [162, 640]; 451 [198, 755]; 426 [163, 741] | supported |
| bge-multilingual-gemma2 (H4) | last | 42 | 357 [162, 586]; 421 [182, 708]; 404 [160, 668] | supported |
| e5-base (secondary) | mean | 12 | -634 [-869, -433]; -703 [-973, -436]; -703 [-956, -472] | NOT supported |
| bge-m3 (secondary) | mean | 24 | -715 [-945, -541]; -787 [-996, -605]; -610 [-789, -422] | NOT supported |

H4 overall: NOT supported.

## H5: lat/lon R² mostly reflects knowing the country (descriptive)

Best layer, `latlon` target, mean pooling, random-item; seed means. Clipped R² (primary) clips predictions to valid latitude and longitude.

| Model | Layer | R² lat / lon clipped | R² lat / lon raw | Country centroid R² lat / lon |
| --- | --- | --- | --- | --- |
| qwen3-4b-base | 34 | 0.87 / 0.92 | 0.87 / 0.92 | 0.95 / 0.98 |
| qwen3-4b-embedding | 33 | 0.89 / 0.93 | 0.89 / 0.93 | 0.95 / 0.98 |
| qwen3-8b-base | 33 | 0.89 / 0.93 | 0.89 / 0.93 | 0.95 / 0.98 |
| qwen3-8b-embedding | 33 | 0.91 / 0.94 | 0.91 / 0.94 | 0.95 / 0.98 |
| mistral-7b-base | 27 | 0.91 / 0.94 | 0.91 / 0.94 | 0.95 / 0.98 |
| e5-mistral | 25 | 0.91 / 0.94 | 0.91 / 0.94 | 0.95 / 0.98 |
| gemma2-9b-base | 35 | 0.93 / 0.95 | 0.93 / 0.95 | 0.95 / 0.98 |
| bge-multilingual-gemma2 | 33 | 0.93 / 0.95 | 0.93 / 0.95 | 0.95 / 0.98 |
| xlmr-base | 12 | 0.79 / 0.88 | 0.79 / 0.88 | 0.95 / 0.98 |
| e5-base | 8 | 0.86 / 0.90 | 0.86 / 0.90 | 0.95 / 0.98 |
| xlmr-large | 3 | 0.83 / 0.89 | 0.83 / 0.89 | 0.95 / 0.98 |
| bge-m3-retromae | 16 | 0.83 / 0.89 | 0.83 / 0.89 | 0.95 / 0.98 |
| bge-m3-unsupervised | 16 | 0.88 / 0.93 | 0.88 / 0.93 | 0.95 / 0.98 |
| bge-m3 | 16 | 0.88 / 0.93 | 0.88 / 0.93 | 0.95 / 0.98 |
| xlmr-xl | 32 | 0.86 / 0.91 | 0.86 / 0.91 | 0.95 / 0.98 |
| qwen3-0.6b-random | 0 | 0.79 / 0.87 | 0.79 / 0.87 | 0.95 / 0.98 |
| xlmr-base-random | 0 | 0.74 / 0.83 | 0.74 / 0.83 | 0.95 / 0.98 |

## H6: the within-country signal is not limited to famous places

Smallest population quintile, H1 cells, seeds pooled.

| Model | spatial-block | random-item | Verdict |
| --- | --- | --- | --- |
| qwen3-4b-base | 93 [47, 148] | 72 [19, 134] | supported |
| qwen3-4b-embedding | 74 [32, 119] | 79 [33, 132] | supported |
| qwen3-8b-base | 95 [33, 161] | 119 [57, 193] | supported |
| qwen3-8b-embedding | 102 [54, 161] | 138 [82, 204] | supported |
| mistral-7b-base | 134 [69, 204] | 182 [104, 276] | supported |
| e5-mistral | 131 [71, 197] | 168 [95, 256] | supported |
| gemma2-9b-base | 193 [120, 270] | 224 [140, 320] | supported |
| bge-multilingual-gemma2 | 189 [121, 261] | 211 [135, 301] | supported |

H6 overall: supported.
