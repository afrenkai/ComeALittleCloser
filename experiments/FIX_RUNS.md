# Closing the gap: fixes that make cosine reflect within-country geography

Exploratory run log (started 2026-11-04, branch `research/fixes-and-results`). The test results show that retrieval models' output vectors reflect about a third of the possible within-country geography (`experiments/REFLECTED_V1_RESULTS.md`, R2). Can a cheap change to frozen models close more of the gap without a geocoder and without hurting semantic relevance? Ideas come from what we observed and from a literature review (`docs/FIX_LITERATURE.md`).

**Data discipline.** Cities: fit on training cities and evaluate on validation cities (layers splits, seed 1729 unless noted); the test cities are untouched. Stations: all 11,833 are used for evaluation only, never for fitting city-trained components. They were already used in the earlier probe-fix exploration, so they are a development set, not held out.

**Metrics** are the same as `experiments/REFLECTED_RUNS.md`:

- Cities: within-country kNN gain (ceiling about 0.74), rho, and global mean km to the 10 nearest.
- Stations, the same three plus genre agreement: the share of the 10 neighbours sharing a genre tag (the semantic cost).

Every fix is compared with plain cosine, the country-lookup baseline (append the country-centroid position, w=0.1 and 0.5) and the geocoder oracle (append true positions, w=0.2).

**Fixes** (script `outputs/diagnostics/fixes.py`, not in git):

- **F1, layer fusion:** add the centred cosine of an intermediate layer (layer chosen from the validation depth profile, so it is optimistic).
- **F2, distance probe:** a rank-k linear map B fit on training cities with a triplet loss. Each anchor is pulled toward one of its 10 geographically nearest same-country training cities and pushed from a random same-country city. The score is cos - lambda × normalised squared distance in the B space. This follows Hewitt & Manning's structural probe, with a ranking loss.
- **F3, pseudo-positions:** append the mean true position of the 10 most similar training cities, either global or restricted to the same country.
- **F4, per-country centring** (from the literature review): subtract each country's mean vector. Cities use training-city means; stations use their own country means, which are unsupervised and need no coordinates.

## Jobs

| Job | Run | Outcome | Elapsed |
| --- | --- | --- | --- |
| 3312765-3312768 | F1-F3, rank 32, random-item, seed 1729: e5-mistral (mid 26), qwen3-8b-embedding (32), bge-multilingual-gemma2 (32), qwen3-4b-embedding (32) | Completed | about 3:30 each |
| 3312869-3312874 | Round 2 (F1, F2, F4, and F1+F2). e5-mistral rank 32 on random-item and on spatial-block; bge-gemma2 rank 32 on spatial-block; e5-mistral rank 128 and bge-gemma2 rank 128 on random-item; qwen3-8b-embedding rank 32 on spatial-block | Completed | 1:47-4:06 |
| 3313206-3313229 | Round 3 as 24 separate jobs (failure-only email) | Cancelled; resubmitted as an array with full email notifications | |
| 3313230 (array 0-23) | Round 3: F1 (mid layer chosen on 8,000 training cities) and F2 rank 32. Four models × {random-item, spatial-block} × seeds {1729, 2718, 3141} | Completed, 24 of 24 | |
| 3313258 (array 0-5) | Instruction re-encoding (`instruct_extract.py`): e5-mistral, bge-gemma2 and qwen3-8b-embedding × {geo, generic} instruction on both sides; last-token states, upper half of layers; gtml L40S | Completed | 15:35-21:26 |
| 3314426 (array 0-5) | Harness on the instructed vectors (seed 1729, random-item; F1, F2) | Completed | 2:26-15:19 |
| 3315040 (array 0-5) | Re-encoding with two alternative geography phrasings (geo2, geo3) × 3 models | Running | |
| 3315041 (array 0-17) | Harness on geo-instructed vectors × 3 models × 2 splits × 3 seeds | Running | |

## Round 1 (random-item, seed 1729)

| Model | Cosine: cities kNN / stations kNN / genre | Country baseline w=0.1 | Oracle w=0.2 | F1 mid only | F2 cos - 0.1·dist | F2 cos - 0.3·dist |
| --- | --- | --- | --- | --- | --- | --- |
| e5-mistral | 0.240 / 0.188 / 0.520 | 0.240 / 0.188 / 0.511 | 0.289 / 0.257 / 0.505 | 0.294 / 0.187 / 0.574 | 0.280 / 0.219 / 0.493 | 0.289 / 0.227 / 0.463 |
| bge-multilingual-gemma2 | 0.192 / 0.190 / 0.511 | 0.192 / 0.190 / 0.496 | 0.233 / 0.253 / 0.492 | 0.240 / 0.165 / 0.574 | 0.267 / 0.224 / 0.487 | 0.300 / 0.241 / 0.458 |
| qwen3-8b-embedding | 0.132 / 0.166 / 0.639 | 0.132 / 0.166 / 0.620 | 0.192 / 0.231 / 0.613 | 0.156 / 0.140 / 0.633 | 0.156 / 0.177 / 0.619 | 0.168 / 0.185 / 0.587 |
| qwen3-4b-embedding | 0.110 / 0.127 / 0.554 | 0.110 / 0.127 / 0.538 | 0.180 / 0.218 / 0.533 | 0.121 / 0.113 / 0.590 | 0.123 / 0.141 / 0.538 | 0.128 / 0.142 / 0.510 |

F3 is marginal: cities kNN +0.004 to +0.018.

- **F2 is the first fix that improves within-country structure,** which the country baseline cannot change at all. It does so for all four models on cities, and it transfers to stations: the B map is fit on cities only, yet station within-country kNN rises in every model. The genre cost at lambda = 0.1 is 0.5-2.7 points. For Gemma at lambda = 0.3, city kNN gain goes from 0.192 to 0.300, above the mild oracle blend.
- **F1** (e5-mistral layer 26, Gemma layer 32) improves city kNN gain and genre agreement together, but lowers station within-country kNN for Gemma and both Qwen models.

## Round 2 (seed 1729)

Cities kNN gain / stations kNN gain / genre agreement:

| Model, split | Cosine | Country w=0.1 | Oracle w=0.2 | F2 lambda=0.1 | F2 lambda=0.3 | F1(w=1)+F2 lambda=0.1 | F1(w=1)+F2 lambda=0.3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e5-mistral, random-item | 0.240 / 0.188 / 0.520 | 0.240 / 0.188 / 0.511 | 0.289 / 0.257 / 0.505 | 0.280 / 0.219 / 0.493 | 0.289 / 0.227 / 0.463 | 0.299 / 0.213 / 0.536 | 0.302 / 0.223 / 0.502 |
| e5-mistral, spatial-block | 0.222 / 0.188 / 0.520 | 0.222 / 0.188 / 0.510 | 0.244 / 0.257 / 0.505 | 0.256 / 0.216 / 0.497 | 0.264 / 0.216 / 0.469 | 0.275 / 0.210 / 0.539 | 0.277 / 0.216 / 0.508 |
| bge-gemma2, spatial-block | 0.190 / 0.190 / 0.511 | 0.190 / 0.190 / 0.496 | 0.205 / 0.253 / 0.492 | 0.244 / 0.211 / 0.494 | 0.271 / 0.218 / 0.470 | 0.258 / 0.209 / 0.548 | 0.276 / 0.219 / 0.508 |
| qwen3-8b-emb, spatial-block | 0.113 / 0.166 / 0.639 | 0.113 / 0.166 / 0.619 | 0.149 / 0.231 / 0.613 | 0.129 / 0.170 / 0.621 | 0.138 / 0.166 / 0.591 | 0.138 / 0.158 / 0.626 | 0.144 / 0.163 / 0.589 |

- **F2 generalises to held-out regions.** On spatial-block, where validation cities lie in 10-degree blocks with no training cities, the distance probe still raises within-country kNN gain (e5-mistral +0.034 to +0.042, Gemma +0.054 to +0.081). Nearest-neighbour geolocation failed this test (REFLECTED_RUNS finding 10), so F2 is not memorising regions.
- **F1+F2 is the best combination for e5-mistral and Gemma.** City kNN gain rises by 0.06-0.09 and station within-country kNN gain by 0.02-0.03, while genre agreement stays at or above plain cosine (e5-mistral 0.536 vs 0.520; Gemma 0.548 vs 0.511). For Qwen3-8B the gains are small.
- **Rank 128 adds nothing over rank 32.**
- **F4 (per-country centring) raises genre agreement a lot** (e5-mistral 0.52 to 0.60; Qwen3-8B 0.64 to 0.69) but destroys cross-country geography: global neighbour km goes from 893 to 2,268-3,633. It is not a geography fix. Its genre effect, where removing the country component lets genre dominate, is worth noting for the semantic side.
- **Caveat:** F1's layer was chosen from the validation depth profile. Round 3 chooses it on training cities only.

## Round 3: three seeds × two splits, with F1's layer chosen on training cities

The layer with the highest centred-cosine kNN gain on 8,000 training cities was e5-mistral 25-26, bge-gemma2 32, Qwen3-8B 32-34 and Qwen3-4B 32-35. These match the validation choices, so F1 was not overfit. Values are means over three seeds; the range is the per-seed change vs plain cosine.

| Model | Split | Cosine: city / station / genre | F1+F2 lambda=0.1 change: city, station, genre | F2 lambda=0.1 change: city, station, genre | Oracle w=0.2 change: city, station, genre |
| --- | --- | --- | --- | --- | --- |
| e5-mistral | random-item | 0.252 / 0.188 / 0.520 | +0.060..+0.062, +0.025..+0.028, +0.013..+0.017 | +0.039..+0.045, +0.032..+0.035, -0.028..-0.024 | +0.049..+0.051, +0.069, -0.015 |
| e5-mistral | spatial-block | 0.237 / 0.188 / 0.520 | +0.054..+0.072, +0.022..+0.028, +0.016..+0.017 | +0.034..+0.045, +0.025..+0.034, -0.024..-0.023 | +0.022..+0.038, +0.069, -0.015 |
| bge-gemma2 | random-item | 0.203 / 0.190 / 0.511 | +0.072..+0.083, +0.025..+0.026, +0.029..+0.032 | +0.073..+0.080, +0.031..+0.036, -0.025..-0.023 | +0.041..+0.044, +0.063, -0.019 |
| bge-gemma2 | spatial-block | 0.213 / 0.190 / 0.511 | +0.068..+0.080, +0.018..+0.021, +0.034..+0.036 | +0.056..+0.066, +0.022..+0.028, -0.022..-0.017 | +0.015..+0.028, +0.063, -0.019 |
| qwen3-8b-emb | random-item | 0.133 / 0.166 / 0.639 | +0.034..+0.040, +0.000..+0.006, -0.018..-0.015 | +0.024..+0.033, +0.006..+0.010, -0.020..-0.019 | +0.060..+0.069, +0.065, -0.026 |
| qwen3-8b-emb | spatial-block | 0.123 / 0.166 / 0.639 | +0.019..+0.025, -0.005..+0.009, -0.015..-0.012 | +0.012..+0.017, +0.007..+0.009, -0.020..-0.018 | +0.036..+0.054, +0.065, -0.026 |
| qwen3-4b-emb | random-item | 0.109 / 0.127 / 0.554 | +0.015..+0.021, +0.000..+0.011, +0.007..+0.009 | +0.014..+0.021, -0.002..+0.014, -0.018..-0.016 | +0.070..+0.079, +0.091, -0.021 |
| qwen3-4b-emb | spatial-block | 0.105 / 0.127 / 0.554 | +0.008..+0.011, -0.002..+0.001, -0.001..+0.010 | +0.008..+0.012, -0.006..+0.003, -0.019..-0.016 | +0.037..+0.066, +0.091, -0.021 |

- **For e5-mistral and bge-gemma2, F1+F2 (lambda = 0.1, F1 weight 1) is better than plain cosine on all three measures in all 12 runs** (2 models × 2 splits × 3 seeds):
  - city within-country kNN gain +0.054 to +0.083, about 25-40% relative;
  - station within-country kNN gain +0.018 to +0.028, for a probe fit on cities only;
  - genre agreement +0.013 to +0.036.

  The country-lookup baseline cannot change within-country gain at all and costs 0.9-1.5 genre points. On cities, F1+F2 exceeds the mild geocoder blend (oracle w=0.2). On stations, the geocoder still wins, since it has the true coordinates.
- **It holds on spatial-block** (validation cities in 10-degree blocks with no training cities), so it does not depend on memorising regions.
- **For the Qwen models the gains are smaller:** city +0.008 to +0.040, station about 0. Genre is about neutral for 4B and -0.015 for 8B. Their training-chosen mid layers (32-35) sit close to the output layer (36).
- **Still to check:** lambda and the F1 weight were chosen by looking at validation and stations, so a final evaluation needs held-out cities (the test split) and a semantic set not used for selection.

## F5: task instructions (seed 1729, random-item; `instruct_extract.py`)

Re-encode both cities and stations with the model's native instruction format: e5-mistral `Instruct: {task}\nQuery: {text}`, Qwen3-Embedding `Instruct: {task}\nQuery:{text}`, bge-gemma2 `<instruct>{task}\n<query>{text}`. Two instructions:

- **geo:** "Given the name of a place, retrieve places that are geographically closest to it".
- **generic** (control): "Given a text, retrieve semantically similar text".

City kNN gain / station kNN gain / genre agreement:

| Model | No instruction (output) | Generic instruction | Geo instruction | Geo + F1 + F2 (lambda=0.1) | Oracle w=0.2 (no instruction) |
| --- | --- | --- | --- | --- | --- |
| e5-mistral | 0.240 / 0.188 / 0.520 | 0.188 / 0.133 / 0.617 | 0.316 / 0.206 / 0.515 | 0.349 / 0.231 / 0.516 | 0.289 / 0.257 / 0.505 |
| bge-gemma2 | 0.192 / 0.190 / 0.511 | 0.171 / 0.158 / 0.594 | 0.254 / 0.226 / 0.521 | 0.329 / 0.247 / 0.522 | 0.233 / 0.253 / 0.492 |
| qwen3-8b-emb | 0.132 / 0.166 / 0.639 | 0.064 / 0.083 / 0.679 | 0.210 / 0.200 / 0.543 | 0.229 / 0.209 / 0.537 | 0.192 / 0.231 / 0.613 |

- **The instruction steers which facet cosine similarity tracks.** The geo instruction raises within-country kNN gain on cities (+0.06 to +0.08) and stations (+0.02 to +0.04). The generic instruction lowers geography below the uninstructed vector and raises genre agreement (+0.06 to +0.10). Both are the same frozen model; only the instruction text differs. This is a causal handle on the trade-off that the reflected-geography finding describes.
- **The geo instruction combines with F1+F2.** For e5-mistral and bge-gemma2, city kNN gain is 0.349 and 0.329 (+45% and +62% over the uninstructed output), above the geocoder blend. Station within-country gain is 0.231-0.247, near the blend's 0.253-0.257. Genre agreement is unchanged (0.516 vs 0.520; 0.522 vs 0.511).
- **For Qwen3-8B, the geo instruction trades genre for geography:** genre 0.639 to 0.543.
- **Not yet checked:** other seeds and splits, other phrasings, and whether a geography instruction harms retrieval when the query is about something other than place. Instructions would be applied only to geographic queries, but the index side (documents) here is also instructed.
