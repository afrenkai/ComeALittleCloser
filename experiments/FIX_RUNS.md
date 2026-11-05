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
| 3315040 (array 0-5) | Re-encoding with two alternative geography phrasings (geo2, geo3) × 3 models | Completed | |
| 3315041 (array 0-17) | Harness on geo-instructed vectors × 3 models × 2 splits × 3 seeds | Completed | |
| 3316551 (array 0-5) | Harness on geo2 and geo3 vectors (seed 1729, random-item) | Completed | 1:24-3:57 |

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

## F5 robustness

**Seeds and splits** (geo instruction, job 3315041). Change vs uninstructed cosine, range over three seeds; city kNN gain, station kNN gain, genre agreement:

| Model | Split | Geo instruction alone | Geo + F1 + F2 (lambda=0.1) |
| --- | --- | --- | --- |
| e5-mistral | random-item | city +0.071..+0.079, station +0.018, genre -0.005 | city +0.106..+0.112 (0.361), station +0.041..+0.043, genre -0.004..-0.002 |
| e5-mistral | spatial-block | city +0.074..+0.084, station +0.018, genre -0.005 | city +0.114..+0.131 (0.358), station +0.039..+0.041, genre -0.003..+0.001 |
| bge-gemma2 | random-item | city +0.062..+0.075, station +0.036, genre +0.010 | city +0.136..+0.149 (0.344), station +0.055..+0.059, genre +0.011..+0.013 |
| bge-gemma2 | spatial-block | city +0.044..+0.055, station +0.036, genre +0.010 | city +0.122..+0.140 (0.344), station +0.054..+0.056, genre +0.012..+0.014 |
| qwen3-8b-emb | random-item | city +0.078..+0.099, station +0.034, genre -0.096 | city +0.097..+0.127 (0.247), station +0.040..+0.043, genre -0.103..-0.082 |
| qwen3-8b-emb | spatial-block | city +0.067..+0.082, station +0.034, genre -0.096 | city +0.076..+0.102 (0.215), station +0.036..+0.042, genre -0.102..-0.099 |

Station and genre changes are identical across seeds for the instruction alone, because stations are not split. For e5-mistral and bge-gemma2, the geo instruction with F1+F2 raises city within-country kNN gain by 45-70% relative and station within-country gain by 0.04-0.06, with genre agreement within ±0.014, in all 12 runs. That includes held-out regions. Qwen3-8B gains similar geography but loses about 0.1 genre agreement.

**Phrasing** (seed 1729, random-item). City kNN gain / station kNN gain / genre agreement, uninstructed vs three geographic wordings:

| Model | None | geo: "...retrieve places that are geographically closest to it" | geo2: "Represent this place for finding places located near it" | geo3: "...retrieve other locations in the same region of the country" |
| --- | --- | --- | --- | --- |
| e5-mistral | 0.240 / 0.188 / 0.520 | 0.316 / 0.206 / 0.515 | 0.280 / 0.197 / 0.511 | 0.329 / 0.206 / 0.512 |
| bge-gemma2 | 0.192 / 0.190 / 0.511 | 0.254 / 0.226 / 0.521 | 0.240 / 0.202 / 0.516 | 0.276 / 0.221 / 0.516 |
| qwen3-8b-emb | 0.132 / 0.166 / 0.639 | 0.210 / 0.200 / 0.543 | 0.146 / 0.150 / 0.551 | 0.171 / 0.182 / 0.493 |

All three geographic wordings raise within-country geography in all three models. The size depends on wording: +0.04 to +0.09 on cities for e5-mistral and bge-gemma2. With F1+F2 added, e5-mistral reaches 0.324-0.352 and bge-gemma2 0.314-0.346 across the three wordings. Wording is a free parameter, so a frozen evaluation must fix it in advance; we would fix "geo", the first and pre-chosen wording.

## Where this stands (exploratory, validation)

The best coordinate-free combination is a geography instruction on both sides, plus fusion with an intermediate layer, plus a city-trained distance probe. For e5-mistral and bge-gemma2 it raises within-country neighbour quality from 0.20-0.25 to 0.34-0.36. That is about half of the 0.74 ceiling, up from about a third, with no genre cost. It also improves stations, whose texts it was never fit on.

Remaining before claiming it:

1. A frozen test on held-out cities.
2. A semantic cost measured on data not used for choosing lambda, the weight, the layer or the instruction. The stations have been used for selection.
3. A check that instructed documents do not hurt non-geographic queries in a realistic retrieval setting.
4. Turning `fixes.py` into a tested module.

## fix-v1: module, freeze and test run

The fix is now `evaluation/geo_fix.py`, with instruction-conditioned extraction in `evaluation/instructed_extract.py`, schemas in `evaluation/fix_schemas.py`, tests in `tests/test_geo_fix.py` and GPU jobs via `ops/fix-extract.sbatch`. Claims and criteria were frozen in `docs/FIX_FREEZE.md` (commit `ed4a5bf`) before the module produced any number. The plans are `experiments/fix-v1-dev.json` (validation, layers-v5 splits) and `experiments/fix-v1.json` (test, layers-v7; SHA-256 `0e8267f599ae978bfec6a7c5d12df9d2b326ab139d233f9ca9563dd59cdd6ff2`). Their settings are identical. The semantic-cost set is STS-Benchmark (`data/raw/stsbenchmark`, dev 1,500 pairs and test 1,379 pairs, checksums in the plan).

| Job | Role | Outcome | Elapsed |
| --- | --- | --- | --- |
| 3318253 (array 0-2) | Module extraction (instructed cities, stations, STS dev/test plain and instructed) for e5-mistral, bge-gemma2, qwen3-8b-embedding | Failed, all three. A guard refused instructed station texts over 256 tokens; the original layer pipeline truncates and records the count (stations: 10-47 truncated per model). Code fixed to record `truncated_items` in the manifest, matching the original pipeline, and to skip collections already extracted; plan unchanged. Partial city outputs were re-extracted | 10:49-12:07 |
| 3318624 (array 0-2) | Module extraction, resubmitted | Completed, 18 encodings with manifests. Truncated instructed stations: e5-mistral 89, bge-gemma2 21, qwen3 35 | 15:21-15:46 |
| 3319730 | Module dev run (`fix-v1-dev.json`, `short`) | Cancelled while pending. Slurm estimated a 3-hour wait for 128 GB on `short`, so the job was resubmitted on the idle `gtml` node. Before submission, a code fix sliced the plain city and station arrays to the manifest's kept layers so they match the STS arrays, and added a check that the plain and instructed layers agree. Plain fusion candidates are therefore the kept layers (upper half, every second layer, plus output), as for instructed | pending 15:48-16:51 |
| 3322692 | Module dev run, resubmitted on `gtml` (L40S requested; CPU work) | Completed. Reproduces the exploration: e5-mistral full fix 0.360 / 0.357 city kNN (random-item / spatial-block) vs cosine 0.252 / 0.237; bge-gemma2 0.344 / 0.341 vs 0.204 / 0.213; qwen3-8b 0.246 / 0.216 vs 0.133 / 0.123. The paired CI is above 0 in all 18 runs. STS-B dev Spearman change of the full fix vs cosine: e5 +0.020 to +0.022, gemma +0.030 to +0.031, qwen -0.009 to -0.012. Genre change: e5 -0.005 to -0.001, gemma +0.010 to +0.016, qwen about -0.10. The plain cosine baseline is uninstructed, while e5 and bge are usually run with a task instruction, so part of the STS gain is likely any-instruction rather than geography. Report: `outputs/fix-v1-dev/dev/report.md` | 16:52-17:28 |
| 3325419 | Check suite (ruff, black, pytest) at `69de7bd` | Completed, 74 passed | 0:28 |
| 3325420 | **Final (test), run once**: `fix-v1.json` (SHA-256 verified `0e8267f5…cdd6ff2`), on `gtml`, `afterok:3325419`. `outputs/fix-v1` did not exist before submission | Completed. F1 supported (12/12, lowest CI bound +0.084); F2 supported (12/12, STS +0.013 to +0.028). Results: `experiments/FIX_V1_RESULTS.md` | 35:20 |

Two details of fix-v1 that `docs/FIX_FREEZE.md` does not spell out:
- **Fusion candidates.** These are the kept layers (every second layer of the upper half, plus the output), not every layer.
- **Distance normalisation.** The distance probe's squared distances are divided by their mean over the set being scored (the evaluated cities, all stations, or the STS sentences). That uses no labels, but it is computed on the scored set, not on training cities. Both behaviours were fixed before the test run, and the code at the frozen commit is the reference. The paper's method section describes both.

## fix-baselines-v1: validation-only baselines and ablations (2026-11-04)

These answer reviewer questions about fix-v1. Everything here is validation only: test cities and STS-B test are not read. `experiments/fix-generic-v1-dev.json` lists only the STS dev split, and extraction now encodes only the STS splits a plan lists. These are not frozen claims; the fix-v1 test is already used, so any new claim needs fresh held-out places.

- **Similarity-instruction baseline.** Each model's standard STS instruction, "Retrieve semantically similar text.", on both sides (`experiments/fix-generic-v1-dev.json`, features in `data/processed/instructed-generic-v1`). This separates an any-instruction effect from the geographic wording. The exploration used a different control wording ("Given a text, retrieve semantically similar text").
- **Factorial ablation.** Instruction × layer fusion × distance probe. "Probe" alone is the distance map on the final layer with no fusion.
- **Feuerstein et al. (2026) axes.** Their Eq. 2, d = Σ(p₁ − p₂), on plain unit-norm output vectors (no centring, as in the paper). Latitude and longitude axes come from training cities. The paper uses K = 2 unspecified seed pairs, so we use every training city in a pair (upper half against lower half), which is at least as stable. There are two variants: global pairs, and pairs within each country. Scores are min-max mapped to the training coordinate range (their mapping) and turned into positions. They are used alone, and blended with cosine at w = 0.2 and 1.0 in the oracle's form.

| Job | Role | Outcome | Elapsed |
| --- | --- | --- | --- |
| 3331578 (array 0-2) | Similarity-instruction extraction (cities, stations, STS dev plain and instructed) for e5-mistral, bge-gemma2, qwen3-8b-embedding; gtml L40S | Submitted | |
| 3331775 | `fix_baselines --config experiments/fix-baselines-v1-dev.json`, on `gtml`, `afterok:3331578` | Submitted | |
