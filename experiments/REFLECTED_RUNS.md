# Reflected geography: is within-country geography visible to cosine similarity?

Exploratory run log (started 2026-11-04, branch `research/reflected-geography`). The layer probes show that within-country geography is *decodable* from 4B+ decoder models and their retrieval versions (`experiments/LAYERS_V7_CLAIMS.md`). This line asks whether it is also *reflected* in the similarity that retrieval actually uses: are a city's cosine nearest neighbours geographically close, beyond being in the same country?

**Data discipline.** Everything here uses only the **validation** cities of the layers random-item splits (seeds 1729, 2718, 3141; 6,839 cities each). Centring and principal components are fit on the training cities of the same split. The test cities are not used, so a later frozen experiment can still be scored on them once.

**Measures** (per country with at least 30 validation cities, then averaged over countries; intervals bootstrap over countries):

- *rho*: Spearman correlation, over all same-country city pairs, between similarity and negative great-circle distance.
- *kNN gain*: 1 - (mean km to a city's 10 most similar same-country cities) / (mean km to all its same-country cities). 0 = no better than random, the oracle (true 10 nearest) reaches about 0.74.
- Similarities compared: cosine of the model's last-layer vector (native pooling for retrieval models: last token for decoders, mean for E5/BGE-M3), the probe-decoded positions from the same vector (`country-residual` probe, saved validation predictions), a character 2-4-gram TF-IDF baseline on the text, and a shuffled-position null.

Caveats: E5-base documents are encoded without the `passage: ` prefix and BGE-M3 with mean rather than CLS pooling, so the encoder retrieval rows are not exactly native. Base-model last-token states at the last layer are next-token states, not designed for similarity.

## Jobs

Scripts: `outputs/diagnostics/reflected.py`, `fix.py`, `knn_geo.py` (exploratory, not in git; logs next to them). The analysis is now a module, `evaluation/layer_reflected.py`, with plan `experiments/reflected-v1-dev.json` (validation, reading the layers-v5 dev predictions).

| Job | Role | Outcome | Elapsed |
| --- | --- | --- | --- |
| 3312223 | first pass, seed 1729, retrieval models + some bases | Completed | 0:37 |
| 3312231 | all three seeds, base and retrieval at the same pooling, paired differences | Completed | 1:30 |
| 3312234 | adds centring and all-but-the-top (remove top 5 PCs, fit on training cities) controls | Completed | 20:03 |
| 3312265 | `fix.py`: probe-based fixes on cities (validation) and stations (all; city-trained probe transferred), seed 1729 | Completed | 6:14 |
| 3312273 | `fix.py` rerun adding the country-centroid baseline | Completed | 7:53 |
| 3312274 | `knn_geo.py`: nearest-neighbour geolocation vs the linear probe, random-item, three seeds | Completed | 2:21 |
| 3312282 | `knn_geo.py --split spatial-block` | Completed | 2:19 |
| 3312289, 3312297, 3312310 | `evaluation.layer_reflected --config experiments/reflected-v1-dev.json --mode dev` (the module; 26 cells, both splits, three seeds). 3312297 adds the true-distance ceiling; 3312310 adds the crossover interaction | Completed; `outputs/reflected-v1-dev/dev/report.md`. Reproduces the exploratory numbers | about 3:05 each |
| 3312326 | `reflected_depth.py`: centred-cosine kNN gain at every layer, base vs retrieval (validation, three seeds) | Cancelled after start: layer-0 last-token states are identical across cities, so centring divided by zero. Guarded and resubmitted | |
| 3312327 | `reflected_depth.py` (guarded) | Completed | 30:29 |

## Findings so far (validation, exploratory)

1. Lexical similarity and the shuffled null are at 0. The effect is not shared surface form.
2. **Retrieval training makes cosine reflect within-country geography.** Retrieval minus base, same pooling, paired by country: kNN gain and rho rise in all four 4B+ pairs, at both poolings, on all three seeds, with intervals above 0. The largest jump is Mistral to e5-mistral at last-token pooling: kNN gain 0.04 to 0.24, rho 0.02 to 0.24. There is no change for Qwen3-0.6B or the two encoder pairs, which had little to reflect.
3. **In base models geography is decodable but barely reflected; in retrieval models cosine reflects as much as the probe decodes, or more.** e5-mistral last-token: cosine kNN gain 0.24 vs probe-decoded 0.18. Mistral base last-token: cosine 0.04 vs probe 0.11.
4. **There is headroom.** The best cosine reaches about one third of the oracle kNN gain (e5-mistral 0.24 of 0.74). Qwen3-8B-Embedding reaches 0.13 (18%).
5. **Not an isotropy artefact.** Base-model embeddings are anisotropic, which could hide geography from raw cosine. Centring, or removing the top 5 principal components (fit on training cities), raises base-model kNN gain only modestly; Qwen3-8B base mean-pooled goes from 0.07 to 0.10. After centring, the retrieval model still beats its base in every 4B+ pair, pooling and seed, on both rho and kNN gain (intervals above 0). After all-but-the-top, the same holds at last-token pooling for Qwen3-4B, Qwen3-8B and Mistral. The exceptions are Gemma at last-token pooling (kNN gain about +0.01, intervals spanning 0; rho still above 0) and Qwen3-4B/8B at mean pooling (kNN gain above 0, but rho spans 0 on one or two seeds). e5-mistral (last token) stays +0.20 and Qwen3-8B +0.10.

**Working interpretation (revised by finding 12):** retrieval training does not add within-country geographic knowledge (the probe gains are unchanged, H2). Its output vectors reflect that knowledge in cosine similarity far more than the base model's output vectors do, but only partially: about one third of the oracle at best. At mean pooling this is mostly preservation, not creation; see finding 12.

## Cheap fixes (validation, seed 1729; `fix.py`)

A ridge probe from the unit-normalised last-layer vector to the unit-sphere position is fit on training cities, with alpha chosen on validation. It is applied two ways:

- appending the probe's predicted position to the vector, with weight w;
- amplifying the probe's 3-dimensional subspace by a factor 1 + b.

It is applied unchanged to all 11,833 stations, so stations are fully held out from fitting. Station genre agreement is the share of a station's 10 cosine neighbours that share at least one genre tag (9,371 tagged stations).

| Model | Station neighbour km: cosine / probe boost b=10 / country append w=0.1 / oracle w=0.2 | Genre agreement: same order | City within-country kNN gain: cosine / boost b=10 |
| --- | --- | --- | --- |
| qwen3-4b-embedding | 1175 / 908 / 784 / 546 | 0.554 / 0.537 / 0.538 / 0.533 | 0.110 / 0.113 |
| qwen3-8b-embedding | 1446 / 1033 / 845 / 617 | 0.639 / 0.621 / 0.620 / 0.613 | 0.132 / 0.136 |
| e5-mistral | 893 / 776 / 652 / 434 | 0.520 / 0.515 / 0.511 / 0.505 | 0.240 / 0.244 |
| bge-multilingual-gemma2 | 1046 / 770 / 708 / 493 | 0.511 / 0.499 / 0.496 / 0.492 | 0.192 / 0.206 |

6. **The cheap linear fix is dominated by a trivial country baseline on stations.** Appending the country position from the country code lowers neighbour distance more than the probe fix, at about the same genre cost. Within countries, the probe fix barely changes neighbourhoods (kNN gain +0.003 to +0.014), because the probe's own positions are less locally accurate than cosine already is. The oracle (true coordinates, i.e. a geocoder) is far stronger. As specified so far, the "cheap fix" does not stand as a contribution.

## Nearest-neighbour geolocation vs the linear probe (validation; `knn_geo.py`)

Predict a validation city's position as the mean of its 10 most cosine-similar training cities in the same country (falling back to the country centroid if none). Score gain over the training country centroid with the same block-mean metric and bootstrap as the probes. Last layer, native pooling for retrieval models, mean pooling for bases.

| Model | Probe gain (three seeds) | Nearest-neighbour gain (three seeds) |
| --- | --- | --- |
| qwen3-4b-base (mean) | 98 / 95 / 120 | 70 / 79 / 87 |
| qwen3-4b-embedding | 133 / 112 / 165 | 176 / 166 / 214 |
| qwen3-8b-base (mean) | 130 / 120 / 185 | 77 / 100 / 123 |
| qwen3-8b-embedding | 178 / 157 / 219 | 193 / 202 / 255 |
| mistral-7b-base (mean) | 199 / 191 / 243 | 149 / 174 / 177 |
| e5-mistral | 251 / 239 / 308 | 285 / 281 / 345 |
| gemma2-9b-base (mean) | 219 / 241 / 275 | 202 / 218 / 239 |
| bge-multilingual-gemma2 | 238 / 234 / 289 | 256 / 255 / 331 |
| xlmr-large (mean) | 0 / 1 / 0 | 12 / 25 / 25 |
| bge-m3 (mean) | 0 / 3 / 0 | 22 / 33 / 22 |
| qwen3-0.6b-random (mean) | 0 / 0 / -1 | -4 / 0 / -10 |

7. **A crossover between base and retrieval models (direction consistent; per-run significance mixed).** In point estimates, the linear probe extracts more within-country geography than nearest neighbours in every base model (mean pooling, 12 of 12 model-seed runs). Nearest neighbours in cosine space extract more than the probe in every retrieval model (native pooling, 12 of 12). The paired per-place difference (module run 3312297) has an interval excluding 0 in only 7 of 12 retrieval runs and 4 of 12 base runs. The claim worth testing is therefore the interaction: retrieval training shifts the balance from the probe toward nearest neighbours. For retrieval models, linear-probe gains may be a lower bound on the geographic information present.
8. Encoders show a small nearest-neighbour gain (12-33 km; some seed intervals above 0) where the probe shows none. This is consistent with the narrow H3 failure on the test split: encoders are not exactly zero.
9. Without the same-country restriction, nearest neighbours from base models are far worse than the country centroid (-700 to -1,900 km): their cosine neighbours are often in other countries. Retrieval models' unrestricted neighbours beat the centroid (+207 to +552 km).
10. **The crossover is about interpolation.** On the spatial-block split, which holds out whole 10-degree blocks, the probe beats nearest neighbours for every model and seed. Retrieval models: probe 131-348 km vs nearest-neighbour 68-232. Neighbours can only point to places seen in training, while a linear map can extrapolate. The retrieval-model advantage in finding 7 holds for unseen cities inside known regions.
11. **The crossover as an interaction** (module, paired by place). Compute retrieval minus base of (probe error - neighbour error). On random-item it is positive in all 24 runs (4 pairs × 2 poolings × 3 seeds), with the interval above 0 in 17. It is strongest for Qwen3-4B (6 of 6, +60 to +115 km) and Mistral last-token (3 of 3, +128 to +175 km). It is weakest for Mistral mean-pooled (0 of 3, +14 to +20 km). On spatial-block it is mostly not significant (3 of 24 above 0), consistent with finding 10.
12. **Where the difference arises (depth profile; centred-cosine kNN gain; validation, mean of three seeds).**

    | Pair | Pooling | Base: best layer / last layer | Retrieval: best layer / last layer |
    | --- | --- | --- | --- |
    | Qwen3-4B | mean | 0.094 (L32) / 0.064 | 0.106 (L36) / 0.106 |
    | Qwen3-4B | last | 0.038 (L20) / 0.024 | 0.118 (L32) / 0.113 |
    | Qwen3-8B | mean | 0.131 (L32) / 0.084 | 0.145 (L36) / 0.145 |
    | Qwen3-8B | last | 0.050 (L23) / 0.030 | 0.161 (L32) / 0.141 |
    | Mistral | mean | 0.194 (L27) / 0.130 | 0.219 (L26) / 0.147 |
    | Mistral | last | 0.061 (L12) / 0.044 | 0.308 (L26) / 0.259 |
    | Gemma | mean | 0.240 (L38) / 0.177 | 0.234 (L35) / 0.216 |
    | Gemma | last | 0.181 (L41) / 0.177 | 0.245 (L32) / 0.217 |
    | XLM-R large to BGE-M3 | mean | 0.048 (L7) / 0.034 | 0.046 (L14) / 0.041 |

    With mean pooling, base models' best intermediate layers already reflect within-country geography in cosine about as well as the retrieval models do. Base models lose much of it in their last few layers (Qwen3-8B 0.131 to 0.084; Gemma 0.240 to 0.177), while retrieval models keep it to the output. With last-token pooling, the base model's last token barely reflects geography at any depth (except Gemma's final layers), while the retrieval model's last token does from mid-depth on. This fits embedding training turning the last token into a summary of the text. The comparison in findings 2 and 5 is between output (last-layer) vectors. "Retrieval training surfaces geography" is accurate for those, but at mean pooling the mechanism is mostly that retrieval training stops the final layers from discarding it.
