# fix-v1 test results (run once, 2026-11-04)

Job 3325420 ran `experiments/fix-v1.json` (SHA-256 `0e8267f599ae978bfec6a7c5d12df9d2b326ab139d233f9ca9563dd59cdd6ff2`) at commit `69de7bd`, once, after check suite 3325419 passed (74 tests). Claims and criteria are in `docs/FIX_FREEZE.md`, written and approved before the module produced any number. The report below is copied unchanged from `outputs/fix-v1/final/report.md`.

## Verdicts

| # | Criterion (frozen) | Result | Verdict |
| --- | --- | --- | --- |
| F1 | e5-mistral and bge-multilingual-gemma2: full fix minus cosine, city within-country kNN gain paired by country, 95% CI above 0 in all 12 runs | 12/12. Differences +0.105 to +0.131 (e5) and +0.120 to +0.149 (gemma); lowest CI bound +0.084 | **Supported** |
| F2 | Same models: STS-B test Spearman of the full fix at least cosine's minus 0.03, in all 12 runs | 12/12. Change +0.013 to +0.014 (e5) and +0.023 to +0.028 (gemma); STS did not drop | **Supported** |
| F3 | Descriptive | See below | Reported |

## Descriptive (F3)

- **Components.** For e5-mistral, the geography instruction alone gives most of the gain: random-item city kNN goes from 0.256 to 0.333, and to 0.370 with all three components. For bge-gemma2, instruction alone (0.275) and fusion plus probe alone (0.276) contribute about equally, and together they reach 0.345.
- **Baselines.**
  - The country-code append cannot change within-country rankings, and its city kNN equals cosine's.
  - On cities, the full fix exceeds the geocoder blend with true coordinates at w=0.2 (e5 0.370 vs 0.307; gemma 0.345 vs 0.246).
  - On stations the geocoder stays ahead (e5 0.257 vs 0.231; gemma 0.253 vs 0.247).
- **Qwen3-Embedding-8B.** City kNN rises in all 6 runs, with every CI above 0 (+0.077 to +0.121). STS changes by -0.020 to -0.022, which is within the tolerance, but station genre agreement falls by about 0.10 (0.639 to 0.535). Here the fix trades semantic structure for geography.
- **Transfer and genre.** Station kNN rises for all three models (e5 0.188 to 0.231, gemma 0.190 to 0.247, qwen 0.166 to 0.206). Genre agreement changes by -0.005 to +0.016 for e5 and gemma. Stations were used during selection, so these are not claims.
- **Caveat on F2.** The cosine baseline is uninstructed, and e5 and bge are normally run with a task instruction. The STS gain is therefore likely an any-instruction effect, not a geographic one. For gemma, the full fix scores below instruction alone (0.846 vs 0.860), so fusion plus probe costs some STS relative to the instruction by itself.

## Report

# Coordinate-free fixes, fix-v1 final

Instruction: "Given the name of a place, retrieve places that are geographically closest to it". Within-country kNN gain for cities (evaluated split) and stations (all), and station genre agreement; mean over seeds [1729, 2718, 3141].

| Model | Split | Method | City kNN | Station kNN | Genre | STS rho | Fusion layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e5-mistral | random-item | cosine | 0.256 | 0.188 | 0.520 | 0.845 | None / None / None |
| e5-mistral | random-item | country | 0.256 | 0.188 | 0.511 | n/a | None / None / None |
| e5-mistral | random-item | oracle | 0.307 | 0.257 | 0.505 | n/a | None / None / None |
| e5-mistral | random-item | fusion-probe | 0.321 | 0.212 | 0.536 | 0.863 | 26 / 26 / 26 |
| e5-mistral | random-item | instruction | 0.333 | 0.206 | 0.515 | 0.859 | None / None / None |
| e5-mistral | random-item | instruction-fusion-probe | 0.370 | 0.231 | 0.516 | 0.858 | 26 / 28 / 26 |
| e5-mistral | spatial-block | cosine | 0.260 | 0.188 | 0.520 | 0.845 | None / None / None |
| e5-mistral | spatial-block | country | 0.260 | 0.188 | 0.510 | n/a | None / None / None |
| e5-mistral | spatial-block | oracle | 0.295 | 0.257 | 0.505 | n/a | None / None / None |
| e5-mistral | spatial-block | fusion-probe | 0.326 | 0.209 | 0.538 | 0.862 | 26 / 28 / 26 |
| e5-mistral | spatial-block | instruction | 0.345 | 0.206 | 0.515 | 0.859 | None / None / None |
| e5-mistral | spatial-block | instruction-fusion-probe | 0.382 | 0.227 | 0.518 | 0.858 | 26 / 26 / 26 |
| bge-multilingual-gemma2 | random-item | cosine | 0.202 | 0.190 | 0.511 | 0.822 | None / None / None |
| bge-multilingual-gemma2 | random-item | country | 0.202 | 0.190 | 0.497 | n/a | None / None / None |
| bge-multilingual-gemma2 | random-item | oracle | 0.246 | 0.253 | 0.492 | n/a | None / None / None |
| bge-multilingual-gemma2 | random-item | fusion-probe | 0.276 | 0.212 | 0.540 | 0.844 | 33 / 33 / 33 |
| bge-multilingual-gemma2 | random-item | instruction | 0.275 | 0.226 | 0.521 | 0.860 | None / None / None |
| bge-multilingual-gemma2 | random-item | instruction-fusion-probe | 0.345 | 0.247 | 0.523 | 0.846 | 31 / 31 / 31 |
| bge-multilingual-gemma2 | spatial-block | cosine | 0.232 | 0.190 | 0.511 | 0.822 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | country | 0.232 | 0.190 | 0.496 | n/a | None / None / None |
| bge-multilingual-gemma2 | spatial-block | oracle | 0.261 | 0.253 | 0.492 | n/a | None / None / None |
| bge-multilingual-gemma2 | spatial-block | fusion-probe | 0.302 | 0.210 | 0.546 | 0.844 | 31 / 33 / 33 |
| bge-multilingual-gemma2 | spatial-block | instruction | 0.291 | 0.226 | 0.521 | 0.860 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | instruction-fusion-probe | 0.364 | 0.241 | 0.525 | 0.848 | 31 / 33 / 31 |
| qwen3-8b-embedding | random-item | cosine | 0.133 | 0.166 | 0.639 | 0.883 | None / None / None |
| qwen3-8b-embedding | random-item | country | 0.133 | 0.166 | 0.620 | n/a | None / None / None |
| qwen3-8b-embedding | random-item | oracle | 0.199 | 0.231 | 0.613 | n/a | None / None / None |
| qwen3-8b-embedding | random-item | fusion-probe | 0.166 | 0.168 | 0.622 | 0.891 | 32 / 32 / 32 |
| qwen3-8b-embedding | random-item | instruction | 0.232 | 0.200 | 0.543 | 0.863 | None / None / None |
| qwen3-8b-embedding | random-item | instruction-fusion-probe | 0.251 | 0.206 | 0.535 | 0.861 | 32 / 32 / 32 |
| qwen3-8b-embedding | spatial-block | cosine | 0.142 | 0.166 | 0.639 | 0.883 | None / None / None |
| qwen3-8b-embedding | spatial-block | country | 0.142 | 0.166 | 0.620 | n/a | None / None / None |
| qwen3-8b-embedding | spatial-block | oracle | 0.194 | 0.231 | 0.613 | n/a | None / None / None |
| qwen3-8b-embedding | spatial-block | fusion-probe | 0.160 | 0.167 | 0.624 | 0.891 | 32 / 32 / 32 |
| qwen3-8b-embedding | spatial-block | instruction | 0.218 | 0.200 | 0.543 | 0.863 | None / None / None |
| qwen3-8b-embedding | spatial-block | instruction-fusion-probe | 0.231 | 0.205 | 0.538 | 0.862 | 32 / 32 / 34 |

## Full fix minus cosine, city kNN gain paired by country (95% CI)

| Model | Split | Seed | Difference | Genre change | STS change |
| --- | --- | --- | --- | --- | --- |
| e5-mistral | random-item | 1729 | +0.105 [+0.086, +0.123] | -0.003 | +0.013 |
| e5-mistral | random-item | 2718 | +0.120 [+0.101, +0.141] | -0.004 | +0.013 |
| e5-mistral | random-item | 3141 | +0.116 [+0.097, +0.133] | -0.005 | +0.014 |
| e5-mistral | spatial-block | 1729 | +0.108 [+0.084, +0.133] | -0.001 | +0.013 |
| e5-mistral | spatial-block | 2718 | +0.128 [+0.106, +0.151] | -0.003 | +0.013 |
| e5-mistral | spatial-block | 3141 | +0.131 [+0.107, +0.152] | -0.002 | +0.014 |
| bge-multilingual-gemma2 | random-item | 1729 | +0.147 [+0.130, +0.165] | +0.010 | +0.025 |
| bge-multilingual-gemma2 | random-item | 2718 | +0.134 [+0.109, +0.158] | +0.012 | +0.023 |
| bge-multilingual-gemma2 | random-item | 3141 | +0.149 [+0.128, +0.170] | +0.013 | +0.024 |
| bge-multilingual-gemma2 | spatial-block | 1729 | +0.120 [+0.097, +0.144] | +0.014 | +0.025 |
| bge-multilingual-gemma2 | spatial-block | 2718 | +0.129 [+0.107, +0.153] | +0.016 | +0.028 |
| bge-multilingual-gemma2 | spatial-block | 3141 | +0.146 [+0.123, +0.169] | +0.012 | +0.025 |
| qwen3-8b-embedding | random-item | 1729 | +0.120 [+0.095, +0.143] | -0.103 | -0.021 |
| qwen3-8b-embedding | random-item | 2718 | +0.121 [+0.095, +0.146] | -0.106 | -0.022 |
| qwen3-8b-embedding | random-item | 3141 | +0.114 [+0.086, +0.144] | -0.103 | -0.022 |
| qwen3-8b-embedding | spatial-block | 1729 | +0.077 [+0.048, +0.109] | -0.101 | -0.021 |
| qwen3-8b-embedding | spatial-block | 2718 | +0.087 [+0.062, +0.113] | -0.103 | -0.021 |
| qwen3-8b-embedding | spatial-block | 3141 | +0.103 [+0.080, +0.127] | -0.101 | -0.020 |
