# fresh-v1 results (test part, run once on 2026-11-04)

Claims: `docs/FRESH_FREEZE.md` (approved before the check part was scored). Job 3349703, 55:46. Runs log: `experiments/FRESH_RUNS.md`. Raw scores: `outputs/fresh-v1/test/scores.jsonl`.

## Verdicts

| Claim | Criterion | Result | Verdict |
| --- | --- | --- | --- |
| G1 | Fix minus plain cosine, CI above 0 in all 12 runs (e5-mistral, bge-gemma2) | 12 of 12. e5 +0.084 to +0.109, gemma +0.098 to +0.135; lowest bound +0.057 | **Supported** |
| G2 | Fix minus the same pipeline with the similarity instruction, CI above 0 in all 12 | 12 of 12. e5 +0.078 to +0.109, gemma +0.057 to +0.080; lowest bound +0.043 | **Supported** |
| G3 | Fix minus within-country seed-pair axes blended at w = 0.2, CI above 0 in all 12 | 12 of 12. e5 +0.079 to +0.095, gemma +0.083 to +0.114; lowest bound +0.057 | **Supported** |

## G4 (descriptive)

- **Qwen3-Embedding-8B** also clears all three criteria in 6 of 6 runs (G1 +0.063 to +0.089, G2 +0.055 to +0.081, G3 +0.062 to +0.089; lowest bound +0.033).
- **Levels.** On towns the fix reaches 0.27-0.30 within-country kNN gain (e5, gemma), up from 0.15-0.21 for cosine. On the original test cities it was 0.35-0.38, up from 0.20-0.26. Gains are lower for towns, as stated in advance, but similar in size.
- **The similarity instruction again lowers geography below no instruction** in every model and split. For example, e5-mistral RI goes from 0.172 to 0.121.
- **Layer fusion helps** in all 18 runs (+0.006 to +0.047; every interval above 0).
- **The distance probe adds little on towns.** It helps bge-gemma2 in 6 of 6 runs (+0.008 to +0.013), e5-mistral in 3 of 6 (−0.002 to +0.005), and Qwen3-8B in 0 of 6 (−0.001 to +0.002). On validation cities it helped in 15 of 18 runs. Its benefit transfers less to small towns than fusion and the instruction do.
- **The fix beats the geocoder oracle** (cosine blended with true coordinates, w = 0.2) in every model and split: by +0.049 to +0.085 for e5 and gemma, and by +0.006 to +0.009 for Qwen3-8B.
- **Feuerstein axes.** Alone they score 0.02-0.10. Blended at w = 0.2, the within-country axes add at most +0.022 over cosine (bge-gemma2 RI), and the global axes lower it slightly (−0.001 to −0.005).

## Report

# Fresh towns, fresh-v1, test part

GeoNames towns of 5,000-14,999 people that are not in `cities15000`, scored with components fit on the original training cities. Random-item scores all towns of the part; spatial-block scores only towns outside that split's training blocks. Means over seeds [1729, 2718, 3141].

## Claims (e5-mistral, bge-multilingual-gemma2)

| Claim | Runs with CI above 0 | Verdict |
| --- | --- | --- |
| G1 | 12 of 12 | Supported |
| G2 | 12 of 12 | Supported |
| G3 | 12 of 12 | Supported |

| Model | Split | Method | Weight | Towns | Countries | City kNN | City rho | Fusion layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5-mistral | random-item | cosine |  | 24378 | 85 | 0.172 | 0.146 | None / None / None |
| e5-mistral | random-item | fusion |  | 24378 | 85 | 0.219 | 0.186 | 26 / 26 / 26 |
| e5-mistral | random-item | probe |  | 24378 | 85 | 0.207 | 0.195 | None / None / None |
| e5-mistral | random-item | fusion-probe |  | 24378 | 85 | 0.226 | 0.203 | 26 / 26 / 26 |
| e5-mistral | random-item | instruction |  | 24378 | 85 | 0.235 | 0.202 | None / None / None |
| e5-mistral | random-item | instruction-fusion |  | 24378 | 85 | 0.265 | 0.244 | 26 / 28 / 26 |
| e5-mistral | random-item | instruction-probe |  | 24378 | 85 | 0.254 | 0.237 | None / None / None |
| e5-mistral | random-item | instruction-fusion-probe |  | 24378 | 85 | 0.270 | 0.255 | 26 / 28 / 26 |
| e5-mistral | random-item | generic-instruction |  | 24378 | 85 | 0.121 | 0.109 | None / None / None |
| e5-mistral | random-item | generic-instruction-fusion-probe |  | 24378 | 85 | 0.181 | 0.172 | 26 / 26 / 26 |
| e5-mistral | random-item | oracle | 0.2 | 24378 | 85 | 0.213 | 0.192 | None / None / None |
| e5-mistral | random-item | feuerstein-global |  | 24378 | 85 | 0.035 | 0.038 | None / None / None |
| e5-mistral | random-item | feuerstein-global | 0.2 | 24378 | 85 | 0.167 | 0.128 | None / None / None |
| e5-mistral | random-item | feuerstein-global | 1.0 | 24378 | 85 | 0.151 | 0.094 | None / None / None |
| e5-mistral | random-item | feuerstein-country |  | 24378 | 85 | 0.100 | 0.115 | None / None / None |
| e5-mistral | random-item | feuerstein-country | 0.2 | 24378 | 85 | 0.184 | 0.163 | None / None / None |
| e5-mistral | random-item | feuerstein-country | 1.0 | 24378 | 85 | 0.166 | 0.139 | None / None / None |
| e5-mistral | spatial-block | cosine |  | 10605 | 34 | 0.207 | 0.171 | None / None / None |
| e5-mistral | spatial-block | fusion |  | 10605 | 34 | 0.253 | 0.213 | 26 / 28 / 26 |
| e5-mistral | spatial-block | probe |  | 10605 | 34 | 0.235 | 0.206 | None / None / None |
| e5-mistral | spatial-block | fusion-probe |  | 10605 | 34 | 0.257 | 0.224 | 26 / 28 / 26 |
| e5-mistral | spatial-block | instruction |  | 10605 | 34 | 0.271 | 0.234 | None / None / None |
| e5-mistral | spatial-block | instruction-fusion |  | 10605 | 34 | 0.302 | 0.273 | 26 / 26 / 26 |
| e5-mistral | spatial-block | instruction-probe |  | 10605 | 34 | 0.286 | 0.254 | None / None / None |
| e5-mistral | spatial-block | instruction-fusion-probe |  | 10605 | 34 | 0.303 | 0.276 | 26 / 26 / 26 |
| e5-mistral | spatial-block | generic-instruction |  | 10605 | 34 | 0.147 | 0.135 | None / None / None |
| e5-mistral | spatial-block | generic-instruction-fusion-probe |  | 10605 | 34 | 0.209 | 0.186 | 20 / 26 / 26 |
| e5-mistral | spatial-block | oracle | 0.2 | 10605 | 34 | 0.253 | 0.222 | None / None / None |
| e5-mistral | spatial-block | feuerstein-global |  | 10605 | 34 | 0.049 | 0.053 | None / None / None |
| e5-mistral | spatial-block | feuerstein-global | 0.2 | 10605 | 34 | 0.202 | 0.154 | None / None / None |
| e5-mistral | spatial-block | feuerstein-global | 1.0 | 10605 | 34 | 0.180 | 0.112 | None / None / None |
| e5-mistral | spatial-block | feuerstein-country |  | 10605 | 34 | 0.102 | 0.117 | None / None / None |
| e5-mistral | spatial-block | feuerstein-country | 0.2 | 10605 | 34 | 0.217 | 0.180 | None / None / None |
| e5-mistral | spatial-block | feuerstein-country | 1.0 | 10605 | 34 | 0.195 | 0.148 | None / None / None |
| bge-multilingual-gemma2 | random-item | cosine |  | 24378 | 85 | 0.152 | 0.095 | None / None / None |
| bge-multilingual-gemma2 | random-item | fusion |  | 24378 | 85 | 0.198 | 0.135 | 33 / 33 / 33 |
| bge-multilingual-gemma2 | random-item | probe |  | 24378 | 85 | 0.201 | 0.158 | None / None / None |
| bge-multilingual-gemma2 | random-item | fusion-probe |  | 24378 | 85 | 0.220 | 0.168 | 33 / 33 / 33 |
| bge-multilingual-gemma2 | random-item | instruction |  | 24378 | 85 | 0.192 | 0.166 | None / None / None |
| bge-multilingual-gemma2 | random-item | instruction-fusion |  | 24378 | 85 | 0.255 | 0.218 | 31 / 31 / 31 |
| bge-multilingual-gemma2 | random-item | instruction-probe |  | 24378 | 85 | 0.231 | 0.218 | None / None / None |
| bge-multilingual-gemma2 | random-item | instruction-fusion-probe |  | 24378 | 85 | 0.267 | 0.240 | 31 / 31 / 31 |
| bge-multilingual-gemma2 | random-item | generic-instruction |  | 24378 | 85 | 0.111 | 0.093 | None / None / None |
| bge-multilingual-gemma2 | random-item | generic-instruction-fusion-probe |  | 24378 | 85 | 0.203 | 0.169 | 31 / 31 / 31 |
| bge-multilingual-gemma2 | random-item | oracle | 0.2 | 24378 | 85 | 0.182 | 0.126 | None / None / None |
| bge-multilingual-gemma2 | random-item | feuerstein-global |  | 24378 | 85 | 0.024 | 0.019 | None / None / None |
| bge-multilingual-gemma2 | random-item | feuerstein-global | 0.2 | 24378 | 85 | 0.148 | 0.088 | None / None / None |
| bge-multilingual-gemma2 | random-item | feuerstein-global | 1.0 | 24378 | 85 | 0.136 | 0.064 | None / None / None |
| bge-multilingual-gemma2 | random-item | feuerstein-country |  | 24378 | 85 | 0.090 | 0.098 | None / None / None |
| bge-multilingual-gemma2 | random-item | feuerstein-country | 0.2 | 24378 | 85 | 0.173 | 0.129 | None / None / None |
| bge-multilingual-gemma2 | random-item | feuerstein-country | 1.0 | 24378 | 85 | 0.155 | 0.116 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | cosine |  | 10605 | 34 | 0.184 | 0.118 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | fusion |  | 10605 | 34 | 0.232 | 0.152 | 31 / 33 / 33 |
| bge-multilingual-gemma2 | spatial-block | probe |  | 10605 | 34 | 0.228 | 0.173 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | fusion-probe |  | 10605 | 34 | 0.250 | 0.181 | 31 / 33 / 33 |
| bge-multilingual-gemma2 | spatial-block | instruction |  | 10605 | 34 | 0.230 | 0.203 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | instruction-fusion |  | 10605 | 34 | 0.289 | 0.249 | 31 / 33 / 31 |
| bge-multilingual-gemma2 | spatial-block | instruction-probe |  | 10605 | 34 | 0.262 | 0.244 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | instruction-fusion-probe |  | 10605 | 34 | 0.299 | 0.266 | 31 / 33 / 31 |
| bge-multilingual-gemma2 | spatial-block | generic-instruction |  | 10605 | 34 | 0.132 | 0.111 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | generic-instruction-fusion-probe |  | 10605 | 34 | 0.227 | 0.180 | 31 / 31 / 31 |
| bge-multilingual-gemma2 | spatial-block | oracle | 0.2 | 10605 | 34 | 0.221 | 0.154 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | feuerstein-global |  | 10605 | 34 | 0.032 | 0.027 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | feuerstein-global | 0.2 | 10605 | 34 | 0.181 | 0.112 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | feuerstein-global | 1.0 | 10605 | 34 | 0.164 | 0.082 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | feuerstein-country |  | 10605 | 34 | 0.083 | 0.085 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | feuerstein-country | 0.2 | 10605 | 34 | 0.197 | 0.135 | None / None / None |
| bge-multilingual-gemma2 | spatial-block | feuerstein-country | 1.0 | 10605 | 34 | 0.176 | 0.112 | None / None / None |
| qwen3-8b-embedding | random-item | cosine |  | 24378 | 85 | 0.072 | 0.029 | None / None / None |
| qwen3-8b-embedding | random-item | fusion |  | 24378 | 85 | 0.089 | 0.050 | 32 / 32 / 32 |
| qwen3-8b-embedding | random-item | probe |  | 24378 | 85 | 0.082 | 0.045 | None / None / None |
| qwen3-8b-embedding | random-item | fusion-probe |  | 24378 | 85 | 0.093 | 0.058 | 32 / 32 / 32 |
| qwen3-8b-embedding | random-item | instruction |  | 24378 | 85 | 0.122 | 0.079 | None / None / None |
| qwen3-8b-embedding | random-item | instruction-fusion |  | 24378 | 85 | 0.137 | 0.102 | 32 / 32 / 32 |
| qwen3-8b-embedding | random-item | instruction-probe |  | 24378 | 85 | 0.127 | 0.093 | None / None / None |
| qwen3-8b-embedding | random-item | instruction-fusion-probe |  | 24378 | 85 | 0.139 | 0.108 | 32 / 32 / 32 |
| qwen3-8b-embedding | random-item | generic-instruction |  | 24378 | 85 | 0.050 | 0.024 | None / None / None |
| qwen3-8b-embedding | random-item | generic-instruction-fusion-probe |  | 24378 | 85 | 0.082 | 0.052 | 32 / 32 / 32 |
| qwen3-8b-embedding | random-item | oracle | 0.2 | 24378 | 85 | 0.130 | 0.059 | None / None / None |
| qwen3-8b-embedding | random-item | feuerstein-global |  | 24378 | 85 | 0.022 | 0.030 | None / None / None |
| qwen3-8b-embedding | random-item | feuerstein-global | 0.2 | 24378 | 85 | 0.071 | 0.032 | None / None / None |
| qwen3-8b-embedding | random-item | feuerstein-global | 1.0 | 24378 | 85 | 0.062 | 0.033 | None / None / None |
| qwen3-8b-embedding | random-item | feuerstein-country |  | 24378 | 85 | 0.034 | 0.041 | None / None / None |
| qwen3-8b-embedding | random-item | feuerstein-country | 0.2 | 24378 | 85 | 0.071 | 0.040 | None / None / None |
| qwen3-8b-embedding | random-item | feuerstein-country | 1.0 | 24378 | 85 | 0.063 | 0.043 | None / None / None |
| qwen3-8b-embedding | spatial-block | cosine |  | 10605 | 34 | 0.094 | 0.046 | None / None / None |
| qwen3-8b-embedding | spatial-block | fusion |  | 10605 | 34 | 0.115 | 0.070 | 32 / 32 / 32 |
| qwen3-8b-embedding | spatial-block | probe |  | 10605 | 34 | 0.103 | 0.059 | None / None / None |
| qwen3-8b-embedding | spatial-block | fusion-probe |  | 10605 | 34 | 0.117 | 0.076 | 32 / 32 / 32 |
| qwen3-8b-embedding | spatial-block | instruction |  | 10605 | 34 | 0.152 | 0.106 | None / None / None |
| qwen3-8b-embedding | spatial-block | instruction-fusion |  | 10605 | 34 | 0.167 | 0.126 | 32 / 32 / 34 |
| qwen3-8b-embedding | spatial-block | instruction-probe |  | 10605 | 34 | 0.157 | 0.116 | None / None / None |
| qwen3-8b-embedding | spatial-block | instruction-fusion-probe |  | 10605 | 34 | 0.167 | 0.130 | 32 / 32 / 34 |
| qwen3-8b-embedding | spatial-block | generic-instruction |  | 10605 | 34 | 0.063 | 0.034 | None / None / None |
| qwen3-8b-embedding | spatial-block | generic-instruction-fusion-probe |  | 10605 | 34 | 0.100 | 0.065 | 32 / 32 / 32 |
| qwen3-8b-embedding | spatial-block | oracle | 0.2 | 10605 | 34 | 0.162 | 0.079 | None / None / None |
| qwen3-8b-embedding | spatial-block | feuerstein-global |  | 10605 | 34 | 0.034 | 0.042 | None / None / None |
| qwen3-8b-embedding | spatial-block | feuerstein-global | 0.2 | 10605 | 34 | 0.094 | 0.050 | None / None / None |
| qwen3-8b-embedding | spatial-block | feuerstein-global | 1.0 | 10605 | 34 | 0.084 | 0.050 | None / None / None |
| qwen3-8b-embedding | spatial-block | feuerstein-country |  | 10605 | 34 | 0.049 | 0.051 | None / None / None |
| qwen3-8b-embedding | spatial-block | feuerstein-country | 0.2 | 10605 | 34 | 0.095 | 0.055 | None / None / None |
| qwen3-8b-embedding | spatial-block | feuerstein-country | 1.0 | 10605 | 34 | 0.086 | 0.058 | None / None / None |

## G1: Fix vs cosine (instruction-fusion-probe minus cosine, city kNN gain paired by country)

| Model | Split | Seed | Difference [95% CI] |
| --- | --- | --- | --- |
| e5-mistral | random-item | 1729 | +0.099 [+0.079, +0.118] |
| e5-mistral | random-item | 2718 | +0.096 [+0.077, +0.116] |
| e5-mistral | random-item | 3141 | +0.099 [+0.079, +0.118] |
| e5-mistral | spatial-block | 1729 | +0.084 [+0.057, +0.117] |
| e5-mistral | spatial-block | 2718 | +0.095 [+0.073, +0.119] |
| e5-mistral | spatial-block | 3141 | +0.109 [+0.082, +0.133] |
| bge-multilingual-gemma2 | random-item | 1729 | +0.116 [+0.092, +0.138] |
| bge-multilingual-gemma2 | random-item | 2718 | +0.114 [+0.090, +0.137] |
| bge-multilingual-gemma2 | random-item | 3141 | +0.116 [+0.092, +0.138] |
| bge-multilingual-gemma2 | spatial-block | 1729 | +0.113 [+0.085, +0.147] |
| bge-multilingual-gemma2 | spatial-block | 2718 | +0.098 [+0.073, +0.126] |
| bge-multilingual-gemma2 | spatial-block | 3141 | +0.135 [+0.106, +0.163] |
| qwen3-8b-embedding | random-item | 1729 | +0.068 [+0.049, +0.086] |
| qwen3-8b-embedding | random-item | 2718 | +0.067 [+0.049, +0.085] |
| qwen3-8b-embedding | random-item | 3141 | +0.068 [+0.050, +0.086] |
| qwen3-8b-embedding | spatial-block | 1729 | +0.089 [+0.064, +0.117] |
| qwen3-8b-embedding | spatial-block | 2718 | +0.063 [+0.040, +0.085] |
| qwen3-8b-embedding | spatial-block | 3141 | +0.068 [+0.045, +0.091] |

## G2: Geographic vs similarity instruction, both with fusion and probe (instruction-fusion-probe minus generic-instruction-fusion-probe, city kNN gain paired by country)

| Model | Split | Seed | Difference [95% CI] |
| --- | --- | --- | --- |
| e5-mistral | random-item | 1729 | +0.089 [+0.072, +0.105] |
| e5-mistral | random-item | 2718 | +0.087 [+0.070, +0.103] |
| e5-mistral | random-item | 3141 | +0.090 [+0.072, +0.106] |
| e5-mistral | spatial-block | 1729 | +0.078 [+0.054, +0.104] |
| e5-mistral | spatial-block | 2718 | +0.094 [+0.074, +0.116] |
| e5-mistral | spatial-block | 3141 | +0.109 [+0.086, +0.133] |
| bge-multilingual-gemma2 | random-item | 1729 | +0.065 [+0.053, +0.077] |
| bge-multilingual-gemma2 | random-item | 2718 | +0.063 [+0.051, +0.076] |
| bge-multilingual-gemma2 | random-item | 3141 | +0.065 [+0.053, +0.077] |
| bge-multilingual-gemma2 | spatial-block | 1729 | +0.079 [+0.061, +0.102] |
| bge-multilingual-gemma2 | spatial-block | 2718 | +0.057 [+0.043, +0.071] |
| bge-multilingual-gemma2 | spatial-block | 3141 | +0.080 [+0.066, +0.097] |
| qwen3-8b-embedding | random-item | 1729 | +0.058 [+0.042, +0.074] |
| qwen3-8b-embedding | random-item | 2718 | +0.056 [+0.040, +0.073] |
| qwen3-8b-embedding | random-item | 3141 | +0.058 [+0.041, +0.075] |
| qwen3-8b-embedding | spatial-block | 1729 | +0.081 [+0.053, +0.113] |
| qwen3-8b-embedding | spatial-block | 2718 | +0.055 [+0.033, +0.078] |
| qwen3-8b-embedding | spatial-block | 3141 | +0.066 [+0.042, +0.090] |

## G3: Fix vs within-country seed-pair axes blended with cosine (instruction-fusion-probe minus feuerstein-country w=0.2, city kNN gain paired by country)

| Model | Split | Seed | Difference [95% CI] |
| --- | --- | --- | --- |
| e5-mistral | random-item | 1729 | +0.086 [+0.070, +0.103] |
| e5-mistral | random-item | 2718 | +0.085 [+0.069, +0.102] |
| e5-mistral | random-item | 3141 | +0.085 [+0.069, +0.102] |
| e5-mistral | spatial-block | 1729 | +0.083 [+0.057, +0.111] |
| e5-mistral | spatial-block | 2718 | +0.079 [+0.061, +0.099] |
| e5-mistral | spatial-block | 3141 | +0.095 [+0.074, +0.116] |
| bge-multilingual-gemma2 | random-item | 1729 | +0.095 [+0.074, +0.115] |
| bge-multilingual-gemma2 | random-item | 2718 | +0.091 [+0.072, +0.111] |
| bge-multilingual-gemma2 | random-item | 3141 | +0.095 [+0.076, +0.114] |
| bge-multilingual-gemma2 | spatial-block | 1729 | +0.107 [+0.080, +0.137] |
| bge-multilingual-gemma2 | spatial-block | 2718 | +0.083 [+0.059, +0.110] |
| bge-multilingual-gemma2 | spatial-block | 3141 | +0.114 [+0.089, +0.140] |
| qwen3-8b-embedding | random-item | 1729 | +0.068 [+0.050, +0.086] |
| qwen3-8b-embedding | random-item | 2718 | +0.068 [+0.051, +0.085] |
| qwen3-8b-embedding | random-item | 3141 | +0.067 [+0.050, +0.086] |
| qwen3-8b-embedding | spatial-block | 1729 | +0.089 [+0.066, +0.116] |
| qwen3-8b-embedding | spatial-block | 2718 | +0.062 [+0.041, +0.084] |
| qwen3-8b-embedding | spatial-block | 3141 | +0.067 [+0.045, +0.090] |

## descriptive: Is layer fusion needed? (instruction-fusion-probe minus instruction-probe, city kNN gain paired by country)

| Model | Split | Seed | Difference [95% CI] |
| --- | --- | --- | --- |
| e5-mistral | random-item | 1729 | +0.014 [+0.009, +0.019] |
| e5-mistral | random-item | 2718 | +0.014 [+0.010, +0.018] |
| e5-mistral | random-item | 3141 | +0.018 [+0.013, +0.022] |
| e5-mistral | spatial-block | 1729 | +0.015 [+0.007, +0.024] |
| e5-mistral | spatial-block | 2718 | +0.017 [+0.011, +0.024] |
| e5-mistral | spatial-block | 3141 | +0.017 [+0.012, +0.022] |
| bge-multilingual-gemma2 | random-item | 1729 | +0.037 [+0.027, +0.045] |
| bge-multilingual-gemma2 | random-item | 2718 | +0.034 [+0.025, +0.043] |
| bge-multilingual-gemma2 | random-item | 3141 | +0.037 [+0.028, +0.046] |
| bge-multilingual-gemma2 | spatial-block | 1729 | +0.036 [+0.024, +0.049] |
| bge-multilingual-gemma2 | spatial-block | 2718 | +0.029 [+0.020, +0.037] |
| bge-multilingual-gemma2 | spatial-block | 3141 | +0.047 [+0.036, +0.057] |
| qwen3-8b-embedding | random-item | 1729 | +0.011 [+0.004, +0.019] |
| qwen3-8b-embedding | random-item | 2718 | +0.012 [+0.004, +0.019] |
| qwen3-8b-embedding | random-item | 3141 | +0.013 [+0.006, +0.020] |
| qwen3-8b-embedding | spatial-block | 1729 | +0.013 [+0.005, +0.022] |
| qwen3-8b-embedding | spatial-block | 2718 | +0.013 [+0.006, +0.020] |
| qwen3-8b-embedding | spatial-block | 3141 | +0.006 [+0.002, +0.010] |

## descriptive: Is the probe needed? (instruction-fusion-probe minus instruction-fusion, city kNN gain paired by country)

| Model | Split | Seed | Difference [95% CI] |
| --- | --- | --- | --- |
| e5-mistral | random-item | 1729 | +0.004 [+0.002, +0.006] |
| e5-mistral | random-item | 2718 | +0.005 [+0.003, +0.008] |
| e5-mistral | random-item | 3141 | +0.004 [+0.002, +0.006] |
| e5-mistral | spatial-block | 1729 | -0.002 [-0.006, +0.002] |
| e5-mistral | spatial-block | 2718 | +0.004 [-0.000, +0.008] |
| e5-mistral | spatial-block | 3141 | +0.001 [-0.002, +0.005] |
| bge-multilingual-gemma2 | random-item | 1729 | +0.013 [+0.010, +0.016] |
| bge-multilingual-gemma2 | random-item | 2718 | +0.011 [+0.008, +0.014] |
| bge-multilingual-gemma2 | random-item | 3141 | +0.013 [+0.010, +0.015] |
| bge-multilingual-gemma2 | spatial-block | 1729 | +0.008 [+0.004, +0.013] |
| bge-multilingual-gemma2 | spatial-block | 2718 | +0.013 [+0.008, +0.017] |
| bge-multilingual-gemma2 | spatial-block | 3141 | +0.009 [+0.006, +0.012] |
| qwen3-8b-embedding | random-item | 1729 | +0.002 [-0.001, +0.004] |
| qwen3-8b-embedding | random-item | 2718 | +0.002 [-0.001, +0.004] |
| qwen3-8b-embedding | random-item | 3141 | +0.002 [-0.001, +0.005] |
| qwen3-8b-embedding | spatial-block | 1729 | -0.001 [-0.004, +0.002] |
| qwen3-8b-embedding | spatial-block | 2718 | -0.000 [-0.004, +0.003] |
| qwen3-8b-embedding | spatial-block | 3141 | +0.002 [-0.001, +0.006] |
