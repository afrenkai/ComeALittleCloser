# Reflected geography, reflected-v1 final

Cities `name-country`, last layer. Per country with at least 30 evaluated cities: rho = Spearman of similarity against closeness; kNN gain = 1 - (km to the 10 most similar) / (km to all). Mean over countries; one value per seed.

Ceiling (similarity = true closeness), kNN gain per seed: 0.746 / 0.734 / 0.733.

## Reflection by model

| Model | Pooling | cosine rho | cosine kNN gain | centred cosine kNN gain | probe-decoded kNN gain |
| --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-base | mean | 0.034 / 0.053 / 0.034 | 0.033 / 0.049 / 0.048 | 0.038 / 0.042 / 0.042 | 0.021 / 0.027 / 0.024 |
| qwen3-0.6b-base | last | 0.006 / 0.014 / 0.012 | 0.013 / 0.012 / 0.004 | 0.011 / 0.008 / 0.007 | 0.015 / 0.014 / 0.003 |
| qwen3-0.6b-embedding | mean | 0.027 / 0.042 / 0.025 | 0.038 / 0.046 / 0.037 | 0.050 / 0.052 / 0.048 | 0.018 / 0.021 / 0.021 |
| qwen3-0.6b-embedding | last | 0.002 / 0.020 / -0.006 | 0.024 / 0.033 / 0.025 | 0.030 / 0.039 / 0.031 | 0.021 / 0.021 / 0.021 |
| qwen3-4b-base | mean | 0.044 / 0.049 / 0.048 | 0.059 / 0.057 / 0.063 | 0.062 / 0.065 / 0.060 | 0.087 / 0.077 / 0.073 |
| qwen3-4b-base | last | 0.006 / 0.016 / 0.012 | 0.018 / 0.024 / 0.020 | 0.019 / 0.027 / 0.022 | 0.044 / 0.044 / 0.036 |
| qwen3-4b-embedding | mean | 0.091 / 0.109 / 0.084 | 0.109 / 0.110 / 0.103 | 0.118 / 0.120 / 0.116 | 0.070 / 0.064 / 0.066 |
| qwen3-4b-embedding | last | 0.076 / 0.067 / 0.073 | 0.119 / 0.104 / 0.111 | 0.128 / 0.113 / 0.121 | 0.085 / 0.076 / 0.093 |
| qwen3-8b-base | mean | 0.038 / 0.050 / 0.046 | 0.063 / 0.073 / 0.072 | 0.081 / 0.081 / 0.074 | 0.098 / 0.105 / 0.100 |
| qwen3-8b-base | last | 0.009 / 0.018 / 0.002 | 0.026 / 0.027 / 0.021 | 0.036 / 0.032 / 0.030 | 0.048 / 0.039 / 0.032 |
| qwen3-8b-embedding | mean | 0.099 / 0.108 / 0.090 | 0.137 / 0.144 / 0.132 | 0.156 / 0.156 / 0.145 | 0.100 / 0.104 / 0.110 |
| qwen3-8b-embedding | last | 0.096 / 0.081 / 0.076 | 0.144 / 0.125 / 0.129 | 0.150 / 0.135 / 0.140 | 0.121 / 0.098 / 0.111 |
| mistral-7b-base | mean | 0.083 / 0.091 / 0.079 | 0.129 / 0.132 / 0.125 | 0.139 / 0.135 / 0.127 | 0.174 / 0.155 / 0.144 |
| mistral-7b-base | last | 0.034 / 0.034 / 0.013 | 0.050 / 0.051 / 0.043 | 0.053 / 0.054 / 0.044 | 0.131 / 0.114 / 0.110 |
| e5-mistral | mean | 0.123 / 0.119 / 0.105 | 0.157 / 0.152 / 0.142 | 0.158 / 0.149 / 0.135 | 0.145 / 0.143 / 0.126 |
| e5-mistral | last | 0.269 / 0.233 / 0.235 | 0.274 / 0.245 / 0.249 | 0.281 / 0.255 / 0.260 | 0.195 / 0.175 / 0.183 |
| gemma2-9b-base | mean | 0.129 / 0.131 / 0.119 | 0.184 / 0.176 / 0.172 | 0.184 / 0.174 / 0.176 | 0.193 / 0.179 / 0.178 |
| gemma2-9b-base | last | 0.095 / 0.083 / 0.082 | 0.179 / 0.169 / 0.168 | 0.190 / 0.178 / 0.178 | 0.222 / 0.200 / 0.202 |
| bge-multilingual-gemma2 | mean | 0.165 / 0.160 / 0.141 | 0.213 / 0.199 / 0.185 | 0.222 / 0.216 / 0.202 | 0.168 / 0.154 / 0.161 |
| bge-multilingual-gemma2 | last | 0.143 / 0.143 / 0.137 | 0.211 / 0.198 / 0.196 | 0.227 / 0.211 / 0.211 | 0.189 / 0.177 / 0.167 |
| xlmr-base | mean | 0.012 / 0.035 / 0.015 | 0.020 / 0.044 / 0.025 | 0.024 / 0.045 / 0.028 | 0.004 / 0.021 / 0.007 |
| e5-base | mean | 0.037 / 0.032 / 0.035 | 0.049 / 0.042 / 0.047 | 0.045 / 0.049 / 0.047 | 0.026 / 0.010 / 0.013 |
| xlmr-large | mean | 0.005 / 0.017 / 0.023 | 0.021 / 0.023 / 0.027 | 0.024 / 0.031 / 0.025 | 0.009 / 0.009 / 0.010 |
| bge-m3 | mean | 0.035 / 0.017 / 0.022 | 0.041 / 0.035 / 0.042 | 0.042 / 0.035 / 0.047 | 0.012 / 0.009 / 0.011 |
| xlmr-base-random | mean | -0.001 / 0.011 / -0.003 | 0.020 / 0.019 / 0.014 | 0.025 / 0.020 / 0.016 | 0.003 / 0.011 / 0.010 |
| qwen3-0.6b-random | mean | 0.006 / 0.008 / 0.007 | 0.022 / 0.011 / 0.014 | 0.018 / 0.007 / 0.014 | 0.008 / 0.008 / 0.010 |
| xlmr-xl | mean | 0.051 / 0.030 / 0.018 | 0.048 / 0.036 / 0.025 | 0.040 / 0.044 / 0.033 | 0.020 / 0.017 / 0.023 |

## Retrieval minus base, same pooling (countries paired, bootstrap CI)

| Pair | Pooling | Similarity | Seed | rho | kNN gain |
| --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | cosine | 1729 | -0.007 [-0.024, +0.010] | +0.006 [-0.006, +0.018] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | cosine | 2718 | -0.012 [-0.039, +0.010] | -0.003 [-0.016, +0.010] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | cosine | 3141 | -0.009 [-0.027, +0.010] | -0.010 [-0.027, +0.005] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | centered-cosine | 1729 | +0.007 [-0.011, +0.025] | +0.012 [-0.002, +0.026] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | centered-cosine | 2718 | +0.011 [-0.010, +0.031] | +0.009 [-0.004, +0.021] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | centered-cosine | 3141 | +0.004 [-0.012, +0.021] | +0.006 [-0.006, +0.017] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | cosine | 1729 | -0.004 [-0.024, +0.014] | +0.011 [-0.007, +0.028] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | cosine | 2718 | +0.006 [-0.014, +0.026] | +0.021 [+0.001, +0.039] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | cosine | 3141 | -0.018 [-0.036, -0.001] | +0.020 [-0.001, +0.038] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | centered-cosine | 1729 | -0.001 [-0.025, +0.025] | +0.019 [-0.002, +0.039] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | centered-cosine | 2718 | +0.012 [-0.016, +0.038] | +0.031 [+0.013, +0.048] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | centered-cosine | 3141 | +0.006 [-0.011, +0.021] | +0.023 [+0.003, +0.041] |
| qwen3-4b-base → qwen3-4b-embedding | mean | cosine | 1729 | +0.047 [+0.028, +0.067] | +0.050 [+0.025, +0.075] |
| qwen3-4b-base → qwen3-4b-embedding | mean | cosine | 2718 | +0.059 [+0.036, +0.084] | +0.054 [+0.033, +0.074] |
| qwen3-4b-base → qwen3-4b-embedding | mean | cosine | 3141 | +0.035 [+0.008, +0.060] | +0.040 [+0.018, +0.063] |
| qwen3-4b-base → qwen3-4b-embedding | mean | centered-cosine | 1729 | +0.037 [+0.014, +0.058] | +0.055 [+0.033, +0.079] |
| qwen3-4b-base → qwen3-4b-embedding | mean | centered-cosine | 2718 | +0.048 [+0.028, +0.068] | +0.055 [+0.033, +0.078] |
| qwen3-4b-base → qwen3-4b-embedding | mean | centered-cosine | 3141 | +0.038 [+0.016, +0.062] | +0.056 [+0.031, +0.082] |
| qwen3-4b-base → qwen3-4b-embedding | last | cosine | 1729 | +0.070 [+0.045, +0.095] | +0.101 [+0.067, +0.133] |
| qwen3-4b-base → qwen3-4b-embedding | last | cosine | 2718 | +0.051 [+0.019, +0.085] | +0.080 [+0.044, +0.116] |
| qwen3-4b-base → qwen3-4b-embedding | last | cosine | 3141 | +0.061 [+0.037, +0.084] | +0.091 [+0.058, +0.128] |
| qwen3-4b-base → qwen3-4b-embedding | last | centered-cosine | 1729 | +0.080 [+0.049, +0.112] | +0.109 [+0.075, +0.145] |
| qwen3-4b-base → qwen3-4b-embedding | last | centered-cosine | 2718 | +0.055 [+0.023, +0.087] | +0.086 [+0.049, +0.124] |
| qwen3-4b-base → qwen3-4b-embedding | last | centered-cosine | 3141 | +0.069 [+0.041, +0.095] | +0.099 [+0.064, +0.137] |
| qwen3-8b-base → qwen3-8b-embedding | mean | cosine | 1729 | +0.062 [+0.037, +0.086] | +0.074 [+0.050, +0.098] |
| qwen3-8b-base → qwen3-8b-embedding | mean | cosine | 2718 | +0.058 [+0.027, +0.088] | +0.071 [+0.050, +0.094] |
| qwen3-8b-base → qwen3-8b-embedding | mean | cosine | 3141 | +0.044 [+0.015, +0.073] | +0.060 [+0.032, +0.091] |
| qwen3-8b-base → qwen3-8b-embedding | mean | centered-cosine | 1729 | +0.043 [+0.016, +0.069] | +0.075 [+0.049, +0.103] |
| qwen3-8b-base → qwen3-8b-embedding | mean | centered-cosine | 2718 | +0.055 [+0.028, +0.082] | +0.075 [+0.051, +0.101] |
| qwen3-8b-base → qwen3-8b-embedding | mean | centered-cosine | 3141 | +0.040 [+0.016, +0.065] | +0.071 [+0.043, +0.103] |
| qwen3-8b-base → qwen3-8b-embedding | last | cosine | 1729 | +0.087 [+0.055, +0.118] | +0.118 [+0.078, +0.155] |
| qwen3-8b-base → qwen3-8b-embedding | last | cosine | 2718 | +0.063 [+0.032, +0.090] | +0.098 [+0.060, +0.137] |
| qwen3-8b-base → qwen3-8b-embedding | last | cosine | 3141 | +0.074 [+0.045, +0.104] | +0.108 [+0.072, +0.149] |
| qwen3-8b-base → qwen3-8b-embedding | last | centered-cosine | 1729 | +0.079 [+0.040, +0.116] | +0.114 [+0.073, +0.154] |
| qwen3-8b-base → qwen3-8b-embedding | last | centered-cosine | 2718 | +0.064 [+0.023, +0.102] | +0.104 [+0.063, +0.146] |
| qwen3-8b-base → qwen3-8b-embedding | last | centered-cosine | 3141 | +0.089 [+0.054, +0.126] | +0.110 [+0.070, +0.153] |
| mistral-7b-base → e5-mistral | mean | cosine | 1729 | +0.040 [+0.018, +0.060] | +0.028 [+0.014, +0.042] |
| mistral-7b-base → e5-mistral | mean | cosine | 2718 | +0.028 [+0.012, +0.043] | +0.020 [+0.009, +0.032] |
| mistral-7b-base → e5-mistral | mean | cosine | 3141 | +0.026 [+0.006, +0.044] | +0.017 [+0.004, +0.031] |
| mistral-7b-base → e5-mistral | mean | centered-cosine | 1729 | +0.028 [+0.012, +0.043] | +0.018 [+0.006, +0.031] |
| mistral-7b-base → e5-mistral | mean | centered-cosine | 2718 | +0.023 [+0.008, +0.039] | +0.014 [+0.003, +0.026] |
| mistral-7b-base → e5-mistral | mean | centered-cosine | 3141 | +0.024 [+0.009, +0.038] | +0.008 [-0.006, +0.022] |
| mistral-7b-base → e5-mistral | last | cosine | 1729 | +0.235 [+0.192, +0.280] | +0.224 [+0.181, +0.268] |
| mistral-7b-base → e5-mistral | last | cosine | 2718 | +0.199 [+0.164, +0.238] | +0.194 [+0.151, +0.242] |
| mistral-7b-base → e5-mistral | last | cosine | 3141 | +0.222 [+0.176, +0.267] | +0.205 [+0.157, +0.257] |
| mistral-7b-base → e5-mistral | last | centered-cosine | 1729 | +0.258 [+0.218, +0.299] | +0.228 [+0.187, +0.273] |
| mistral-7b-base → e5-mistral | last | centered-cosine | 2718 | +0.215 [+0.175, +0.257] | +0.201 [+0.156, +0.248] |
| mistral-7b-base → e5-mistral | last | centered-cosine | 3141 | +0.232 [+0.182, +0.281] | +0.216 [+0.168, +0.269] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | cosine | 1729 | +0.036 [+0.015, +0.055] | +0.029 [+0.014, +0.044] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | cosine | 2718 | +0.029 [+0.004, +0.055] | +0.023 [+0.008, +0.039] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | cosine | 3141 | +0.022 [+0.000, +0.045] | +0.013 [-0.006, +0.032] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | centered-cosine | 1729 | +0.041 [+0.020, +0.059] | +0.038 [+0.023, +0.055] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | centered-cosine | 2718 | +0.043 [+0.025, +0.061] | +0.042 [+0.028, +0.057] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | centered-cosine | 3141 | +0.034 [+0.017, +0.053] | +0.026 [+0.009, +0.044] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | cosine | 1729 | +0.048 [+0.021, +0.072] | +0.032 [+0.008, +0.056] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | cosine | 2718 | +0.060 [+0.039, +0.081] | +0.029 [+0.009, +0.047] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | cosine | 3141 | +0.055 [+0.033, +0.078] | +0.028 [+0.005, +0.049] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | centered-cosine | 1729 | +0.082 [+0.054, +0.105] | +0.037 [+0.015, +0.061] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | centered-cosine | 2718 | +0.089 [+0.068, +0.110] | +0.033 [+0.014, +0.051] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | centered-cosine | 3141 | +0.077 [+0.051, +0.106] | +0.033 [+0.012, +0.056] |
| xlmr-base → e5-base | mean | cosine | 1729 | +0.025 [+0.003, +0.053] | +0.029 [+0.013, +0.046] |
| xlmr-base → e5-base | mean | cosine | 2718 | -0.003 [-0.028, +0.019] | -0.001 [-0.017, +0.012] |
| xlmr-base → e5-base | mean | cosine | 3141 | +0.021 [-0.002, +0.050] | +0.022 [+0.004, +0.043] |
| xlmr-base → e5-base | mean | centered-cosine | 1729 | +0.016 [-0.007, +0.040] | +0.021 [+0.006, +0.038] |
| xlmr-base → e5-base | mean | centered-cosine | 2718 | -0.000 [-0.016, +0.015] | +0.004 [-0.009, +0.016] |
| xlmr-base → e5-base | mean | centered-cosine | 3141 | +0.010 [-0.012, +0.033] | +0.018 [+0.003, +0.034] |
| xlmr-large → bge-m3 | mean | cosine | 1729 | +0.030 [+0.015, +0.048] | +0.020 [+0.007, +0.034] |
| xlmr-large → bge-m3 | mean | cosine | 2718 | +0.000 [-0.024, +0.025] | +0.012 [-0.003, +0.026] |
| xlmr-large → bge-m3 | mean | cosine | 3141 | -0.001 [-0.022, +0.018] | +0.016 [+0.002, +0.030] |
| xlmr-large → bge-m3 | mean | centered-cosine | 1729 | +0.013 [-0.002, +0.030] | +0.018 [+0.006, +0.031] |
| xlmr-large → bge-m3 | mean | centered-cosine | 2718 | -0.000 [-0.020, +0.017] | +0.003 [-0.011, +0.018] |
| xlmr-large → bge-m3 | mean | centered-cosine | 3141 | +0.007 [-0.014, +0.028] | +0.022 [+0.008, +0.035] |

**R1 (primary pairs, kNN gain, raw and centred cosine, both poolings, every seed): 46 of 48 paired intervals above 0; R1 NOT supported.**

## Nearest-neighbour geolocation vs the linear probe

Gain over the training country centroid, km, block mean [95% CI]. Difference = nearest-neighbour minus probe, paired by place.

| Model | Pooling | Split | Seed | Probe | Nearest neighbour | Difference |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-base | mean | random-item | 1729 | 34 [14, 57] | 54 [15, 105] | +20 [-18, +66] |
| qwen3-0.6b-base | mean | random-item | 2718 | 13 [-3, 31] | 38 [0, 86] | +25 [-12, +68] |
| qwen3-0.6b-base | mean | random-item | 3141 | 16 [-1, 35] | 47 [15, 88] | +31 [+4, +65] |
| qwen3-0.6b-base | mean | spatial-block | 1729 | 58 [24, 99] | 32 [-4, 77] | -26 [-62, +11] |
| qwen3-0.6b-base | mean | spatial-block | 2718 | 66 [3, 150] | 42 [-12, 104] | -24 [-89, +39] |
| qwen3-0.6b-base | mean | spatial-block | 3141 | 8 [-5, 23] | 1 [-21, 24] | -7 [-26, +14] |
| qwen3-0.6b-base | last | random-item | 1729 | 3 [-6, 14] | 6 [-17, 30] | +3 [-18, +23] |
| qwen3-0.6b-base | last | random-item | 2718 | -2 [-10, 7] | 14 [-5, 36] | +16 [-2, +35] |
| qwen3-0.6b-base | last | random-item | 3141 | -0 [-1, 0] | 5 [-12, 27] | +6 [-12, +27] |
| qwen3-0.6b-base | last | spatial-block | 1729 | 3 [-16, 26] | 18 [-7, 48] | +15 [-7, +38] |
| qwen3-0.6b-base | last | spatial-block | 2718 | 16 [-28, 64] | -1 [-33, 44] | -17 [-53, +19] |
| qwen3-0.6b-base | last | spatial-block | 3141 | -7 [-17, 4] | 3 [-23, 34] | +10 [-14, +37] |
| qwen3-0.6b-embedding | mean | random-item | 1729 | 19 [0, 38] | 76 [38, 120] | +58 [+27, +90] |
| qwen3-0.6b-embedding | mean | random-item | 2718 | 7 [-9, 21] | 55 [21, 97] | +48 [+19, +87] |
| qwen3-0.6b-embedding | mean | random-item | 3141 | 14 [-2, 32] | 68 [32, 110] | +54 [+23, +90] |
| qwen3-0.6b-embedding | mean | spatial-block | 1729 | 26 [1, 55] | 19 [-39, 69] | -7 [-54, +30] |
| qwen3-0.6b-embedding | mean | spatial-block | 2718 | 52 [-14, 136] | 86 [-2, 221] | +34 [-32, +114] |
| qwen3-0.6b-embedding | mean | spatial-block | 3141 | 10 [-19, 40] | 16 [-18, 54] | +6 [-23, +34] |
| qwen3-0.6b-embedding | last | random-item | 1729 | 26 [8, 48] | 64 [30, 105] | +38 [+11, +66] |
| qwen3-0.6b-embedding | last | random-item | 2718 | 17 [-4, 39] | 48 [14, 85] | +32 [+5, +63] |
| qwen3-0.6b-embedding | last | random-item | 3141 | 23 [-2, 54] | 33 [2, 66] | +10 [-16, +34] |
| qwen3-0.6b-embedding | last | spatial-block | 1729 | 24 [-18, 66] | 21 [-20, 65] | -4 [-38, +32] |
| qwen3-0.6b-embedding | last | spatial-block | 2718 | 99 [18, 219] | 37 [-3, 81] | -62 [-175, +7] |
| qwen3-0.6b-embedding | last | spatial-block | 3141 | 16 [-2, 36] | 28 [-32, 119] | +12 [-42, +100] |
| qwen3-4b-base | mean | random-item | 1729 | 135 [81, 201] | 105 [53, 171] | -30 [-69, +7] |
| qwen3-4b-base | mean | random-item | 2718 | 111 [64, 163] | 62 [18, 121] | -49 [-95, -10] |
| qwen3-4b-base | mean | random-item | 3141 | 114 [58, 183] | 101 [55, 159] | -13 [-56, +24] |
| qwen3-4b-base | mean | spatial-block | 1729 | 187 [115, 269] | 60 [14, 116] | -127 [-211, -60] |
| qwen3-4b-base | mean | spatial-block | 2718 | 237 [93, 464] | 89 [2, 206] | -148 [-288, -50] |
| qwen3-4b-base | mean | spatial-block | 3141 | 114 [45, 192] | 18 [-9, 45] | -97 [-156, -46] |
| qwen3-4b-base | last | random-item | 1729 | 39 [11, 71] | 11 [-10, 32] | -28 [-60, +2] |
| qwen3-4b-base | last | random-item | 2718 | 33 [7, 61] | 8 [-12, 29] | -25 [-48, -2] |
| qwen3-4b-base | last | random-item | 3141 | 26 [-9, 65] | 4 [-14, 23] | -23 [-58, +8] |
| qwen3-4b-base | last | spatial-block | 1729 | 91 [18, 166] | 16 [-10, 47] | -75 [-136, -11] |
| qwen3-4b-base | last | spatial-block | 2718 | 184 [48, 347] | 2 [-33, 50] | -182 [-319, -65] |
| qwen3-4b-base | last | spatial-block | 3141 | 41 [-5, 89] | 1 [-21, 23] | -40 [-74, -6] |
| qwen3-4b-embedding | mean | random-item | 1729 | 142 [87, 210] | 222 [148, 311] | +80 [+40, +123] |
| qwen3-4b-embedding | mean | random-item | 2718 | 105 [58, 157] | 142 [83, 207] | +37 [+0, +74] |
| qwen3-4b-embedding | mean | random-item | 3141 | 114 [66, 172] | 188 [128, 269] | +73 [+38, +113] |
| qwen3-4b-embedding | mean | spatial-block | 1729 | 155 [96, 223] | 115 [58, 186] | -40 [-89, +10] |
| qwen3-4b-embedding | mean | spatial-block | 2718 | 260 [82, 511] | 244 [42, 559] | -16 [-113, +91] |
| qwen3-4b-embedding | mean | spatial-block | 3141 | 128 [51, 215] | 83 [25, 154] | -44 [-80, -11] |
| qwen3-4b-embedding | last | random-item | 1729 | 173 [104, 257] | 228 [149, 315] | +55 [+26, +82] |
| qwen3-4b-embedding | last | random-item | 2718 | 136 [82, 196] | 182 [112, 263] | +46 [+14, +82] |
| qwen3-4b-embedding | last | random-item | 3141 | 129 [73, 204] | 183 [124, 264] | +54 [+27, +80] |
| qwen3-4b-embedding | last | spatial-block | 1729 | 204 [130, 284] | 135 [71, 218] | -69 [-120, -19] |
| qwen3-4b-embedding | last | spatial-block | 2718 | 300 [90, 597] | 255 [52, 572] | -45 [-139, +55] |
| qwen3-4b-embedding | last | spatial-block | 3141 | 159 [62, 269] | 117 [36, 217] | -42 [-93, +2] |
| qwen3-8b-base | mean | random-item | 1729 | 169 [97, 254] | 111 [63, 174] | -58 [-100, -15] |
| qwen3-8b-base | mean | random-item | 2718 | 159 [103, 220] | 101 [55, 157] | -58 [-93, -27] |
| qwen3-8b-base | mean | random-item | 3141 | 152 [90, 230] | 119 [70, 186] | -33 [-75, +8] |
| qwen3-8b-base | mean | spatial-block | 1729 | 194 [94, 307] | 77 [24, 142] | -117 [-189, -47] |
| qwen3-8b-base | mean | spatial-block | 2718 | 254 [70, 498] | 88 [12, 190] | -166 [-352, -42] |
| qwen3-8b-base | mean | spatial-block | 3141 | 141 [46, 254] | 36 [1, 78] | -105 [-185, -33] |
| qwen3-8b-base | last | random-item | 1729 | 74 [43, 112] | 50 [18, 92] | -23 [-51, +5] |
| qwen3-8b-base | last | random-item | 2718 | 54 [23, 86] | 49 [21, 85] | -5 [-32, +21] |
| qwen3-8b-base | last | random-item | 3141 | 52 [24, 86] | 22 [3, 43] | -31 [-56, -8] |
| qwen3-8b-base | last | spatial-block | 1729 | 108 [49, 172] | 15 [-16, 46] | -93 [-158, -31] |
| qwen3-8b-base | last | spatial-block | 2718 | 158 [46, 312] | 2 [-30, 39] | -155 [-288, -53] |
| qwen3-8b-base | last | spatial-block | 3141 | 54 [-23, 156] | -1 [-27, 28] | -55 [-150, +20] |
| qwen3-8b-embedding | mean | random-item | 1729 | 206 [139, 283] | 249 [171, 338] | +43 [+13, +74] |
| qwen3-8b-embedding | mean | random-item | 2718 | 201 [131, 277] | 207 [145, 279] | +6 [-35, +39] |
| qwen3-8b-embedding | mean | random-item | 3141 | 199 [130, 286] | 213 [150, 301] | +15 [-25, +50] |
| qwen3-8b-embedding | mean | spatial-block | 1729 | 232 [123, 364] | 150 [64, 255] | -82 [-149, -17] |
| qwen3-8b-embedding | mean | spatial-block | 2718 | 350 [111, 701] | 285 [57, 629] | -65 [-117, -21] |
| qwen3-8b-embedding | mean | spatial-block | 3141 | 156 [48, 294] | 133 [51, 245] | -23 [-80, +28] |
| qwen3-8b-embedding | last | random-item | 1729 | 218 [152, 301] | 240 [169, 322] | +21 [-4, +47] |
| qwen3-8b-embedding | last | random-item | 2718 | 221 [149, 305] | 248 [168, 337] | +27 [+1, +55] |
| qwen3-8b-embedding | last | random-item | 3141 | 219 [143, 316] | 245 [176, 338] | +27 [-4, +53] |
| qwen3-8b-embedding | last | spatial-block | 1729 | 246 [133, 386] | 256 [103, 456] | +10 [-66, +100] |
| qwen3-8b-embedding | last | spatial-block | 2718 | 336 [119, 645] | 304 [61, 660] | -32 [-99, +41] |
| qwen3-8b-embedding | last | spatial-block | 3141 | 190 [73, 325] | 139 [51, 258] | -50 [-98, -6] |
| mistral-7b-base | mean | random-item | 1729 | 228 [160, 307] | 189 [127, 272] | -39 [-79, -1] |
| mistral-7b-base | mean | random-item | 2718 | 231 [158, 309] | 185 [127, 252] | -46 [-95, -4] |
| mistral-7b-base | mean | random-item | 3141 | 211 [140, 293] | 153 [100, 229] | -57 [-106, -12] |
| mistral-7b-base | mean | spatial-block | 1729 | 268 [133, 419] | 89 [41, 143] | -179 [-298, -72] |
| mistral-7b-base | mean | spatial-block | 2718 | 272 [67, 550] | 160 [46, 308] | -113 [-254, -9] |
| mistral-7b-base | mean | spatial-block | 3141 | 202 [97, 328] | 87 [24, 175] | -115 [-173, -60] |
| mistral-7b-base | last | random-item | 1729 | 172 [109, 247] | 66 [33, 103] | -107 [-159, -55] |
| mistral-7b-base | last | random-item | 2718 | 152 [87, 220] | 53 [24, 82] | -99 [-153, -51] |
| mistral-7b-base | last | random-item | 3141 | 147 [88, 217] | 39 [10, 73] | -108 [-164, -60] |
| mistral-7b-base | last | spatial-block | 1729 | 237 [123, 369] | 52 [-1, 119] | -185 [-279, -101] |
| mistral-7b-base | last | spatial-block | 2718 | 229 [15, 488] | 46 [-7, 106] | -184 [-402, -6] |
| mistral-7b-base | last | spatial-block | 3141 | 158 [56, 286] | 14 [-14, 43] | -144 [-256, -52] |
| e5-mistral | mean | random-item | 1729 | 210 [143, 289] | 197 [132, 272] | -13 [-52, +29] |
| e5-mistral | mean | random-item | 2718 | 198 [131, 271] | 169 [113, 235] | -29 [-79, +16] |
| e5-mistral | mean | random-item | 3141 | 191 [124, 272] | 161 [107, 241] | -30 [-82, +15] |
| e5-mistral | mean | spatial-block | 1729 | 230 [132, 343] | 92 [37, 154] | -138 [-210, -76] |
| e5-mistral | mean | spatial-block | 2718 | 236 [64, 455] | 165 [41, 338] | -71 [-127, -11] |
| e5-mistral | mean | spatial-block | 3141 | 178 [58, 331] | 76 [25, 138] | -102 [-207, -21] |
| e5-mistral | last | random-item | 1729 | 288 [206, 391] | 334 [246, 432] | +46 [+10, +78] |
| e5-mistral | last | random-item | 2718 | 292 [209, 377] | 331 [243, 426] | +39 [+3, +75] |
| e5-mistral | last | random-item | 3141 | 267 [185, 375] | 315 [233, 422] | +48 [+2, +87] |
| e5-mistral | last | spatial-block | 1729 | 312 [192, 450] | 258 [125, 432] | -53 [-140, +31] |
| e5-mistral | last | spatial-block | 2718 | 304 [100, 597] | 317 [89, 658] | +13 [-57, +99] |
| e5-mistral | last | spatial-block | 3141 | 269 [114, 459] | 191 [80, 344] | -78 [-155, -14] |
| gemma2-9b-base | mean | random-item | 1729 | 278 [200, 365] | 251 [183, 331] | -27 [-64, +11] |
| gemma2-9b-base | mean | random-item | 2718 | 271 [191, 358] | 227 [165, 303] | -43 [-88, -1] |
| gemma2-9b-base | mean | random-item | 3141 | 264 [190, 358] | 202 [146, 277] | -62 [-122, -6] |
| gemma2-9b-base | mean | spatial-block | 1729 | 295 [157, 461] | 190 [70, 345] | -105 [-174, -46] |
| gemma2-9b-base | mean | spatial-block | 2718 | 303 [117, 541] | 143 [39, 278] | -160 [-325, -53] |
| gemma2-9b-base | mean | spatial-block | 3141 | 247 [106, 417] | 100 [33, 179] | -147 [-260, -59] |
| gemma2-9b-base | last | random-item | 1729 | 352 [261, 456] | 327 [245, 423] | -25 [-54, +2] |
| gemma2-9b-base | last | random-item | 2718 | 327 [227, 441] | 337 [232, 459] | +10 [-24, +45] |
| gemma2-9b-base | last | random-item | 3141 | 337 [246, 455] | 321 [239, 433] | -15 [-51, +16] |
| gemma2-9b-base | last | spatial-block | 1729 | 315 [144, 529] | 244 [90, 437] | -72 [-161, +25] |
| gemma2-9b-base | last | spatial-block | 2718 | 348 [92, 666] | 313 [77, 645] | -35 [-111, +54] |
| gemma2-9b-base | last | spatial-block | 3141 | 298 [148, 489] | 200 [54, 408] | -98 [-159, -39] |
| bge-multilingual-gemma2 | mean | random-item | 1729 | 241 [167, 323] | 268 [195, 353] | +27 [-16, +64] |
| bge-multilingual-gemma2 | mean | random-item | 2718 | 244 [169, 326] | 250 [178, 328] | +6 [-36, +47] |
| bge-multilingual-gemma2 | mean | random-item | 3141 | 232 [166, 312] | 232 [163, 323] | -0 [-51, +45] |
| bge-multilingual-gemma2 | mean | spatial-block | 1729 | 306 [185, 460] | 183 [69, 334] | -123 [-204, -50] |
| bge-multilingual-gemma2 | mean | spatial-block | 2718 | 326 [146, 550] | 215 [50, 426] | -111 [-194, -25] |
| bge-multilingual-gemma2 | mean | spatial-block | 3141 | 222 [93, 381] | 130 [40, 260] | -92 [-152, -32] |
| bge-multilingual-gemma2 | last | random-item | 1729 | 277 [196, 368] | 302 [227, 386] | +25 [-11, +58] |
| bge-multilingual-gemma2 | last | random-item | 2718 | 276 [187, 367] | 310 [227, 406] | +34 [-2, +68] |
| bge-multilingual-gemma2 | last | random-item | 3141 | 270 [193, 360] | 276 [207, 361] | +6 [-33, +44] |
| bge-multilingual-gemma2 | last | spatial-block | 1729 | 372 [233, 539] | 200 [66, 384] | -172 [-258, -85] |
| bge-multilingual-gemma2 | last | spatial-block | 2718 | 382 [147, 689] | 220 [60, 429] | -161 [-333, -30] |
| bge-multilingual-gemma2 | last | spatial-block | 3141 | 258 [125, 431] | 150 [43, 292] | -108 [-177, -46] |
| xlmr-base | mean | random-item | 1729 | 0 [-0, 0] | 29 [-3, 64] | +29 [-3, +64] |
| xlmr-base | mean | random-item | 2718 | -5 [-15, 4] | 25 [-4, 61] | +30 [+2, +62] |
| xlmr-base | mean | random-item | 3141 | 0 [-0, 0] | 25 [2, 54] | +25 [+2, +54] |
| xlmr-base | mean | spatial-block | 1729 | 11 [-10, 32] | 14 [-18, 52] | +4 [-22, +32] |
| xlmr-base | mean | spatial-block | 2718 | 12 [-13, 44] | 3 [-31, 42] | -9 [-39, +17] |
| xlmr-base | mean | spatial-block | 3141 | 0 [-14, 16] | -9 [-30, 13] | -10 [-31, +8] |
| e5-base | mean | random-item | 1729 | 7 [-6, 22] | 46 [13, 85] | +39 [+11, +71] |
| e5-base | mean | random-item | 2718 | -7 [-25, 12] | 32 [0, 71] | +40 [+11, +70] |
| e5-base | mean | random-item | 3141 | 0 [-0, 0] | 45 [15, 78] | +45 [+15, +78] |
| e5-base | mean | spatial-block | 1729 | 18 [-19, 54] | 20 [-23, 72] | +2 [-49, +53] |
| e5-base | mean | spatial-block | 2718 | 51 [-15, 132] | 32 [-4, 79] | -18 [-76, +38] |
| e5-base | mean | spatial-block | 3141 | 17 [-1, 38] | -6 [-28, 13] | -24 [-45, -5] |
| xlmr-large | mean | random-item | 1729 | 0 [-0, 0] | 14 [-6, 34] | +14 [-6, +34] |
| xlmr-large | mean | random-item | 2718 | -0 [-7, 6] | 29 [4, 56] | +29 [+6, +56] |
| xlmr-large | mean | random-item | 3141 | -0 [-0, 0] | 12 [-9, 33] | +12 [-9, +33] |
| xlmr-large | mean | spatial-block | 1729 | 1 [-9, 11] | -6 [-56, 38] | -7 [-54, +38] |
| xlmr-large | mean | spatial-block | 2718 | 14 [2, 28] | 46 [-13, 125] | +32 [-20, +105] |
| xlmr-large | mean | spatial-block | 3141 | 14 [-9, 42] | 29 [-11, 82] | +16 [-23, +59] |
| bge-m3 | mean | random-item | 1729 | 0 [-0, 1] | 41 [16, 70] | +41 [+16, +70] |
| bge-m3 | mean | random-item | 2718 | -13 [-27, 0] | 15 [-15, 45] | +28 [-0, +55] |
| bge-m3 | mean | random-item | 3141 | -0 [-1, 0] | 40 [16, 70] | +41 [+17, +71] |
| bge-m3 | mean | spatial-block | 1729 | -6 [-19, 6] | 2 [-22, 27] | +8 [-17, +33] |
| bge-m3 | mean | spatial-block | 2718 | 0 [-1, 1] | -4 [-41, 35] | -4 [-41, +34] |
| bge-m3 | mean | spatial-block | 3141 | -6 [-23, 12] | -2 [-27, 20] | +4 [-21, +26] |
| xlmr-base-random | mean | random-item | 1729 | -0 [-0, 0] | 5 [-18, 33] | +5 [-18, +33] |
| xlmr-base-random | mean | random-item | 2718 | 0 [-0, 0] | 7 [-15, 35] | +7 [-15, +35] |
| xlmr-base-random | mean | random-item | 3141 | 0 [-0, 0] | 0 [-18, 18] | +0 [-18, +19] |
| xlmr-base-random | mean | spatial-block | 1729 | 0 [-0, 0] | 17 [-46, 82] | +17 [-46, +82] |
| xlmr-base-random | mean | spatial-block | 2718 | -0 [-0, 0] | 13 [-27, 69] | +13 [-27, +69] |
| xlmr-base-random | mean | spatial-block | 3141 | -0 [-0, 0] | 34 [-16, 110] | +34 [-16, +110] |
| qwen3-0.6b-random | mean | random-item | 1729 | -0 [-1, 0] | 4 [-16, 23] | +4 [-16, +24] |
| qwen3-0.6b-random | mean | random-item | 2718 | 0 [-0, 1] | -2 [-20, 16] | -2 [-21, +16] |
| qwen3-0.6b-random | mean | random-item | 3141 | 0 [-1, 1] | -7 [-28, 13] | -7 [-27, +12] |
| qwen3-0.6b-random | mean | spatial-block | 1729 | -3 [-6, -1] | -9 [-36, 20] | -6 [-34, +24] |
| qwen3-0.6b-random | mean | spatial-block | 2718 | -8 [-29, 8] | -8 [-28, 14] | -1 [-25, +27] |
| qwen3-0.6b-random | mean | spatial-block | 3141 | -0 [-1, 1] | 18 [-16, 61] | +18 [-16, +62] |
| xlmr-xl | mean | random-item | 1729 | 8 [-4, 20] | 40 [16, 70] | +33 [+11, +57] |
| xlmr-xl | mean | random-item | 2718 | -4 [-16, 9] | 33 [5, 63] | +37 [+14, +64] |
| xlmr-xl | mean | random-item | 3141 | 5 [-2, 12] | 34 [11, 61] | +29 [+8, +54] |
| xlmr-xl | mean | spatial-block | 1729 | 14 [-3, 34] | 18 [-32, 88] | +4 [-39, +62] |
| xlmr-xl | mean | spatial-block | 2718 | 40 [-7, 94] | 34 [-19, 109] | -5 [-66, +60] |
| xlmr-xl | mean | spatial-block | 3141 | -1 [-16, 15] | 10 [-18, 40] | +11 [-18, +43] |

## Crossover: does retrieval training shift the balance toward neighbours?

Retrieval minus base of (probe error - neighbour error), km, paired by place, block mean [95% CI]. Positive = retrieval training favours nearest neighbours relative to the probe.

| Pair | Pooling | Split | Seed | Interaction |
| --- | --- | --- | --- | --- |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | random-item | 1729 | +38 [+3, +71] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | random-item | 2718 | +24 [+5, +43] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | random-item | 3141 | +22 [+2, +45] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | spatial-block | 1729 | +19 [-38, +66] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | spatial-block | 2718 | +58 [-14, +153] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | mean | spatial-block | 3141 | +12 [-14, +37] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | random-item | 1729 | +35 [+1, +73] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | random-item | 2718 | +16 [-12, +47] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | random-item | 3141 | +4 [-26, +30] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | spatial-block | 1729 | -18 [-56, +19] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | spatial-block | 2718 | -46 [-153, +26] |
| qwen3-0.6b-base → qwen3-0.6b-embedding | last | spatial-block | 3141 | +3 [-46, +68] |
| qwen3-4b-base → qwen3-4b-embedding | mean | random-item | 1729 | +110 [+67, +157] |
| qwen3-4b-base → qwen3-4b-embedding | mean | random-item | 2718 | +86 [+54, +117] |
| qwen3-4b-base → qwen3-4b-embedding | mean | random-item | 3141 | +87 [+44, +136] |
| qwen3-4b-base → qwen3-4b-embedding | mean | spatial-block | 1729 | +87 [+28, +156] |
| qwen3-4b-base → qwen3-4b-embedding | mean | spatial-block | 2718 | +132 [+6, +333] |
| qwen3-4b-base → qwen3-4b-embedding | mean | spatial-block | 3141 | +52 [+15, +96] |
| qwen3-4b-base → qwen3-4b-embedding | last | random-item | 1729 | +82 [+42, +128] |
| qwen3-4b-base → qwen3-4b-embedding | last | random-item | 2718 | +71 [+35, +106] |
| qwen3-4b-base → qwen3-4b-embedding | last | random-item | 3141 | +76 [+38, +122] |
| qwen3-4b-base → qwen3-4b-embedding | last | spatial-block | 1729 | +6 [-51, +63] |
| qwen3-4b-base → qwen3-4b-embedding | last | spatial-block | 2718 | +138 [+25, +284] |
| qwen3-4b-base → qwen3-4b-embedding | last | spatial-block | 3141 | -2 [-47, +45] |
| qwen3-8b-base → qwen3-8b-embedding | mean | random-item | 1729 | +101 [+57, +148] |
| qwen3-8b-base → qwen3-8b-embedding | mean | random-item | 2718 | +64 [+37, +89] |
| qwen3-8b-base → qwen3-8b-embedding | mean | random-item | 3141 | +47 [+6, +87] |
| qwen3-8b-base → qwen3-8b-embedding | mean | spatial-block | 1729 | +35 [-31, +103] |
| qwen3-8b-base → qwen3-8b-embedding | mean | spatial-block | 2718 | +101 [-32, +299] |
| qwen3-8b-base → qwen3-8b-embedding | mean | spatial-block | 3141 | +82 [+27, +146] |
| qwen3-8b-base → qwen3-8b-embedding | last | random-item | 1729 | +45 [+4, +82] |
| qwen3-8b-base → qwen3-8b-embedding | last | random-item | 2718 | +32 [-3, +69] |
| qwen3-8b-base → qwen3-8b-embedding | last | random-item | 3141 | +57 [+23, +90] |
| qwen3-8b-base → qwen3-8b-embedding | last | spatial-block | 1729 | +103 [-0, +233] |
| qwen3-8b-base → qwen3-8b-embedding | last | spatial-block | 2718 | +124 [-10, +303] |
| qwen3-8b-base → qwen3-8b-embedding | last | spatial-block | 3141 | +5 [-60, +77] |
| mistral-7b-base → e5-mistral | mean | random-item | 1729 | +27 [+2, +55] |
| mistral-7b-base → e5-mistral | mean | random-item | 2718 | +17 [-9, +42] |
| mistral-7b-base → e5-mistral | mean | random-item | 3141 | +27 [-2, +56] |
| mistral-7b-base → e5-mistral | mean | spatial-block | 1729 | +41 [-23, +109] |
| mistral-7b-base → e5-mistral | mean | spatial-block | 2718 | +42 [-32, +145] |
| mistral-7b-base → e5-mistral | mean | spatial-block | 3141 | +13 [-64, +72] |
| mistral-7b-base → e5-mistral | last | random-item | 1729 | +153 [+100, +209] |
| mistral-7b-base → e5-mistral | last | random-item | 2718 | +138 [+83, +195] |
| mistral-7b-base → e5-mistral | last | random-item | 3141 | +156 [+109, +211] |
| mistral-7b-base → e5-mistral | last | spatial-block | 1729 | +132 [+28, +241] |
| mistral-7b-base → e5-mistral | last | spatial-block | 2718 | +196 [-14, +474] |
| mistral-7b-base → e5-mistral | last | spatial-block | 3141 | +65 [-15, +152] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | random-item | 1729 | +54 [+27, +84] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | random-item | 2718 | +50 [+22, +82] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | random-item | 3141 | +62 [+24, +103] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | spatial-block | 1729 | -18 [-91, +41] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | spatial-block | 2718 | +49 [-52, +184] |
| gemma2-9b-base → bge-multilingual-gemma2 | mean | spatial-block | 3141 | +55 [-6, +145] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | random-item | 1729 | +49 [+21, +81] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | random-item | 2718 | +23 [-18, +65] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | random-item | 3141 | +22 [-16, +57] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | spatial-block | 1729 | -100 [-198, -10] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | spatial-block | 2718 | -127 [-329, +13] |
| gemma2-9b-base → bge-multilingual-gemma2 | last | spatial-block | 3141 | -10 [-90, +58] |
| xlmr-base → e5-base | mean | random-item | 1729 | +10 [-20, +37] |
| xlmr-base → e5-base | mean | random-item | 2718 | +9 [-18, +34] |
| xlmr-base → e5-base | mean | random-item | 3141 | +20 [-14, +52] |
| xlmr-base → e5-base | mean | spatial-block | 1729 | -2 [-46, +43] |
| xlmr-base → e5-base | mean | spatial-block | 2718 | -9 [-53, +38] |
| xlmr-base → e5-base | mean | spatial-block | 3141 | -14 [-37, +6] |
| xlmr-large → bge-m3 | mean | random-item | 1729 | +28 [+3, +53] |
| xlmr-large → bge-m3 | mean | random-item | 2718 | -1 [-36, +27] |
| xlmr-large → bge-m3 | mean | random-item | 3141 | +29 [+5, +55] |
| xlmr-large → bge-m3 | mean | spatial-block | 1729 | +15 [-42, +65] |
| xlmr-large → bge-m3 | mean | spatial-block | 2718 | -36 [-91, +9] |
| xlmr-large → bge-m3 | mean | spatial-block | 3141 | -12 [-68, +34] |
