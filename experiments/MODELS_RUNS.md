# More models, small towns and station search (models-v1)

Claims and design: `docs/MODELS_FREEZE.md`. Plans:

- `experiments/models-v1.json` (SHA-256 `9f855887a8f95433ee7ffc27a949794878f4dd5376e5a1a7cf8c5328cbf858eb`);
- `experiments/fix-v2.json` (`908fb70e…`) and `experiments/fix-generic-v2.json` (`cdfe1224…`), the test copies of `fix-v2-dev.json` and `fix-generic-v2-dev.json`;
- `experiments/fix-baselines-v2-dev.json`.

## Inputs

| File | SHA-256 | Note |
| --- | --- | --- |
| `data/raw/cities1000.zip` | `621da017da3e4eebd0d1012a641d4150703fe870e44b4828224d58e1dca666a2` | Downloaded 2026-11-05 from download.geonames.org |

Checkpoints were downloaded to the BC Hugging Face cache on 2026-11-05 at the revisions in `models/layer_catalog.py`. harrier-27b has 11 shards (54.0 GB).

Small towns (1,000-4,999 people, not in `cities5000` or `cities15000`, unique text) number 68,202; checksum `11fa57195c80a0ead0fbcf69b114b2c398571dfaa4b04622acff40c930f327f5`. The 5k-15k town checksum is unchanged, and no town text appears in both bands.

| Part | Random-item | Spatial-block (seeds 1729 / 2718 / 3141) |
| --- | --- | --- |
| Check | 13,640 towns, 50 countries with 30 or more | 21 / 34 / 35 countries |
| Test (before the 25,000 cap) | 54,562 towns, 88 countries | 42 / 53 / 53 countries |

## Jobs

| Job | Role | Outcome | Elapsed |
| --- | --- | --- | --- |
| 3361908 (array 0-3) | Parity: qwen3-8b-embedding (control), qwen3-4b-embedding, harrier-0.6b, octen-8b | 0-2 passed (min cosine 0.9999999). 3 failed: Octen's saved Normalize module has an argument Sentence Transformers 6.1 rejects | 0:19-0:33 |
| 3362053 (array 0-1) | Parity: octen-8b with Sentence Transformers' Transformer and Pooling modules; nemotron-8b (mean, `passage: `) | Nemotron passed (0.9999999). Octen reference mismatched (min 0.566): the module-built reference applies Octen's chat template, which the card's usage does not. Replaced by the card's Transformers example | 0:24-0:26 |
| 3362411 (array 0-1) | Parity against the model card's Transformers example: octen-8b, and qwen3-8b-embedding as control | Both passed (0.9999999) | |
| 3362220 | Refactor check: fresh-v1 check part rerun into `outputs/refactor-checks/` | Cancelled at once: submitted with failure-only email, resubmitted with all notifications | 0:00 |
| 3362223 | Same, resubmitted | Completed. All 306 rows equal `outputs/fresh-v1/check/scores.jsonl` in kNN gain, rho, town counts and fusion layers, so the generalized scorer reproduces fresh-v1 | 45:26 |
| 3362618 (array 0-3) | Extraction, `fix-v2-dev.json` standard jobs (plain and geographic cities and stations, STS validation): qwen3-4b-embedding, harrier-0.6b, octen-8b, nemotron-8b | Completed; logs clean | 0:03:53-0:29:31 |
| 3362619 (array 0-3) | Same with `fix-generic-v2-dev.json` (similarity instruction) | Completed; logs clean | 0:03:24-0:26:00 |
| 3362620 (array 0-3) | Towns and small towns, plain and geographic (`--towns plain-and-instructed`) | Completed; logs clean | 0:06:20-0:43:56 |
| 3362621 (array 0-3) | Towns and small towns, similarity instruction | Completed; logs clean | 0:03:05-0:22:34 |
| 3362622 (array 0-2) | Small towns for e5-mistral, bge-multilingual-gemma2, qwen3-8b-embedding with `fix-v1.json` | Completed; logs clean | 0:29:52-0:34:40 |
| 3362623 (array 0-2) | Same with `fix-generic-v1-dev.json` | Completed; logs clean | 0:14:43-0:18:06 |
| 3362624 | harrier-27b on one H200 (`short`, 320 GB): parity, then all four extraction steps in sequence | Cancelled while pending: the gtml route (3385945) passed parity and is extracting | 0:00 |
| 3363090 (array 0-2) | Search features (documents, check and test queries; plain and geographic) for e5-mistral, bge-gemma2, qwen3-8b with `fix-v1.json` | Completed; logs clean | 0:14:19-0:17:51 |
| 3363091 (array 0-3) | Same for qwen3-4b, harrier-0.6b, octen-8b, nemotron-8b with `fix-v2-dev.json` | Completed; logs clean | 0:01:45-0:15:06 |
| 3363092 (array 0-5) | Web-search-instruction queries (`search-instruction-v1.json`) for the six runnable instruction models | Completed; logs clean | 0:00:15-0:00:39 |
| 3363093 | harrier-27b search features on one H200 | Cancelled while pending: the gtml route (3385945) passed parity and is extracting | 0:00 |
| 3363197 | harrier-27b on 4 gtml L40S, fp32 spread over GPUs (`spread_over_gpus`): parity, then all extraction steps. Starts after the L40S arrays; whichever of this and the H200 jobs finishes first is used | Failed at parity load: `device_map="auto"` needs `accelerate`, which was not installed. Added `accelerate` to the `ampere` extra (1.15.0; lock otherwise unchanged) and synced `.venv-ampere` | 0:00:12 |
| 3378323 (array 0-2) | Early validation without harrier-27b, on copies in `outputs/early-checks/`: `fix_baselines`, `fresh_places --part check`, `station_search --part check` | Completed. Pipelines and reports ran for all models; no code bug found. Check-part findings: P1 and P2 fail for harrier-0.6b (its instruction format matches the model card). P5 fails for most models, driven by mislocated stations (see the addendum in `docs/MODELS_FREEZE.md`). No setting changed | 1:04:50-1:51:05 |
| 3378448 | Same as 3363197, resubmitted with `accelerate` | Failed in parity extraction: Gemma-3 hidden states exceed the float16 range. Added a per-model `feature_dtype`; harrier-27b now stores float32 features, other models unchanged | 0:00:35 |
| 3385728 | harrier-27b on 4 gtml L40S, as 3378448, with float32 features | Failed at the parity reference: our extraction ran, but Sentence Transformers 6.1 loads this Gemma-3 text model through `AutoProcessor` and finds no image processor. The tokenizer adds `<bos>` and `<eos>`, as our spec does, so the model card's Transformers example is used as the reference, as for octen-8b; it now spreads over GPUs too | 0:00:47 |
| 3385745 | Search check part with mislocated stations removed from the pools (`outputs/early-checks/search-v1-located.json`, descriptive) | Completed. Without the 55 mislocated stations, instruction + fusion + probe has a higher mean search gain than standard use for every model. Intervals above 0 (of 6): e5-mistral 6, qwen3-8b 6, octen-8b 5, qwen3-4b 4, harrier-0.6b 4, bge-gemma2 3 | 1:03:19 |
| 3385945 | Same, parity against the model card's Transformers example (`--without-saved-modules`) | Completed. Parity passed (min cosine 0.9999998, 60 texts); all features extracted | 7:22:25 |
| 3390538 (array 0-3) | Official validation and check runs with harrier-27b, after 3385945 succeeds: `fix_baselines` (`fix-baselines-v2-dev.json`), `fresh_places` (`models-v1.json`, check), `station_search` (`search-v1.json` and `search-v1-located.json`, check) | Completed. Every row shared with the early checks (3378323, 3385745) is identical; the extra rows are harrier-27b. On the check part harrier-27b passes P1 and P2 (18 of 18 each) and P5 (6 of 6) | 1:17:19-2:13:08 |
| 3405093 | **Test, once:** `fresh_places --config experiments/models-v1.json --part test` (P1-P4; cities test split, towns test part, first 25,000 small towns of the test part) | Completed (`outputs/models-v1/test/`). P1 supported for harrier-27b, octen-8b and nemotron-8b (18 of 18); not supported for qwen3-4b (16 of 18; small-towns random-item 3141 and spatial-block 2718 intervals cross 0) or harrier-0.6b (6 of 18). P2 supported for qwen3-4b, harrier-27b and octen-8b; not for harrier-0.6b (13 of 18). P3 supported (12 of 12) | 2:39:18 |
| 3407636 (array 0-1) | **Test, once** (approved by Tian-hao Zhang after reading the addendum): `station_search --part test` with `search-v1.json` (P5) and `search-v1-located.json` (descriptive) | Completed (`outputs/search-v1/test/`, `outputs/search-v1-located/test/`). P5 (frozen pools) supported for bge-gemma2 and harrier-27b (6 of 6); not for e5-mistral (5 of 6: random-item 3141, +0.108 [−0.013, +0.230]), harrier-0.6b (2), octen-8b (2), qwen3-4b (1). Located pools, descriptive (intervals above 0 of 6): e5, qwen3-8b, octen, harrier-27b 6; bge, qwen3-4b, harrier-0.6b 5 | 1:24:48-1:26:30 |
| 3407923 | size-v1 (`docs/SIZE_FREEZE.md`): parity, then plain, geographic and similarity-instruction features of qwen3-0.6b-embedding for cities, stations, STS validation, towns and small towns | Completed. Parity passed (min cosine 0.99999994, 60 texts) | 0:17:44 |
| 3407924 | size-v1 check part (`fresh_places --config experiments/size-v1.json --part check`), after 3407923 succeeds | Completed; pipeline ran for all sets. Plain cosine is 0.023-0.051, in line with this model's last-token cosine in `REFLECTED_V1_RESULTS.md` (−0.006 to 0.039), so the features are consistent. No setting changed | 0:08:15 |
| 3408121 | **Test, once:** `fresh_places --config experiments/size-v1.json --part test` | Completed (`outputs/size-v1/test/`). S1 not supported (5 of 18), S2 not supported (2 of 18). S3 prediction held: fix gain 0.6B < 4B < 8B on all three sets | 0:12:09 |
| 3408561 (array 0-2) | search-v2 (`docs/SEARCH_V2_FREEZE.md`): plain and geographic small-town query features, e5-mistral, bge-gemma2, qwen3-8b (`fix-v1.json`) | Completed; logs clean | 0:01:47-0:02:31 |
| 3408562 (array 0-3) | Same for qwen3-4b, harrier-0.6b, octen-8b, nemotron-8b (`fix-v2-dev.json`) | Completed; logs clean | 0:00:41-0:01:45 |
| 3408563 (array 0-5) | Web-search-instruction small-town queries for the six runnable instruction models | Completed; logs clean | 0:00:27-0:01:16 |
| 3408564 | Same three query sets for harrier-27b on 4 gtml L40S | Completed; logs clean | 0:07:21 |
| 3408565 | search-v2 check part, after all four succeed | Completed (`outputs/search-v2/check/`). Pipeline ran for all models: 1,107 queries in 17 countries (random-item) as frozen, the same true-distance ceiling for every model (0.512 / 0.504). Search gains are lower than in search-v1 for every method. Check-part P6 intervals above 0 (of 6): e5-mistral and harrier-27b 6, octen-8b 5, qwen3-4b 4, harrier-0.6b 2, bge-gemma2 1. No setting changed | 1:17:50 |
| 3409349 | **Test, once:** `station_search --config experiments/search-v2.json --part test` (P6) | Submitted | |
