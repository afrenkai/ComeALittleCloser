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
| 3362223 | Same, resubmitted | Submitted | |
| 3362618 (array 0-3) | Extraction, `fix-v2-dev.json` standard jobs (plain and geographic cities and stations, STS validation): qwen3-4b-embedding, harrier-0.6b, octen-8b, nemotron-8b | Submitted | |
| 3362619 (array 0-3) | Same with `fix-generic-v2-dev.json` (similarity instruction) | Submitted | |
| 3362620 (array 0-3) | Towns and small towns, plain and geographic (`--towns plain-and-instructed`) | Submitted | |
| 3362621 (array 0-3) | Towns and small towns, similarity instruction | Submitted | |
| 3362622 (array 0-2) | Small towns for e5-mistral, bge-multilingual-gemma2, qwen3-8b-embedding with `fix-v1.json` | Submitted | |
| 3362623 (array 0-2) | Same with `fix-generic-v1-dev.json` | Submitted | |
| 3362624 | harrier-27b on one H200 (`short`, 320 GB): parity, then all four extraction steps in sequence | Submitted | |
| 3363090 (array 0-2) | Search features (documents, check and test queries; plain and geographic) for e5-mistral, bge-gemma2, qwen3-8b with `fix-v1.json` | Submitted | |
| 3363091 (array 0-3) | Same for qwen3-4b, harrier-0.6b, octen-8b, nemotron-8b with `fix-v2-dev.json` | Submitted | |
| 3363092 (array 0-5) | Web-search-instruction queries (`search-instruction-v1.json`) for the six runnable instruction models | Submitted | |
| 3363093 | harrier-27b search features on one H200 | Submitted | |
| 3363197 | harrier-27b on 4 gtml L40S, fp32 spread over GPUs (`spread_over_gpus`): parity, then all extraction steps. Starts after the L40S arrays; whichever of this and the H200 jobs finishes first is used | Submitted | |
