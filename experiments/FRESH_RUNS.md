# Fresh-towns runs (fresh-v1)

Claims and design: `docs/FRESH_FREEZE.md`. Plan: `experiments/fresh-v1.json` (SHA-256 `a1958a7682c16581084a9048dbae90a07c0045b9ebe5c72976a3a12da46e1875`).

## Inputs

| File | SHA-256 | Note |
| --- | --- | --- |
| `data/raw/cities5000.zip` | `38c3d7ee3a0ebb7abe3245d5831a60e14a38a0f36a7418b6e8442de59365b79f` | Downloaded 2026-11-04 from download.geonames.org |
| `data/raw/cities15000.zip` | `c7b7482b115463874798824f76381f84103233c0a9db9d43efe8501719200dc9` | Original snapshot (2026-11-02) |
| `data/raw/countryInfo.txt` | `79f8f59ea01cbb40438bf1856dd6c2ff201f03f8fc72d0530b2646aaf603ef04` | |

How the 32,148 towns were selected from the 70,000 rows of `cities5000`:

| Step | Rows |
| --- | --- |
| Start: `cities5000` | 70,000 |
| Not in our `cities15000` snapshot | 35,804 |
| Population 5,000-14,999 (drops 64 that grew past 15,000 since the snapshot and 632 places under 5,000, mostly administrative seats) | 35,108 |
| Text not equal to a `cities15000` text, and unique among towns | 32,148 |

Scored towns (countries with at least 30 scored towns):

| Part | Random-item | Spatial-block (seeds 1729 / 2718 / 3141) |
| --- | --- | --- |
| Check | 5,391 towns in 38 countries | 18 / 25 / 25 countries |
| Test | 24,378 towns in 85 countries | 34 / 53 / 47 countries |

## Jobs

| Job | Role | Outcome | Elapsed |
| --- | --- | --- | --- |
| 3343705 (array 0-2) | Town extraction, plain and geographic instruction (`fix-v1.json`, `--towns plain-and-instructed`), e5-mistral, bge-gemma2, qwen3-8b; gtml L40S | Submitted | |
| 3343706 (array 0-2) | Town extraction, similarity instruction (`fix-generic-v1-dev.json`, `--towns instructed`) | Submitted | |
| 3343707 | `fresh_places --part check`, `afterok:3343705:3343706`, gtml, 160 GB | Submitted | |
