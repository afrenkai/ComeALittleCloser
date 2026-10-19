# Location experiment execution record

All runs used the frozen Radio Browser checksum `0a7dd5c2090dd20e1d99087b8e451fcebb178bafb04a357c4e2da71eb5db1c48` and the five recorded checkpoint revisions. Research runs used turing for offline batch computation. SSH sessions for status and file transfer disabled forwarding and connection persistence; the research web service and forwarding tunnel remained off. The user's final clarification leaves the existing desktop-managed Codex SSH control connection enabled.

| Job | Role | Outcome |
| --- | --- | --- |
| 2387783 | Six-tag pilot preparation | Completed, exit 0, 2m14s |
| 2387796 | Pilot evaluation | Completed, exit 0, 53s |
| 2387801 | Expanded preparation with batch/native equivalence guard | Failed, exit 1, 30s; Qwen did not meet cosine 0.99999; no query artifact published |
| 2387802 | Evaluation depending on failed preparation | Cancelled |
| 2387814 | Expanded preparation using native single-query calls | Completed, exit 0, 1m16s, gpu-5-27 |
| 2387816 | Evaluation briefly attached to an incorrect dependency | Cancelled before running, elapsed 0s |
| 2387817 | Correct expanded evaluation | Completed, exit 0, 2m39s, compute-2-03 |

Pilot plan SHA-256: `788cd8b3e83eddfc219b5c7543b89770a907bf1136f0543eccbf4413021152d7`.

Expanded plan SHA-256: `f6023065fb402ff6c1edc3beb6e729bd51e2cf230aadc564a021ea048df51cfd`.

The expanded run contains 84 probe records, 10,800 retrieval cases, and 59,400 dependent origin-pair records. All 2,925 model-query encodings are below the token limit. Original explicit-coordinate documents exceeding the limit are recorded per model; the largest count is 217/11,479 for CLAP. Per-station tokenizer audits remain in `data/processed/location-v1/` and `location-v2/`.

The three spatial partitions each use 131 training, 44 validation, and 44 test blocks. Station counts differ because the split is by occupied geographic block. Test targets never enter probe fitting or alpha selection. Local checks: 18 tests, Ruff, Black, and the pinned Lean build pass. Report validation checks complete grids, no-origin zero controls, frozen plan/provenance, and taxonomy candidate counts. Figures and report were generated locally after transferring the completed files.
