# Explorer execution record

The 2026-10-17 Radio Browser snapshot contains 13,498 raw records. Validation and exact-stream deduplication retain 11,479 stations, 179 nonempty country codes, and 859 exact rock tags. These are catalog counts, not audited labels.

Job 2387361 failed the unit-norm guard after reduced-precision inference and published no cache. Float64 renormalization corrected it. Job 2387363 completed Qwen3, BGE-M3, and E5 in 1m42s on gpu-5-27. Jobs 2387646 and 2387647 exposed missing Gemma image-processor dependencies; the official SentenceTransformers image extra supplied them. Job 2387648 completed Gemma and CLAP text embeddings in 1m22s on gpu-5-21. Each artifact contains aligned finite unit vectors and a pinned checkpoint revision.

Check jobs 2387399 and 2387649 passed the checks then present (nine and twelve tests respectively), formatting, lint, and artifact validation. The proposal compiled to eight pages and the Lean build passed. Local browser checks covered query replacement, three-dimensional projection, full-catalog globe, station playback, and empty startup. Physical-device pinch validation remains pending.

The NYC rock example had overlap@10 = 0 and a first embedding neighbor 1,928.1 km away. This was a pilot example, not held-out evidence. The subsequent location experiment is recorded separately in LOCATION_RUNS.md.

The historical remote web job 2387364 and research forwarding tunnel were stopped at the user's request. The web launcher now refuses execution. Turing is reserved for batch computation and file transfer; the existing Codex SSH control connection remains enabled. The explorer runs locally from local artifacts.
