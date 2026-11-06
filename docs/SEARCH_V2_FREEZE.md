# Station search on fresh origins with clean pools: freeze list (search-v2)

**Status: approved 2026-11-05.** Tian-hao Zhang approved this follow-up in advance, to run only if search-v1 came back mixed ("Yes, if search is mixed"). Claude decided the claim. This file was written and committed after the search-v1 test and before any search-v2 feature or score existed. Runs are logged in `experiments/MODELS_RUNS.md`.

## Why, and what we already know

The frozen P5 test (`docs/MODELS_FREEZE.md`) was mixed:

- It was supported for bge-gemma2 and harrier-27b (6 of 6).
- It was not supported for:
  - e5-mistral (5 of 6);
  - octen-8b and harrier-0.6b (2 of 6);
  - qwen3-4b (1 of 6).

The check run had already shown why. Two Dutch stations geolocated in the Seychelles, and other stations listed in another country, dominate some countries' search gain. The post hoc located-pool analysis removed the 55 such stations among genre stations. On the same test queries, the interval was then above 0 in 5 or 6 of 6 runs for every model.

That analysis was chosen after we had seen the check part. search-v2 therefore repeats it as a frozen test on query towns that no search run has used. The search-v1 results stand as reported.

## Design

Everything is as in search-v1 (`experiments/search-v1.json`), with three changes fixed before any v2 data:

1. **Clean pools.** Stations whose nearest `cities15000` place lies outside their catalog country are removed from the pools and from the per-country station counts. This removes 247 of 11,833 stations, 55 of them with a search genre. The documents and their features are unchanged.
2. **Fresh origins.** Origin towns come from the small-town band (1,000-4,999 people, `SmallTownSource`). The check part gives the check run; the models-v1 test part, capped at its first 25,000 towns, gives the test. No search run has used these towns.
3. **More queries.** Up to 25 origins per genre and country instead of 10, to narrow the intervals. This gives 1,107 check queries and 1,105 test queries in 17 countries.

The code is `SmallSearchSource` in `evaluation/fix_schemas.py` and the plan is `experiments/search-v2.json`. The search-v1 queries are byte-identical after the change: their query IDs equal the extracted manifests.

## Claim

| # | Claim | Supported if |
| --- | --- | --- |
| P6, each of e5-mistral, bge-gemma2, qwen3-4b, harrier-0.6b, harrier-27b and octen-8b | The fix improves station search over standard use | Fix minus standard use in within-country search gain, paired by country, 95% block-bootstrap interval above 0 in all 6 runs (2 splits × 3 seeds) |

Descriptive, as in search-v1:

- plain cosine;
- the geography instruction alone;
- plain fusion and probe;
- Nemotron with its native prefixes;
- Qwen3-Embedding-8B;
- the true-distance ceiling.

## What runs

- **Extraction:** plain and geographic query features for the claim models plus Qwen3-8B and Nemotron, and web-search-instruction queries for the instruction models.
- **Check part:** `station_search --config experiments/search-v2.json --part check`. Pipeline checks only.
- **Test, once:** `station_search --config experiments/search-v2.json --part test`.

Every outcome will be reported next to search-v1, not instead of it.
