# More models, smaller towns and station search: freeze list (models-v1)

**Status: approved 2026-11-05.** Tian-hao Zhang delegated the claims, the search-task design and the Nemotron handling to Claude ("can i let u decide that?"). This file was written and committed before any validation or test score of the new models existed. Only parity checks had run (`outputs/parity/`), and they compare our features with each model's own embedding, not with geography. Runs are logged in `experiments/MODELS_RUNS.md`.

## Why

The fix was tested on three retrieval models. Here we ask three questions:

- Does it hold for current models from other teams?
- Does it hold for towns smaller than any we have tested?
- Does it help a realistic genre-and-place search?

The fix itself is frozen. Every setting comes from `experiments/fix-v1.json` and is fit on the original training cities.

## Models

| Model | Checkpoint @ revision | Pooling | Plain text | Instruction format |
| --- | --- | --- | --- | --- |
| qwen3-4b-embedding | Qwen/Qwen3-Embedding-4B @ 5cf2132 | last token | as is | `Instruct: {instruction}\nQuery:{text}` |
| harrier-0.6b | microsoft/harrier-oss-v1-0.6b @ f9b9dc8 | last token | as is | `Instruct: {instruction}\nQuery: {text}` |
| harrier-27b | microsoft/harrier-oss-v1-27b @ 0c0fc62 | last token | as is | `Instruct: {instruction}\nQuery: {text}` |
| octen-8b | Octen/Octen-Embedding-8B @ 88695e4 | last token | as is | `Instruct: {instruction}\nQuery:{text}` |
| nemotron-8b | nvidia/Nemotron-3-Embed-8B-BF16 @ d1f2f25 | mean, bidirectional | `passage: {text}` | none native; see below |

- **Parity.** Before any scoring, our pooled output layer must equal each model's own embedding. Cosine ≥ 0.9999 on 60 texts passed for qwen3-4b, harrier-0.6b, octen-8b and nemotron-8b. The control, qwen3-8b, also passed. harrier-27b is checked on the H200 before its extraction.
- **Nemotron.** It accepts only fixed `query: ` / `passage: ` prefixes, with no task instruction. Its frozen claim therefore covers fusion and probe only. Its instruction variants use the non-native `query: {instruction}\n{text}` and are reported descriptively.
- **harrier-27b.** It needs an H200 (fp32 weights, 108 GB). If no H200 run completes, its claims are reported as "not run", not dropped.
- **Plain text** follows each model's own document format. Octen's model card suggests a `"- "` prefix for documents. We do not use it, because Qwen3-Embedding-8B, its base, was also run without one.

## Scored sets

Components are fit on the original training cities. Spatial-block scores only places outside that split's training blocks.

| Set | Places | Status |
| --- | --- | --- |
| cities | Original test split (15,000+ people) | **Reused.** Used by layers-v7, reflected-v1 and fix-v1 |
| towns | Test part of fresh-v1 (5,000-14,999 people) | **Reused.** Used once by fresh-v1 |
| small-towns | GeoNames `cities1000` towns of 1,000-4,999 people (see below) | **Fresh** |

- **Why reuse is acceptable here.** The fix is frozen, and the new models were never evaluated on these places. Reusing a test set to evaluate a frozen method on new models is standard benchmark practice. It is weaker evidence than a fresh set, and the paper will say so.
- **Small-towns filters.** Towns must be absent from `cities5000` (2026-11-04) and `cities15000` (2026-11-02). Their name-country text must be unique and differ from every text in those files.
- **Small-towns sizes.** 68,202 towns remain (`cities1000` SHA-256 `621da017…a666a2`; checksum `11fa5719…f327f5`). The same permutation seed as fresh-v1 (20261104) gives a 20% check part and an 80% test part. The test part is capped at its first 25,000 towns in permutation order, to bound compute.

## Claims (city within-country kNN gain, paired by country, 95% block-bootstrap CI)

Each claim needs the interval above 0 in **every** run: 3 sets × 2 splits × 3 seeds = 18 runs per model, unless stated.

| # | Claim | Supported if |
| --- | --- | --- |
| P1 (each of qwen3-4b, harrier-0.6b, harrier-27b, octen-8b) | The frozen fix beats plain cosine | Instruction + fusion + probe minus cosine, CI above 0 in all 18 runs. One verdict per model |
| P1 nemotron-8b | Fusion and probe beat plain cosine without an instruction | Fusion + probe minus cosine, CI above 0 in all 18 runs |
| P2 (each of the four instruction models) | The geographic wording, not using an instruction, drives the gain | Fix minus the same pipeline with "Retrieve semantically similar text.", CI above 0 in all 18 runs |
| P3 | The fix still helps on towns of 1,000-4,999 people | For e5-mistral and bge-multilingual-gemma2: fix minus cosine on small towns, CI above 0 in all 6 runs each |
| P4 (descriptive) | Population trend | Per model, cosine and fix kNN gain across cities, towns and small towns. Also reported: fusion and probe contributions, Feuerstein axes, the geocoder oracle, Qwen3-Embedding-8B and Nemotron's instruction variants |
| P5 (each of e5-mistral, bge-gemma2, qwen3-4b, harrier-0.6b, harrier-27b, octen-8b) | The fix improves a realistic station search | Fix minus standard use in within-country search gain, CI above 0 in all 6 runs (2 splits × 3 seeds) |

Qwen3-Embedding-8B stays descriptive throughout, as in fix-v1 and fresh-v1.

## P5: station search task

A query asks for radio stations of a genre near a town. Genre is matched lexically, the way the explorer's retriever already does it. The embedding model ranks the matching stations by closeness.

- **Documents (11,833 catalog stations):** `Station: {name}\nLocation: {place}, {country}\nGenres: {tags}`. The place is the `cities15000` place nearest to the station's coordinates, like the city a station directory would list. Coordinates are never shown to a model.
- **Queries:** `{genre} radio stations near {town}, {country}`.
  - Genres are the 8 most frequent station genres from the frozen 45-genre list of `experiments/location-v2.json`: pop, rock, dance, pop rock, jazz, classic rock, electronic and house. Raw tags also include non-genre words such as "radio" and "estación". This was corrected on 2026-11-05, before any search code or result existed.
  - For each genre, we use countries with at least 20 stations carrying it.
  - Origins come from the fresh-v1 towns: the check part for validation runs, the test part for the test run. Spatial-block keeps only towns outside the training blocks. Up to 10 towns per genre and country are drawn with seed 20261104.
- **Pool:** stations in the query's country carrying the query's genre.
- **Metric:** search gain = 1 − (mean km to the 10 top-ranked pool stations) / (mean km to all pool stations). It is averaged per country, paired by country between methods, and bootstrapped over countries as above.
- **Standard use:** the query is wrapped in each model's own web-search instruction, "Given a web search query, retrieve relevant passages that answer the query". Documents are plain text, scored by cosine.
- **Fix:**
  - The geography instruction wraps both the query and the documents. The index is built once for geographic search.
  - Fusion and the distance probe are added, fit on training cities as everywhere else.
- **Also reported:** plain query and document cosine, the geography instruction alone, true-distance ranking (the ceiling), and Nemotron with its native prefixes plus fusion and probe.

Its validation run uses check-part towns and must pass pipeline checks before the test run. No setting of the task is changed after its validation run, except to fix a bug, which would be logged.

## What runs

- **Extraction:**
  - New models: plain, geographic and similarity-instruction features for cities, stations, STS validation, towns and small towns. Search queries and documents follow once the search module exists.
  - Old claim models: small towns.
- **Validation:**
  - `fix_baselines --config experiments/fix-baselines-v2-dev.json`, covering validation cities, stations and STS-B validation.
  - `fresh_places --config experiments/models-v1.json --part check`, covering validation cities, the towns check part and the small-towns check part.
- **Test, once:** `fresh_places --config experiments/models-v1.json --part test`, plus the search-task test. The module refuses a second test run.

Every outcome will be reported, including models or claims that fail.
