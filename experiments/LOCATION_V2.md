# Expanded genre evaluation: location-v2

The user requested a broader **genre-only** evaluation after the six-tag pilot was submitted. The pilot is retained as location-v1. This amendment was specified before inspecting the expanded retrieval results. Geographic probes do not depend on the genre battery; they still use every catalog station and the same three spatial splits.

The vocabulary is the official [MusicBrainz genre list](https://musicbrainz.org/genres), snapshotted locally. Its acquisition checksum and all 2,215 reference genre names/identifiers are in `data/musicbrainz-genres.json`. It is a nonexclusive genre vocabulary, not a set of disjoint top-level classes. MusicBrainz genres do not establish that individual station tags are correct.

Match catalog tags to reference names after NFKC normalization, case folding, conversion of hyphens/underscores to spaces, conversion of `&` to `and`, and whitespace collapse. Do not infer synonyms, merge parent/subgenre styles, or add country/language/brand tags. Keep exact original matching tags in the frozen plan so every candidate set can be reconstructed. The query uses the canonical genre name; both learned and Haversine rankings use the same union of its matching raw tags.

Include every matching genre with at least 50 stations (five times k=10). This minimum prevents near-exhaustive candidate pools from producing trivially high overlap. It is a data-coverage rule fixed without model scores, not a ceiling on the number of genres. The rule selects **45 genres**. Preserve all sparse and unmatched vocabulary entries in `experiments/genre-coverage.csv`; do not claim complete coverage of music genres or radio formats. Genres overlap and vary in granularity, so macro-averaging is a defined weighting scheme rather than sampling independent classes.

Freeze `experiments/location-v2.json` and its SHA-256 hash. It holds the selected genres and spelling map, all five models, twelve unchanged origins, two document conditions, k=10, three split seeds, ridge grid, and bootstrap seeds. Reuse the unchanged metadata-only document caches from location-v1, checking station order, source hash, and model revisions. Encode all 585 queries using the exact native single-query path used by the application.

Execution amendment: preparation job `2387801` stopped before publishing query vectors because Qwen's batched result failed the prespecified native-path cosine equivalence threshold of 0.99999. The final run uses native single-query encoding for every model. The threshold was not relaxed; query texts, candidate sets, document vectors, and the frozen plan remain unchanged. The failed attempt and its dependent evaluation job are retained in the execution record.

The unchanged protocol in `LOCATION_PROTOCOL.md` defines the numerical probes, controls, retrieval metrics, origin bootstrap, interpretation limits, and computation-only boundary. The expanded run produces 10,800 retrieval cases and 59,400 pairwise response records. Origin pairs remain dependent; their summaries are descriptive. A single exact Haversine system on the shared candidate set remains the spatial oracle.

Turing remains restricted to offline batch computation and short SSH job-control/file-transfer sessions with forwarding disabled. Generate and inspect the report locally after transferring completed files.
