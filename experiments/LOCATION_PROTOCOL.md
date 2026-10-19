# Location experiment v1

This is a frozen pilot analysis plan following exploratory inspection of the radio explorer. It is not a preregistration preceding all observation. Freeze the exact JSON bytes, source checksum, model revisions, split manifests, tokenizer audit, and software lock before examining experiment results.

## Question and scope

Does a final station embedding support linear geographic decoding, and does changing only a query origin move cosine neighbors toward that origin? Recoverability, retrieval sensitivity, and causal use are distinct. This experiment measures the first two. It does not establish internal mechanisms or causal mediation. No audio is collected or embedded here; CLAP uses its text tower.

## Inputs

Use all 11,479 stations in the frozen, exact-stream-deduplicated Radio Browser catalog. All five checkpoints and their native retrieval prompts remain fixed. The original condition contains station name, country code, tags, and numeric latitude/longitude. The ablation removes only the two coordinate fields, preserving name, country, and tags. It is **metadata-only**, not names-only or location-free: geographic names and country codes can still reveal location.

Encode metadata-only documents at the same token limit as the original cache. Audit untruncated native-tokenizer lengths for both conditions and every query, including native prompt prefixes. Count any over-limit document conservatively as not guaranteed to retain its complete coordinate suffix. Save per-station counts for a subsequent truncation sensitivity analysis; do not silently discard long documents or imply all numeric coordinates were retained.

## Geographic decoding

Partition occupied 10-degree latitude/longitude cells into 60% training, 20% validation, and 20% test blocks using each of the three fixed seeds. Fractions apply to blocks, not stations. A block never crosses partitions; +180 and -180 longitude share a block. Splits are reused by model and condition. There is no exclusion buffer across adjacent block boundaries and no station-family grouping beyond exact-stream deduplication; report these limitations.

Fit an affine ridge map from the final normalized embedding to the three-dimensional unit-sphere coordinate vector. Center features and targets using training data only. Penalize weights, not the intercept. Select alpha from the frozen grid by validation block-mean angular error; keep the training-only fitted coefficients for test prediction. Normalize predictions to the sphere; use the training coordinate centroid for a zero prediction. No test target participates in fitting or alpha selection.

Controls: (1) the training global coordinate centroid; (2) per-country training centroids with a training-global fallback for unseen/empty countries; (3) identical probes after permuting only the training labels; (4) independent unit-normalized Gaussian features of the same dimension and the same split/alpha search; (5) true unit coordinates as an explicit positive control. The country baseline uses known country codes because both document conditions contain them. Permutation/random controls measure this implementation's capacity and chance behavior; they are not a reproduction of word-type control tasks.

Primary metric: mean of test-block mean great-circle errors in km. Also report station median and p90, all selected penalties, station/block counts, and paired improvement over the country baseline. Use a 95% percentile bootstrap over test blocks (1,000 draws, fixed seed). Seed 1729 is primary; the other two are split-sensitivity checks. Intervals are conditional on these data and partitions; do not pool overlapping test sets as independent samples or interpret exploratory multiple-model intervals as adjusted significance tests.

## Retrieval and origin-only counterfactuals

Use six prespecified exact tags and twelve fixed approximate city origins in the JSON. Construct `nearest {genre} station` and append five-decimal latitude/longitude exactly as in the app. Hold every other character fixed while changing origin. Include the same query without any appended origin as a negative control. Native model prompt prefixes remain unchanged. Use full-dimensional cosine, not PCA.

For the spatial comparison, restrict both learned retrieval and exact Haversine retrieval to **the same exact-tag candidate set**. Report overlap@10, mean retrieved/reference distance, and excess mean kilometres (geographic regret). Also retrieve without genre filtering and report exact-tag precision@10 separately. These are metadata-tag measurements, not human genre ground truth. Each genre must have at least ten candidates; fail rather than select a replacement tag after results.

Measure pairwise top-10 turnover, query cosine distance, and movement toward the new origin. Movement is the mean of the two cross-origin distance improvements: each origin's own query neighbors are compared with the other origin's query neighbors. Positive values indicate better alignment with the assigned origin; arbitrary neighbor changes can yield zero or negative values. The no-origin condition must yield identical neighbors, zero turnover, and zero movement, up to numeric tolerance. Exact Haversine is the spatial oracle with overlap one and regret zero.

Aggregate retrieval metrics equally over the six genres and twelve origins. Bootstrap paired changes in regret over origin sites (averaging genres within each site), not over individual stations. Pairwise response metrics are descriptive: origin pairs share sites and must not be treated as independent trials. Do not tune prompts, genres, origins, model selection, or alpha grids after inspecting results.

## Interpretation and reproducibility

A successful probe on explicit-coordinate text may reflect coordinate transcription. A metadata-only improvement over country centroids supports finer recoverable geographic information for this corpus; it does not demonstrate novel world knowledge. Poor retrieval with a useful probe supports a representation/retrieval mismatch, not proof that the model ignores all spatial information. Compare probe and retrieval results before characterizing the initial UI observation.

Save raw per-station test errors, per-origin ranked identifiers, pairwise responses, token counts, exact plan bytes/hash, split assignments, manifests, CSV summaries, and a standalone scientific figure. All new mathematical definitions are formalized in `formal/ComeCloser/LocationExperiment.lean`; these proofs do not verify the numerical eigensolver, tokenizer audit, or model outputs.

Methodological references: [Hewitt and Liang (2019), control tasks](https://aclanthology.org/D19-1275/); [Elazar et al. (2021), amnesic probing](https://doi.org/10.1162/tacl_a_00359). Our label permutations and random features are capacity controls adapted to regression; this run does not implement amnesic interventions.

## Execution boundary

Turing is computation-only for this project. Research SSH sessions submit/check batch jobs and transfer files with forwarding disabled. Research web services and forwarding tunnels must remain off; `ops/web.sbatch` refuses execution. The user explicitly leaves the existing desktop-managed Codex SSH control connection enabled. Batch model loading uses cached pinned checkpoints with Hugging Face/Transformers offline flags. The local explorer reads local artifacts and does not depend on a turing web service.
