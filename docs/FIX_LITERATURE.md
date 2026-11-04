# Fixing within-country geography in frozen-embedding cosine: literature-driven ideas

Literature scan, 2026-11-04. Ideas are ranked by expected payoff × feasibility. "Verified" means I opened the paper page or abstract during this scan. "Not re-checked" means a well-known paper that I cite from memory and did not open this session.

**Evaluation rule for every idea fitted on training cities:** compute kNN gain only among held-out cities. Also run a *spatial-block* holdout, where whole admin-1 regions or grid cells are held out. Our kNN-regression result only held "within known regions", so a method that merely memorises regions has to be caught. The minimum controls are (a) appending the country centroid and (b) the geocoder oracle. Report the station genre-agreement cost for every idea.

---

## 1. Mid-layer selection and layer fusion (training-free, uses existing states)
- **Mechanism:** Use an intermediate layer's pooled state instead of the output layer, or a weighted concatenation of a mid layer and the output layer. Both vectors are L2-normalised first, and the cosine is computed on the concatenation, which equals a weighted sum of per-layer cosines.
- **Why within-country:** We already see e5-mistral last-token peak at L26 (0.31) against 0.26 at L32. Base models' mid layers match retrieval outputs at mean pooling, and final layers discard the signal. Fusion can keep the output layer's semantics for stations while borrowing the mid layer's geography.
- **Cites:** Skean et al., "Layer by Layer: Uncovering Hidden Representations in Language Models", ICML 2025, https://arxiv.org/abs/2502.02013 (verified; mid layers beat the final layer on 32 embedding tasks).
- **Test:** Choose the layer and the fusion weight α on training cities. Report held-out kNN gain and station genre agreement across a grid of α. Also try fusing base-LLM mid layers with the paired retrieval model's output. No GPU needed.
- **Kill:** The fused curve is Pareto-dominated by the single output layer on (kNN gain, genre agreement), or the L26 gain does not survive the held-out layer choice.

## 2. Within-country low-rank metric learning (structural-probe style), added to cosine
- **Mechanism:** Learn a rank-r map B (r ≈ 16–128) on country-centred embeddings, using only same-country pairs of training cities. The distance ‖B(x_i−x_j)‖² is trained to rank geographic neighbours correctly. Options are a regression to log-km as in the structural probe, a triplet/LMNN loss on near vs. far pairs, or closed-form supervised Locality Preserving Projections built from a geographic kNN graph. The final score is cos(x_i,x_j) − λ·‖B(x_i−x_j)‖².
- **Why within-country:** The two failed attempts fitted a 3-D *coordinate regression* and amplified that 3-D subspace. A pairwise, locally weighted objective targets neighbour ranking directly. A rank >3 subspace can also represent a curved or folded geographic manifold. That would explain why cosine kNN is locally better than the linear probe. Restricting training to same-country pairs means the map cannot gain by encoding country identity.
- **Cites:** Hewitt & Manning, "A Structural Probe for Finding Syntax in Word Representations", NAACL 2019, https://aclanthology.org/N19-1419/ (method confirmed through secondary sources). Weinberger & Saul, LMNN, JMLR 2009, https://jmlr.org/papers/v10/weinberger09a.html (not re-checked). He & Niyogi, "Locality Preserving Projections", NeurIPS 2003, https://papers.nips.cc/paper/2359-locality-preserving-projections (verified).
- **Test:** CPU or a small GPU. Start with closed-form LPP: W_ij = exp(−d_ij/σ) for the k nearest same-country training cities, then solve a generalised eigenproblem on PCA-256 residuals. Then try a triplet loss. Sweep r and λ. Run both the random and the spatial-block holdout.
- **Kill:** The gain is ≤ +0.02 on random holdout or disappears on spatial-block holdout. Either result means it memorises regions, as kNN regression did.

## 3. Per-country centring + within-country whitening / all-but-the-top
- **Mechanism:** Subtract the country mean, which needs only the country code already in the text. Then whiten with the pooled within-country covariance, or remove its top-k principal components (all-but-the-top), before taking the cosine.
- **Why within-country:** Uncentred cosine includes c·r cross-terms between the shared country vector c and each city's residual r. Those terms reward cities with a large country component (likely popularity or frequency), not proximity. The top residual directions are probably also non-geographic (name form, population). Removing them lets the low-variance geographic directions count. The method is training-free and does not use coordinates.
- **Cites:** Su et al., "Whitening Sentence Representations…", https://arxiv.org/abs/2103.15316 (not re-checked). Mu & Viswanath, "All-but-the-Top", https://arxiv.org/abs/1702.01417 (not re-checked).
- **Test:** CPU. For stations, centre on the country code and check whether genre agreement drops; whitening may hurt it. Sweep the number of removed PCs, 0–64.
- **Kill:** Centring alone barely changes within-country ranking (|Δ| < 0.01) and whitening hurts genre agreement.

## 4. Geography instructions for instruction-tuned embedders (e5-mistral, Qwen3-Embedding, bge-gemma2)
- **Mechanism:** Re-extract with a task instruction on *both* sides, since the task is symmetric. Example: `Instruct: Given a place name, retrieve places that are geographically closest to it\nQuery: Name, Country`. Variants: "…in the same province/state", or "represent the location of this place". For stations, compute a geo-instructed vector and a generic vector, then concatenate or score-fuse them so genre semantics survive.
- **Why within-country:** Retrieval models already show the most geographic structure in cosine. Instructions are the one supported knob that changes which facet the similarity weights. It costs nothing at query time and needs no coordinates.
- **Cites:** Wang et al., e5-mistral, https://arxiv.org/abs/2401.00368 (not re-checked). Su et al., INSTRUCTOR, https://arxiv.org/abs/2212.09741 (verified). For the risk that embedders follow instructions poorly: Weller et al., FollowIR, https://arxiv.org/abs/2403.15246 (not re-checked).
- **Test:** GPU re-extraction, about 46k short texts per model and instruction, which is cheap. Try 3–5 instruction wordings and also the per-layer states (combine with idea 1).
- **Kill:** Spread across instruction wordings is below run-to-run noise. Embedders often treat instructions as a near-constant offset for very short inputs.

## 5. kNN-transported pseudo-coordinates via training-city anchors
- **Mechanism:** For each city, compute a softmax-weighted average (Nadaraya–Watson) of the 3-D positions of its top-k cosine-similar *training* cities. Append the result with weight β, or use distance between pseudo-positions to rerank the cosine top-50. This is the relative-representation idea with anchor coordinates attached.
- **Why within-country:** This builds directly on our strongest positive finding: cosine-kNN regression beats the ridge probe in retrieval models. The earlier "append predicted position" failed partly because the *probe's* predictions are poor locally. Only training-city coordinates are used, which is allowed.
- **Cites:** Moschella et al., "Relative representations enable zero-shot latent space communication", ICLR 2023, https://arxiv.org/abs/2209.15430 (not re-checked).
- **Test:** CPU. Fit k and temperature on training cities, and exclude the query city itself. Run spatial-block holdout without fail. The point is to measure how much of the gain is region lookup.
- **Kill:** The gain collapses under spatial-block holdout, or does not beat the country-centroid append. Watch for a reviewer framing: "a geocoder built from the training set".

## 6. Conditional prompting for base LLMs (PonTE) + Self-Contrastive Steering
- **Mechanism:** Run a base or instruct LLM on a prompt such as `Express this text "Name, Country" in one word in terms of its location within the country: "` and take the last-token state, with mid layers included. A region-eliciting variant is `"Name, Country" is located in the state/province of`. Self-Contrastive Steering then subtracts the unconditional embedding's influence, so that the vector keeps the condition (location) rather than general meaning.
- **Why within-country:** The next-token target is an admin-1 or region name, which is a within-country variable. The model is asked to bring that knowledge up rather than leaving it buried. Base mid layers already hold the signal, and this aims it at the output token.
- **Cites:** Yamada & Zhang, PonTE, https://arxiv.org/abs/2504.16411 (verified; on C-STS, unsupervised Llama-3-8B-Inst at 37.1 Spearman matches supervised e5-mistral at 34.8). Cheng et al., "Focusing Condition: Inference-Time Self-Contrastive Steering…", ACL 2026, https://aclanthology.org/2026.acl-long.50/ (verified abstract; the combination formula was not on the landing page). Jiang et al., PromptEOL, https://arxiv.org/abs/2307.16645 (not re-checked).
- **Test:** GPU re-extraction on Qwen3-8B, Mistral-7B and Gemma-2-9B, all layers, last token. Compare with the mean-pooled baseline layers. For stations, the conditional vector is geo-only, so score-fuse it with the generic vector.
- **Kill:** It is no better than plain mean-pooled mid layers. Or it only sharpens country identity: check that the country-centroid control does not explain the gain.

## 7. Neighbourhood reranking (k-reciprocal / α-QE / DBA / graph diffusion), within country
- **Mechanism:** Rerank cosine neighbours using shared-neighbour structure. Options are a Jaccard distance on k-reciprocal sets, or replacing each vector with a similarity-weighted mean of its top-k neighbours (α-query expansion and database-side augmentation). A third option is diffusion on the kNN graph.
- **Why within-country:** Local cosine structure is partly right, as idea 5 already relies on. Mutual-neighbour constraints remove one-off coincidental neighbours, such as cities that look alike by name, while keeping consistent local clusters. It is training-free and coordinate-free.
- **Cites:** Zhong et al., k-reciprocal re-ranking, CVPR 2017, https://arxiv.org/abs/1701.08398 (verified). Iscen et al., "Efficient Diffusion on Region Manifolds", CVPR 2017, https://openaccess.thecvf.com/content_cvpr_2017/poster/758_POSTER.pdf (poster verified). Radenović et al., α-QE / DBA, https://arxiv.org/abs/1711.02512 (not re-checked).
- **Test:** CPU, inside each country's index. Sweep k1, k2 and α. Check genre agreement on stations.
- **Kill:** It amplifies systematic errors, for example clusters of names sharing a "San …" prefix, so gain is ~0 or negative.

## 8. Hubness correction (CSLS / mutual proximity / f-norm)
- **Mechanism:** Rescale similarities by each point's mean similarity to its k neighbours (CSLS), or turn them into mutual-proximity probabilities.
- **Why within-country:** In ~4k-dim spaces, a few "hub" cities (large or famous places with strong country components) can appear in many neighbour lists. That pulls the mean km of the top-10 toward national hubs. Fixing it is training-free.
- **Cites:** Nielsen & Hansen, "Hubness Reduction Improves Sentence-BERT Semantic Spaces", NLDL 2024, https://arxiv.org/abs/2311.18364 (verified; f-norm+MP cut hubness ~75% and kNN error ~9%). Conneau et al., CSLS, https://arxiv.org/abs/1710.04087 (not re-checked).
- **Test:** CPU. Diagnose first with k-occurrence skewness and the correlation of hub frequency with population. Then apply CSLS or MP within each country.
- **Kill:** Low k-occurrence skewness, meaning there are no hubs to fix.

## 9. Cross-model / cross-layer ensemble
- **Mechanism:** Concatenate L2-normalised vectors, which equals averaging cosines, from e5-mistral L26, Qwen3-Emb-8B, bge-gemma2 and the best base mid layer.
- **Why within-country:** Different models' non-geographic "noise neighbours" (tokenisation, name semantics) are likely partly independent, while the geographic signal is shared. Averaging should therefore raise the geographic share of the similarity.
- **Cites:** No specific source found; this is standard ensembling practice.
- **Test:** CPU. Fit the weights on training cities.
- **Kill:** The ensemble does not beat the best single layer (errors are correlated).

## 10. Nuisance-concept erasure (LEACE used to *remove* confounders, not to amplify geography)
- **Mechanism:** Use LEACE to erase variables that may drive within-country cosine but are not geographic. Candidates are log-population, name length, first characters / shared subword tokens, script or language of the name, and admin "type" words.
- **Why within-country:** Amplifying geography has already failed. If within-country neighbours are partly determined by population rank or surface form, erasing those directions raises the geographic share without any coordinate model.
- **Cites:** Belrose et al., "LEACE: Perfect linear concept erasure in closed form", NeurIPS 2023, https://arxiv.org/abs/2306.03819 (verified).
- **Test:** CPU. First run a diagnostic regressing neighbour-pair similarity on |Δlog pop|, character n-gram overlap and km. Erase only the confounders that are significant.
- **Kill:** The diagnostic shows confounders explain little of the variance in neighbour selection.

## 11. Residual adapter with similarity preservation (Search-Adaptor / embedding adapters)
- **Mechanism:** Train f(x) = x + U·g(Vx), with a small low-rank MLP g, on a geographic contrastive loss over same-country training pairs. Add a distillation term that keeps the original cosine on held-in stations and genres, plus a ranking-preservation term.
- **Why within-country:** This is the nonlinear version of idea 2, for the case where the geographic manifold is not linearly unfoldable. The preservation term targets the genre cost directly.
- **Cites:** Yoon et al., "Search-Adaptor", ACL 2024, https://aclanthology.org/2024.acl-long.661 (verified). Chroma, "Embedding Adapters", 2024, https://research.trychroma.com/embedding-adapters (verified; technical report).
- **Test:** Small GPU or CPU. Run only if idea 2 shows a linear signal but appears limited by capacity.
- **Kill:** Same as idea 2: no gain under spatial-block holdout.

## 12. Input formatting / echo / richer context
- **Mechanism:** (a) Echo embeddings: repeat the text so that, under causal attention, early tokens see the country. (b) Put the country first ("Country: X. City: Y"). (c) A diagnostic only: LLM-generated one-line context ("Y is a town in X"), HyDE-style.
- **Why within-country:** With mean pooling in causal models, the name tokens never see the country, so homonyms and ambiguous names blur. Feuerstein et al. report that fuller context (Wikipedia text instead of names) improves geographic recovery.
- **Cites:** Springer et al., "Repetition Improves Language Model Embeddings", https://arxiv.org/abs/2402.15449 (not re-checked). Gao et al., HyDE, https://arxiv.org/abs/2212.10496 (not re-checked). Feuerstein et al. 2026, https://arxiv.org/abs/2609.05721 (verified).
- **Test:** GPU re-extraction for (a) and (b). Treat (c) as an upper-bound diagnostic only, because it is close to geocoding.
- **Kill:** (a) and (b) change kNN gain by <0.01. If (c) helps a lot, the knowledge is in the generator but not in the embedder, which is still a useful finding but not an allowed fix.

---

## Scoop check (as of 2026-11-04; not exhaustive)
I found no paper that compares retrieval-tuned with base models on **within-country cosine-neighbour geography**, and none that proposes a coordinate-free fix for it. The closest work:
- **Feuerstein et al. 2026**, https://arxiv.org/abs/2609.05721 (verified full text). Uses commercial embedding APIs (ada-002, text-embedding-004, Voyage, Cohere) and seed-pair axes d = Σ(p₁−p₂). Reports only *global* lat/lon rank correlations (WorldPlaces lon τ≈0.80). No within-country or kNN evaluation, and no base-vs-retrieval comparison. Notes that fuller context helps and that longitude may be circular.
- **Srinivasagan & Georges 2026**, https://arxiv.org/abs/2609.00454 (verified abstract). Trains with lat/lon in the inputs, so it is not frozen and not coordinate-free. TSD 2026.
- **Gurnee & Tegmark 2024**, https://arxiv.org/abs/2310.02207 (not re-checked). Linear probes on base LLMs only; no cosine.
- **De Sabbata, Mizzaro & Roitero 2025**, https://arxiv.org/abs/2505.03368 (verified abstract). Spatial autocorrelation of LLM probe/SAE features for place names; analysis only, no fix.
- **Neira (blog, undated)**, https://mateoneira.com/probing-geographic-world-model/. Llama-3.1-8B ridge probe on 6.1k cities. Within-country deviations correlate 0.35 (E–W) and 0.57 (N–S), but the probe beats a country-centroid lookup in only 6/41 countries. This is consistent with our probe findings; it is a blog post, not peer-reviewed.
- **"Evaluation of Geographical Distortions in Language Models"**, 2024, https://arxiv.org/abs/2404.17401 (search snippet only). Global semantic-vs-geographic distance correlation is low for all models; ada beats Llama-2.
- **Xia et al. 2025**, https://arxiv.org/abs/2505.20874 (verified). Coordinates become linearly or MLP decodable only after continual pre-training; not frozen.
- **Liétard et al. 2021**, https://arxiv.org/abs/2109.07971; **Hofmann et al. TACL 2024 (geoadaptation)**, https://aclanthology.org/2024.tacl-1.23/. Both are older or training-based.

Risk: the closest potential scoop is a follow-up from the Feuerstein group, which already names "geographic ranking/tagging in retrieval" as future work.

## Ideas I'd skip and why
- **Global steering vectors / activation addition (ActAdd, CAA):** a constant offset barely changes *relative* within-country geometry. Ideas 4 and 6 are the prompt-level version, with better grounding.
- **LEACE-inverted or probe-subspace amplification of geography:** this is essentially what already failed (3-D subspace amplification).
- **Seed-axis projections (Feuerstein):** 1-D global axes are coarser than our ridge probe, which already fails locally.
- **GeoLM, SpaBERT, ERNIE-GeoL, geoadaptation, secondary embeddings, GeoLLM-style "nearby places" prompts:** they need training or fine-tuning, or coordinates/OSM context at input, so they break the frozen and coordinate-free constraints.
- **Global BERT-whitening/flow without country centring:** dominated by idea 3. Global whitening mainly reshapes the between-country variance.
