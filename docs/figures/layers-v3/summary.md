# Layerwise probe summary (dev mode)

All numbers are **validation (dev-mode) results** from the `layers-v3` plan; they are not held-out test results. Best layer = maximal seed-mean gain over the country-centroid control (over pooling and layer), chosen on the same validation items, so gains are optimistic. CIs are seed-means of 95% block-bootstrap bounds.

## Models and revisions

| model | family | stage | checkpoint | revision |
|---|---|---|---|---|
| bge-m3 | xlmr-large | retrieval | BAAI/bge-m3 | 5617a9f61b028005a4858fdac845db406aefb181 |
| bge-m3-retromae | xlmr-large | long-context pretraining | BAAI/bge-m3-retromae | 95c4f81ef8f7911dcac8e384532489c32c8dea64 |
| bge-m3-unsupervised | xlmr-large | unsupervised contrastive | BAAI/bge-m3-unsupervised | 46f03bc86361cf88102b0b517b36c8259f2946b1 |
| e5-base | xlmr-base | retrieval | intfloat/multilingual-e5-base | d128750597153bb5987e10b1c3493a34e5a4502a |
| qwen3-0.6b-base | qwen3-0.6b | base | Qwen/Qwen3-0.6B-Base | da87bfb608c14b7cf20ba1ce41287e8de496c0cd |
| qwen3-0.6b-embedding | qwen3-0.6b | retrieval | Qwen/Qwen3-Embedding-0.6B | 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 |
| qwen3-0.6b-random | qwen3-0.6b | random | Qwen/Qwen3-0.6B-Base | da87bfb608c14b7cf20ba1ce41287e8de496c0cd (random init) |
| qwen3-4b-base | qwen3-4b | base | Qwen/Qwen3-4B-Base | 906bfd4b4dc7f14ee4320094d8b41684abff8539 |
| qwen3-4b-embedding | qwen3-4b | retrieval | Qwen/Qwen3-Embedding-4B | 5cf2132abc99cad020ac570b19d031efec650f2b |
| qwen3-8b-base | qwen3-8b | base | Qwen/Qwen3-8B-Base | 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 |
| qwen3-8b-embedding | qwen3-8b | retrieval | Qwen/Qwen3-Embedding-8B | 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af |
| xlmr-base | xlmr-base | base | FacebookAI/xlm-roberta-base | e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 |
| xlmr-base-random | xlmr-base | random | FacebookAI/xlm-roberta-base | e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 (random init) |
| xlmr-large | xlmr-large | base | FacebookAI/xlm-roberta-large | c23d21b0620b635a76227c604d44e43a9f0ee389 |

## Best layer per model, dataset, condition, and split kind (validation)

Errors are block-mean km (lower is better). Probe and permuted-labels errors are at the best layer; country centroid is the control.

| model | dataset | condition | split | pooling | layer | rel. depth | gain over country (km) [CI] | probe km | permuted-labels km | country-centroid km |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b-random | cities | name | random-item | mean | 0 | 0.00 | -4,938 [-5,421, -4,458] | 6,135 | 7,293 | 1,197 |
| qwen3-0.6b-base | cities | name | random-item | mean | 18 | 0.64 | -3,186 [-3,662, -2,739] | 4,383 | 7,299 | 1,197 |
| qwen3-0.6b-embedding | cities | name | random-item | last | 23 | 0.82 | -3,200 [-3,692, -2,752] | 4,397 | 7,302 | 1,197 |
| qwen3-4b-base | cities | name | random-item | mean | 24 | 0.67 | -2,115 [-2,517, -1,746] | 3,312 | 7,296 | 1,197 |
| qwen3-4b-embedding | cities | name | random-item | last | 31 | 0.86 | -2,028 [-2,412, -1,676] | 3,225 | 7,308 | 1,197 |
| qwen3-8b-base | cities | name | random-item | mean | 32 | 0.89 | -1,860 [-2,238, -1,496] | 3,057 | 7,324 | 1,197 |
| qwen3-8b-embedding | cities | name | random-item | last | 32 | 0.89 | -1,517 [-1,912, -1,128] | 2,714 | 7,393 | 1,197 |
| xlmr-base-random | cities | name | random-item | mean | 6 | 0.50 | -5,338 [-5,830, -4,841] | 6,535 | 7,294 | 1,197 |
| xlmr-base | cities | name | random-item | mean | 0 | 0.00 | -4,132 [-4,658, -3,629] | 5,330 | 7,292 | 1,197 |
| e5-base | cities | name | random-item | mean | 11 | 0.92 | -3,396 [-3,904, -2,904] | 4,593 | 7,291 | 1,197 |
| xlmr-large | cities | name | random-item | mean | 6 | 0.25 | -3,901 [-4,402, -3,413] | 5,099 | 7,288 | 1,197 |
| bge-m3-retromae | cities | name | random-item | mean | 6 | 0.25 | -3,831 [-4,349, -3,338] | 5,028 | 7,292 | 1,197 |
| bge-m3-unsupervised | cities | name | random-item | mean | 24 | 1.00 | -3,665 [-4,199, -3,182] | 4,862 | 7,289 | 1,197 |
| bge-m3 | cities | name | random-item | mean | 12 | 0.50 | -3,693 [-4,211, -3,209] | 4,890 | 7,294 | 1,197 |
| qwen3-0.6b-random | cities | name | spatial-block | mean | 0 | 0.00 | -4,083 [-5,127, -3,120] | 7,001 | 7,860 | 2,918 |
| qwen3-0.6b-base | cities | name | spatial-block | mean | 21 | 0.75 | -2,405 [-3,394, -1,483] | 5,323 | 7,862 | 2,918 |
| qwen3-0.6b-embedding | cities | name | spatial-block | last | 23 | 0.82 | -2,464 [-3,432, -1,560] | 5,382 | 7,877 | 2,918 |
| qwen3-4b-base | cities | name | spatial-block | mean | 24 | 0.67 | -1,356 [-2,241, -491] | 4,274 | 7,829 | 2,918 |
| qwen3-4b-embedding | cities | name | spatial-block | last | 33 | 0.92 | -1,265 [-2,142, -372] | 4,183 | 7,906 | 2,918 |
| qwen3-8b-base | cities | name | spatial-block | mean | 30 | 0.83 | -1,070 [-1,962, -168] | 3,987 | 7,882 | 2,918 |
| qwen3-8b-embedding | cities | name | spatial-block | last | 30 | 0.83 | -724 [-1,691, 246] | 3,642 | 7,929 | 2,918 |
| xlmr-base-random | cities | name | spatial-block | mean | 7 | 0.58 | -4,385 [-5,416, -3,418] | 7,303 | 7,861 | 2,918 |
| xlmr-base | cities | name | spatial-block | mean | 0 | 0.00 | -3,508 [-4,537, -2,587] | 6,425 | 7,850 | 2,918 |
| e5-base | cities | name | spatial-block | mean | 11 | 0.92 | -2,895 [-3,916, -1,938] | 5,812 | 7,868 | 2,918 |
| xlmr-large | cities | name | spatial-block | mean | 4 | 0.17 | -3,258 [-4,266, -2,338] | 6,176 | 7,866 | 2,918 |
| bge-m3-retromae | cities | name | spatial-block | mean | 6 | 0.25 | -3,219 [-4,251, -2,273] | 6,136 | 7,869 | 2,918 |
| bge-m3-unsupervised | cities | name | spatial-block | mean | 24 | 1.00 | -3,028 [-4,026, -2,117] | 5,946 | 7,839 | 2,918 |
| bge-m3 | cities | name | spatial-block | mean | 24 | 1.00 | -3,082 [-4,092, -2,143] | 6,000 | 7,853 | 2,918 |
| qwen3-0.6b-random | cities | name-country | random-item | mean | 0 | 0.00 | -1,441 [-1,790, -1,158] | 2,638 | 7,291 | 1,197 |
| qwen3-0.6b-base | cities | name-country | random-item | last | 8 | 0.29 | -533 [-755, -319] | 1,730 | 7,293 | 1,197 |
| qwen3-0.6b-embedding | cities | name-country | random-item | last | 21 | 0.75 | -162 [-363, 57] | 1,359 | 7,345 | 1,197 |
| qwen3-4b-base | cities | name-country | random-item | last | 13 | 0.36 | -113 [-294, 92] | 1,310 | 7,294 | 1,197 |
| qwen3-4b-embedding | cities | name-country | random-item | last | 32 | 0.89 | 231 [30, 472] | 966 | 7,345 | 1,197 |
| qwen3-8b-base | cities | name-country | random-item | last | 15 | 0.42 | -22 [-238, 221] | 1,219 | 7,298 | 1,197 |
| qwen3-8b-embedding | cities | name-country | random-item | last | 32 | 0.89 | 311 [89, 565] | 886 | 7,397 | 1,197 |
| xlmr-base-random | cities | name-country | random-item | mean | 1 | 0.08 | -1,826 [-2,177, -1,522] | 3,024 | 7,293 | 1,197 |
| xlmr-base | cities | name-country | random-item | mean | 1 | 0.08 | -1,387 [-1,734, -1,106] | 2,584 | 7,293 | 1,197 |
| e5-base | cities | name-country | random-item | mean | 11 | 0.92 | -741 [-1,023, -488] | 1,938 | 7,294 | 1,197 |
| xlmr-large | cities | name-country | random-item | mean | 1 | 0.04 | -1,187 [-1,525, -912] | 2,384 | 7,294 | 1,197 |
| bge-m3-retromae | cities | name-country | random-item | mean | 24 | 1.00 | -993 [-1,299, -750] | 2,190 | 7,289 | 1,197 |
| bge-m3-unsupervised | cities | name-country | random-item | mean | 23 | 0.96 | -547 [-800, -311] | 1,745 | 7,293 | 1,197 |
| bge-m3 | cities | name-country | random-item | mean | 22 | 0.92 | -682 [-951, -447] | 1,879 | 7,294 | 1,197 |
| qwen3-0.6b-random | cities | name-country | spatial-block | mean | 0 | 0.00 | -1,206 [-1,756, -670] | 4,124 | 7,868 | 2,918 |
| qwen3-0.6b-base | cities | name-country | spatial-block | mean | 15 | 0.54 | 128 [-470, 803] | 2,790 | 7,861 | 2,918 |
| qwen3-0.6b-embedding | cities | name-country | spatial-block | last | 22 | 0.79 | 943 [263, 1,735] | 1,975 | 7,864 | 2,918 |
| qwen3-4b-base | cities | name-country | spatial-block | mean | 20 | 0.56 | 841 [75, 1,724] | 2,077 | 7,868 | 2,918 |
| qwen3-4b-embedding | cities | name-country | spatial-block | last | 32 | 0.89 | 1,487 [632, 2,487] | 1,430 | 7,887 | 2,918 |
| qwen3-8b-base | cities | name-country | spatial-block | last | 15 | 0.42 | 1,086 [371, 1,900] | 1,831 | 7,875 | 2,918 |
| qwen3-8b-embedding | cities | name-country | spatial-block | last | 31 | 0.86 | 1,626 [743, 2,684] | 1,292 | 7,943 | 2,918 |
| xlmr-base-random | cities | name-country | spatial-block | mean | 0 | 0.00 | -1,524 [-2,070, -1,026] | 4,442 | 7,860 | 2,918 |
| xlmr-base | cities | name-country | spatial-block | mean | 0 | 0.00 | -957 [-1,487, -464] | 3,875 | 7,865 | 2,918 |
| e5-base | cities | name-country | spatial-block | mean | 12 | 1.00 | -124 [-624, 440] | 3,042 | 7,848 | 2,918 |
| xlmr-large | cities | name-country | spatial-block | mean | 1 | 0.04 | -744 [-1,264, -261] | 3,662 | 7,867 | 2,918 |
| bge-m3-retromae | cities | name-country | spatial-block | mean | 24 | 1.00 | -696 [-1,310, -117] | 3,614 | 7,869 | 2,918 |
| bge-m3-unsupervised | cities | name-country | spatial-block | mean | 24 | 1.00 | 5 [-467, 587] | 2,913 | 7,860 | 2,918 |
| bge-m3 | cities | name-country | spatial-block | mean | 22 | 0.92 | -194 [-732, 387] | 3,112 | 7,847 | 2,918 |
| qwen3-0.6b-random | stations | metadata | random-item | mean | 0 | 0.00 | -2,262 [-2,728, -1,828] | 3,924 | 7,810 | 1,662 |
| qwen3-0.6b-base | stations | metadata | random-item | mean | 21 | 0.75 | -1,629 [-2,030, -1,254] | 3,291 | 7,821 | 1,662 |
| qwen3-0.6b-embedding | stations | metadata | random-item | last | 23 | 0.82 | -1,283 [-1,629, -944] | 2,945 | 7,818 | 1,662 |
| qwen3-4b-base | stations | metadata | random-item | mean | 24 | 0.67 | -757 [-1,066, -442] | 2,420 | 7,805 | 1,662 |
| qwen3-4b-embedding | stations | metadata | random-item | last | 33 | 0.92 | -227 [-486, 77] | 1,889 | 7,860 | 1,662 |
| qwen3-8b-base | stations | metadata | random-item | mean | 14 | 0.39 | -650 [-961, -331] | 2,312 | 7,814 | 1,662 |
| qwen3-8b-embedding | stations | metadata | random-item | last | 32 | 0.89 | -217 [-489, 82] | 1,880 | 7,877 | 1,662 |
| xlmr-base-random | stations | metadata | random-item | mean | 0 | 0.00 | -2,403 [-2,845, -1,945] | 4,065 | 7,810 | 1,662 |
| xlmr-base | stations | metadata | random-item | mean | 0 | 0.00 | -2,113 [-2,558, -1,678] | 3,776 | 7,767 | 1,662 |
| e5-base | stations | metadata | random-item | mean | 12 | 1.00 | -1,607 [-1,996, -1,246] | 3,269 | 7,816 | 1,662 |
| xlmr-large | stations | metadata | random-item | mean | 0 | 0.00 | -1,778 [-2,193, -1,384] | 3,440 | 7,798 | 1,662 |
| bge-m3-retromae | stations | metadata | random-item | mean | 0 | 0.00 | -1,768 [-2,196, -1,365] | 3,431 | 7,794 | 1,662 |
| bge-m3-unsupervised | stations | metadata | random-item | mean | 0 | 0.00 | -1,766 [-2,199, -1,359] | 3,428 | 7,795 | 1,662 |
| bge-m3 | stations | metadata | random-item | mean | 23 | 0.96 | -1,751 [-2,143, -1,369] | 3,414 | 7,806 | 1,662 |
| qwen3-0.6b-random | stations | metadata | spatial-block | mean | 0 | 0.00 | -1,002 [-1,758, -239] | 4,962 | 8,171 | 3,960 |
| qwen3-0.6b-base | stations | metadata | spatial-block | mean | 21 | 0.75 | -138 [-855, 605] | 4,097 | 8,126 | 3,960 |
| qwen3-0.6b-embedding | stations | metadata | spatial-block | mean | 24 | 0.86 | 156 [-602, 983] | 3,804 | 8,137 | 3,960 |
| qwen3-4b-base | stations | metadata | spatial-block | mean | 33 | 0.92 | 529 [-133, 1,255] | 3,431 | 8,131 | 3,960 |
| qwen3-4b-embedding | stations | metadata | spatial-block | last | 32 | 0.89 | 1,120 [314, 1,999] | 2,840 | 8,201 | 3,960 |
| qwen3-8b-base | stations | metadata | spatial-block | mean | 35 | 0.97 | 637 [-110, 1,425] | 3,323 | 8,163 | 3,960 |
| qwen3-8b-embedding | stations | metadata | spatial-block | last | 33 | 0.92 | 985 [259, 1,776] | 2,974 | 8,206 | 3,960 |
| xlmr-base-random | stations | metadata | spatial-block | mean | 0 | 0.00 | -1,053 [-1,752, -329] | 5,013 | 7,995 | 3,960 |
| xlmr-base | stations | metadata | spatial-block | mean | 1 | 0.08 | -743 [-1,584, 57] | 4,703 | 8,164 | 3,960 |
| e5-base | stations | metadata | spatial-block | mean | 11 | 0.92 | -203 [-903, 517] | 4,163 | 8,173 | 3,960 |
| xlmr-large | stations | metadata | spatial-block | mean | 0 | 0.00 | -551 [-1,285, 161] | 4,511 | 8,156 | 3,960 |
| bge-m3-retromae | stations | metadata | spatial-block | mean | 23 | 0.96 | -427 [-1,156, 350] | 4,387 | 8,167 | 3,960 |
| bge-m3-unsupervised | stations | metadata | spatial-block | mean | 24 | 1.00 | -352 [-1,058, 384] | 4,311 | 8,132 | 3,960 |
| bge-m3 | stations | metadata | spatial-block | mean | 24 | 1.00 | -317 [-979, 384] | 4,277 | 8,159 | 3,960 |

## Probe credibility: country-residual gain and lat/lon R-squared (validation, dev)

All values are validation (dev) results, seed means, with the best layer chosen on the same validation items (optimistic). Country-residual: the probe predicts the offset from the training-only country centroid; gain is over the country centroid (km), best layer by gain. Lat/lon R-squared (ridge onto latitude and longitude in degrees, comparable to Gurnee & Tegmark) is at the layer with the highest mean of the two R-squared values. Latlon R-squared is computed from the raw ridge outputs in degrees (before clipping or wrapping); country-centroid R-squared is computed from the centroid's unit vectors converted to latitude and longitude. Longitude R-squared uses raw degrees and ignores wraparound. Country-centroid R-squared is the reference: what knowing the country alone explains. Alpha often selects the grid's lower edge (0.01) for latlon and sphere probes; with many more items than features that is effectively ordinary least squares. † Country-residual results for the `name` condition are given the true country label (not present in the text). On stations spatial-block splits the country centroid is unreliable: when a country's main block is held out, its training centroid can come from a few distant stations, so gains there partly reflect fixing broken centroids (see the coverage table below).

| model | dataset | condition | split | residual layer | residual gain (km) [CI] | latlon layer | latlon R2 lat | latlon R2 lon | country-centroid R2 lat | country-centroid R2 lon |
|---|---|---|---|---|---|---|---|---|---|---|
| bge-m3 | cities | name | random-item | 24 | 0 [-0, 1] † | 15 | 0.384 | 0.584 | 0.953 | 0.981 |
| bge-m3-retromae | cities | name | random-item | 1 | 0 [-0, 0] † | 21 | 0.356 | 0.539 | 0.953 | 0.981 |
| bge-m3-unsupervised | cities | name | random-item | 1 | 0 [-0, 0] † | 22 | 0.396 | 0.586 | 0.953 | 0.981 |
| e5-base | cities | name | random-item | 1 | 0 [-3, 4] † | 11 | 0.440 | 0.617 | 0.953 | 0.981 |
| qwen3-0.6b-base | cities | name | random-item | 22 | 2 [-7, 12] † | 18 | 0.452 | 0.641 | 0.953 | 0.981 |
| qwen3-0.6b-embedding | cities | name | random-item | 24 | 7 [-7, 21] † | 23 | 0.451 | 0.637 | 0.953 | 0.981 |
| qwen3-0.6b-random | cities | name | random-item | 1 | 0 [-0, 1] † | 0 | 0.126 | 0.287 | 0.953 | 0.981 |
| qwen3-4b-base | cities | name | random-item | 31 | 51 [13, 92] † | 31 | 0.625 | 0.731 | 0.953 | 0.981 |
| qwen3-4b-embedding | cities | name | random-item | 33 | 117 [65, 178] † | 31 | 0.639 | 0.742 | 0.953 | 0.981 |
| qwen3-8b-base | cities | name | random-item | 32 | 88 [39, 145] † | 32 | 0.671 | 0.765 | 0.953 | 0.981 |
| qwen3-8b-embedding | cities | name | random-item | 33 | 181 [116, 256] † | 32 | 0.691 | 0.780 | 0.953 | 0.981 |
| xlmr-base | cities | name | random-item | 12 | 0 [-1, 2] † | 1 | 0.316 | 0.469 | 0.953 | 0.981 |
| xlmr-base-random | cities | name | random-item | 0 | 1 [-1, 3] † | 0 | 0.084 | 0.192 | 0.953 | 0.981 |
| xlmr-large | cities | name | random-item | 6 | 0 [-1, 1] † | 8 | 0.345 | 0.525 | 0.953 | 0.981 |
| bge-m3 | cities | name | spatial-block | 16 | 8 [-13, 33] † | 17 | 0.294 | 0.546 | 0.864 | 0.914 |
| bge-m3-retromae | cities | name | spatial-block | 19 | 9 [-6, 25] † | 13 | 0.274 | 0.506 | 0.864 | 0.914 |
| bge-m3-unsupervised | cities | name | spatial-block | 16 | 8 [-14, 34] † | 22 | 0.307 | 0.547 | 0.864 | 0.914 |
| e5-base | cities | name | spatial-block | 6 | 12 [-9, 37] † | 11 | 0.347 | 0.587 | 0.864 | 0.914 |
| qwen3-0.6b-base | cities | name | spatial-block | 13 | 20 [-10, 53] † | 19 | 0.374 | 0.608 | 0.864 | 0.914 |
| qwen3-0.6b-embedding | cities | name | spatial-block | 23 | 35 [1, 77] † | 21 | 0.367 | 0.596 | 0.864 | 0.914 |
| qwen3-0.6b-random | cities | name | spatial-block | 2 | 6 [-7, 20] † | 0 | 0.044 | 0.237 | 0.864 | 0.914 |
| qwen3-4b-base | cities | name | spatial-block | 13 | 109 [30, 205] † | 31 | 0.548 | 0.695 | 0.864 | 0.914 |
| qwen3-4b-embedding | cities | name | spatial-block | 33 | 161 [64, 281] † | 27 | 0.538 | 0.699 | 0.864 | 0.914 |
| qwen3-8b-base | cities | name | spatial-block | 14 | 145 [65, 242] † | 33 | 0.588 | 0.721 | 0.864 | 0.914 |
| qwen3-8b-embedding | cities | name | spatial-block | 35 | 201 [89, 343] † | 27 | 0.610 | 0.731 | 0.864 | 0.914 |
| xlmr-base | cities | name | spatial-block | 1 | 11 [-6, 32] † | 1 | 0.227 | 0.407 | 0.864 | 0.914 |
| xlmr-base-random | cities | name | spatial-block | 0 | 2 [-3, 7] † | 2 | 0.034 | 0.118 | 0.864 | 0.914 |
| xlmr-large | cities | name | spatial-block | 23 | 8 [-10, 28] † | 8 | 0.253 | 0.477 | 0.864 | 0.914 |
| bge-m3 | cities | name-country | random-item | 16 | 11 [-8, 32] | 23 | 0.892 | 0.931 | 0.953 | 0.981 |
| bge-m3-retromae | cities | name-country | random-item | 16 | 2 [-7, 11] | 24 | 0.843 | 0.906 | 0.953 | 0.981 |
| bge-m3-unsupervised | cities | name-country | random-item | 16 | 14 [-5, 36] | 23 | 0.908 | 0.946 | 0.953 | 0.981 |
| e5-base | cities | name-country | random-item | 8 | 3 [-13, 21] | 12 | 0.887 | 0.934 | 0.953 | 0.981 |
| qwen3-0.6b-base | cities | name-country | random-item | 24 | 12 [-6, 31] | 8 | 0.906 | 0.943 | 0.953 | 0.981 |
| qwen3-0.6b-embedding | cities | name-country | random-item | 24 | 24 [1, 48] | 23 | 0.939 | 0.961 | 0.953 | 0.981 |
| qwen3-0.6b-random | cities | name-country | random-item | 1 | 1 [-1, 2] | 0 | 0.782 | 0.870 | 0.953 | 0.981 |
| qwen3-4b-base | cities | name-country | random-item | 34 | 120 [68, 178] | 4 | 0.938 | 0.965 | 0.953 | 0.981 |
| qwen3-4b-embedding | cities | name-country | random-item | 32 | 152 [90, 223] | 32 | 0.957 | 0.972 | 0.953 | 0.981 |
| qwen3-8b-base | cities | name-country | random-item | 33 | 175 [110, 254] | 13 | 0.944 | 0.967 | 0.953 | 0.981 |
| qwen3-8b-embedding | cities | name-country | random-item | 32 | 201 [128, 285] | 32 | 0.959 | 0.975 | 0.953 | 0.981 |
| xlmr-base | cities | name-country | random-item | 12 | 1 [-2, 4] | 1 | 0.802 | 0.873 | 0.953 | 0.981 |
| xlmr-base-random | cities | name-country | random-item | 0 | 1 [-2, 4] | 3 | 0.729 | 0.833 | 0.953 | 0.981 |
| xlmr-large | cities | name-country | random-item | 3 | 1 [-3, 5] | 1 | 0.835 | 0.896 | 0.953 | 0.981 |
| bge-m3 | cities | name-country | spatial-block | 18 | 21 [-9, 54] | 24 | 0.828 | 0.901 | 0.864 | 0.914 |
| bge-m3-retromae | cities | name-country | spatial-block | 6 | 6 [-1, 13] | 19 | 0.767 | 0.870 | 0.864 | 0.914 |
| bge-m3-unsupervised | cities | name-country | spatial-block | 19 | 26 [-11, 67] | 24 | 0.838 | 0.912 | 0.864 | 0.914 |
| e5-base | cities | name-country | spatial-block | 8 | 31 [-2, 72] | 12 | 0.838 | 0.910 | 0.864 | 0.914 |
| qwen3-0.6b-base | cities | name-country | spatial-block | 28 | 48 [8, 94] | 8 | 0.866 | 0.912 | 0.864 | 0.914 |
| qwen3-0.6b-embedding | cities | name-country | spatial-block | 22 | 48 [2, 100] | 23 | 0.906 | 0.939 | 0.864 | 0.914 |
| qwen3-0.6b-random | cities | name-country | spatial-block | 4 | 4 [-4, 12] | 0 | 0.666 | 0.808 | 0.864 | 0.914 |
| qwen3-4b-base | cities | name-country | spatial-block | 35 | 158 [73, 259] | 13 | 0.902 | 0.939 | 0.864 | 0.914 |
| qwen3-4b-embedding | cities | name-country | spatial-block | 35 | 192 [83, 321] | 29 | 0.919 | 0.949 | 0.864 | 0.914 |
| qwen3-8b-base | cities | name-country | spatial-block | 32 | 206 [97, 333] | 13 | 0.905 | 0.942 | 0.864 | 0.914 |
| qwen3-8b-embedding | cities | name-country | spatial-block | 33 | 226 [102, 380] | 26 | 0.924 | 0.949 | 0.864 | 0.914 |
| xlmr-base | cities | name-country | spatial-block | 8 | 19 [-4, 43] | 1 | 0.739 | 0.833 | 0.864 | 0.914 |
| xlmr-base-random | cities | name-country | spatial-block | 0 | 1 [-3, 5] | 1 | 0.633 | 0.775 | 0.864 | 0.914 |
| xlmr-large | cities | name-country | spatial-block | 15 | 8 [-19, 37] | 1 | 0.778 | 0.866 | 0.864 | 0.914 |
| bge-m3 | stations | metadata | random-item | 0 | 27 [-32, 103] | 24 | 0.686 | 0.802 | 0.923 | 0.956 |
| bge-m3-retromae | stations | metadata | random-item | 0 | 27 [-31, 102] | 0 | 0.682 | 0.760 | 0.923 | 0.956 |
| bge-m3-unsupervised | stations | metadata | random-item | 0 | 27 [-31, 103] | 24 | 0.669 | 0.790 | 0.923 | 0.956 |
| e5-base | stations | metadata | random-item | 0 | 13 [-35, 70] | 12 | 0.706 | 0.798 | 0.923 | 0.956 |
| qwen3-0.6b-base | stations | metadata | random-item | 28 | 32 [-11, 79] | 28 | 0.673 | 0.761 | 0.923 | 0.956 |
| qwen3-0.6b-embedding | stations | metadata | random-item | 0 | 25 [-20, 81] | 21 | 0.741 | 0.805 | 0.923 | 0.956 |
| qwen3-0.6b-random | stations | metadata | random-item | 0 | 12 [-17, 53] | 0 | 0.675 | 0.751 | 0.923 | 0.956 |
| qwen3-4b-base | stations | metadata | random-item | 33 | 108 [19, 203] | 33 | 0.823 | 0.851 | 0.923 | 0.956 |
| qwen3-4b-embedding | stations | metadata | random-item | 34 | 120 [29, 217] | 32 | 0.893 | 0.901 | 0.923 | 0.956 |
| qwen3-8b-base | stations | metadata | random-item | 33 | 170 [83, 264] | 35 | 0.840 | 0.869 | 0.923 | 0.956 |
| qwen3-8b-embedding | stations | metadata | random-item | 33 | 179 [82, 287] | 32 | 0.894 | 0.909 | 0.923 | 0.956 |
| xlmr-base | stations | metadata | random-item | 1 | 12 [-27, 60] | 0 | 0.639 | 0.730 | 0.923 | 0.956 |
| xlmr-base-random | stations | metadata | random-item | 0 | 6 [-17, 34] | 0 | 0.622 | 0.710 | 0.923 | 0.956 |
| xlmr-large | stations | metadata | random-item | 0 | 33 [-27, 111] | 0 | 0.680 | 0.755 | 0.923 | 0.956 |
| bge-m3 | stations | metadata | spatial-block | 10 | 109 [-8, 262] | 24 | 0.548 | 0.657 | 0.052 | 0.727 |
| bge-m3-retromae | stations | metadata | spatial-block | 3 | 121 [-3, 286] | 14 | 0.561 | 0.599 | 0.052 | 0.727 |
| bge-m3-unsupervised | stations | metadata | spatial-block | 7 | 109 [-15, 276] | 24 | 0.551 | 0.652 | 0.052 | 0.727 |
| e5-base | stations | metadata | spatial-block | 3 | 137 [18, 290] | 12 | 0.487 | 0.642 | 0.052 | 0.727 |
| qwen3-0.6b-base | stations | metadata | spatial-block | 28 | 135 [18, 277] | 19 | 0.579 | 0.613 | 0.052 | 0.727 |
| qwen3-0.6b-embedding | stations | metadata | spatial-block | 24 | 137 [1, 312] | 12 | 0.537 | 0.626 | 0.052 | 0.727 |
| qwen3-0.6b-random | stations | metadata | spatial-block | 3 | 44 [-13, 102] | 0 | 0.441 | 0.580 | 0.052 | 0.727 |
| qwen3-4b-base | stations | metadata | spatial-block | 34 | 290 [84, 561] | 18 | 0.710 | 0.675 | 0.052 | 0.727 |
| qwen3-4b-embedding | stations | metadata | spatial-block | 34 | 279 [67, 545] | 28 | 0.729 | 0.772 | 0.052 | 0.727 |
| qwen3-8b-base | stations | metadata | spatial-block | 33 | 301 [97, 579] | 24 | 0.725 | 0.750 | 0.052 | 0.727 |
| qwen3-8b-embedding | stations | metadata | spatial-block | 34 | 330 [100, 622] | 24 | 0.763 | 0.763 | 0.052 | 0.727 |
| xlmr-base | stations | metadata | spatial-block | 0 | 81 [-13, 199] | 0 | 0.484 | 0.595 | 0.052 | 0.727 |
| xlmr-base-random | stations | metadata | spatial-block | 0 | 51 [-29, 152] | 0 | 0.365 | 0.535 | 0.052 | 0.727 |
| xlmr-large | stations | metadata | spatial-block | 4 | 115 [-15, 288] | 6 | 0.566 | 0.631 | 0.052 | 0.727 |

## Country training coverage (validation, dev)

Share of evaluated (validation) items whose country has fewer than 10 training items (or no country code), averaged over seeds. These items get a thin or fallback country-centroid.

| dataset | split | share of evaluated items |
|---|---|---|
| cities | spatial-block | 0.143 |
| cities | random-item | 0.014 |
| stations | spatial-block | 0.346 |
| stations | random-item | 0.055 |
