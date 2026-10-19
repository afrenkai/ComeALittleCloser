# Geographic recovery and retrieval: initial findings

The expanded experiment compares five frozen text embedding models on 11,479 stations, 45 MusicBrainz genres, and twelve origins. It separates three questions: linear coordinate recovery, sensitivity to query coordinates, and agreement with exact genre-matched geographic neighbors. The results support origin sensitivity more strongly than precise geographic recovery or retrieval.

On the primary geographic split, metadata-only linear probes have block-mean errors of 3,478–5,488 km, compared with 4,005 km for training-only country centroids. None has a positive lower bound on the paired 95% bootstrap improvement over that country control. Gemma's metadata-only gain is 526 km, with interval −164 to 1,255 km. Gemma improves on a second split, but the advantage is not consistent across all three recorded split intervals. These results do not establish fine geographic information beyond country metadata. True-coordinate and shuffled-label controls confirm that the decoder can recover spatial structure and that the learned features outperform the permutation null.

With coordinate-bearing station documents, origin changes replace about 8–21% of top-ten neighbors across origin pairs. They produce much smaller directional improvements: approximately −11 to 101 km depending on the model. Mean regret relative to genre-matched Haversine retrieval remains 3,668–4,141 km. Thus, changing an embedding or its neighbor set does not by itself demonstrate useful geographic alignment.

| Model | Regret reduction from adding query origin, km | Exploratory 95% interval |
| --- | ---: | ---: |
| Qwen3 | 55 | −4 to 114 |
| BGE-M3 | 32 | −13 to 75 |
| E5 | 192 | 79 to 307 |
| EmbeddingGemma 2 | 385 | 2 to 711 |
| CLAP text | −170 | −319 to −4 |

The table uses coordinate-bearing documents. Positive reduction means the origin-bearing query performs better than the identical query without its origin. Intervals resample origin sites after averaging genres within each site. They are exploratory, conditional on this corpus and these twelve origins, and are not adjusted for multiple comparisons. Metadata-only results, all split sensitivities, ranked identifiers, and complete controls are in the full report and raw files.

The initial interpretation—precise geography is encoded but outweighed by genre—is not established. A narrower supported observation is that coordinate text changes retrieval, while geographic accuracy often changes little. Country metadata remains a strong decoding control. Audio, internal layer probes, human relevance labels, station-family grouping, and buffered geographic splits remain future work.
