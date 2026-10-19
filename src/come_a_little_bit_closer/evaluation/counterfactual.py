from itertools import combinations

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.evaluation.location_schemas import (
    IndexVector,
    LocationPlan,
    QueryManifest,
    QueryMatrix,
    RankedIndices,
    ResponsivenessScore,
    RetrievalScore,
)
from come_a_little_bit_closer.geometry.distances import haversine_distances
from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix, ScoreVector


@jaxtyped(typechecker=beartype)
def rank_candidates(
    scores: ScoreVector, candidates: IndexVector, station_ids: list[str], limit: int
) -> RankedIndices:
    candidate_ids = np.asarray(station_ids)[candidates]
    return candidates[np.lexsort((candidate_ids, -scores[candidates]))[:limit]]


@jaxtyped(typechecker=beartype)
def neighbor_turnover(first: IndexVector, second: IndexVector) -> float:
    if not len(first) or len(first) != len(second):
        raise ValueError("Turnover requires equal, nonempty neighbor sets.")
    return 1.0 - len(set(first.tolist()).intersection(second.tolist())) / len(first)


@jaxtyped(typechecker=beartype)
def movement_toward_origin(
    first_neighbors: IndexVector,
    second_neighbors: IndexVector,
    first_distances: ScoreVector,
    second_distances: ScoreVector,
) -> float:
    return float(
        (
            second_distances[first_neighbors].mean()
            - second_distances[second_neighbors].mean()
            + first_distances[second_neighbors].mean()
            - first_distances[first_neighbors].mean()
        )
        / 2
    )


@jaxtyped(typechecker=beartype)
def evaluate_counterfactuals(
    stations: list[Station],
    documents: EmbeddingMatrix,
    queries: QueryMatrix,
    manifest: QueryManifest,
    plan: LocationPlan,
    model_id: str,
    condition: str,
) -> tuple[list[RetrievalScore], list[ResponsivenessScore]]:
    station_ids = [station.station_id for station in stations]
    coordinates = np.asarray(
        [
            [station.coordinates.latitude, station.coordinates.longitude]
            for station in stations
        ],
        dtype=np.float64,
    )
    all_candidates = np.arange(len(stations), dtype=np.int64)
    distances = [
        haversine_distances(
            np.asarray(
                [origin.coordinates.latitude, origin.coordinates.longitude],
                dtype=np.float64,
            ),
            coordinates,
            plan.geometry,
        )
        for origin in plan.origins
    ]
    query_positions = {
        query.query_id: index for index, query in enumerate(manifest.queries)
    }
    scores = queries @ documents.T
    retrieval_rows: list[RetrievalScore] = []
    response_rows: list[ResponsivenessScore] = []
    for genre in plan.genres:
        matching_tags = plan.genre_tags.get(genre, [genre])
        candidates = np.asarray(
            [
                index
                for index, station in enumerate(stations)
                if any(tag in matching_tags for tag in station.genre_tags)
            ],
            dtype=np.int64,
        )
        if len(candidates) < max(plan.neighbors, plan.minimum_genre_candidates):
            raise ValueError(f"Frozen genre {genre!r} has fewer than k candidates.")
        for query_condition in ["origin", "no-origin"]:
            neighbors: list[IndexVector] = []
            query_indices: list[int] = []
            for origin_index, origin in enumerate(plan.origins):
                suffix = str(origin_index) if query_condition == "origin" else "none"
                query_index = query_positions[f"{genre}:{suffix}"]
                query_indices.append(query_index)
                ranked = rank_candidates(
                    scores[query_index], candidates, station_ids, plan.neighbors
                )
                neighbors.append(ranked)
                reference = rank_candidates(
                    -distances[origin_index], candidates, station_ids, plan.neighbors
                )
                unfiltered = rank_candidates(
                    scores[query_index], all_candidates, station_ids, plan.neighbors
                )
                retrieved_mean = float(distances[origin_index][ranked].mean())
                reference_mean = float(distances[origin_index][reference].mean())
                retrieval_rows.append(
                    RetrievalScore(
                        model_id=model_id,
                        condition=condition,
                        query_condition=query_condition,
                        genre=genre,
                        origin_name=origin.name,
                        latitude=origin.coordinates.latitude,
                        longitude=origin.coordinates.longitude,
                        candidates=len(candidates),
                        genre_precision_at_k=sum(
                            any(
                                tag in matching_tags
                                for tag in stations[int(index)].genre_tags
                            )
                            for index in unfiltered
                        )
                        / len(unfiltered),
                        geographic_overlap_at_k=1.0
                        - neighbor_turnover(ranked, reference),
                        retrieved_mean_km=retrieved_mean,
                        reference_mean_km=reference_mean,
                        regret_km=retrieved_mean - reference_mean,
                        retrieved_ids=[station_ids[int(index)] for index in ranked],
                        reference_ids=[station_ids[int(index)] for index in reference],
                    )
                )
            for first_index, second_index in combinations(range(len(plan.origins)), 2):
                query_cosine = float(
                    np.dot(
                        queries[query_indices[first_index]],
                        queries[query_indices[second_index]],
                    )
                )
                response_rows.append(
                    ResponsivenessScore(
                        model_id=model_id,
                        condition=condition,
                        query_condition=query_condition,
                        genre=genre,
                        first_origin=plan.origins[first_index].name,
                        second_origin=plan.origins[second_index].name,
                        query_cosine_distance=float(np.clip(1.0 - query_cosine, 0, 2)),
                        neighbor_turnover=neighbor_turnover(
                            neighbors[first_index], neighbors[second_index]
                        ),
                        movement_toward_origin_km=movement_toward_origin(
                            neighbors[first_index],
                            neighbors[second_index],
                            distances[first_index],
                            distances[second_index],
                        ),
                    )
                )
    return retrieval_rows, response_rows
