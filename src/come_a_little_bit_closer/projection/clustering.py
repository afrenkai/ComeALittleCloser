from collections import Counter

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix
from come_a_little_bit_closer.projection.schemas import (
    ClusterLabels,
    ProjectionCluster,
    ProjectionSettings,
)


@jaxtyped(typechecker=beartype)
def cluster_embeddings(
    embeddings: EmbeddingMatrix, settings: ProjectionSettings
) -> ClusterLabels:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if len(embeddings) == 0 or not np.isfinite(embeddings).all() or np.any(norms == 0):
        raise ValueError("Clustering requires finite, nonzero embedding vectors.")
    unit_vectors = embeddings / norms
    cluster_count = min(settings.cluster_count, len(unit_vectors))
    generator = np.random.default_rng(settings.random_seed)
    centers = np.empty((cluster_count, unit_vectors.shape[1]), dtype=np.float64)
    centers[0] = unit_vectors[generator.integers(len(unit_vectors))]
    nearest_distances = np.clip(1 - unit_vectors @ centers[0], 0, 2)
    for center_index in range(1, cluster_count):
        total_distance = float(nearest_distances.sum())
        selected_position = (
            int(
                generator.choice(
                    len(unit_vectors), p=nearest_distances / total_distance
                )
            )
            if total_distance > 1e-12
            else 0
        )
        centers[center_index] = unit_vectors[selected_position]
        nearest_distances = np.minimum(
            nearest_distances,
            np.clip(1 - unit_vectors @ centers[center_index], 0, 2),
        )
    previous_labels: ClusterLabels | None = None
    for iteration in range(1, settings.maximum_iterations + 1):
        labels = np.asarray(np.argmax(unit_vectors @ centers.T, axis=1), dtype=np.int64)
        if iteration > 1 and np.array_equal(labels, previous_labels):
            break
        previous_labels = labels.copy()
        for center_index in range(cluster_count):
            members = unit_vectors[labels == center_index]
            if len(members) == 0:
                continue
            center = members.mean(axis=0)
            center_norm = np.linalg.norm(center)
            if center_norm > 0:
                centers[center_index] = center / center_norm
    return labels


@jaxtyped(typechecker=beartype)
def describe_clusters(
    stations: list[Station], labels: ClusterLabels
) -> list[ProjectionCluster]:
    if len(stations) != len(labels):
        raise ValueError("Station order must match cluster-label order.")
    descriptions: list[ProjectionCluster] = []
    for cluster_id in np.unique(labels):
        members = [
            station
            for position, station in enumerate(stations)
            if labels[position] == cluster_id
        ]
        tag_counts = Counter(tag for station in members for tag in station.genre_tags)
        leading_tags = sorted(tag_counts, key=lambda tag: (-tag_counts[tag], tag))[:2]
        descriptions.append(
            ProjectionCluster(
                cluster_id=int(cluster_id),
                label=" / ".join(leading_tags) or f"Cluster {cluster_id + 1}",
                size=len(members),
            )
        )
    return descriptions
