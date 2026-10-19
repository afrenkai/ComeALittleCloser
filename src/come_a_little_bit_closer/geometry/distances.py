import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.geometry.schemas import (
    CoordinateMatrix,
    CoordinateVector,
    EmbeddingMatrix,
    EmbeddingVector,
    GeometrySettings,
    ScoreVector,
    SpatialMatrix,
)


@jaxtyped(typechecker=beartype)
def validate_coordinates(coordinates: CoordinateMatrix) -> None:
    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite.")
    if np.any(np.abs(coordinates[:, 0]) > 90):
        raise ValueError("Latitude must lie between -90 and 90 degrees.")
    if np.any(np.abs(coordinates[:, 1]) > 180):
        raise ValueError("Longitude must lie between -180 and 180 degrees.")


@jaxtyped(typechecker=beartype)
def spherical_vectors(coordinates: CoordinateMatrix) -> SpatialMatrix:
    validate_coordinates(coordinates)
    radians = np.deg2rad(coordinates)
    latitudes = radians[:, 0]
    longitudes = radians[:, 1]
    return np.column_stack(
        (
            np.cos(latitudes) * np.cos(longitudes),
            np.cos(latitudes) * np.sin(longitudes),
            np.sin(latitudes),
        )
    )


@jaxtyped(typechecker=beartype)
def haversine_distances(
    origin: CoordinateVector,
    locations: CoordinateMatrix,
    settings: GeometrySettings,
) -> ScoreVector:
    validate_coordinates(origin[np.newaxis, :])
    validate_coordinates(locations)
    origin_radians = np.deg2rad(origin)
    location_radians = np.deg2rad(locations)
    latitude_offsets = location_radians[:, 0] - origin_radians[0]
    longitude_offsets = location_radians[:, 1] - origin_radians[1]
    haversines = np.sin(latitude_offsets / 2) ** 2 + (
        np.cos(origin_radians[0])
        * np.cos(location_radians[:, 0])
        * np.sin(longitude_offsets / 2) ** 2
    )
    clipped_haversines = np.clip(haversines, 0, 1)
    angles = 2 * np.arctan2(
        np.sqrt(clipped_haversines), np.sqrt(1 - clipped_haversines)
    )
    return settings.earth_radius_km * angles


@jaxtyped(typechecker=beartype)
def cosine_similarities(
    query: EmbeddingVector, candidates: EmbeddingMatrix
) -> ScoreVector:
    if not np.isfinite(query).all() or not np.isfinite(candidates).all():
        raise ValueError("Embeddings must be finite.")
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    if query_norm == 0 or np.any(candidate_norms == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return np.clip((candidates @ query) / (candidate_norms * query_norm), -1, 1)
