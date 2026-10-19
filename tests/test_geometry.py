import numpy as np
import pytest

from come_a_little_bit_closer.geometry.distances import (
    cosine_similarities,
    haversine_distances,
    spherical_vectors,
)
from come_a_little_bit_closer.geometry.schemas import GeometrySettings


def test_coincident_and_antipodal_distances() -> None:
    settings = GeometrySettings()
    distances = haversine_distances(
        np.array([0.0, 0.0]), np.array([[0.0, 0.0], [0.0, 180.0]]), settings
    )
    np.testing.assert_allclose(distances, [0, np.pi * settings.earth_radius_km])


def test_antimeridian_and_polar_rankings() -> None:
    origin = np.array([89.0, 179.9])
    locations = np.array([[89.0, -179.9], [88.0, 179.9], [0.0, 0.0]])
    distances = haversine_distances(origin, locations, GeometrySettings())
    similarities = cosine_similarities(
        spherical_vectors(origin[np.newaxis, :])[0], spherical_vectors(locations)
    )
    assert distances[0] < 1
    assert np.argsort(distances).tolist() == np.argsort(-similarities).tolist()


def test_invalid_coordinates_and_zero_embeddings() -> None:
    with pytest.raises(ValueError, match="Latitude"):
        spherical_vectors(np.array([[91.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        spherical_vectors(np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError, match="zero vectors"):
        cosine_similarities(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]))
