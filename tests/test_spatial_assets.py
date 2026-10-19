import hashlib
from pathlib import Path

import h3
import numpy as np
import pytest

from come_a_little_bit_closer.geometry.distances import haversine_distances
from come_a_little_bit_closer.geometry.schemas import GeometrySettings
from tests.schemas import FeatureCollection, PolygonGeometry


def test_h3_round_trip_and_antimeridian_candidates() -> None:
    cell = h3.latlng_to_cell(40.73, -73.94, 7)
    latitude, longitude = h3.cell_to_latlng(cell)
    assert h3.latlng_to_cell(latitude, longitude, 7) == cell
    assert h3.get_resolution(cell) == 7
    assert len(h3.grid_disk(cell, 1)) == 7
    assert h3.latlng_to_cell(0, -179.9999, 7) in h3.grid_disk(
        h3.latlng_to_cell(0, 179.9999, 7), 1
    )
    origin = np.array([40.73, -73.94])
    locations = np.array([[40.731, -73.94], [40.735, -73.94], [-33.87, 151.21]])
    ring = set(h3.grid_disk(cell, 1))
    candidates = [
        index
        for index, position in enumerate(locations)
        if h3.latlng_to_cell(float(position[0]), float(position[1]), 7) in ring
    ]
    distances = haversine_distances(origin, locations, GeometrySettings())
    assert candidates == [0, 1]
    assert min(candidates, key=lambda index: distances[index]) == int(
        np.argmin(distances)
    )


@pytest.mark.parametrize(
    "filename, expected_count",
    [
        ("ne_110m_admin_0_countries.geojson", 177),
        ("ne_50m_admin_0_countries.geojson", 242),
    ],
)
def test_vendored_geojson_geometry_and_checksum(
    filename: str, expected_count: int
) -> None:
    vendor = Path("src/come_a_little_bit_closer/web/static/vendor")
    source = (vendor / filename).read_bytes()
    assert hashlib.sha256(source).hexdigest() in (vendor / "ASSETS.md").read_text()
    collection = FeatureCollection.model_validate_json(source)
    assert len(collection.features) == expected_count
    for feature in collection.features:
        polygons = (
            [feature.geometry.coordinates]
            if isinstance(feature.geometry, PolygonGeometry)
            else feature.geometry.coordinates
        )
        assert polygons
        for polygon in polygons:
            assert polygon
            for ring in polygon:
                assert len(ring) >= 4
                assert ring[0] == ring[-1]
                assert all(
                    abs(position[0]) <= 180 and abs(position[1]) <= 90
                    for position in ring
                )
