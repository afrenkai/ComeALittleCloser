import numpy as np

from come_a_little_bit_closer.data.schemas import StationDataset
from come_a_little_bit_closer.evaluation.schemas import (
    BaselineReport,
    BaselineRequest,
    RankedStation,
)
from come_a_little_bit_closer.geometry.distances import (
    cosine_similarities,
    haversine_distances,
    spherical_vectors,
)


def compare_geographic_rankings(
    dataset: StationDataset, request: BaselineRequest
) -> BaselineReport:
    genre = request.genre.strip().casefold()
    eligible = [station for station in dataset.stations if genre in station.genre_tags]
    if not eligible:
        raise ValueError(f"No stations have the exact genre tag {genre!r}.")
    origin = np.array(
        [request.origin.latitude, request.origin.longitude], dtype=np.float64
    )
    coordinates = np.array(
        [
            [station.coordinates.latitude, station.coordinates.longitude]
            for station in eligible
        ],
        dtype=np.float64,
    )
    distances = haversine_distances(origin, coordinates, request.geometry)
    scores = cosine_similarities(
        spherical_vectors(origin[np.newaxis, :])[0], spherical_vectors(coordinates)
    )
    count = min(request.limit, len(eligible))
    distance_order = sorted(
        range(len(eligible)),
        key=lambda station_index: (
            float(distances[station_index]),
            eligible[station_index].station_id,
        ),
    )[:count]
    cosine_order = sorted(
        range(len(eligible)),
        key=lambda station_index: (
            -float(scores[station_index]),
            eligible[station_index].station_id,
        ),
    )[:count]
    ranked_stations = [
        RankedStation(
            station_id=station.station_id,
            name=station.name,
            country_code=station.country_code,
            distance_km=float(distances[station_index]),
            spherical_cosine=float(scores[station_index]),
        )
        for station_index, station in enumerate(eligible)
    ]
    angular_distances = request.geometry.earth_radius_km * np.arccos(scores)
    return BaselineReport(
        data_summary=dataset.summary,
        origin=request.origin,
        genre=genre,
        eligible_stations=len(eligible),
        evaluated_neighbors=count,
        overlap_at_k=len(set(distance_order).intersection(cosine_order)) / count,
        maximum_distance_discrepancy_km=float(
            np.max(np.abs(angular_distances - distances))
        ),
        haversine_neighbors=[ranked_stations[index] for index in distance_order],
        spherical_cosine_neighbors=[ranked_stations[index] for index in cosine_order],
    )
