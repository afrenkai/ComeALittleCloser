from statistics import fmean

from come_a_little_bit_closer.data.genre_schemas import GenreSelection
from come_a_little_bit_closer.web.schemas import Neighbor, RetrievalMetrics


def comparison_metrics(
    geographic: list[Neighbor],
    embedding: list[Neighbor],
    unfiltered: list[Neighbor],
    selection: GenreSelection,
) -> RetrievalMetrics:
    geographic_mean = (
        fmean(station.distance_km for station in geographic) if geographic else None
    )
    embedding_mean = (
        fmean(station.distance_km for station in embedding) if embedding else None
    )
    precision = (
        sum(
            any(tag in selection.station_tags for tag in station.genre_tags)
            for station in unfiltered
        )
        / len(unfiltered)
        if selection.source != "none" and unfiltered
        else None
    )
    return RetrievalMetrics(
        unfiltered_genre_precision_at_k=precision,
        geographic_mean_distance_km=geographic_mean,
        embedding_mean_distance_km=embedding_mean,
        excess_distance_km=(
            embedding_mean - geographic_mean
            if embedding_mean is not None and geographic_mean is not None
            else None
        ),
    )
