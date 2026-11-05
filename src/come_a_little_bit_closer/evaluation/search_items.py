import hashlib
import json
from collections import Counter

import numpy as np

from come_a_little_bit_closer.data.cities import load_country_names, load_places
from come_a_little_bit_closer.data.schemas import Station, TownSource
from come_a_little_bit_closer.evaluation.fix_schemas import (
    FreshPart,
    SearchQuery,
    SearchSource,
)
from come_a_little_bit_closer.evaluation.layer_extract import town_items
from come_a_little_bit_closer.evaluation.splits import holdout_parts
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.storage.catalog import StationCatalog, initialize_catalog
from come_a_little_bit_closer.storage.schemas import DatabaseSettings


def catalog_stations() -> list[Station]:
    source = DatabaseSettings()
    initialize_catalog(source)
    return StationCatalog(source.database_path).stations()


def search_genres(stations: list[Station], source: SearchSource) -> list[str]:
    """The most frequent genres of the frozen location-v2 genre list."""
    genres = json.loads(source.genre_plan.read_text())["genres"]
    counts = Counter(tag for station in stations for tag in station.genre_tags)
    return sorted(genres, key=lambda genre: (-counts[genre], genre))[: source.genres]


def station_documents(stations: list[Station]) -> list[str]:
    """Station name, the nearest cities15000 place with its country, and genres."""
    source = TownSource()
    places = load_places(source.reference_archives[0], source.country_info)
    countries = load_country_names(source.country_info)
    place_vectors = spherical_vectors(
        np.asarray(
            [
                [place.coordinates.latitude, place.coordinates.longitude]
                for place in places
            ]
        )
    ).astype(np.float32)
    station_vectors = spherical_vectors(
        np.asarray(
            [
                [station.coordinates.latitude, station.coordinates.longitude]
                for station in stations
            ]
        )
    ).astype(np.float32)
    nearest = np.concatenate(
        [
            np.argmax(station_vectors[start : start + 1024] @ place_vectors.T, axis=1)
            for start in range(0, len(stations), 1024)
        ]
    )
    return [
        f"Station: {station.name}\n"
        f"Location: {places[nearest[position]].name}, "
        f"{countries.get(station.country_code, station.country_code)}\n"
        f"Genres: {', '.join(station.genre_tags)}"
        for position, station in enumerate(stations)
    ]


def search_queries(
    stations: list[Station], source: SearchSource, part: FreshPart
) -> list[SearchQuery]:
    """Up to `origins_per_pair` towns of the part per (genre, country) pair whose
    country has at least `minimum_stations` stations of the genre."""
    towns = town_items(TownSource())
    check, test = holdout_parts(
        len(towns.item_ids), source.partition_seed, source.check_fraction
    )
    part_rows = check if part == "check" else test
    town_codes = np.asarray(towns.country_codes)
    generator = np.random.default_rng(source.sample_seed)
    queries = []
    for genre in search_genres(stations, source):
        per_country = Counter(
            station.country_code for station in stations if genre in station.genre_tags
        )
        for code in sorted(
            code
            for code, count in per_country.items()
            if count >= source.minimum_stations
        ):
            candidates = part_rows[town_codes[part_rows] == code]
            chosen = np.sort(
                generator.choice(
                    candidates,
                    min(source.origins_per_pair, len(candidates)),
                    replace=False,
                )
            )
            queries += [
                SearchQuery(
                    query_id=f"{genre}|{towns.item_ids[row]}",
                    genre=genre,
                    town_id=towns.item_ids[row],
                    country_code=code,
                    text=f"{genre} radio stations near {towns.texts[row]}",
                    latitude=towns.coordinates[row].latitude,
                    longitude=towns.coordinates[row].longitude,
                )
                for row in chosen
            ]
    return queries


def text_checksum(texts: list[str]) -> str:
    return hashlib.sha256("\n\x1e".join(texts).encode()).hexdigest()
