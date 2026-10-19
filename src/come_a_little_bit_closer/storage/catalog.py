import argparse
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import duckdb

from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.data.stations import load_stations
from come_a_little_bit_closer.geometry.schemas import Coordinates
from come_a_little_bit_closer.storage.schemas import CatalogStats, DatabaseSettings


def snapshot_sha256(source_path: Path) -> str:
    with source_path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


def initialize_catalog(settings: DatabaseSettings) -> CatalogStats:
    if settings.database_path.exists():
        catalog = StationCatalog(settings.database_path)
        stats = catalog.stats()
        if settings.source_path.exists():
            if snapshot_sha256(settings.source_path) != stats.source_sha256:
                raise ValueError("Source snapshot differs from the catalog provenance.")
        return stats
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=settings.database_path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / "stations.duckdb"
        build_catalog(temporary_path, settings.source_path)
        os.replace(temporary_path, settings.database_path)
    return StationCatalog(settings.database_path).stats()


def build_catalog(database_path: Path, source_path: Path) -> None:
    dataset = load_stations(source_path)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("BEGIN TRANSACTION")
        connection.execute("""CREATE TABLE stations (
                station_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL,
                stream_url VARCHAR NOT NULL, country_code VARCHAR NOT NULL,
                genre_tags VARCHAR[] NOT NULL, latitude DOUBLE NOT NULL,
                longitude DOUBLE NOT NULL
            )""")
        connection.executemany(
            "INSERT INTO stations VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    station.station_id,
                    station.name,
                    station.stream_url,
                    station.country_code,
                    station.genre_tags,
                    station.coordinates.latitude,
                    station.coordinates.longitude,
                )
                for station in dataset.stations
            ],
        )
        connection.execute(
            "CREATE TABLE provenance (source_sha256 VARCHAR, data_summary JSON)"
        )
        connection.execute(
            "INSERT INTO provenance VALUES (?, ?)",
            [snapshot_sha256(source_path), dataset.summary.model_dump_json()],
        )
        connection.execute("COMMIT")


class StationCatalog:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def stats(self) -> CatalogStats:
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            counts = connection.execute(
                """SELECT count(*), count(DISTINCT nullif(country_code, '')),
                       count(*) FILTER (WHERE list_contains(genre_tags, 'rock'))
                   FROM stations"""
            ).fetchone()
            provenance = connection.execute(
                "SELECT source_sha256 FROM provenance"
            ).fetchone()
        if counts is None or provenance is None:
            raise ValueError("Station catalog is missing its source provenance.")
        return CatalogStats(
            station_count=cast(int, counts[0]),
            country_count=cast(int, counts[1]),
            rock_count=cast(int, counts[2]),
            source_sha256=cast(str, provenance[0]),
        )

    def stations(self, genre: str = "") -> list[Station]:
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            rows = connection.execute(
                """SELECT station_id, name, stream_url, country_code, genre_tags,
                          latitude, longitude
                   FROM stations
                   WHERE ? = '' OR list_contains(genre_tags, ?)
                   ORDER BY station_id""",
                [genre.strip().casefold(), genre.strip().casefold()],
            ).fetchall()
        return [
            Station(
                station_id=cast(str, row[0]),
                name=cast(str, row[1]),
                stream_url=cast(str, row[2]),
                country_code=cast(str, row[3]),
                genre_tags=cast(list[str], row[4]),
                coordinates=Coordinates(
                    latitude=cast(float, row[5]), longitude=cast(float, row[6])
                ),
            )
            for row in rows
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen DuckDB station catalog."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--database", type=Path)
    arguments = parser.parse_args()
    defaults = DatabaseSettings()
    settings = DatabaseSettings(
        source_path=arguments.source or defaults.source_path,
        database_path=arguments.database or defaults.database_path,
    )
    print(initialize_catalog(settings).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
