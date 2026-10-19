import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from uuid import UUID

from come_a_little_bit_closer.data.genre_matching import normalized_genre
from come_a_little_bit_closer.data.genre_schemas import (
    GenreCoverage,
    GenreVocabulary,
    ReferenceGenre,
)
from come_a_little_bit_closer.evaluation.location_schemas import LocationPlan
from come_a_little_bit_closer.storage.catalog import StationCatalog
from come_a_little_bit_closer.storage.schemas import DatabaseSettings


class GenreParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_id: str | None = None
        self.current_text: list[str] = []
        self.genres: dict[str, ReferenceGenre] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href and href.startswith("/genre/"):
            identifier = href.split("/")[2]
            try:
                UUID(identifier)
            except ValueError:
                return
            self.current_id = identifier
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.current_id is not None:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current_id is not None:
            name = "".join(self.current_text).strip().casefold()
            if name:
                self.genres[self.current_id] = ReferenceGenre(
                    genre_id=self.current_id, name=name
                )
            self.current_id = None
            self.current_text = []


def select_genres(
    source_path: Path,
    base_plan_path: Path,
    output_plan_path: Path,
    minimum_stations: int,
) -> None:
    parser = GenreParser()
    parser.feed(source_path.read_text())
    if len(parser.genres) < 100:
        raise ValueError(
            "Genre snapshot does not contain the expected vocabulary links."
        )
    references: dict[str, list[ReferenceGenre]] = {}
    for genre in parser.genres.values():
        references.setdefault(normalized_genre(genre.name), []).append(genre)
    stations = StationCatalog(DatabaseSettings().database_path).stations()
    matching_tags: dict[str, set[str]] = {}
    for station in stations:
        for tag in station.genre_tags:
            name = normalized_genre(tag)
            if name in references:
                matching_tags.setdefault(name, set()).add(tag)
    coverage: list[GenreCoverage] = []
    for name, entries in sorted(references.items()):
        aliases = sorted(matching_tags.get(name, set()))
        count = sum(
            any(tag in aliases for tag in station.genre_tags) for station in stations
        )
        coverage.append(
            GenreCoverage(
                genre=min(entry.name for entry in entries),
                station_count=count,
                station_tags=aliases,
                musicbrainz_ids=sorted(entry.genre_id for entry in entries),
                selected=count >= minimum_stations,
                status=(
                    "selected"
                    if count >= minimum_stations
                    else "sparse" if count else "no-matching-tag"
                ),
            )
        )
    selected = [row for row in coverage if row.selected]
    if not selected:
        raise ValueError("No genre has sufficient catalog coverage.")
    manifest_path = Path("data/musicbrainz-genres.json")
    vocabulary = GenreVocabulary(
        source_url="https://musicbrainz.org/genres",
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        reference_genres=len(parser.genres),
        normalized_genres=len(references),
        selected_genres=len(selected),
        minimum_stations=minimum_stations,
        normalization=(
            "NFKC/casefold; hyphens/underscores to spaces; & to and; "
            "collapse whitespace; no synonym inference"
        ),
        genres=sorted(
            parser.genres.values(), key=lambda genre: (genre.name, genre.genre_id)
        ),
    )
    manifest_path.write_text(vocabulary.model_dump_json(indent=2) + "\n")
    plan = LocationPlan.model_validate_json(base_plan_path.read_bytes()).model_copy(
        update={
            "experiment_id": "location-v2",
            "genres": [row.genre for row in selected],
            "genre_tags": {row.genre: row.station_tags for row in selected},
            "minimum_genre_candidates": minimum_stations,
            "artifact_directory": Path("data/processed/location-v2"),
            "metadata_directory": Path("data/processed/location-v1/metadata-only"),
            "output_directory": Path("outputs/location-v2"),
            "taxonomy_manifest": manifest_path,
        }
    )
    if (
        output_plan_path.exists()
        and output_plan_path.read_text() != plan.model_dump_json(indent=2) + "\n"
    ):
        raise ValueError("Expanded plan already exists with different settings.")
    output_plan_path.write_text(plan.model_dump_json(indent=2) + "\n")
    with (output_plan_path.parent / "genre-coverage.csv").open(
        "w", newline=""
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "genre",
                "station_count",
                "station_tags",
                "musicbrainz_ids",
                "selected",
                "status",
            ]
        )
        for row in coverage:
            writer.writerow(
                [
                    row.genre,
                    row.station_count,
                    json.dumps(row.station_tags),
                    json.dumps(row.musicbrainz_ids),
                    row.selected,
                    row.status,
                ]
            )
    print(
        f"Selected {len(selected)} of {len(references)} normalized genres "
        f"at >= {minimum_stations} stations",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze genre queries from the official MusicBrainz snapshot."
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/raw/musicbrainz-genres.html")
    )
    parser.add_argument(
        "--base-plan", type=Path, default=Path("experiments/location-v1.json")
    )
    parser.add_argument(
        "--output-plan", type=Path, default=Path("experiments/location-v2.json")
    )
    parser.add_argument("--minimum-stations", type=int, default=50)
    arguments = parser.parse_args()
    select_genres(
        arguments.source,
        arguments.base_plan,
        arguments.output_plan,
        arguments.minimum_stations,
    )


if __name__ == "__main__":
    main()
