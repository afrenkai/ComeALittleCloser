import re
import unicodedata
from pathlib import Path

from come_a_little_bit_closer.data.genre_schemas import GenreSelection, GenreVocabulary
from come_a_little_bit_closer.data.schemas import Station


def normalized_genre(name: str) -> str:
    value = unicodedata.normalize("NFKC", name).casefold().replace("&", " and ")
    return " ".join(re.sub(r"[-_‐‑–]+", " ", value).split())


class GenreMatcher:
    def __init__(self, vocabulary_path: Path, stations: list[Station]) -> None:
        vocabulary = GenreVocabulary.model_validate_json(vocabulary_path.read_bytes())
        self.names: dict[str, str] = {}
        for genre in sorted(vocabulary.genres, key=lambda entry: entry.name):
            self.names.setdefault(normalized_genre(genre.name), genre.name)
        if not self.names:
            raise ValueError("The genre vocabulary is empty.")
        self.station_tags: dict[str, set[str]] = {}
        for station in stations:
            for tag in station.genre_tags:
                self.station_tags.setdefault(normalized_genre(tag), set()).add(tag)
        alternatives = sorted(self.names, key=lambda name: (-len(name), name))
        self.pattern = re.compile(
            r"(?<!\w)(?:"
            + "|".join(re.escape(name) for name in alternatives)
            + r")(?!\w)"
        )

    def resolve(self, query: str, genre_filter: str) -> GenreSelection:
        matches = sorted(set(self.pattern.findall(normalized_genre(query))))
        selected = normalized_genre(genre_filter)
        if selected:
            if matches and selected not in matches:
                raise ValueError(
                    "The station tag conflicts with the query genre. "
                    "Clear the tag or update the query."
                )
            return GenreSelection(
                genre=self.names.get(selected, genre_filter.strip().casefold()),
                source="filter",
                station_tags=sorted(self.station_tags.get(selected, set())),
            )
        if len(matches) > 1:
            raise ValueError(
                "Several genres were detected: "
                + ", ".join(self.names[name] for name in matches)
                + ". Choose one with the station tag filter."
            )
        if not matches:
            return GenreSelection()
        selected = matches[0]
        return GenreSelection(
            genre=self.names[selected],
            source="query",
            station_tags=sorted(self.station_tags.get(selected, set())),
        )
