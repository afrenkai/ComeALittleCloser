from typing import Literal

from pydantic import BaseModel, Field


class GenreSelection(BaseModel):
    genre: str = ""
    source: Literal["query", "filter", "none"] = "none"
    station_tags: list[str] = Field(default_factory=list)


class ReferenceGenre(BaseModel):
    genre_id: str
    name: str


class GenreCoverage(BaseModel):
    genre: str
    station_count: int
    station_tags: list[str]
    musicbrainz_ids: list[str]
    selected: bool
    status: str


class GenreVocabulary(BaseModel):
    source_url: str
    source_sha256: str
    retrieved_at_utc: str
    reference_genres: int
    normalized_genres: int
    selected_genres: int
    minimum_stations: int
    normalization: str
    genres: list[ReferenceGenre]
