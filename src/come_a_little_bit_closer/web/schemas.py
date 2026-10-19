from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from come_a_little_bit_closer.data.genre_schemas import GenreSelection
from come_a_little_bit_closer.geometry.schemas import Coordinates, GeometrySettings
from come_a_little_bit_closer.projection.schemas import (
    EmbeddingProjection,
    ProjectionSettings,
)
from come_a_little_bit_closer.storage.schemas import DatabaseSettings


class WebSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLOSER_", env_nested_delimiter="__")
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding_directory: Path = Path("data/processed/embeddings")
    query_device: str = "cpu"
    genre_vocabulary_path: Path = Path("data/musicbrainz-genres.json")
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    projection: ProjectionSettings = Field(default_factory=ProjectionSettings)


class SearchParameters(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    genre: str = Field(default="", max_length=100)
    query: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=80)
    limit: int = Field(default=10, ge=1, le=50)


class Neighbor(BaseModel):
    station_id: str
    name: str
    country_code: str
    latitude: float
    longitude: float
    distance_km: float
    similarity: float | None
    genre_tags: list[str]
    stream_url: str


class GlobeStation(BaseModel):
    station_id: str
    name: str
    country_code: str
    latitude: float
    longitude: float
    stream_url: str


class RetrievalMetrics(BaseModel):
    unfiltered_genre_precision_at_k: float | None = None
    geographic_mean_distance_km: float | None = None
    embedding_mean_distance_km: float | None = None
    excess_distance_km: float | None = None


class SearchResult(BaseModel):
    origin: Coordinates
    query: str
    genre: str
    genre_selection: GenreSelection
    model_id: str
    model_label: str
    model_description: str
    is_learned: bool
    eligible_stations: int
    overlap_at_k: float
    geographic_neighbors: list[Neighbor]
    embedding_neighbors: list[Neighbor]
    unfiltered_embedding_neighbors: list[Neighbor] = Field(default_factory=list)
    metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    model_available: bool
    message: str | None = None
    projection: EmbeddingProjection | None = None
