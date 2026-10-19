from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_path: Path = Path("data/raw/radio-browser-geolocated.json")
    database_path: Path = Path("data/processed/stations.duckdb")


class CatalogStats(BaseModel):
    station_count: int = Field(ge=0)
    country_count: int = Field(ge=0)
    rock_count: int = Field(ge=0)
    source_sha256: str
