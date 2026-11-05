from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from come_a_little_bit_closer.geometry.schemas import Coordinates

type PlaceCondition = Literal["name", "name-country"]


class RadioBrowserRecord(BaseModel):
    stationuuid: str
    name: str
    url_resolved: str
    countrycode: str
    tags: str
    geo_lat: float | None = None
    geo_long: float | None = None


class Station(BaseModel):
    station_id: str
    name: str
    stream_url: str
    country_code: str
    genre_tags: list[str]
    coordinates: Coordinates


class DataSummary(BaseModel):
    source_records: int = Field(ge=0)
    valid_coordinate_records: int = Field(ge=0)
    duplicate_stream_records: int = Field(ge=0)
    empty_stream_records: int = Field(ge=0)
    retained_stations: int = Field(ge=0)
    countries: int = Field(ge=0)
    rock_stations: int = Field(ge=0)


class StationDataset(BaseModel):
    stations: list[Station]
    summary: DataSummary


class Place(BaseModel):
    place_id: str
    name: str
    country_code: str
    country_name: str
    population: int = Field(ge=0)
    coordinates: Coordinates


class TownSource(BaseModel):
    """GeoNames towns absent from the reference extract, within a population band."""

    archive: Path = Path("data/raw/cities5000.zip")
    reference_archive: Path = Path("data/raw/cities15000.zip")
    country_info: Path = Path("data/raw/countryInfo.txt")
    minimum_population: int = Field(default=5000, ge=0)
    maximum_population: int = Field(default=15000, gt=0)
