from pathlib import Path

from pydantic import BaseModel, Field

from come_a_little_bit_closer.data.schemas import DataSummary
from come_a_little_bit_closer.geometry.schemas import Coordinates, GeometrySettings


class BaselineRequest(BaseModel):
    source_path: Path
    output_path: Path
    origin: Coordinates
    genre: str = "rock"
    limit: int = Field(default=10, gt=0)
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)


class RankedStation(BaseModel):
    station_id: str
    name: str
    country_code: str
    distance_km: float
    spherical_cosine: float


class BaselineReport(BaseModel):
    baseline: str = "deterministic spherical coordinates; no learned encoder"
    data_summary: DataSummary
    origin: Coordinates
    genre: str
    eligible_stations: int
    evaluated_neighbors: int
    overlap_at_k: float = Field(ge=0, le=1)
    maximum_distance_discrepancy_km: float = Field(ge=0)
    haversine_neighbors: list[RankedStation]
    spherical_cosine_neighbors: list[RankedStation]
