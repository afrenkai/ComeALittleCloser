import numpy as np
from jaxtyping import Float64
from pydantic import BaseModel, ConfigDict, Field

type CoordinateVector = Float64[np.ndarray, "2"]
type CoordinateMatrix = Float64[np.ndarray, "locations 2"]
type SpatialMatrix = Float64[np.ndarray, "locations 3"]
type EmbeddingVector = Float64[np.ndarray, "features"]
type EmbeddingMatrix = Float64[np.ndarray, "locations features"]
type ScoreVector = Float64[np.ndarray, "locations"]


class Coordinates(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class GeometrySettings(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    earth_radius_km: float = Field(default=6371.0088, gt=0)
