from typing import Literal

import numpy as np
from jaxtyping import Float64, Int64
from pydantic import BaseModel, Field

type ProjectionBasis = Float64[np.ndarray, "features 3"]
type ProjectionMatrix = Float64[np.ndarray, "locations 3"]
type ProjectionVector = Float64[np.ndarray, "3"]
type ClusterLabels = Int64[np.ndarray, "locations"]


class ProjectionSettings(BaseModel):
    cluster_count: int = Field(default=8, ge=1, le=10)
    random_seed: int = 1729
    maximum_iterations: int = Field(default=25, ge=1)


class ProjectionCluster(BaseModel):
    cluster_id: int = Field(ge=0)
    label: str
    size: int = Field(gt=0)


class ProjectionQuery(BaseModel):
    x: float
    y: float
    z: float


class ProjectionPoint(ProjectionQuery):
    station_id: str
    name: str
    country_code: str
    stream_url: str
    latitude: float
    longitude: float
    distance_km: float
    similarity: float | None
    geographic_rank: int | None
    embedding_rank: int | None
    cluster_id: int = Field(ge=0)


class EmbeddingProjection(BaseModel):
    method: Literal["PCA"] = "PCA"
    description: str
    explained_variance_ratio: list[float]
    query: ProjectionQuery
    points: list[ProjectionPoint]
    frame_center: ProjectionQuery
    frame_radius: float = Field(ge=0)
    clusters: list[ProjectionCluster]
    clustering_method: Literal["spherical k-means"] = "spherical k-means"
