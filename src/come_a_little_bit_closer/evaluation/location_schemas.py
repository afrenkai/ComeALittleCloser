from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float64, Int64
from pydantic import BaseModel, Field

from come_a_little_bit_closer.geometry.schemas import Coordinates, GeometrySettings
from come_a_little_bit_closer.models.schemas import EmbeddingManifest

type IndexVector = Int64[np.ndarray, "indices"]
type TrainingIndices = Int64[np.ndarray, "training"]
type ValidationIndices = Int64[np.ndarray, "validation"]
type RankedIndices = Int64[np.ndarray, "neighbors"]
type QueryMatrix = Float64[np.ndarray, "queries features"]
type BlockScores = Float64[np.ndarray, "blocks"]
type ProbeWeights = Float64[np.ndarray, "features 3"]
type ProbeTargets = Float64[np.ndarray, "samples 3"]
type ProbeFeatures = Float64[np.ndarray, "samples features"]
type DocumentCondition = Literal["explicit-coordinates", "metadata-only"]
type SplitName = Literal["train", "validation", "test"]


class ExperimentOrigin(BaseModel):
    name: str
    coordinates: Coordinates


class ProbeSettings(BaseModel):
    split_seeds: list[int]
    ridge_alphas: list[float]
    block_degrees: float = Field(default=10, gt=0, le=90)
    train_fraction: float = Field(default=0.6, gt=0, lt=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    bootstrap_replicates: int = Field(default=1000, gt=0)
    bootstrap_seed: int = 1618
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)


class LocationPlan(ProbeSettings):
    experiment_id: str
    model_ids: list[str]
    genres: list[str]
    genre_tags: dict[str, list[str]] = Field(default_factory=dict)
    minimum_genre_candidates: int = Field(default=1, gt=0)
    origins: list[ExperimentOrigin]
    neighbors: int = Field(default=10, gt=0)
    batch_size: int = Field(default=32, gt=0)
    artifact_directory: Path = Path("data/processed/location-v1")
    metadata_directory: Path | None = None
    taxonomy_manifest: Path | None = None
    output_directory: Path = Path("outputs/location-v1")


class StationSplit(BaseModel):
    station_id: str
    block_id: str
    split: SplitName


class SplitManifest(BaseModel):
    seed: int
    source_sha256: str
    plan_sha256: str
    block_degrees: float
    assignments: list[StationSplit]


class ExperimentQuery(BaseModel):
    query_id: str
    genre: str
    origin: ExperimentOrigin | None
    text: str


class TokenAudit(BaseModel):
    model_id: str
    max_sequence_length: int
    station_ids: list[str]
    explicit_document_lengths: list[int]
    metadata_document_lengths: list[int]
    query_lengths: list[int]


class QueryManifest(BaseModel):
    encoder_manifest: EmbeddingManifest
    plan_sha256: str
    queries: list[ExperimentQuery]
    encoding_path: Literal["native-single"] = "native-single"


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: Float64[np.ndarray, "features"]
    target_mean: Float64[np.ndarray, "3"]
    weights: ProbeWeights
    alpha: float


class ProbeScore(BaseModel):
    model_id: str
    condition: str
    control: str
    seed: int
    dimensions: int
    train_stations: int
    validation_stations: int
    test_stations: int
    test_blocks: int
    selected_alpha: float | None
    block_mean_km: float
    block_mean_ci_low: float
    block_mean_ci_high: float
    median_km: float
    p90_km: float
    improvement_over_country_km: float
    improvement_ci_low: float
    improvement_ci_high: float


class RetrievalScore(BaseModel):
    model_id: str
    condition: str
    query_condition: Literal["origin", "no-origin"]
    genre: str
    origin_name: str
    latitude: float
    longitude: float
    candidates: int
    genre_precision_at_k: float
    geographic_overlap_at_k: float
    retrieved_mean_km: float
    reference_mean_km: float
    regret_km: float
    retrieved_ids: list[str]
    reference_ids: list[str]


class ResponsivenessScore(BaseModel):
    model_id: str
    condition: str
    query_condition: Literal["origin", "no-origin"]
    genre: str
    first_origin: str
    second_origin: str
    query_cosine_distance: float
    neighbor_turnover: float
    movement_toward_origin_km: float
