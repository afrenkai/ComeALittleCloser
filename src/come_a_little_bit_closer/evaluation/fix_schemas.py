from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float32
from pydantic import BaseModel, Field, model_validator

from come_a_little_bit_closer.evaluation.layer_schemas import (
    CountryReflection,
    EvaluationMode,
    SplitKind,
)

type UnitMatrix = Float32[np.ndarray, "items features"]
type ScoreMatrix = Float32[np.ndarray, "items items"]
type DistanceProbe = Float32[np.ndarray, "features rank"]
type FixCollection = Literal[
    "cities-name-country",
    "stations-metadata",
    "sts-dev",
    "sts-test",
    "towns-name-country",
]
type FreshPart = Literal["check", "test"]
type TownExtraction = Literal["plain-and-instructed", "instructed"]
type FixMethod = Literal[
    "cosine",
    "country",
    "oracle",
    "fusion",
    "probe",
    "fusion-probe",
    "instruction",
    "instruction-fusion",
    "instruction-probe",
    "instruction-fusion-probe",
    "generic-instruction",
    "generic-instruction-fusion-probe",
    "feuerstein-global",
    "feuerstein-country",
]


class InstructedModel(BaseModel):
    """A retrieval model and its native instruction format ({instruction}, {text})."""

    model_id: str
    template: str

    @model_validator(mode="after")
    def template_has_fields(self) -> "InstructedModel":
        if "{instruction}" not in self.template or "{text}" not in self.template:
            raise ValueError("Templates need {instruction} and {text}.")
        return self


class FixPlan(BaseModel):
    """Coordinate-free fixes for within-country geography in cosine similarity."""

    experiment_id: str
    layers_plan: Path
    models: list[InstructedModel]
    instruction: str
    instructed_directory: Path
    output_directory: Path
    sts_files: dict[Literal["dev", "test"], Path]
    sts_sha256: dict[Literal["dev", "test"], str]
    split_kinds: list[SplitKind] = ["random-item", "spatial-block"]
    fusion_weight: float = Field(default=1.0, ge=0)
    distance_weight: float = Field(default=0.1, ge=0)
    country_weight: float = Field(default=0.1, ge=0)
    oracle_weight: float = Field(default=0.2, ge=0)
    probe_rank: int = Field(default=32, gt=0)
    probe_steps: int = Field(default=3000, gt=0)
    probe_batch: int = Field(default=2048, gt=0)
    probe_positives: int = Field(default=10, gt=0)
    probe_learning_rate: float = Field(default=1e-3, gt=0)
    probe_margin: float = Field(default=1.0, gt=0)
    probe_seed: int = 0
    layer_sample: int = Field(default=8000, gt=0)
    minimum_cities: int = Field(default=30, gt=1)
    neighbors: int = Field(default=10, gt=0)
    batch_size: int = Field(default=16, gt=0)
    max_length: int = Field(default=256, gt=0)


class InstructedManifest(BaseModel):
    model_id: str
    checkpoint: str
    revision: str
    collection: FixCollection
    instruction: str | None
    template: str
    source_sha256: str
    item_ids: list[str]
    kept_layers: list[int]
    total_layers: int
    hidden: int
    max_length: int
    truncated_items: int
    transformers_version: str
    created_at: str


class FixScore(BaseModel):
    model_id: str
    mode: EvaluationMode
    split_kind: SplitKind
    seed: int
    method: FixMethod
    fusion_layer: int | None
    city_knn_gain: float
    city_rho: float
    city_neighbor_km: float
    station_knn_gain: float
    station_rho: float
    station_neighbor_km: float
    genre_agreement: float
    sts_spearman: float | None
    city_countries: list[CountryReflection]
    blend_weight: float | None = None


class BaselinePlan(BaseModel):
    """Validation-only baselines and ablations for the fix (no final mode)."""

    experiment_id: str
    fix_plan: Path
    generic_plan: Path
    output_directory: Path
    feuerstein_weights: list[float] = [0.2, 1.0]


class FreshPlan(BaseModel):
    """Fresh held-out towns, scored with components fit on the original training cities.

    Towns are split once into a check part (pipeline checks) and a test part (run once).
    """

    experiment_id: str
    fix_plan: Path
    generic_plan: Path
    output_directory: Path
    towns_sha256: str
    check_fraction: float = Field(default=0.2, gt=0, lt=1)
    partition_seed: int
    claim_models: list[str]
    feuerstein_weights: list[float] = [0.2, 1.0]
    feuerstein_claim_weight: float = 0.2


class TownScore(BaseModel):
    model_id: str
    part: FreshPart
    split_kind: SplitKind
    seed: int
    method: FixMethod
    blend_weight: float | None = None
    fusion_layer: int | None = None
    towns: int
    city_knn_gain: float
    city_rho: float
    city_countries: list[CountryReflection]


@dataclass(frozen=True)
class StsPairs:
    sentences: list[str]
    first: np.ndarray
    second: np.ndarray
    scores: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class FittedFix:
    """Components fit on training cities and applied unchanged to any text set."""

    layer: int
    layer_center: np.ndarray
    output_center: np.ndarray
    probe: np.ndarray


@dataclass(frozen=True)
class FixTargets:
    """What every score matrix is judged against, for one split and seed."""

    city_truth: np.ndarray
    city_codes: np.ndarray
    station_truth: np.ndarray
    station_codes: np.ndarray
    genres: list[set[str]]
    sts: StsPairs


@dataclass(frozen=True)
class FeuersteinMap:
    """Latitude and longitude axes from seed pairs, min-max mapped to coordinates."""

    axes: np.ndarray
    projection_low: np.ndarray
    projection_high: np.ndarray
    coordinate_low: np.ndarray
    coordinate_high: np.ndarray
