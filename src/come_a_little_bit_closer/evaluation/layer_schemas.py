from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from come_a_little_bit_closer.evaluation.location_schemas import (
    IndexVector,
    ProbeSettings,
    SubsetRidgeTargets,
)
from come_a_little_bit_closer.geometry.schemas import (
    CoordinateMatrix,
    Coordinates,
    ScoreVector,
)
from come_a_little_bit_closer.models.layer_schemas import LayerModelSpec

type LayerDataset = Literal["cities", "stations"]
type TextCondition = Literal["name", "name-country", "metadata"]
type Pooling = Literal["mean", "last"]
type SplitKind = Literal["spatial-block", "random-item"]
type EvaluationMode = Literal["dev", "final"]
type LayerControl = Literal["probe", "permuted-labels"]
type ScoreControl = LayerControl | Literal["country-centroid"]
type ProbeTarget = Literal["sphere", "latlon", "country-residual"]


class FinalCell(BaseModel):
    model_id: str
    dataset: LayerDataset
    condition: TextCondition
    pooling: Pooling
    layer: int = Field(ge=0)


class DatasetItems(BaseModel):
    item_ids: list[str]
    texts: list[str]
    coordinates: list[Coordinates]
    country_codes: list[str]
    source_sha256: str


class LayerPlan(ProbeSettings):
    experiment_id: str
    model_ids: list[str]
    datasets: dict[LayerDataset, list[TextCondition]]
    split_kinds: list[SplitKind]
    probe_targets: list[ProbeTarget] = Field(default_factory=lambda: ["sphere"])
    max_length: int = Field(default=256, gt=1)
    batch_size: int = Field(default=256, gt=0)
    feature_directory: Path
    output_directory: Path
    final_cells: list[FinalCell] = Field(default_factory=list)
    prediction_cells: list[FinalCell] = Field(default_factory=list)

    @model_validator(mode="after")
    def conditions_and_final_cells_are_consistent(self) -> "LayerPlan":
        allowed: dict[LayerDataset, set[TextCondition]] = {
            "cities": {"name", "name-country"},
            "stations": {"metadata"},
        }
        for dataset, conditions in self.datasets.items():
            if not set(conditions) <= allowed[dataset]:
                raise ValueError(f"Invalid conditions for dataset {dataset}.")
        if len({cell.model_dump_json() for cell in self.final_cells}) != len(
            self.final_cells
        ):
            raise ValueError("Final cells must be unique.")
        for cell in self.final_cells:
            if cell.condition not in self.datasets.get(cell.dataset, []):
                raise ValueError(f"Final cell outside the plan: {cell}")
        if len({cell.model_dump_json() for cell in self.prediction_cells}) != len(
            self.prediction_cells
        ):
            raise ValueError("Prediction cells must be unique.")
        for cell in self.prediction_cells:
            if cell.model_id not in self.model_ids or cell.condition not in (
                self.datasets.get(cell.dataset, [])
            ):
                raise ValueError(f"Prediction cell outside the plan: {cell}")
        return self


class FeatureManifest(BaseModel):
    model_id: str
    model_spec: LayerModelSpec
    checkpoint: str
    revision: str
    random_init: bool
    dataset: LayerDataset
    condition: TextCondition
    pooling: Pooling
    source_sha256: str
    item_ids: list[str]
    layers: int
    hidden: int
    dtype: str
    max_length: int
    truncated_items: int
    transformers_version: str
    created_at: str


class LayerScoreValues(BaseModel):
    selected_alpha: float | None
    evaluated_items: int = Field(gt=0)
    evaluated_blocks: int = Field(gt=0)
    block_mean_km: float
    block_mean_ci_low: float
    block_mean_ci_high: float
    gain_over_country_km: float
    gain_ci_low: float
    gain_ci_high: float
    r2_latitude: float | None = None
    r2_longitude: float | None = None


class RunIdentity(BaseModel):
    model_id: str
    family: str
    stage: str
    stage_order: int
    dataset: LayerDataset
    condition: TextCondition
    mode: EvaluationMode


class LayerProbeScore(LayerScoreValues, RunIdentity):
    pooling: Pooling | None
    layer: int | None
    relative_depth: float | None
    split_kind: SplitKind
    seed: int
    control: ScoreControl
    probe_target: ProbeTarget = "sphere"

    @model_validator(mode="after")
    def country_rows_have_no_layer(self) -> "LayerProbeScore":
        if self.control == "country-centroid" and self.probe_target != "sphere":
            raise ValueError("Country-centroid rows carry the sphere target.")
        if self.control == "permuted-labels" and self.probe_target != "sphere":
            raise ValueError("Permuted-labels controls run for the sphere target.")
        fitted = [
            self.pooling,
            self.layer,
            self.relative_depth,
            self.selected_alpha,
        ]
        if self.control == "country-centroid" and any(
            value is not None for value in fitted
        ):
            raise ValueError("Country-centroid rows carry no probe fields.")
        if self.control != "country-centroid" and any(
            value is None for value in fitted
        ):
            raise ValueError("Probe rows require pooling, layer, depth, and alpha.")
        return self


class ScoreSelection(BaseModel):
    dataset: LayerDataset
    condition: TextCondition
    split_kind: SplitKind
    control: ScoreControl
    pooling: Pooling | None = None
    probe_target: ProbeTarget = "sphere"


class BlockScore(BaseModel):
    block_id: str
    items: int = Field(gt=0)
    mean_km: float
    country_mean_km: float


class LayerBlockScores(RunIdentity):
    pooling: Pooling
    layer: int
    split_kind: SplitKind
    seed: int
    control: LayerControl
    probe_target: ProbeTarget
    blocks: list[BlockScore]


class PopularityCell(BaseModel):
    model_id: str
    cell: str
    mode: EvaluationMode
    split_kind: SplitKind
    probe_target: ProbeTarget


class PopularityGain(PopularityCell):
    quantile: int = Field(gt=0)
    population_min: int
    population_max: int
    items: int = Field(gt=0)
    blocks: int = Field(gt=0)
    gain_over_country_km: float
    gain_ci_low: float
    gain_ci_high: float


class ClaimRoster(BaseModel):
    """Models behind each frozen claim, as fixed in docs/FREEZE.md."""

    h1_models: list[str] = [
        "qwen3-4b-base",
        "qwen3-4b-embedding",
        "qwen3-8b-base",
        "qwen3-8b-embedding",
        "mistral-7b-base",
        "e5-mistral",
        "gemma2-9b-base",
        "bge-multilingual-gemma2",
    ]
    h2_pairs: list[tuple[str, str]] = [
        ("qwen3-4b-base", "qwen3-4b-embedding"),
        ("qwen3-8b-base", "qwen3-8b-embedding"),
        ("mistral-7b-base", "e5-mistral"),
        ("gemma2-9b-base", "bge-multilingual-gemma2"),
    ]
    secondary_pairs: list[tuple[str, str]] = [
        ("qwen3-0.6b-base", "qwen3-0.6b-embedding"),
        ("xlmr-base", "e5-base"),
        ("xlmr-large", "bge-m3"),
    ]
    h3_models: list[str] = [
        "xlmr-base",
        "e5-base",
        "xlmr-large",
        "bge-m3-retromae",
        "bge-m3-unsupervised",
        "bge-m3",
        "xlmr-xl",
    ]
    h3_gain_floor_km: float = 20.0
    h4_models: list[str] = [
        "qwen3-4b-embedding",
        "qwen3-8b-embedding",
        "e5-mistral",
        "bge-multilingual-gemma2",
    ]
    h4_secondary_models: list[str] = ["e5-base", "bge-m3"]
    random_init_models: list[str] = ["qwen3-0.6b-random", "xlmr-base-random"]


type ReflectedSimilarity = Literal["cosine", "centered-cosine", "probe-decoded"]


class ReflectedCell(BaseModel):
    model_id: str
    pooling: Pooling


class ReflectedPlan(BaseModel):
    """Is within-country geography reflected in similarity? Cities, name-country."""

    experiment_id: str
    layers_plan: Path
    cells: list[ReflectedCell]
    pairs: list[tuple[str, str]]
    minimum_cities: int = Field(default=30, gt=1)
    neighbors: int = Field(default=10, gt=0)
    output_directory: Path

    @model_validator(mode="after")
    def pairs_use_planned_cells(self) -> "ReflectedPlan":
        planned = {cell.model_id for cell in self.cells}
        for base_id, retrieval_id in self.pairs:
            if not {base_id, retrieval_id} <= planned:
                raise ValueError(f"Pair outside the plan: {base_id}, {retrieval_id}")
        return self


class CountryReflection(BaseModel):
    country_code: str
    cities: int = Field(gt=1)
    rho: float
    knn_gain: float


class ReflectedGeography(BaseModel):
    model_id: str
    pooling: Pooling
    layer: int
    mode: EvaluationMode
    seed: int
    similarity: ReflectedSimilarity
    rho: float
    knn_gain: float
    countries: list[CountryReflection]


class NeighborGeolocation(BaseModel):
    model_id: str
    pooling: Pooling
    layer: int
    mode: EvaluationMode
    split_kind: SplitKind
    seed: int
    neighbor_gain_km: float
    neighbor_ci_low: float
    neighbor_ci_high: float
    probe_gain_km: float
    probe_ci_low: float
    probe_ci_high: float
    difference_km: float
    difference_ci_low: float
    difference_ci_high: float


class NeighborCrossover(BaseModel):
    """Retrieval minus base of (probe error - neighbour error), paired by place."""

    base_id: str
    retrieval_id: str
    pooling: Pooling
    mode: EvaluationMode
    split_kind: SplitKind
    seed: int
    interaction_km: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PlaceErrors:
    """Per-place outcomes of one fitted probe on the evaluated rows."""

    rows: IndexVector
    block_ids: list[str]
    predicted_coordinates: CoordinateMatrix
    raw_outputs: SubsetRidgeTargets
    country_coordinates: CoordinateMatrix
    errors_km: ScoreVector
    country_errors_km: ScoreVector


class SeedRange(BaseModel):
    mean: float
    low: float
    high: float


class SeedMeanScore(BaseModel):
    model_id: str
    family: str
    stage: str
    stage_order: int
    pooling: Pooling | None
    layer: int | None
    relative_depth: float | None
    block_mean_km: float
    gain_over_country_km: float
    gain_ci_low: float
    gain_ci_high: float
    r2_latitude: SeedRange | None = None
    r2_longitude: SeedRange | None = None
    seeds: int


class StagePalette(BaseModel):
    """Okabe-Ito colors keyed by stage name, shared by every report figure."""

    colors: dict[str, str] = Field(
        default_factory=lambda: {
            "random": "#6e6e6e",
            "base": "#0072B2",
            "long-context pretraining": "#009E73",
            "unsupervised contrastive": "#CC79A7",
            "retrieval": "#D55E00",
        }
    )
    stage_names: dict[str, str] = Field(
        default_factory=lambda: {
            "random": "random init",
            "base": "base",
            "long-context pretraining": "long-context (RetroMAE)",
            "unsupervised contrastive": "unsupervised contrastive",
            "retrieval": "retrieval-tuned",
        }
    )
    family_order: list[str] = Field(
        default_factory=lambda: [
            "qwen3-0.6b",
            "qwen3-4b",
            "qwen3-8b",
            "mistral-7b",
            "gemma2-9b",
            "xlmr-base",
            "xlmr-large",
            "xlmr-xl",
        ]
    )
    arrow_sample: int = 1500
    min_country_train_items: int = 10
    font_settings: dict[str, float | bool] = Field(
        default_factory=lambda: {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


@dataclass(frozen=True)
class DistortionPanel:
    score: SeedMeanScore
    true_coordinates: CoordinateMatrix
    predicted_coordinates: CoordinateMatrix
    errors_km: ScoreVector
