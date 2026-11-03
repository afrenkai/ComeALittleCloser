from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from come_a_little_bit_closer.evaluation.location_schemas import ProbeSettings
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
    max_length: int = Field(default=256, gt=1)
    batch_size: int = Field(default=256, gt=0)
    feature_directory: Path
    output_directory: Path
    final_cells: list[FinalCell] = Field(default_factory=list)

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
    control: LayerControl | Literal["country-centroid"]

    @model_validator(mode="after")
    def country_rows_have_no_layer(self) -> "LayerProbeScore":
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
    control: LayerControl | Literal["country-centroid"]
    pooling: Pooling | None = None


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
        default_factory=lambda: ["qwen3-0.6b", "qwen3-4b", "xlmr-base", "xlmr-large"]
    )
    arrow_sample: int = 1500
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
