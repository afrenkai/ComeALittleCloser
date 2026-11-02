from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from come_a_little_bit_closer.evaluation.location_schemas import ProbeSettings
from come_a_little_bit_closer.models.layer_schemas import LayerModelSpec

type LayerDataset = Literal["cities", "stations"]
type TextCondition = Literal["name", "name-country", "metadata"]
type Pooling = Literal["mean", "last"]
type SplitKind = Literal["spatial-block", "random-item"]
type EvaluationMode = Literal["dev", "final"]


class FinalCell(BaseModel):
    model_id: str
    dataset: LayerDataset
    condition: TextCondition
    pooling: Pooling
    layer: int = Field(ge=0)


class LayerPlan(ProbeSettings):
    experiment_id: str
    model_ids: list[str]
    datasets: dict[LayerDataset, list[TextCondition]]
    split_kinds: list[SplitKind]
    max_length: int = Field(default=64, gt=1)
    batch_size: int = Field(default=256, gt=0)
    feature_directory: Path
    output_directory: Path
    final_cells: list[FinalCell] = Field(default_factory=list)


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
