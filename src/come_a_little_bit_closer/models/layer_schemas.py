from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float16, Float32
from pydantic import BaseModel, ConfigDict

type HiddenStates = Float32[torch.Tensor, "batch tokens hidden"]
type TokenMask = Bool[torch.Tensor, "batch tokens"]
type PooledStates = Float32[torch.Tensor, "batch hidden"]
type LayerFeatures = Float16[np.ndarray, "layers items hidden"]


class LayerModelSpec(BaseModel):
    model_id: str
    family: str
    stage: str
    stage_order: int
    checkpoint: str
    revision: str
    architecture: Literal["decoder", "encoder"]
    random_init: bool = False
    random_seed: int = 1729
    start_token: str | None = None
    end_token: str | None = None
    attention_implementation: Literal["eager"] | None = None
    spread_over_gpus: bool = False


class EncodedBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor


class ParityResult(BaseModel):
    """Agreement between our pooled output layer and the model's own embedding."""

    model_id: str
    pooling: str
    prefix: str
    texts: int
    minimum_cosine: float
    mean_cosine: float
