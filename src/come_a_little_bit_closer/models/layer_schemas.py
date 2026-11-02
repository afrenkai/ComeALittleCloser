from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float16, Float32
from pydantic import BaseModel

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
    end_token: str | None = None
