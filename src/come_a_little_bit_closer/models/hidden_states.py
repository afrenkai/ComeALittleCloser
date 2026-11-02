import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float16, Float32, jaxtyped
from transformers import AutoConfig, AutoModel, AutoTokenizer

from come_a_little_bit_closer.models.schemas import (
    HiddenStates,
    LayerFeatures,
    LayerModelSpec,
    PooledStates,
    TokenMask,
)

RANDOM_INIT_SEED = 1729


@jaxtyped(typechecker=beartype)
def mean_pool(states: HiddenStates, mask: TokenMask) -> PooledStates:
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every item needs at least one pooled token.")
    kept = states.masked_fill(~mask.unsqueeze(-1), 0.0)
    return kept.sum(dim=1) / counts.unsqueeze(-1).to(states.dtype)


@jaxtyped(typechecker=beartype)
def last_token_pool(states: HiddenStates, attention_mask: TokenMask) -> PooledStates:
    if bool((attention_mask.sum(dim=1) == 0).any()):
        raise ValueError("Every item needs at least one attended token.")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_index = (attention_mask * positions).argmax(dim=1)
    rows = torch.arange(states.shape[0], device=states.device)
    return states[rows, last_index]


@jaxtyped(typechecker=beartype)
def checked_float16(values: Float32[np.ndarray, "..."]) -> Float16[np.ndarray, "..."]:
    if (
        not np.isfinite(values).all()
        or np.abs(values).max(initial=0) > np.finfo(np.float16).max
    ):
        raise ValueError("Values are non-finite or exceed the float16 range.")
    return values.astype(np.float16)


class LayerExtractor:
    def __init__(self, spec: LayerModelSpec, device: str, max_length: int) -> None:
        self.spec = spec
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.checkpoint, revision=spec.revision, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if spec.random_init:
            torch.manual_seed(RANDOM_INIT_SEED)
            config = AutoConfig.from_pretrained(
                spec.checkpoint, revision=spec.revision, trust_remote_code=False
            )
            self.model = AutoModel.from_config(config, trust_remote_code=False)
        else:
            self.model = AutoModel.from_pretrained(
                spec.checkpoint,
                revision=spec.revision,
                torch_dtype=torch.float32,
                trust_remote_code=False,
            )
        self.model.to(device).eval()

    @jaxtyped(typechecker=beartype)
    def extract(self, texts: list[str], batch_size: int) -> dict[str, LayerFeatures]:
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        pooled: dict[str, list[np.ndarray]] = {"mean": []}
        if self.spec.architecture == "decoder":
            pooled["last"] = []
        for start in range(0, len(order), batch_size):
            batch_texts = [texts[index] for index in order[start : start + batch_size]]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            attention_mask = encoded["attention_mask"].bool().to(self.device)
            content_mask = attention_mask & ~encoded["special_tokens_mask"].bool().to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded["input_ids"].to(self.device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            layer_states = [state.float() for state in outputs.hidden_states]
            pooled["mean"].append(
                torch.stack([mean_pool(state, content_mask) for state in layer_states])
                .cpu()
                .numpy()
            )
            if self.spec.architecture == "decoder":
                pooled["last"].append(
                    torch.stack(
                        [
                            last_token_pool(state, attention_mask)
                            for state in layer_states
                        ]
                    )
                    .cpu()
                    .numpy()
                )
        restore = np.argsort(np.asarray(order))
        return {
            pooling: checked_float16(np.concatenate(batches, axis=1)[:, restore])
            for pooling, batches in pooled.items()
        }
