import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, Float32, jaxtyped
from transformers import AutoConfig, AutoModel, AutoTokenizer

from come_a_little_bit_closer.models.layer_schemas import (
    EncodedBatch,
    ExtractedFeatures,
    HiddenStates,
    LayerModelSpec,
    PooledStates,
    TokenMask,
)


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
def checked_features(
    values: Float32[np.ndarray, "..."], dtype: str
) -> Float[np.ndarray, "..."]:
    if (
        not np.isfinite(values).all()
        or np.abs(values).max(initial=0) > np.finfo(dtype).max
    ):
        raise ValueError(f"Values are non-finite or exceed the {dtype} range.")
    return values.astype(dtype)


class LayerExtractor:
    def __init__(self, spec: LayerModelSpec, device: str, max_length: int) -> None:
        self.spec = spec
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.checkpoint, revision=spec.revision, trust_remote_code=False
        )
        self.end_token_id = self.tokenizer.convert_tokens_to_ids(
            spec.end_token or self.tokenizer.eos_token
        )
        self.start_ids = (
            [self.tokenizer.convert_tokens_to_ids(spec.start_token)]
            if spec.start_token
            else []
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if spec.random_init:
            config = AutoConfig.from_pretrained(
                spec.checkpoint, revision=spec.revision, trust_remote_code=False
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(spec.random_seed)
                self.model = AutoModel.from_config(
                    config,
                    dtype=torch.float32,
                    attn_implementation=spec.attention_implementation,
                    trust_remote_code=False,
                )
        else:
            self.model = AutoModel.from_pretrained(
                spec.checkpoint,
                revision=spec.revision,
                dtype=torch.float32,
                attn_implementation=spec.attention_implementation,
                trust_remote_code=False,
                device_map="auto" if spec.spread_over_gpus else None,
            )
        if not spec.spread_over_gpus:
            self.model.to(device)
        self.model.eval()

    def count_truncated(self, texts: list[str]) -> int:
        if self.spec.architecture == "encoder":
            lengths = [
                len(ids)
                for ids in self.tokenizer(texts, add_special_tokens=True)["input_ids"]
            ]
            return sum(length > self.max_length for length in lengths)
        lengths = [
            len(ids)
            for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        ]
        content_limit = self.max_length - 1 - len(self.start_ids)
        return sum(length > content_limit for length in lengths)

    def encode(self, texts: list[str]) -> EncodedBatch:
        if self.spec.architecture == "encoder":
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            attention_mask = encoded["attention_mask"].bool()
            return EncodedBatch(
                input_ids=encoded["input_ids"],
                attention_mask=attention_mask,
                content_mask=attention_mask & ~encoded["special_tokens_mask"].bool(),
            )
        content_ids = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1 - len(self.start_ids),
        )["input_ids"]
        padded = self.tokenizer.pad(
            {
                "input_ids": [
                    self.start_ids + ids + [self.end_token_id] for ids in content_ids
                ]
            },
            return_tensors="pt",
        )
        attention_mask = padded["attention_mask"].bool()
        content_start = len(self.start_ids)
        content_ends = torch.tensor([content_start + len(ids) for ids in content_ids])
        positions = torch.arange(attention_mask.shape[1])
        return EncodedBatch(
            input_ids=padded["input_ids"],
            attention_mask=attention_mask,
            content_mask=attention_mask
            & (positions >= content_start)
            & (positions < content_ends.unsqueeze(1)),
        )

    @jaxtyped(typechecker=beartype)
    def extract(
        self, texts: list[str], batch_size: int
    ) -> dict[str, ExtractedFeatures]:
        if not texts:
            raise ValueError("extract needs at least one text.")
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        decoder = self.spec.architecture == "decoder"
        features: dict[str, np.ndarray] = {}
        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            encoded = self.encode([texts[index] for index in positions])
            attention_mask = encoded.attention_mask.to(self.device)
            content_mask = encoded.content_mask.to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded.input_ids.to(self.device),
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            layer_states = [
                state.float().to(self.device) for state in outputs.hidden_states
            ]
            batch_pooled = {
                "mean": torch.stack(
                    [mean_pool(state, content_mask) for state in layer_states]
                )
            }
            if decoder:
                batch_pooled["last"] = torch.stack(
                    [last_token_pool(state, attention_mask) for state in layer_states]
                )
            for pooling, values in batch_pooled.items():
                if pooling not in features:
                    features[pooling] = np.empty(
                        (values.shape[0], len(texts), values.shape[2]),
                        dtype=self.spec.feature_dtype,
                    )
                features[pooling][:, positions] = checked_features(
                    values.cpu().numpy(), self.spec.feature_dtype
                )
        return features
