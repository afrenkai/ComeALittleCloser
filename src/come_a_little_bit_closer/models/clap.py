from functools import lru_cache
from threading import Lock

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix, EmbeddingVector
from come_a_little_bit_closer.models.encoder import normalize_vectors
from come_a_little_bit_closer.models.schemas import ModelSpec

try:
    import torch
    from transformers import AutoTokenizer, ClapTextModelWithProjection
except ModuleNotFoundError:
    torch = None
    AutoTokenizer = None
    ClapTextModelWithProjection = None


class ClapTextEncoder:
    def __init__(
        self, spec: ModelSpec, revision: str, device: str, max_sequence_length: int
    ) -> None:
        if (
            torch is None
            or AutoTokenizer is None
            or ClapTextModelWithProjection is None
        ):
            raise RuntimeError("Run uv sync --locked for CPU CLAP inference.")
        self.spec = spec
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.checkpoint, revision=revision, trust_remote_code=False
        )
        self.model = (
            ClapTextModelWithProjection.from_pretrained(
                spec.checkpoint, revision=revision
            )
            .eval()
            .to(device)
        )
        self.lock = Lock()

    def token_lengths(self, texts: list[str], document: bool) -> list[int]:
        prefix = self.spec.document_prefix if document else self.spec.query_prefix
        tokens = self.tokenizer(
            [prefix + text for text in texts],
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return [len(token_ids) for token_ids in tokens["input_ids"]]

    @jaxtyped(typechecker=beartype)
    def encode_batch(self, texts: list[str]) -> EmbeddingMatrix:
        with self.lock, torch.inference_mode():
            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            ).to(self.device)
            embeddings = self.model(**tokens).text_embeds.cpu().float().numpy()
        return normalize_vectors(np.asarray(embeddings, dtype=np.float64))

    @jaxtyped(typechecker=beartype)
    def documents(self, texts: list[str], batch_size: int) -> EmbeddingMatrix:
        if not texts or batch_size < 1:
            raise ValueError("Document batches must contain text and a positive size.")
        batches = [
            self.encode_batch(
                [
                    self.spec.document_prefix + text
                    for text in texts[start : start + batch_size]
                ]
            )
            for start in range(0, len(texts), batch_size)
        ]
        return np.concatenate(batches, axis=0)

    @lru_cache(maxsize=128)
    @jaxtyped(typechecker=beartype)
    def query(self, text: str) -> EmbeddingVector:
        return self.encode_batch([self.spec.query_prefix + text])[0]
