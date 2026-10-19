from functools import lru_cache
from threading import Lock

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix, EmbeddingVector
from come_a_little_bit_closer.models.schemas import ModelSpec

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None
    torch = None


def inference_installed() -> bool:
    return SentenceTransformer is not None


@jaxtyped(typechecker=beartype)
def normalize_vectors(embeddings: EmbeddingMatrix) -> EmbeddingMatrix:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or np.any(norms == 0):
        raise ValueError("Encoders must return finite, nonzero embedding vectors.")
    return embeddings / norms


class TextEncoder:
    def __init__(
        self, spec: ModelSpec, revision: str, device: str, max_sequence_length: int
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("Run uv sync --locked for CPU model inference.")
        self.spec = spec
        if spec.text_only:
            self.model = SentenceTransformer(
                spec.checkpoint,
                revision=revision,
                device=device,
                trust_remote_code=False,
                model_kwargs={"torch_dtype": torch.float32},
                config_kwargs={"vision_config": None, "audio_config": None},
            )
        else:
            self.model = SentenceTransformer(
                spec.checkpoint,
                revision=revision,
                device=device,
                trust_remote_code=False,
            )
        self.model.max_seq_length = max_sequence_length
        self.lock = Lock()

    def token_lengths(self, texts: list[str], document: bool) -> list[int]:
        prompt_name = (
            self.spec.document_prompt_name if document else self.spec.query_prompt_name
        ) or self.model.default_prompt_name
        prompt = self.model.prompts.get(prompt_name, "") if prompt_name else ""
        prefix = self.spec.document_prefix if document else self.spec.query_prefix
        tokens = self.model.tokenizer(
            [prompt + prefix + text for text in texts],
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return [len(token_ids) for token_ids in tokens["input_ids"]]

    @jaxtyped(typechecker=beartype)
    def documents(self, texts: list[str], batch_size: int) -> EmbeddingMatrix:
        with self.lock:
            embeddings = self.model.encode(
                [self.spec.document_prefix + text for text in texts],
                batch_size=batch_size,
                prompt_name=self.spec.document_prompt_name,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        return normalize_vectors(np.asarray(embeddings, dtype=np.float64))

    @lru_cache(maxsize=128)
    @jaxtyped(typechecker=beartype)
    def query(self, text: str) -> EmbeddingVector:
        with self.lock:
            embedding = self.model.encode(
                self.spec.query_prefix + text,
                prompt_name=self.spec.query_prompt_name,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return normalize_vectors(
            np.asarray(embedding, dtype=np.float64)[np.newaxis, :]
        )[0]
