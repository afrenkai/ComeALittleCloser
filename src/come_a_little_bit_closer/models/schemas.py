from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix, EmbeddingVector


class ModelEncoder(Protocol):
    def documents(self, texts: list[str], batch_size: int) -> EmbeddingMatrix: ...

    def query(self, text: str) -> EmbeddingVector: ...

    def token_lengths(self, texts: list[str], document: bool) -> list[int]: ...


class ModelSpec(BaseModel):
    model_id: str
    label: str
    checkpoint: str
    revision: str | None = None
    description: str
    query_prefix: str = ""
    document_prefix: str = ""
    text_supported: bool = True
    backend: Literal["sentence-transformers", "clap-text"] = "sentence-transformers"
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    text_only: bool = False


class ModelStatus(BaseModel):
    model_id: str
    label: str
    available: bool
    description: str


class InferenceSettings(BaseModel):
    model_id: str = "qwen3"
    embedding_directory: Path = Path("data/processed/embeddings")
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    max_sequence_length: int = Field(default=256, ge=32)


class EmbeddingManifest(BaseModel):
    model_id: str
    checkpoint: str
    revision: str
    source_sha256: str
    station_count: int = Field(gt=0)
    dimensions: int = Field(gt=0)
    created_at: str
    document_format: str = "station-metadata-explicit-coordinates-v1"
    normalized: bool = True
    max_sequence_length: int
    sentence_transformers_version: str
    encoder_settings: ModelSpec | None = None


class ArtifactValidation(BaseModel):
    artifact_path: Path
    manifest: EmbeddingManifest
    maximum_norm_error: float = Field(ge=0)
