from pydantic import BaseModel, Field

from come_a_little_bit_closer.evaluation.location_schemas import (
    LocationPlan,
    ProbeScore,
    ResponsivenessScore,
    RetrievalScore,
    SplitManifest,
)
from come_a_little_bit_closer.models.schemas import EmbeddingManifest


class ModelProvenance(BaseModel):
    original: EmbeddingManifest
    metadata_only: EmbeddingManifest


class ExecutionRecord(BaseModel):
    created_at_utc: str
    source_sha256: str
    plan_sha256: str
    station_count: int = Field(gt=0)
    model_manifests: list[ModelProvenance]
    software: dict[str, str]
    source_files_sha256: dict[str, str]


class TokenSummary(BaseModel):
    model_id: str
    condition: str
    texts: int = Field(gt=0)
    over_limit: int = Field(ge=0)
    maximum_tokens: int = Field(gt=0)
    token_limit: int = Field(gt=0)


class SplitSummary(BaseModel):
    seed: int
    partition: str
    stations: int
    blocks: int


class RetrievalSummary(BaseModel):
    model_id: str
    condition: str
    query_condition: str
    genres: int
    origins: int
    cases: int
    mean_regret_km: float
    mean_retrieved_km: float
    mean_reference_km: float
    mean_geographic_overlap: float
    mean_genre_precision: float


class PairedRegretChange(BaseModel):
    model_id: str
    condition: str
    genres: int
    origins: int
    mean_reduction_km: float
    ci_low_km: float
    ci_high_km: float


class ResponseSummary(BaseModel):
    model_id: str
    condition: str
    query_condition: str
    genres: int
    origin_pairs: int
    cases: int
    mean_turnover_percent: float
    mean_movement_km: float
    mean_query_cosine_distance: float


class GenreCoverage(BaseModel):
    columns: list[str]
    rows: list[dict[str, str]]


class TaxonomyRecord(BaseModel):
    source_url: str
    source_sha256: str
    retrieved_at_utc: str
    reference_genres: int
    normalized_genres: int
    selected_genres: int
    minimum_stations: int
    normalization: str


class ReportResults(BaseModel):
    plan: LocationPlan
    execution: ExecutionRecord
    probes: list[ProbeScore]
    retrieval: list[RetrievalScore]
    responses: list[ResponsivenessScore]
    tokens: list[TokenSummary]
    splits: list[SplitManifest]
    coverage: GenreCoverage | None = None
    taxonomy: TaxonomyRecord | None = None


class ReportSummaries(BaseModel):
    retrieval: list[RetrievalSummary]
    paired_regret: list[PairedRegretChange]
    responses: list[ResponseSummary]
    splits: list[SplitSummary]
