import argparse
import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.evaluation.location_schemas import (
    ExperimentQuery,
    LocationPlan,
    QueryManifest,
    TokenAudit,
)
from come_a_little_bit_closer.models.artifacts import EmbeddingArtifact, save_embeddings
from come_a_little_bit_closer.models.batch import station_document
from come_a_little_bit_closer.models.catalog import find_model
from come_a_little_bit_closer.models.factory import create_encoder
from come_a_little_bit_closer.storage.catalog import StationCatalog, initialize_catalog
from come_a_little_bit_closer.storage.schemas import DatabaseSettings


def plan_checksum(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def metadata_document(station: Station) -> str:
    return (
        f"Station: {station.name}\nCountry code: {station.country_code}\n"
        f"Genres: {', '.join(station.genre_tags)}"
    )


def experiment_queries(plan: LocationPlan) -> list[ExperimentQuery]:
    queries: list[ExperimentQuery] = []
    for genre in plan.genres:
        text = f"nearest {genre} station"
        queries.append(
            ExperimentQuery(
                query_id=f"{genre}:none", genre=genre, origin=None, text=text
            )
        )
        for origin_index, origin in enumerate(plan.origins):
            queries.append(
                ExperimentQuery(
                    query_id=f"{genre}:{origin_index}",
                    genre=genre,
                    origin=origin,
                    text=(
                        f"{text}\nOrigin latitude: {origin.coordinates.latitude:.5f}\n"
                        f"Origin longitude: {origin.coordinates.longitude:.5f}"
                    ),
                )
            )
    return queries


def prepare_model(config_path: Path, model_id: str, device: str) -> None:
    plan = LocationPlan.model_validate_json(config_path.read_bytes())
    plan.artifact_directory.mkdir(parents=True, exist_ok=True)
    frozen_path = plan.artifact_directory / "frozen-plan.json"
    if frozen_path.exists() and frozen_path.read_bytes() != config_path.read_bytes():
        raise ValueError(
            "The experiment plan has changed; use a new experiment directory."
        )
    frozen_path.write_bytes(config_path.read_bytes())
    if model_id not in plan.model_ids:
        raise ValueError("Model is outside the frozen analysis plan.")
    source = DatabaseSettings()
    stats = initialize_catalog(source)
    stations = StationCatalog(source.database_path).stations()
    original = EmbeddingArtifact(Path("data/processed/embeddings") / f"{model_id}.npz")
    if original.manifest.source_sha256 != stats.source_sha256:
        raise ValueError("Source and embedding provenance differ.")
    station_ids = [station.station_id for station in stations]
    if original.station_ids != station_ids:
        raise ValueError("Experiment requires the exact sorted station catalog.")
    spec = original.manifest.encoder_settings or find_model(model_id)
    encoder = create_encoder(
        spec, original.manifest.revision, device, original.manifest.max_sequence_length
    )
    documents = [metadata_document(station) for station in stations]
    metadata_directory = (
        plan.metadata_directory or plan.artifact_directory / "metadata-only"
    )
    metadata_path = metadata_directory / f"{model_id}.npz"
    expected_format = "station-metadata-without-coordinates-v1"
    if metadata_path.exists():
        metadata = EmbeddingArtifact(metadata_path)
        if (
            metadata.manifest.source_sha256 != stats.source_sha256
            or metadata.manifest.revision != original.manifest.revision
            or metadata.manifest.document_format != expected_format
            or metadata.station_ids != station_ids
        ):
            raise ValueError("Existing ablation cache does not match this run.")
    else:
        vectors = encoder.documents(documents, plan.batch_size)
        manifest = original.manifest.model_copy(
            update={
                "created_at": datetime.now(UTC).isoformat(),
                "document_format": expected_format,
                "encoder_settings": spec,
            }
        )
        save_embeddings(metadata_path, station_ids, vectors, manifest)
    queries = experiment_queries(plan)
    query_vectors = np.asarray(
        [encoder.query(query.text) for query in queries], dtype=np.float64
    )
    query_manifest = QueryManifest(
        encoder_manifest=original.manifest,
        plan_sha256=plan_checksum(config_path),
        queries=queries,
        encoding_path="native-single",
    )
    plan.artifact_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=plan.artifact_directory, suffix=".npz", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        np.savez_compressed(
            temporary_file,
            embeddings=query_vectors,
            manifest=np.asarray(query_manifest.model_dump_json()),
        )
    os.replace(temporary_path, plan.artifact_directory / f"queries-{model_id}.npz")
    audit = TokenAudit(
        model_id=model_id,
        max_sequence_length=original.manifest.max_sequence_length,
        station_ids=station_ids,
        explicit_document_lengths=encoder.token_lengths(
            [station_document(station) for station in stations], document=True
        ),
        metadata_document_lengths=encoder.token_lengths(documents, document=True),
        query_lengths=encoder.token_lengths(
            [query.text for query in queries], document=False
        ),
    )
    (plan.artifact_directory / f"tokens-{model_id}.json").write_text(
        audit.model_dump_json(indent=2) + "\n"
    )
    print(
        f"Prepared {model_id}: {len(stations)} ablations, {len(queries)} queries",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen location experiment."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/location-v1.json")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--ampere", action="store_true")
    arguments = parser.parse_args()
    prepare_model(
        arguments.config, arguments.model, "cuda" if arguments.ampere else "cpu"
    )


if __name__ == "__main__":
    main()
