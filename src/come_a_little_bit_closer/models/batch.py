import argparse
from datetime import UTC, datetime
from importlib.metadata import version

from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.models.artifacts import save_embeddings
from come_a_little_bit_closer.models.catalog import find_model
from come_a_little_bit_closer.models.factory import create_encoder
from come_a_little_bit_closer.models.schemas import EmbeddingManifest, InferenceSettings
from come_a_little_bit_closer.storage.catalog import StationCatalog, initialize_catalog
from come_a_little_bit_closer.storage.schemas import DatabaseSettings


def station_document(station: Station) -> str:
    return (
        f"Station: {station.name}\nCountry code: {station.country_code}\n"
        f"Genres: {', '.join(station.genre_tags)}\n"
        f"Latitude: {station.coordinates.latitude:.5f}\n"
        f"Longitude: {station.coordinates.longitude:.5f}"
    )


def build_embeddings(settings: InferenceSettings) -> EmbeddingManifest:
    database_settings = DatabaseSettings()
    stats = initialize_catalog(database_settings)
    stations = StationCatalog(database_settings.database_path).stations()
    spec = find_model(settings.model_id)
    if not spec.text_supported:
        raise ValueError(f"{spec.label} has no verified text embedding pipeline yet.")
    revision = spec.revision
    if revision is None:
        raise ValueError("An immutable checkpoint revision must be configured first.")
    encoder = create_encoder(
        spec, revision, settings.device, settings.max_sequence_length
    )
    embeddings = encoder.documents(
        [station_document(station) for station in stations], settings.batch_size
    )
    manifest = EmbeddingManifest(
        model_id=spec.model_id,
        checkpoint=spec.checkpoint,
        revision=revision,
        source_sha256=stats.source_sha256,
        station_count=len(stations),
        dimensions=embeddings.shape[1],
        created_at=datetime.now(UTC).isoformat(),
        max_sequence_length=settings.max_sequence_length,
        sentence_transformers_version=version("sentence-transformers"),
        encoder_settings=spec,
    )
    save_embeddings(
        settings.embedding_directory / f"{spec.model_id}.npz",
        [station.station_id for station in stations],
        embeddings,
        manifest,
    )
    print(manifest.model_dump_json(indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen station text embeddings."
    )
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--ampere", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    build_embeddings(
        InferenceSettings(
            model_id=arguments.model,
            device="cuda" if arguments.ampere else "cpu",
            batch_size=arguments.batch_size,
        )
    )


if __name__ == "__main__":
    main()
