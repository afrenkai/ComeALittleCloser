import argparse
import hashlib
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np
import transformers

from come_a_little_bit_closer.data.cities import (
    load_fresh_towns,
    load_places,
    place_text,
)
from come_a_little_bit_closer.data.schemas import TownSource
from come_a_little_bit_closer.evaluation.layer_schemas import (
    DatasetItems,
    FeatureManifest,
    LayerDataset,
    LayerPlan,
    Pooling,
    TextCondition,
)
from come_a_little_bit_closer.evaluation.location_prepare import metadata_document
from come_a_little_bit_closer.models.hidden_states import LayerExtractor
from come_a_little_bit_closer.models.layer_catalog import find_layer_model
from come_a_little_bit_closer.storage.catalog import StationCatalog, initialize_catalog
from come_a_little_bit_closer.storage.schemas import DatabaseSettings


def feature_path(
    plan: LayerPlan,
    model_id: str,
    dataset: LayerDataset,
    condition: TextCondition,
    pooling: Pooling,
) -> Path:
    return plan.feature_directory / model_id / f"{dataset}-{condition}-{pooling}.npy"


def dataset_items(dataset: LayerDataset, condition: TextCondition) -> DatasetItems:
    if dataset == "cities":
        if condition == "metadata":
            raise ValueError("Cities support the name and name-country conditions.")
        archive_path = Path("data/raw/cities15000.zip")
        country_info_path = Path("data/raw/countryInfo.txt")
        places = load_places(archive_path, country_info_path)
        return DatasetItems(
            item_ids=[place.place_id for place in places],
            texts=[place_text(place, condition) for place in places],
            coordinates=[place.coordinates for place in places],
            country_codes=[place.country_code for place in places],
            source_sha256=hashlib.sha256(
                (
                    hashlib.sha256(archive_path.read_bytes()).hexdigest()
                    + hashlib.sha256(country_info_path.read_bytes()).hexdigest()
                ).encode()
            ).hexdigest(),
        )
    if condition != "metadata":
        raise ValueError("Stations support only the metadata condition.")
    source = DatabaseSettings()
    stats = initialize_catalog(source)
    stations = StationCatalog(source.database_path).stations()
    return DatasetItems(
        item_ids=[station.station_id for station in stations],
        texts=[metadata_document(station) for station in stations],
        coordinates=[station.coordinates for station in stations],
        country_codes=[station.country_code for station in stations],
        source_sha256=stats.source_sha256,
    )


def town_items(source: TownSource) -> DatasetItems:
    towns = load_fresh_towns(source)
    return DatasetItems(
        item_ids=[town.place_id for town in towns],
        texts=[place_text(town, "name-country") for town in towns],
        coordinates=[town.coordinates for town in towns],
        country_codes=[town.country_code for town in towns],
        source_sha256=hashlib.sha256(
            "".join(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    source.archive,
                    source.reference_archive,
                    source.country_info,
                ]
            ).encode()
            + source.model_dump_json(
                include={"minimum_population", "maximum_population"}
            ).encode()
        ).hexdigest(),
    )


def existing_features_match(manifest_path: Path, expected: FeatureManifest) -> bool:
    if not manifest_path.exists():
        return False
    existing = FeatureManifest.model_validate_json(manifest_path.read_bytes())
    ignored = {"created_at"}
    if existing.model_dump(exclude=ignored) != expected.model_dump(exclude=ignored):
        raise ValueError("Existing feature file does not match this run.")
    return True


def features_complete(path: Path, expected: FeatureManifest) -> bool:
    if not existing_features_match(path.with_suffix(".json"), expected):
        return False
    return path.exists() and np.load(path, mmap_mode="r").shape == (
        expected.layers,
        len(expected.item_ids),
        expected.hidden,
    )


def atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        try:
            write(handle)
            handle.flush()
            os.fchmod(handle.fileno(), 0o640)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def write_features(
    plan: LayerPlan,
    extractor: LayerExtractor,
    dataset: LayerDataset,
    condition: TextCondition,
    item_ids: list[str],
    texts: list[str],
    source_sha256: str,
) -> list[Path]:
    spec = extractor.spec
    poolings: list[Pooling] = (
        ["mean", "last"] if spec.architecture == "decoder" else ["mean"]
    )
    truncated_items = extractor.count_truncated(texts)
    manifests = {
        pooling: FeatureManifest(
            model_id=spec.model_id,
            model_spec=spec,
            checkpoint=spec.checkpoint,
            revision=spec.revision,
            random_init=spec.random_init,
            dataset=dataset,
            condition=condition,
            pooling=pooling,
            source_sha256=source_sha256,
            item_ids=item_ids,
            layers=extractor.model.config.num_hidden_layers + 1,
            hidden=extractor.model.config.hidden_size,
            dtype="float16",
            max_length=extractor.max_length,
            truncated_items=truncated_items,
            transformers_version=transformers.__version__,
            created_at=datetime.now(UTC).isoformat(),
        )
        for pooling in poolings
    }
    paths = {
        pooling: feature_path(plan, spec.model_id, dataset, condition, pooling)
        for pooling in poolings
    }
    pending = [
        pooling
        for pooling in poolings
        if not features_complete(paths[pooling], manifests[pooling])
    ]
    if pending:
        features = extractor.extract(texts, plan.batch_size)
        for pooling in pending:
            paths[pooling].parent.mkdir(parents=True, exist_ok=True)
            atomic_write(
                paths[pooling], lambda handle: np.save(handle, features[pooling])
            )
            atomic_write(
                paths[pooling].with_suffix(".json"),
                lambda handle: handle.write(
                    (manifests[pooling].model_dump_json(indent=2) + "\n").encode()
                ),
            )
    return [paths[pooling] for pooling in poolings]


def extract_model(plan: LayerPlan, model_id: str, device: str) -> list[Path]:
    if model_id not in plan.model_ids:
        raise ValueError("Model is outside the frozen analysis plan.")
    extractor = LayerExtractor(find_layer_model(model_id), device, plan.max_length)
    written: list[Path] = []
    for dataset, conditions in plan.datasets.items():
        for condition in conditions:
            items = dataset_items(dataset, condition)
            written.extend(
                write_features(
                    plan,
                    extractor,
                    dataset,
                    condition,
                    items.item_ids,
                    items.texts,
                    items.source_sha256,
                )
            )
            print(f"Extracted {model_id} {dataset} {condition}", flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-layer pooled features.")
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/layers-v1.json")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--ampere", action="store_true")
    arguments = parser.parse_args()
    plan = LayerPlan.model_validate_json(arguments.config.read_bytes())
    extract_model(plan, arguments.model, "cuda" if arguments.ampere else "cpu")


if __name__ == "__main__":
    main()
