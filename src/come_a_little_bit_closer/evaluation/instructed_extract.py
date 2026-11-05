import argparse
import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import transformers

from come_a_little_bit_closer.data.schemas import TownSource
from come_a_little_bit_closer.evaluation.fix_schemas import (
    FixCollection,
    FixPlan,
    InstructedManifest,
    InstructedModel,
    StsPairs,
    TownExtraction,
)
from come_a_little_bit_closer.evaluation.layer_extract import (
    atomic_write,
    dataset_items,
    town_items,
)
from come_a_little_bit_closer.models.hidden_states import LayerExtractor
from come_a_little_bit_closer.models.layer_catalog import find_layer_model


def read_sts(path: Path, expected_sha256: str) -> StsPairs:
    """STS-Benchmark pairs; sentences are deduplicated in first-appearance order."""
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"STS file checksum differs: {path}")
    records = [json.loads(line) for line in gzip.decompress(content).splitlines()]
    position: dict[str, int] = {}
    for record in records:
        for key in ["sentence1", "sentence2"]:
            position.setdefault(record[key], len(position))
    return StsPairs(
        sentences=list(position),
        first=np.asarray([position[record["sentence1"]] for record in records]),
        second=np.asarray([position[record["sentence2"]] for record in records]),
        scores=np.asarray([float(record["score"]) for record in records]),
        source_sha256=digest,
    )


def collection_texts(
    plan: FixPlan, collection: FixCollection
) -> tuple[list[str], list[str], str]:
    if collection == "cities-name-country":
        items = dataset_items("cities", "name-country")
        return items.item_ids, items.texts, items.source_sha256
    if collection == "stations-metadata":
        items = dataset_items("stations", "metadata")
        return items.item_ids, items.texts, items.source_sha256
    if collection in ["towns-name-country", "small-towns-name-country"]:
        items = town_items(town_source(collection))
        return items.item_ids, items.texts, items.source_sha256
    split = "dev" if collection == "sts-dev" else "test"
    pairs = read_sts(plan.sts_files[split], plan.sts_sha256[split])
    return (
        [str(index) for index in range(len(pairs.sentences))],
        pairs.sentences,
        pairs.source_sha256,
    )


def town_source(collection: FixCollection) -> TownSource:
    """Fresh towns of 5,000-14,999 people, or the smaller band of 1,000-4,999."""
    if collection == "towns-name-country":
        return TownSource()
    return TownSource(
        archive=Path("data/raw/cities1000.zip"),
        reference_archives=[
            Path("data/raw/cities5000.zip"),
            Path("data/raw/cities15000.zip"),
        ],
        minimum_population=1000,
        maximum_population=5000,
    )


def plan_model(plan: FixPlan, model_id: str) -> InstructedModel:
    for model in plan.models:
        if model.model_id == model_id:
            return model
    raise ValueError(f"Model is outside the fix plan: {model_id}")


def instructed_path(
    plan: FixPlan, model_id: str, collection: FixCollection, instructed: bool
) -> Path:
    variant = "instructed" if instructed else "plain"
    pooling = plan_model(plan, model_id).pooling
    return (
        plan.instructed_directory / model_id / f"{collection}-{variant}-{pooling}.npy"
    )


def kept_layers(total: int) -> list[int]:
    """Every second layer of the upper half, plus the output layer."""
    return sorted(set(range(total // 2, total, 2)) | {total - 1})


def read_instructed(
    plan: FixPlan, model_id: str, collection: FixCollection, instructed: bool
) -> tuple[np.ndarray, InstructedManifest]:
    path = instructed_path(plan, model_id, collection, instructed)
    manifest = InstructedManifest.model_validate_json(
        path.with_suffix(".json").read_bytes()
    )
    if manifest.instruction != (plan.instruction if instructed else None):
        raise ValueError("Features were made with another instruction.")
    item_ids, texts, source_sha256 = collection_texts(plan, collection)
    if manifest.item_ids != item_ids or manifest.source_sha256 != source_sha256:
        raise ValueError("Features do not match the current items.")
    return np.load(path, mmap_mode="r"), manifest


def already_extracted(
    path: Path,
    plan: FixPlan,
    collection: FixCollection,
    instructed: bool,
    model: InstructedModel,
) -> bool:
    manifest_path = path.with_suffix(".json")
    if not path.exists() or not manifest_path.exists():
        return False
    manifest = InstructedManifest.model_validate_json(manifest_path.read_bytes())
    item_ids, texts, source_sha256 = collection_texts(plan, collection)
    return (
        manifest.instruction == (plan.instruction if instructed else None)
        and manifest.template
        == (model.template if instructed else model.plain_template)
        and manifest.pooling == model.pooling
        and manifest.item_ids == item_ids
        and manifest.source_sha256 == source_sha256
        and manifest.max_length == plan.max_length
    )


def standard_jobs(plan: FixPlan) -> list[tuple[FixCollection, bool]]:
    """Instructed cities and stations, both STS variants, and, for plans whose plain
    features do not come from the layer extraction, plain cities and stations."""
    sts_collections: dict[str, FixCollection] = {"dev": "sts-dev", "test": "sts-test"}
    plain: list[tuple[FixCollection, bool]] = (
        []
        if plan.plain_from_layers
        else [("cities-name-country", False), ("stations-metadata", False)]
    )
    return (
        plain
        + [
            ("cities-name-country", True),
            ("stations-metadata", True),
        ]
        + [
            (sts_collections[split], instructed)
            for split in plan.sts_files
            for instructed in [False, True]
        ]
    )


def town_jobs(
    extraction: TownExtraction, collections: list[FixCollection]
) -> list[tuple[FixCollection, bool]]:
    variants = [True] if extraction == "instructed" else [False, True]
    return [
        (collection, instructed)
        for collection in collections
        for instructed in variants
    ]


def extract_instructed(
    plan: FixPlan,
    model: InstructedModel,
    device: str,
    jobs: list[tuple[FixCollection, bool]],
) -> list[Path]:
    spec = find_layer_model(model.model_id)
    if model.pooling == "last" and spec.architecture != "decoder":
        raise ValueError("Last-token pooling needs a decoder.")
    extractor = LayerExtractor(spec, device, plan.max_length)
    written = []
    for collection, instructed in jobs:
        item_ids, raw_texts, source_sha256 = collection_texts(plan, collection)
        texts = [
            (
                model.template.format(instruction=plan.instruction, text=text)
                if instructed
                else model.plain_template.format(text=text)
            )
            for text in raw_texts
        ]
        path = instructed_path(plan, model.model_id, collection, instructed)
        if already_extracted(path, plan, collection, instructed, model):
            print(f"Already extracted {model.model_id} {collection}", flush=True)
            written.append(path)
            continue
        features = extractor.extract(texts, plan.batch_size)[model.pooling]
        layers = kept_layers(features.shape[0])
        manifest = InstructedManifest(
            model_id=spec.model_id,
            pooling=model.pooling,
            checkpoint=spec.checkpoint,
            revision=spec.revision,
            collection=collection,
            instruction=plan.instruction if instructed else None,
            template=model.template if instructed else model.plain_template,
            source_sha256=source_sha256,
            item_ids=item_ids,
            kept_layers=layers,
            total_layers=features.shape[0],
            hidden=features.shape[2],
            max_length=plan.max_length,
            truncated_items=extractor.count_truncated(texts),
            transformers_version=transformers.__version__,
            created_at=datetime.now(UTC).isoformat(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, lambda handle: np.save(handle, features[layers]))
        atomic_write(
            path.with_suffix(".json"),
            lambda handle: handle.write(
                (manifest.model_dump_json(indent=2) + "\n").encode()
            ),
        )
        written.append(path)
        print(
            f"Extracted {model.model_id} {collection} instructed={instructed}",
            flush=True,
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract instruction-conditioned last-token features."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ampere", action="store_true")
    parser.add_argument("--towns", choices=["plain-and-instructed", "instructed"])
    parser.add_argument(
        "--town-collections",
        nargs="+",
        choices=["towns-name-country", "small-towns-name-country"],
        default=["towns-name-country"],
    )
    arguments = parser.parse_args()
    plan = FixPlan.model_validate_json(arguments.config.read_bytes())
    model = plan_model(plan, arguments.model)
    jobs = (
        standard_jobs(plan)
        if arguments.towns is None
        else town_jobs(arguments.towns, arguments.town_collections)
    )
    extract_instructed(plan, model, "cuda" if arguments.ampere else "cpu", jobs)


if __name__ == "__main__":
    main()
