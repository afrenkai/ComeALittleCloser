import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TextIO

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped
from pydantic import BaseModel

from come_a_little_bit_closer.evaluation.counterfactual import evaluate_counterfactuals
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.location_schemas import (
    LocationPlan,
    ProbeFeatures,
    ProbeScore,
    ProbeTargets,
    QueryManifest,
    ResponsivenessScore,
    RetrievalScore,
    SplitManifest,
    TokenAudit,
)
from come_a_little_bit_closer.evaluation.probes import (
    angular_errors,
    coordinate_centroid,
    country_predictions,
    fit_ridge,
    predict_ridge,
    score_probe,
    unit_predictions,
)
from come_a_little_bit_closer.evaluation.splits import spatial_split, split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import ScoreVector
from come_a_little_bit_closer.models.artifacts import EmbeddingArtifact
from come_a_little_bit_closer.storage.catalog import StationCatalog, initialize_catalog
from come_a_little_bit_closer.storage.schemas import DatabaseSettings


def write_records(path: Path, rows: list[BaseModel]) -> None:
    path.write_text(
        json.dumps([row.model_dump(mode="json") for row in rows], indent=2) + "\n"
    )


@jaxtyped(typechecker=beartype)
def evaluate_probe(
    features: ProbeFeatures,
    targets: ProbeTargets,
    manifest: SplitManifest,
    plan: LocationPlan,
    country_errors: ScoreVector,
    model_id: str,
    condition: str,
    control: str,
    predictions_file: TextIO,
) -> ProbeScore:
    train = split_indices(manifest, "train")
    validation = split_indices(manifest, "validation")
    test = split_indices(manifest, "test")
    fitting_targets = targets.copy()
    if control == "permuted-labels":
        fitting_targets[train] = targets[
            np.random.default_rng(manifest.seed).permutation(train)
        ]
    model = fit_ridge(
        features,
        fitting_targets,
        train,
        validation,
        [assignment.block_id for assignment in manifest.assignments],
        plan,
    )
    predictions = unit_predictions(
        predict_ridge(model, features), coordinate_centroid(targets[train])
    )
    score, errors = score_probe(
        predictions,
        targets,
        country_errors,
        manifest,
        plan,
        model_id,
        condition,
        control,
        features.shape[1],
        model.alpha,
    )
    writer = csv.writer(predictions_file)
    for position, test_index in enumerate(test):
        assignment = manifest.assignments[int(test_index)]
        writer.writerow(
            [
                model_id,
                condition,
                control,
                manifest.seed,
                assignment.station_id,
                assignment.block_id,
                float(errors[position]),
                float(predictions[test_index, 0]),
                float(predictions[test_index, 1]),
                float(predictions[test_index, 2]),
            ]
        )
    return score


def run_experiment(config_path: Path) -> None:
    plan = LocationPlan.model_validate_json(config_path.read_bytes())
    checksum = plan_checksum(config_path)
    if (
        plan.artifact_directory / "frozen-plan.json"
    ).read_bytes() != config_path.read_bytes():
        raise ValueError("Prepared artifacts use a different frozen plan.")
    database = DatabaseSettings()
    stats = initialize_catalog(database)
    stations = StationCatalog(database.database_path).stations()
    station_ids = [station.station_id for station in stations]
    country_codes = [station.country_code for station in stations]
    targets = spherical_vectors(
        np.asarray(
            [
                [station.coordinates.latitude, station.coordinates.longitude]
                for station in stations
            ],
            dtype=np.float64,
        )
    )
    plan.output_directory.mkdir(parents=True, exist_ok=True)
    (plan.output_directory / "frozen-plan.json").write_bytes(config_path.read_bytes())
    manifests = [
        spatial_split(stations, plan, seed, stats.source_sha256, checksum)
        for seed in plan.split_seeds
    ]
    for manifest in manifests:
        (plan.output_directory / f"split-{manifest.seed}.json").write_text(
            manifest.model_dump_json(indent=2) + "\n"
        )
    country_errors_by_seed: dict[int, ScoreVector] = {}
    probe_rows: list[ProbeScore] = []
    retrieval_rows: list[RetrievalScore] = []
    response_rows: list[ResponsivenessScore] = []
    token_rows: list[dict[str, str | int]] = []
    provenance: list[dict[str, object]] = []
    with (plan.output_directory / "probe-predictions.csv").open(
        "w", newline=""
    ) as predictions_file:
        csv.writer(predictions_file).writerow(
            [
                "model_id",
                "condition",
                "control",
                "seed",
                "station_id",
                "block_id",
                "error_km",
                "prediction_x",
                "prediction_y",
                "prediction_z",
            ]
        )
        for manifest in manifests:
            train = split_indices(manifest, "train")
            test = split_indices(manifest, "test")
            country = country_predictions(targets, country_codes, train)
            country_errors = angular_errors(
                country[test], targets[test], plan.geometry.earth_radius_km
            )
            country_errors_by_seed[manifest.seed] = country_errors
            global_predictions = np.tile(
                coordinate_centroid(targets[train]), (len(stations), 1)
            )
            for label, predictions in [
                ("country-centroid", country),
                ("global-centroid", global_predictions),
            ]:
                score, errors = score_probe(
                    predictions,
                    targets,
                    country_errors,
                    manifest,
                    plan,
                    "control",
                    "training-only",
                    label,
                    0,
                    None,
                )
                probe_rows.append(score)
            probe_rows.append(
                evaluate_probe(
                    targets,
                    targets,
                    manifest,
                    plan,
                    country_errors,
                    "control",
                    "true-coordinates",
                    "spherical-positive",
                    predictions_file,
                )
            )
        for model_id in plan.model_ids:
            original = EmbeddingArtifact(
                Path("data/processed/embeddings") / f"{model_id}.npz"
            )
            metadata_directory = (
                plan.metadata_directory or plan.artifact_directory / "metadata-only"
            )
            metadata = EmbeddingArtifact(metadata_directory / f"{model_id}.npz")
            for artifact in [original, metadata]:
                if (
                    artifact.manifest.source_sha256 != stats.source_sha256
                    or artifact.station_ids != station_ids
                ):
                    raise ValueError("Artifact source/identifier mismatch.")
            if metadata.manifest.revision != original.manifest.revision:
                raise ValueError("Ablation checkpoint revisions must match.")
            provenance.append(
                {
                    "original": original.manifest.model_dump(),
                    "metadata_only": metadata.manifest.model_dump(),
                }
            )
            with np.load(
                plan.artifact_directory / f"queries-{model_id}.npz", allow_pickle=False
            ) as archive:
                query_manifest = QueryManifest.model_validate_json(
                    str(archive["manifest"].item())
                )
                query_vectors = np.asarray(archive["embeddings"], dtype=np.float64)
            if (
                query_manifest.plan_sha256 != checksum
                or query_manifest.encoder_manifest.source_sha256 != stats.source_sha256
                or query_manifest.encoder_manifest.revision
                != original.manifest.revision
                or query_vectors.shape
                != (len(query_manifest.queries), original.manifest.dimensions)
                or not np.isfinite(query_vectors).all()
                or not np.allclose(np.linalg.norm(query_vectors, axis=1), 1)
            ):
                raise ValueError(
                    "Counterfactual query artifact failed provenance/shape/norm checks."
                )
            audit = TokenAudit.model_validate_json(
                (plan.artifact_directory / f"tokens-{model_id}.json").read_bytes()
            )
            if audit.station_ids != station_ids:
                raise ValueError("Token-audit identifiers differ from the catalog.")
            for label, lengths in [
                ("explicit-coordinates", audit.explicit_document_lengths),
                ("metadata-only", audit.metadata_document_lengths),
                ("queries", audit.query_lengths),
            ]:
                token_rows.append(
                    {
                        "model_id": model_id,
                        "condition": label,
                        "texts": len(lengths),
                        "over_limit": sum(
                            length > audit.max_sequence_length for length in lengths
                        ),
                        "maximum_tokens": max(lengths),
                        "token_limit": audit.max_sequence_length,
                    }
                )
            for condition, features in [
                ("explicit-coordinates", original.embeddings),
                ("metadata-only", metadata.embeddings),
            ]:
                for manifest in manifests:
                    for control in ["real", "permuted-labels"]:
                        probe_rows.append(
                            evaluate_probe(
                                features,
                                targets,
                                manifest,
                                plan,
                                country_errors_by_seed[manifest.seed],
                                model_id,
                                condition,
                                control,
                                predictions_file,
                            )
                        )
                retrieval, responses = evaluate_counterfactuals(
                    stations,
                    features,
                    query_vectors,
                    query_manifest,
                    plan,
                    model_id,
                    condition,
                )
                retrieval_rows.extend(retrieval)
                response_rows.extend(responses)
            for manifest in manifests:
                random = np.random.default_rng(manifest.seed).standard_normal(
                    original.embeddings.shape
                )
                random = random / np.linalg.norm(random, axis=1, keepdims=True)
                probe_rows.append(
                    evaluate_probe(
                        random,
                        targets,
                        manifest,
                        plan,
                        country_errors_by_seed[manifest.seed],
                        model_id,
                        "dimension-control",
                        "random-features",
                        predictions_file,
                    )
                )
            write_records(plan.output_directory / "probes.json", list(probe_rows))
            write_records(
                plan.output_directory / "retrieval.json", list(retrieval_rows)
            )
            write_records(
                plan.output_directory / "responsiveness.json", list(response_rows)
            )
            print(
                f"Evaluated {model_id}: both document conditions and all split seeds",
                flush=True,
            )
    (plan.output_directory / "token-audit.json").write_text(
        json.dumps(token_rows, indent=2) + "\n"
    )
    package_directory = Path(__file__).parents[1]
    execution = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": stats.source_sha256,
        "plan_sha256": checksum,
        "station_count": len(stations),
        "model_manifests": provenance,
        "software": {
            package: version(package)
            for package in ["numpy", "torch", "transformers", "sentence-transformers"]
        },
        "source_files_sha256": {
            str(path.relative_to(package_directory)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for domain in ["evaluation", "models", "geometry"]
            for path in sorted((package_directory / domain).glob("*.py"))
        },
    }
    (plan.output_directory / "execution.json").write_text(
        json.dumps(execution, indent=2) + "\n"
    )
    print(
        f"Complete: {len(probe_rows)} probe rows, "
        f"{len(retrieval_rows)} retrieval cases, {len(response_rows)} origin pairs",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen location probes and counterfactual retrieval."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/location-v1.json")
    )
    arguments = parser.parse_args()
    run_experiment(arguments.config)


if __name__ == "__main__":
    main()
