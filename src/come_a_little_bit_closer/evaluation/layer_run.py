import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.evaluation.layer_extract import (
    dataset_items,
    feature_path,
)
from come_a_little_bit_closer.evaluation.layer_schemas import (
    EvaluationMode,
    FeatureManifest,
    LayerControl,
    LayerDataset,
    LayerPlan,
    LayerProbeScore,
    LayerScoreValues,
    Pooling,
    RunIdentity,
    SplitKind,
    TextCondition,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.location_schemas import (
    IndexVector,
    ProbeFeatures,
    ProbeTargets,
    RidgeModel,
    SplitManifest,
)
from come_a_little_bit_closer.evaluation.probes import (
    angular_errors,
    block_means,
    bootstrap_interval,
    coordinate_centroid,
    country_predictions,
    fit_ridge,
    predict_ridge,
    unit_predictions,
)
from come_a_little_bit_closer.evaluation.splits import (
    random_split,
    spatial_split,
    split_indices,
)
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import Coordinates, ScoreVector
from come_a_little_bit_closer.models.layer_catalog import find_layer_model
from come_a_little_bit_closer.models.layer_schemas import LayerFeatures


def evaluated_indices(manifest: SplitManifest, mode: EvaluationMode) -> IndexVector:
    return split_indices(manifest, "validation" if mode == "dev" else "test")


@jaxtyped(typechecker=beartype)
def fit_layer_probe(
    features: ProbeFeatures,
    targets: ProbeTargets,
    manifest: SplitManifest,
    plan: LayerPlan,
    control: LayerControl,
) -> RidgeModel:
    """Fits on train and selects alpha on validation without reading other rows."""
    train = split_indices(manifest, "train")
    validation = split_indices(manifest, "validation")
    fitting_rows = np.concatenate([train, validation])
    fitting_targets = targets[fitting_rows]
    if control == "permuted-labels":
        fitting_targets[: len(train)] = targets[
            np.random.default_rng(manifest.seed).permutation(train)
        ]
    return fit_ridge(
        features[fitting_rows],
        fitting_targets,
        np.arange(len(train), dtype=np.int64),
        np.arange(len(train), len(fitting_rows), dtype=np.int64),
        [manifest.assignments[int(row)].block_id for row in fitting_rows],
        plan,
    )


@jaxtyped(typechecker=beartype)
def summarize(
    errors: ScoreVector,
    country_errors: ScoreVector,
    blocks: list[str],
    plan: LayerPlan,
    selected_alpha: float | None,
) -> LayerScoreValues:
    means = block_means(errors, blocks)
    gains = block_means(country_errors - errors, blocks)
    low, high = bootstrap_interval(means, plan)
    gain_low, gain_high = bootstrap_interval(gains, plan)
    return LayerScoreValues(
        selected_alpha=selected_alpha,
        evaluated_items=len(errors),
        evaluated_blocks=len(means),
        block_mean_km=float(means.mean()),
        block_mean_ci_low=low,
        block_mean_ci_high=high,
        gain_over_country_km=float(gains.mean()),
        gain_ci_low=gain_low,
        gain_ci_high=gain_high,
    )


@jaxtyped(typechecker=beartype)
def country_control_errors(
    targets: ProbeTargets,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    mode: EvaluationMode,
) -> ScoreVector:
    evaluated = evaluated_indices(manifest, mode)
    predictions = country_predictions(
        targets, country_codes, split_indices(manifest, "train")
    )
    return angular_errors(
        predictions[evaluated], targets[evaluated], plan.geometry.earth_radius_km
    )


@jaxtyped(typechecker=beartype)
def score_layer(
    features: ProbeFeatures,
    targets: ProbeTargets,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    mode: EvaluationMode,
    control: LayerControl,
) -> LayerScoreValues:
    train = split_indices(manifest, "train")
    evaluated = evaluated_indices(manifest, mode)
    model = fit_layer_probe(features, targets, manifest, plan, control)
    predictions = unit_predictions(
        predict_ridge(model, features[evaluated]),
        coordinate_centroid(targets[train]),
    )
    errors = angular_errors(
        predictions, targets[evaluated], plan.geometry.earth_radius_km
    )
    return summarize(
        errors,
        country_control_errors(targets, country_codes, manifest, plan, mode),
        [manifest.assignments[int(row)].block_id for row in evaluated],
        plan,
        model.alpha,
    )


@jaxtyped(typechecker=beartype)
def score_country_control(
    targets: ProbeTargets,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    mode: EvaluationMode,
) -> LayerScoreValues:
    errors = country_control_errors(targets, country_codes, manifest, plan, mode)
    evaluated = evaluated_indices(manifest, mode)
    return summarize(
        errors,
        errors,
        [manifest.assignments[int(row)].block_id for row in evaluated],
        plan,
        None,
    )


def build_split(
    kind: SplitKind,
    item_ids: list[str],
    coordinates: list[Coordinates],
    plan: LayerPlan,
    seed: int,
    source_sha256: str,
    checksum: str,
) -> SplitManifest:
    builder = spatial_split if kind == "spatial-block" else random_split
    return builder(item_ids, coordinates, plan, seed, source_sha256, checksum)


def scored_layers(
    plan: LayerPlan,
    model_id: str,
    mode: EvaluationMode,
    dataset: LayerDataset,
    condition: TextCondition,
    pooling: Pooling,
    layer_count: int,
) -> list[int]:
    if mode == "dev":
        return list(range(layer_count))
    layers = sorted(
        cell.layer
        for cell in plan.final_cells
        if (cell.model_id, cell.dataset, cell.condition, cell.pooling)
        == (model_id, dataset, condition, pooling)
    )
    if layers and layers[-1] >= layer_count:
        raise ValueError("Final cell layer is outside the extracted layers.")
    return layers


@jaxtyped(typechecker=beartype)
def load_features(
    plan: LayerPlan,
    model_id: str,
    dataset: LayerDataset,
    condition: TextCondition,
    pooling: Pooling,
    item_ids: list[str],
    source_sha256: str,
) -> tuple[LayerFeatures, FeatureManifest]:
    path = feature_path(plan, model_id, dataset, condition, pooling)
    manifest = FeatureManifest.model_validate_json(
        path.with_suffix(".json").read_bytes()
    )
    if (manifest.model_id, manifest.dataset, manifest.condition, manifest.pooling) != (
        model_id,
        dataset,
        condition,
        pooling,
    ):
        raise ValueError("Feature manifest identity differs from its path.")
    if manifest.item_ids != item_ids:
        raise ValueError("Feature item IDs differ from the dataset's items.")
    if manifest.source_sha256 != source_sha256:
        raise ValueError("Feature source checksum differs from the current source.")
    features = np.load(path, mmap_mode="r")
    if features.shape != (manifest.layers, len(item_ids), manifest.hidden):
        raise ValueError("Feature array shape differs from its manifest.")
    return features, manifest


def validate_final_cells(
    plan: LayerPlan, model_id: str, poolings: list[Pooling]
) -> set[tuple[LayerDataset, TextCondition, Pooling, int]]:
    invalid: list[str] = []
    cells: set[tuple[LayerDataset, TextCondition, Pooling, int]] = set()
    for cell in plan.final_cells:
        if cell.model_id != model_id:
            continue
        cells.add((cell.dataset, cell.condition, cell.pooling, cell.layer))
        if cell.condition not in plan.datasets.get(cell.dataset, []):
            invalid.append(f"{cell}: dataset/condition outside the plan")
        elif cell.pooling not in poolings:
            invalid.append(f"{cell}: pooling unavailable for this model")
        else:
            manifest_path = feature_path(
                plan, model_id, cell.dataset, cell.condition, cell.pooling
            ).with_suffix(".json")
            layer_count = FeatureManifest.model_validate_json(
                manifest_path.read_bytes()
            ).layers
            if cell.layer >= layer_count:
                invalid.append(f"{cell}: layer outside 0..{layer_count - 1}")
    if invalid:
        raise ValueError("Invalid final cells: " + "; ".join(invalid))
    return cells


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fchmod(handle.fileno(), 0o640)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def write_rows(path: Path, rows: list[LayerProbeScore]) -> None:
    write_atomic(path, "".join(row.model_dump_json() + "\n" for row in rows).encode())


def split_record(manifest: SplitManifest) -> bytes:
    return manifest.model_dump_json(indent=2).encode()


def split_path(
    plan: LayerPlan,
    mode: EvaluationMode,
    dataset: LayerDataset,
    kind: SplitKind,
    seed: int,
) -> Path:
    return plan.output_directory / mode / "splits" / f"{dataset}-{kind}-{seed}.json"


def refuse_differing_final_split(path: Path, manifest: SplitManifest) -> None:
    if path.exists() and path.read_bytes() != split_record(manifest):
        raise FileExistsError(f"A differing final split already exists: {path}")


def run_model(config_path: Path, model_id: str, mode: EvaluationMode) -> Path:
    plan_bytes = config_path.read_bytes()
    plan = LayerPlan.model_validate_json(plan_bytes)
    if model_id not in plan.model_ids:
        raise ValueError("Model is outside the frozen analysis plan.")
    output_path = plan.output_directory / mode / f"{model_id}.jsonl"
    if mode == "final":
        if output_path.exists():
            raise FileExistsError(f"Final results already exist: {output_path}")
        if not any(cell.model_id == model_id for cell in plan.final_cells):
            raise ValueError("Final mode requires frozen cells for this model.")
    spec = find_layer_model(model_id)
    poolings: list[Pooling] = (
        ["mean", "last"] if spec.architecture == "decoder" else ["mean"]
    )
    expected_cells = (
        validate_final_cells(plan, model_id, poolings) if mode == "final" else set()
    )
    controls: list[LayerControl] = ["probe", "permuted-labels"]
    checksum = plan_checksum(config_path)
    rows: list[LayerProbeScore] = []
    split_records: dict[Path, SplitManifest] = {}
    for dataset, conditions in plan.datasets.items():
        items_by_condition = {
            condition: dataset_items(dataset, condition) for condition in conditions
        }
        reference = items_by_condition[conditions[0]]
        targets = spherical_vectors(
            np.asarray(
                [[place.latitude, place.longitude] for place in reference.coordinates],
                dtype=np.float64,
            )
        )
        manifests = {
            (kind, seed): build_split(
                kind,
                reference.item_ids,
                reference.coordinates,
                plan,
                seed,
                reference.source_sha256,
                checksum,
            )
            for kind in plan.split_kinds
            for seed in plan.split_seeds
        }
        for (kind, seed), manifest in manifests.items():
            path = split_path(plan, mode, dataset, kind, seed)
            if mode == "final":
                refuse_differing_final_split(path, manifest)
            split_records[path] = manifest
        for condition in conditions:
            condition_items = items_by_condition[condition]
            if condition_items.item_ids != reference.item_ids:
                raise ValueError("Dataset conditions must share item IDs.")
            controlled = False
            identity = RunIdentity(
                model_id=model_id,
                family=spec.family,
                stage=spec.stage,
                stage_order=spec.stage_order,
                dataset=dataset,
                condition=condition,
                mode=mode,
            )
            for pooling in poolings:
                features, feature_manifest = load_features(
                    plan,
                    model_id,
                    dataset,
                    condition,
                    pooling,
                    condition_items.item_ids,
                    condition_items.source_sha256,
                )
                layers = scored_layers(
                    plan,
                    model_id,
                    mode,
                    dataset,
                    condition,
                    pooling,
                    feature_manifest.layers,
                )
                if layers and not controlled:
                    controlled = True
                    for (kind, seed), manifest in manifests.items():
                        rows.append(
                            LayerProbeScore(
                                **identity.model_dump(),
                                pooling=None,
                                layer=None,
                                relative_depth=None,
                                split_kind=kind,
                                seed=seed,
                                control="country-centroid",
                                **score_country_control(
                                    targets,
                                    reference.country_codes,
                                    manifest,
                                    plan,
                                    mode,
                                ).model_dump(),
                            )
                        )
                for layer in layers:
                    layer_features = np.asarray(features[layer], dtype=np.float64)
                    for (kind, seed), manifest in manifests.items():
                        for control in controls:
                            rows.append(
                                LayerProbeScore(
                                    **identity.model_dump(),
                                    pooling=pooling,
                                    layer=layer,
                                    relative_depth=layer
                                    / (feature_manifest.layers - 1),
                                    split_kind=kind,
                                    seed=seed,
                                    control=control,
                                    **score_layer(
                                        layer_features,
                                        targets,
                                        reference.country_codes,
                                        manifest,
                                        plan,
                                        mode,
                                        control,
                                    ).model_dump(),
                                )
                            )
                    print(
                        f"Scored {model_id} {dataset} {condition} {pooling} {layer}",
                        flush=True,
                    )
    if mode == "final":
        scored_cells = {
            (row.dataset, row.condition, row.pooling, row.layer)
            for row in rows
            if row.control != "country-centroid"
        }
        if scored_cells != expected_cells:
            raise ValueError("Scored cells differ from the frozen final cells.")
    write_atomic(plan.output_directory / mode / f"{model_id}.plan.json", plan_bytes)
    for path, manifest in split_records.items():
        write_atomic(path, split_record(manifest) + b"\n")
    write_rows(output_path, rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score layerwise geographic probes.")
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/layers-v1.json")
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    arguments = parser.parse_args()
    print(run_model(arguments.config, arguments.model, arguments.mode), flush=True)


if __name__ == "__main__":
    main()
