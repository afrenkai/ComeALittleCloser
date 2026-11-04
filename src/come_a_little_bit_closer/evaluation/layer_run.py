import argparse
import io
import os
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.evaluation.layer_extract import (
    dataset_items,
    feature_path,
)
from come_a_little_bit_closer.evaluation.layer_schemas import (
    BlockScore,
    EvaluationMode,
    FeatureManifest,
    FinalCell,
    LayerBlockScores,
    LayerControl,
    LayerDataset,
    LayerPlan,
    LayerProbeScore,
    LayerScoreValues,
    PlaceErrors,
    Pooling,
    ProbeTarget,
    RunIdentity,
    SplitKind,
    TextCondition,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.location_schemas import (
    IndexVector,
    ProbeFeatures,
    ProbeTargets,
    RidgeTargets,
    SplitManifest,
    SubsetRidgeTargets,
    SubsetTargets,
    TargetRidgeModel,
)
from come_a_little_bit_closer.evaluation.probes import (
    angular_errors,
    block_means,
    bootstrap_interval,
    coordinate_centroid,
    country_predictions,
    fit_ridge_path,
    predict_targets,
    r_squared,
    unit_predictions,
    vectors_to_coordinates,
)
from come_a_little_bit_closer.evaluation.splits import (
    random_split,
    spatial_split,
    split_indices,
)
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import (
    CoordinateMatrix,
    Coordinates,
    ScoreVector,
)
from come_a_little_bit_closer.models.layer_catalog import find_layer_model
from come_a_little_bit_closer.models.layer_schemas import LayerFeatures


def evaluated_indices(manifest: SplitManifest, mode: EvaluationMode) -> IndexVector:
    return split_indices(manifest, "validation" if mode == "dev" else "test")


@jaxtyped(typechecker=beartype)
def ridge_targets(
    probe_target: ProbeTarget,
    targets: ProbeTargets,
    coordinates: CoordinateMatrix,
    country_codes: list[str],
    train: IndexVector,
) -> RidgeTargets:
    if probe_target == "latlon":
        return coordinates
    if probe_target == "country-residual":
        return targets - country_predictions(targets, country_codes, train)
    return targets


@jaxtyped(typechecker=beartype)
def decode_predictions(
    raw: RidgeTargets,
    probe_target: ProbeTarget,
    centroids: ProbeTargets,
    fallback: ScoreVector,
) -> ProbeTargets:
    if probe_target == "latlon":
        return spherical_vectors(
            np.column_stack(
                (
                    np.clip(raw[:, 0], -90.0, 90.0),
                    (raw[:, 1] + 180.0) % 360.0 - 180.0,
                )
            )
        )
    if probe_target == "country-residual":
        return unit_predictions(centroids + raw, fallback)
    return unit_predictions(raw, fallback)


@jaxtyped(typechecker=beartype)
def fit_layer_probe(
    features: ProbeFeatures,
    targets: ProbeTargets,
    coordinates: CoordinateMatrix,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    control: LayerControl,
    probe_target: ProbeTarget,
) -> TargetRidgeModel:
    """Fits on train and selects alpha on validation without reading other rows."""
    if control == "permuted-labels" and probe_target != "sphere":
        raise ValueError("Permuted-labels controls run for the sphere target only.")
    train = split_indices(manifest, "train")
    validation = split_indices(manifest, "validation")
    fitting_rows = np.concatenate([train, validation])
    design = ridge_targets(probe_target, targets, coordinates, country_codes, train)
    fitting_targets = design[fitting_rows]
    if control == "permuted-labels":
        fitting_targets[: len(train)] = design[
            np.random.default_rng(manifest.seed).permutation(train)
        ]
    centroids = country_predictions(targets, country_codes, train)
    return fit_ridge_path(
        features[fitting_rows],
        fitting_targets,
        targets[fitting_rows],
        np.arange(len(train), dtype=np.int64),
        np.arange(len(train), len(fitting_rows), dtype=np.int64),
        [manifest.assignments[int(row)].block_id for row in fitting_rows],
        partial(
            decode_predictions,
            probe_target=probe_target,
            centroids=centroids[validation],
            fallback=coordinate_centroid(targets[train]),
        ),
        plan,
        probe_target != "country-residual",
    )


@jaxtyped(typechecker=beartype)
def layer_predictions(
    features: ProbeFeatures,
    targets: ProbeTargets,
    coordinates: CoordinateMatrix,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    rows: IndexVector,
    control: LayerControl,
    probe_target: ProbeTarget,
) -> tuple[SubsetTargets, SubsetRidgeTargets, float]:
    train = split_indices(manifest, "train")
    model = fit_layer_probe(
        features,
        targets,
        coordinates,
        country_codes,
        manifest,
        plan,
        control,
        probe_target,
    )
    centroids = country_predictions(targets, country_codes, train)
    raw = predict_targets(model, features[rows])
    return (
        decode_predictions(
            raw,
            probe_target,
            centroids[rows],
            coordinate_centroid(targets[train]),
        ),
        raw,
        model.alpha,
    )


@jaxtyped(typechecker=beartype)
def summarize(
    errors: ScoreVector,
    country_errors: ScoreVector,
    blocks: list[str],
    plan: LayerPlan,
    selected_alpha: float | None,
    predicted_coordinates: CoordinateMatrix,
    true_coordinates: CoordinateMatrix,
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
        r2_latitude=r_squared(predicted_coordinates[:, 0], true_coordinates[:, 0]),
        r2_longitude=r_squared(predicted_coordinates[:, 1], true_coordinates[:, 1]),
    )


@jaxtyped(typechecker=beartype)
def country_control_predictions(
    targets: ProbeTargets,
    country_codes: list[str],
    manifest: SplitManifest,
    mode: EvaluationMode,
) -> SubsetTargets:
    return country_predictions(
        targets, country_codes, split_indices(manifest, "train")
    )[evaluated_indices(manifest, mode)]


@jaxtyped(typechecker=beartype)
def evaluate_layer(
    features: ProbeFeatures,
    targets: ProbeTargets,
    coordinates: CoordinateMatrix,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    mode: EvaluationMode,
    control: LayerControl,
    probe_target: ProbeTarget,
) -> tuple[LayerScoreValues, PlaceErrors]:
    evaluated = evaluated_indices(manifest, mode)
    predictions, raw, alpha = layer_predictions(
        features,
        targets,
        coordinates,
        country_codes,
        manifest,
        plan,
        evaluated,
        control,
        probe_target,
    )
    errors = angular_errors(
        predictions, targets[evaluated], plan.geometry.earth_radius_km
    )
    country = country_control_predictions(targets, country_codes, manifest, mode)
    country_errors = angular_errors(
        country, targets[evaluated], plan.geometry.earth_radius_km
    )
    block_ids = [manifest.assignments[int(row)].block_id for row in evaluated]
    values = summarize(
        errors,
        country_errors,
        block_ids,
        plan,
        alpha,
        raw if probe_target == "latlon" else vectors_to_coordinates(predictions),
        coordinates[evaluated],
    )
    return values, PlaceErrors(
        rows=evaluated,
        block_ids=block_ids,
        predicted_coordinates=vectors_to_coordinates(predictions),
        raw_outputs=raw,
        country_coordinates=vectors_to_coordinates(country),
        errors_km=errors,
        country_errors_km=country_errors,
    )


@jaxtyped(typechecker=beartype)
def score_layer(
    features: ProbeFeatures,
    targets: ProbeTargets,
    coordinates: CoordinateMatrix,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    mode: EvaluationMode,
    control: LayerControl,
    probe_target: ProbeTarget,
) -> LayerScoreValues:
    return evaluate_layer(
        features,
        targets,
        coordinates,
        country_codes,
        manifest,
        plan,
        mode,
        control,
        probe_target,
    )[0]


def block_scores(places: PlaceErrors) -> list[BlockScore]:
    groups: dict[str, list[int]] = {}
    for position, block_id in enumerate(places.block_ids):
        groups.setdefault(block_id, []).append(position)
    return [
        BlockScore(
            block_id=block_id,
            items=len(groups[block_id]),
            mean_km=float(places.errors_km[groups[block_id]].mean()),
            country_mean_km=float(places.country_errors_km[groups[block_id]].mean()),
        )
        for block_id in sorted(groups)
    ]


def prediction_arrays(
    places: PlaceErrors, split_kind: SplitKind, seed: int, probe_target: ProbeTarget
) -> dict[str, np.ndarray]:
    prefix = f"{split_kind}/{seed}/{probe_target}"
    return {
        f"{prefix}/rows": places.rows,
        f"{prefix}/block_ids": np.asarray(places.block_ids),
        f"{prefix}/predicted_coordinates": places.predicted_coordinates,
        f"{prefix}/raw_outputs": places.raw_outputs,
        f"{prefix}/country_coordinates": places.country_coordinates,
        f"{prefix}/errors_km": places.errors_km,
        f"{prefix}/country_errors_km": places.country_errors_km,
    }


def prediction_path(plan: LayerPlan, mode: EvaluationMode, cell: FinalCell) -> Path:
    return (
        plan.output_directory
        / mode
        / "predictions"
        / cell.model_id
        / f"{cell.dataset}-{cell.condition}-{cell.pooling}-{cell.layer}.npz"
    )


def prediction_cells(
    plan: LayerPlan, model_id: str, mode: EvaluationMode
) -> set[tuple[LayerDataset, TextCondition, Pooling, int]]:
    cells = plan.final_cells if mode == "final" else plan.prediction_cells
    return {
        (cell.dataset, cell.condition, cell.pooling, cell.layer)
        for cell in cells
        if cell.model_id == model_id
    }


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


@jaxtyped(typechecker=beartype)
def score_country_control(
    targets: ProbeTargets,
    coordinates: CoordinateMatrix,
    country_codes: list[str],
    manifest: SplitManifest,
    plan: LayerPlan,
    mode: EvaluationMode,
) -> LayerScoreValues:
    evaluated = evaluated_indices(manifest, mode)
    predictions = country_control_predictions(targets, country_codes, manifest, mode)
    errors = angular_errors(
        predictions, targets[evaluated], plan.geometry.earth_radius_km
    )
    return summarize(
        errors,
        errors,
        [manifest.assignments[int(row)].block_id for row in evaluated],
        plan,
        None,
        vectors_to_coordinates(predictions),
        coordinates[evaluated],
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


def write_rows(
    path: Path, rows: list[LayerProbeScore] | list[LayerBlockScores]
) -> None:
    write_atomic(path, "".join(row.model_dump_json() + "\n" for row in rows).encode())


def split_record(manifest: SplitManifest) -> bytes:
    return manifest.model_dump_json(indent=2).encode() + b"\n"


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
    checksum = plan_checksum(config_path)
    rows: list[LayerProbeScore] = []
    block_rows: list[LayerBlockScores] = []
    predicted_cells = prediction_cells(plan, model_id, mode)
    predictions: dict[Path, dict[str, np.ndarray]] = {}
    split_records: dict[Path, SplitManifest] = {}
    for dataset, conditions in plan.datasets.items():
        items_by_condition = {
            condition: dataset_items(dataset, condition) for condition in conditions
        }
        reference = items_by_condition[conditions[0]]
        coordinates = np.asarray(
            [[place.latitude, place.longitude] for place in reference.coordinates],
            dtype=np.float64,
        )
        targets = spherical_vectors(coordinates)
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
                                    coordinates,
                                    reference.country_codes,
                                    manifest,
                                    plan,
                                    mode,
                                ).model_dump(),
                            )
                        )
                for layer in layers:
                    layer_features = np.asarray(features[layer], dtype=np.float64)
                    cell_arrays: dict[str, np.ndarray] | None = None
                    if (dataset, condition, pooling, layer) in predicted_cells:
                        cell_arrays = {
                            "item_ids": np.asarray(reference.item_ids),
                            "true_coordinates": coordinates,
                            "country_codes": np.asarray(reference.country_codes),
                        }
                        predictions[
                            prediction_path(
                                plan,
                                mode,
                                FinalCell(
                                    model_id=model_id,
                                    dataset=dataset,
                                    condition=condition,
                                    pooling=pooling,
                                    layer=layer,
                                ),
                            )
                        ] = cell_arrays
                    for (kind, seed), manifest in manifests.items():
                        for probe_target in plan.probe_targets:
                            controls: list[LayerControl] = (
                                ["probe", "permuted-labels"]
                                if probe_target == "sphere"
                                else ["probe"]
                            )
                            for control in controls:
                                values, places = evaluate_layer(
                                    layer_features,
                                    targets,
                                    coordinates,
                                    reference.country_codes,
                                    manifest,
                                    plan,
                                    mode,
                                    control,
                                    probe_target,
                                )
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
                                        probe_target=probe_target,
                                        **values.model_dump(),
                                    )
                                )
                                block_rows.append(
                                    LayerBlockScores(
                                        **identity.model_dump(),
                                        pooling=pooling,
                                        layer=layer,
                                        split_kind=kind,
                                        seed=seed,
                                        control=control,
                                        probe_target=probe_target,
                                        blocks=block_scores(places),
                                    )
                                )
                                if cell_arrays is not None and control == "probe":
                                    cell_arrays.update(
                                        prediction_arrays(
                                            places, kind, seed, probe_target
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
        write_atomic(path, split_record(manifest))
    for path, arrays in predictions.items():
        write_atomic(path, npz_bytes(arrays))
    write_rows(
        plan.output_directory / mode / "blocks" / f"{model_id}.jsonl", block_rows
    )
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
