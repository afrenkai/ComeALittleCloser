from functools import partial

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.evaluation.location_schemas import (
    BlockScores,
    IndexVector,
    PredictionDecoder,
    ProbeFeatures,
    ProbeScore,
    ProbeSettings,
    ProbeTargets,
    RidgeModel,
    RidgeTargets,
    SplitManifest,
    TargetRidgeModel,
    TrainingIndices,
    ValidationIndices,
)
from come_a_little_bit_closer.geometry.schemas import CoordinateMatrix, ScoreVector


@jaxtyped(typechecker=beartype)
def unit_predictions(predictions: ProbeTargets, fallback: ScoreVector) -> ProbeTargets:
    norms = np.linalg.norm(predictions, axis=1, keepdims=True)
    if not np.isfinite(predictions).all() or fallback.shape != (3,):
        raise ValueError(
            "Coordinate predictions require finite vectors and a 3D fallback."
        )
    result = predictions / np.maximum(norms, np.finfo(np.float64).eps)
    result[norms[:, 0] <= np.finfo(np.float64).eps] = fallback
    return result


@jaxtyped(typechecker=beartype)
def angular_errors(
    predictions: ProbeTargets, targets: ProbeTargets, radius_km: float
) -> ScoreVector:
    return radius_km * np.arccos(
        np.clip(np.sum(predictions * targets, axis=1), -1.0, 1.0)
    )


@jaxtyped(typechecker=beartype)
def block_means(values: ScoreVector, block_ids: list[str]) -> BlockScores:
    if len(values) != len(block_ids) or not len(values):
        raise ValueError("Block aggregation requires aligned, nonempty samples.")
    groups: dict[str, list[float]] = {}
    for index, block_id in enumerate(block_ids):
        groups.setdefault(block_id, []).append(float(values[index]))
    return np.asarray(
        [np.mean(groups[block]) for block in sorted(groups)], dtype=np.float64
    )


@jaxtyped(typechecker=beartype)
def bootstrap_interval(values: ScoreVector, plan: ProbeSettings) -> tuple[float, float]:
    generator = np.random.default_rng(plan.bootstrap_seed)
    draws = generator.integers(0, len(values), (plan.bootstrap_replicates, len(values)))
    limits = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return float(limits[0]), float(limits[1])


@jaxtyped(typechecker=beartype)
def coordinate_centroid(targets: ProbeTargets) -> ScoreVector:
    mean = np.mean(targets, axis=0)
    norm = np.linalg.norm(mean)
    return (
        mean / norm if norm > np.finfo(np.float64).eps else np.asarray([1.0, 0.0, 0.0])
    )


@jaxtyped(typechecker=beartype)
def country_predictions(
    targets: ProbeTargets, country_codes: list[str], train_indices: IndexVector
) -> ProbeTargets:
    fallback = coordinate_centroid(targets[train_indices])
    groups: dict[str, list[int]] = {}
    for train_index in train_indices:
        code = country_codes[int(train_index)]
        if code:
            groups.setdefault(code, []).append(int(train_index))
    centroids = {
        code: coordinate_centroid(targets[indices]) for code, indices in groups.items()
    }
    return np.asarray(
        [centroids.get(code, fallback) for code in country_codes], dtype=np.float64
    )


@jaxtyped(typechecker=beartype)
def predict_ridge(model: RidgeModel, features: ProbeFeatures) -> ProbeTargets:
    return (features - model.feature_mean) @ model.weights + model.target_mean


@jaxtyped(typechecker=beartype)
def predict_targets(model: TargetRidgeModel, features: ProbeFeatures) -> RidgeTargets:
    return (features - model.feature_mean) @ model.weights + model.target_mean


@jaxtyped(typechecker=beartype)
def fit_ridge_path(
    features: ProbeFeatures,
    targets: RidgeTargets,
    unit_targets: ProbeTargets,
    train_indices: TrainingIndices,
    validation_indices: ValidationIndices,
    block_ids: list[str],
    decoder: PredictionDecoder,
    plan: ProbeSettings,
    center_targets: bool = True,
) -> TargetRidgeModel:
    """Selects alpha by validation block-mean km of decoded unit-vector predictions."""
    if not np.isfinite(features).all() or not all(
        alpha > 0 for alpha in plan.ridge_alphas
    ):
        raise ValueError(
            "Ridge requires finite features and strictly positive penalties."
        )
    training_features = features[train_indices]
    feature_mean = training_features.mean(axis=0)
    target_mean = (
        targets[train_indices].mean(axis=0)
        if center_targets
        else np.zeros(targets.shape[1])
    )
    training = training_features - feature_mean
    eigenvalues, eigenvectors = np.linalg.eigh(training.T @ training)
    transformed_targets = eigenvectors.T @ (
        training.T @ (targets[train_indices] - target_mean)
    )
    centered_validation = features[validation_indices] - feature_mean
    validation_targets = unit_targets[validation_indices]
    validation_blocks = [block_ids[int(index)] for index in validation_indices]
    best_score = float("inf")
    best_model: TargetRidgeModel | None = None
    for alpha in plan.ridge_alphas:
        weights = eigenvectors @ (
            transformed_targets / (np.maximum(eigenvalues, 0) + alpha)[:, None]
        )
        candidate = TargetRidgeModel(feature_mean, target_mean, weights, alpha)
        predictions = decoder(centered_validation @ weights + target_mean)
        errors = angular_errors(
            predictions, validation_targets, plan.geometry.earth_radius_km
        )
        score = float(block_means(errors, validation_blocks).mean())
        if score < best_score:
            best_score, best_model = score, candidate
    if best_model is None:
        raise ValueError("Ridge alpha grid must not be empty.")
    return best_model


@jaxtyped(typechecker=beartype)
def fit_ridge(
    features: ProbeFeatures,
    targets: ProbeTargets,
    train_indices: TrainingIndices,
    validation_indices: ValidationIndices,
    block_ids: list[str],
    plan: ProbeSettings,
) -> RidgeModel:
    fallback = coordinate_centroid(targets[train_indices])
    model = fit_ridge_path(
        features,
        targets,
        targets,
        train_indices,
        validation_indices,
        block_ids,
        partial(unit_predictions, fallback=fallback),
        plan,
    )
    return RidgeModel(model.feature_mean, model.target_mean, model.weights, model.alpha)


@jaxtyped(typechecker=beartype)
def vectors_to_coordinates(vectors: ProbeTargets) -> CoordinateMatrix:
    return np.column_stack(
        (
            np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0))),
            np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])),
        )
    )


@jaxtyped(typechecker=beartype)
def r_squared(predicted: ScoreVector, truth: ScoreVector) -> float:
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total <= 0:
        raise ValueError("R-squared requires variation in the evaluated targets.")
    return 1.0 - float(np.sum((truth - predicted) ** 2)) / total


@jaxtyped(typechecker=beartype)
def score_probe(
    predictions: ProbeTargets,
    targets: ProbeTargets,
    country_errors: ScoreVector,
    manifest: SplitManifest,
    plan: ProbeSettings,
    model_id: str,
    condition: str,
    control: str,
    dimensions: int,
    selected_alpha: float | None,
) -> tuple[ProbeScore, ScoreVector]:
    test_indices = np.asarray(
        [
            index
            for index, item in enumerate(manifest.assignments)
            if item.split == "test"
        ],
        dtype=np.int64,
    )
    test_blocks = [manifest.assignments[int(index)].block_id for index in test_indices]
    errors = angular_errors(
        predictions[test_indices], targets[test_indices], plan.geometry.earth_radius_km
    )
    means = block_means(errors, test_blocks)
    improvement = block_means(country_errors - errors, test_blocks)
    ci_low, ci_high = bootstrap_interval(means, plan)
    gain_low, gain_high = bootstrap_interval(improvement, plan)
    return (
        ProbeScore(
            model_id=model_id,
            condition=condition,
            control=control,
            seed=manifest.seed,
            dimensions=dimensions,
            train_stations=sum(item.split == "train" for item in manifest.assignments),
            validation_stations=sum(
                item.split == "validation" for item in manifest.assignments
            ),
            test_stations=len(test_indices),
            test_blocks=len(means),
            selected_alpha=selected_alpha,
            block_mean_km=float(means.mean()),
            block_mean_ci_low=ci_low,
            block_mean_ci_high=ci_high,
            median_km=float(np.median(errors)),
            p90_km=float(np.quantile(errors, 0.9)),
            improvement_over_country_km=float(improvement.mean()),
            improvement_ci_low=gain_low,
            improvement_ci_high=gain_high,
        ),
        errors,
    )
