import argparse
from pathlib import Path

import numpy as np

from come_a_little_bit_closer.evaluation.fix_schemas import (
    BaselinePlan,
    FeuersteinMap,
    FixMethod,
    FixPlan,
    FixScore,
    FixTargets,
    ScoreMatrix,
    UnitMatrix,
)
from come_a_little_bit_closer.evaluation.geo_fix import (
    component_scores,
    fit_fix,
    fix_score,
    model_features,
    station_genres,
    sts_text,
    unit,
)
from come_a_little_bit_closer.evaluation.instructed_extract import read_sts
from come_a_little_bit_closer.evaluation.layer_extract import dataset_items
from come_a_little_bit_closer.evaluation.layer_run import (
    build_split,
    evaluated_indices,
    write_atomic,
)
from come_a_little_bit_closer.evaluation.layer_schemas import LayerPlan
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.probes import bootstrap_interval
from come_a_little_bit_closer.evaluation.splits import split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors


def feuerstein_axis(
    vectors: UnitMatrix, values: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Feuerstein et al. (2026), Eq. 2, with every training item in a seed pair.

    Within each group, items sorted by value pair upper half against lower half,
    so the summed pair differences equal the upper sum minus the lower sum.
    """
    direction = np.zeros(vectors.shape[1], dtype=np.float64)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        half = len(members) // 2
        if half == 0:
            continue
        order = members[np.argsort(values[members])]
        direction += vectors[order[len(order) - half :]].sum(axis=0)
        direction -= vectors[order[:half]].sum(axis=0)
    return (direction / np.linalg.norm(direction)).astype(np.float32)


def fit_feuerstein(
    vectors: UnitMatrix, latlon: np.ndarray, groups: np.ndarray
) -> FeuersteinMap:
    axes = np.stack(
        [
            feuerstein_axis(vectors, latlon[:, 0], groups),
            feuerstein_axis(vectors, latlon[:, 1], groups),
        ]
    )
    projections = vectors @ axes.T
    return FeuersteinMap(
        axes=axes,
        projection_low=projections.min(axis=0),
        projection_high=projections.max(axis=0),
        coordinate_low=latlon.min(axis=0),
        coordinate_high=latlon.max(axis=0),
    )


def feuerstein_positions(vectors: UnitMatrix, mapping: FeuersteinMap) -> UnitMatrix:
    """Axis scores min-max scaled to the training coordinate range, as in their maps."""
    scaled = (vectors @ mapping.axes.T - mapping.projection_low) / (
        mapping.projection_high - mapping.projection_low
    )
    latlon = mapping.coordinate_low + scaled * (
        mapping.coordinate_high - mapping.coordinate_low
    )
    latlon[:, 0] = np.clip(latlon[:, 0], -90, 90)
    return spherical_vectors(latlon).astype(np.float32)


def blend(cosine: ScoreMatrix, positions: UnitMatrix, weight: float) -> ScoreMatrix:
    return (cosine + weight * positions @ positions.T) / (1 + weight)


def paired_difference(
    first: FixScore, second: FixScore, layers: LayerPlan
) -> tuple[float, float, float]:
    other = {row.country_code: row.knn_gain for row in second.city_countries}
    differences = np.asarray(
        [row.knn_gain - other[row.country_code] for row in first.city_countries]
    )
    low, high = bootstrap_interval(differences, layers)
    return float(differences.mean()), low, high


def run_baselines(config: Path) -> Path:
    plan = BaselinePlan.model_validate_json(config.read_bytes())
    fix = FixPlan.model_validate_json(plan.fix_plan.read_bytes())
    generic = FixPlan.model_validate_json(plan.generic_plan.read_bytes())
    if generic.layers_plan != fix.layers_plan or [
        model.model_id for model in generic.models
    ] != [model.model_id for model in fix.models]:
        raise ValueError(
            "The generic plan must match the fix plan's layers and models."
        )
    layers = LayerPlan.model_validate_json(fix.layers_plan.read_bytes())
    sts = read_sts(fix.sts_files["dev"], fix.sts_sha256["dev"])
    cities = dataset_items("cities", "name-country")
    stations = dataset_items("stations", "metadata")
    city_latlon = np.asarray(
        [[point.latitude, point.longitude] for point in cities.coordinates]
    )
    city_truth = spherical_vectors(city_latlon).astype(np.float32)
    station_truth = spherical_vectors(
        np.asarray(
            [[point.latitude, point.longitude] for point in stations.coordinates]
        )
    ).astype(np.float32)
    city_codes = np.asarray(cities.country_codes)
    genres = station_genres(stations.texts)
    component_methods: list[tuple[FixMethod, str, str]] = [
        ("cosine", "plain", "cosine"),
        ("fusion", "plain", "fusion"),
        ("probe", "plain", "probe"),
        ("fusion-probe", "plain", "fusion-probe"),
        ("instruction", "instructed", "cosine"),
        ("instruction-fusion", "instructed", "fusion"),
        ("instruction-probe", "instructed", "probe"),
        ("instruction-fusion-probe", "instructed", "fusion-probe"),
        ("generic-instruction", "generic", "cosine"),
        ("generic-instruction-fusion-probe", "generic", "fusion-probe"),
    ]
    rows_written: list[FixScore] = []
    for model in fix.models:
        plain, instructed, kept = model_features(fix, layers, model.model_id, "sts-dev")
        generic_instructed, generic_kept = model_features(
            generic, layers, model.model_id, "sts-dev"
        )[1:]
        if generic_kept != kept:
            raise ValueError("Generic and geographic features keep different layers.")
        variants = {
            "plain": plain,
            "instructed": instructed,
            "generic": generic_instructed,
        }
        for kind in fix.split_kinds:
            for seed in layers.split_seeds:
                split = build_split(
                    kind,
                    cities.item_ids,
                    cities.coordinates,
                    layers,
                    seed,
                    cities.source_sha256,
                    plan_checksum(fix.layers_plan),
                )
                rows = evaluated_indices(split, "dev")
                train_rows = split_indices(split, "train")
                targets = FixTargets(
                    city_truth=city_truth[rows],
                    city_codes=city_codes[rows],
                    station_truth=station_truth,
                    station_codes=np.asarray(stations.country_codes),
                    genres=genres,
                    sts=sts,
                )
                identity = (model.model_id, "dev", kind, seed)
                scored = {}
                fusion_layers = {}
                for variant, arrays in variants.items():
                    fitted = fit_fix(
                        arrays["cities"], train_rows, city_truth, city_codes, fix, seed
                    )
                    scored[variant] = {
                        "cities": component_scores(arrays["cities"], rows, fitted, fix),
                        "stations": component_scores(
                            arrays["stations"], None, fitted, fix
                        ),
                        "sts": component_scores(arrays["sts"], None, fitted, fix),
                    }
                    fusion_layers[variant] = kept[fitted.layer]
                rows_written += [
                    fix_score(
                        identity,
                        method,
                        (
                            scored[variant]["cities"][component],
                            scored[variant]["stations"][component],
                            scored[variant]["sts"][component],
                        ),
                        targets,
                        fix,
                        fusion_layers[variant] if "fusion" in component else None,
                    )
                    for method, variant, component in component_methods
                ]
                outputs = {
                    "cities": unit(plain["cities"][-1][rows]),
                    "stations": unit(plain["stations"][-1]),
                    "sts": unit(plain["sts"][-1]),
                }
                feuerstein_groups: list[tuple[FixMethod, np.ndarray]] = [
                    ("feuerstein-global", np.zeros(len(train_rows), dtype=int)),
                    ("feuerstein-country", city_codes[train_rows]),
                ]
                for method, groups in feuerstein_groups:
                    mapping = fit_feuerstein(
                        unit(plain["cities"][-1][train_rows]),
                        city_latlon[train_rows],
                        groups,
                    )
                    positions = {
                        name: feuerstein_positions(vectors, mapping)
                        for name, vectors in outputs.items()
                    }
                    rows_written.append(
                        fix_score(
                            identity,
                            method,
                            (
                                positions["cities"] @ positions["cities"].T,
                                positions["stations"] @ positions["stations"].T,
                                positions["sts"] @ positions["sts"].T,
                            ),
                            targets,
                            fix,
                        )
                    )
                    rows_written += [
                        fix_score(
                            identity,
                            method,
                            (
                                blend(
                                    scored["plain"]["cities"]["cosine"],
                                    positions["cities"],
                                    weight,
                                ),
                                blend(
                                    scored["plain"]["stations"]["cosine"],
                                    positions["stations"],
                                    weight,
                                ),
                                blend(
                                    scored["plain"]["sts"]["cosine"],
                                    positions["sts"],
                                    weight,
                                ),
                            ),
                            targets,
                            fix,
                            blend_weight=weight,
                        )
                        for weight in plan.feuerstein_weights
                    ]
                print(f"Scored {model.model_id} {kind} {seed}", flush=True)
    output = plan.output_directory / "dev"
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output / "scores.jsonl",
        "".join(row.model_dump_json() + "\n" for row in rows_written).encode(),
    )
    report_path = output / "report.md"
    write_atomic(
        report_path,
        baseline_report(plan, fix, generic, layers, rows_written).encode(),
    )
    return report_path


def baseline_report(
    plan: BaselinePlan,
    fix: FixPlan,
    generic: FixPlan,
    layers: LayerPlan,
    rows: list[FixScore],
) -> str:
    index = {
        (row.model_id, row.split_kind, row.seed, row.method, row.blend_weight): row
        for row in rows
    }
    method_keys = list(dict.fromkeys((row.method, row.blend_weight) for row in rows))
    lines = [
        f"# Fix baselines and ablations, {plan.experiment_id} (validation only)",
        "",
        f'Geographic instruction: "{fix.instruction}". Similarity instruction:'
        f' "{generic.instruction}". Feuerstein rows without a weight rank by the'
        " projected positions alone; with a weight w they score"
        " (cosine + w * position similarity) / (1 + w), like the oracle. Means over"
        f" seeds {layers.split_seeds}.",
        "",
        "| Model | Split | Method | Weight | City kNN | Station kNN | Genre | STS rho"
        " | Fusion layer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in fix.models:
        for kind in fix.split_kinds:
            for method, weight in method_keys:
                runs = [
                    index[(model.model_id, kind, seed, method, weight)]
                    for seed in layers.split_seeds
                ]
                lines.append(
                    f"| {model.model_id} | {kind} | {method}"
                    f" | {'' if weight is None else weight}"
                    f" | {np.mean([run.city_knn_gain for run in runs]):.3f}"
                    f" | {np.mean([run.station_knn_gain for run in runs]):.3f}"
                    f" | {np.mean([run.genre_agreement for run in runs]):.3f}"
                    f" | {sts_text(runs)}"
                    f" | {' / '.join(str(run.fusion_layer) for run in runs)} |"
                )
    comparisons: list[tuple[str, FixMethod, FixMethod]] = [
        ("Is layer fusion needed?", "instruction-fusion-probe", "instruction-probe"),
        ("Is the probe needed?", "instruction-fusion-probe", "instruction-fusion"),
        (
            "Does the geographic wording matter?",
            "instruction-fusion-probe",
            "generic-instruction-fusion-probe",
        ),
        (
            "Instruction vs similarity instruction alone",
            "instruction",
            "generic-instruction",
        ),
    ]
    for title, first, second in comparisons:
        lines += [
            "",
            f"## {title} ({first} minus {second}, city kNN gain paired by country)",
            "",
            "| Model | Split | Seed | Difference [95% CI] | Station kNN change"
            " | Genre change | STS change |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for model in fix.models:
            for kind in fix.split_kinds:
                for seed in layers.split_seeds:
                    one = index[(model.model_id, kind, seed, first, None)]
                    two = index[(model.model_id, kind, seed, second, None)]
                    mean, low, high = paired_difference(one, two, layers)
                    lines.append(
                        f"| {model.model_id} | {kind} | {seed}"
                        f" | {mean:+.3f} [{low:+.3f}, {high:+.3f}]"
                        f" | {one.station_knn_gain - two.station_knn_gain:+.3f}"
                        f" | {one.genre_agreement - two.genre_agreement:+.3f}"
                        f" | {one.sts_spearman - two.sts_spearman:+.3f} |"
                    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only baselines and ablations for the geography fix."
    )
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    print(run_baselines(arguments.config), flush=True)


if __name__ == "__main__":
    main()
