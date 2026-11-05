import argparse
from pathlib import Path

import numpy as np

from come_a_little_bit_closer.data.schemas import TownSource
from come_a_little_bit_closer.evaluation.fix_baselines import (
    blend,
    feuerstein_positions,
    fit_feuerstein,
)
from come_a_little_bit_closer.evaluation.fix_schemas import (
    FixMethod,
    FixPlan,
    FreshPart,
    FreshPlan,
    ScoreMatrix,
    TownScore,
)
from come_a_little_bit_closer.evaluation.geo_fix import (
    component_scores,
    country_reflections,
    fit_fix,
    unit,
)
from come_a_little_bit_closer.evaluation.instructed_extract import read_instructed
from come_a_little_bit_closer.evaluation.layer_extract import (
    dataset_items,
    feature_path,
    town_items,
)
from come_a_little_bit_closer.evaluation.layer_run import build_split, write_atomic
from come_a_little_bit_closer.evaluation.layer_schemas import (
    CountryReflection,
    LayerPlan,
    SplitKind,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.location_schemas import SplitManifest
from come_a_little_bit_closer.evaluation.probes import bootstrap_interval
from come_a_little_bit_closer.evaluation.splits import spatial_block, split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import Coordinates


def town_partition(count: int, plan: FreshPlan) -> dict[FreshPart, np.ndarray]:
    order = np.random.default_rng(plan.partition_seed).permutation(count)
    check_count = int(round(count * plan.check_fraction))
    return {"check": np.sort(order[:check_count]), "test": np.sort(order[check_count:])}


def scored_town_rows(
    kind: SplitKind,
    split: SplitManifest,
    part_rows: np.ndarray,
    coordinates: list[Coordinates],
) -> np.ndarray:
    """All towns of the part; for spatial-block, only those outside training blocks."""
    if kind == "random-item":
        return part_rows
    training_blocks = {
        item.block_id for item in split.assignments if item.split == "train"
    }
    return np.asarray(
        [
            row
            for row in part_rows
            if spatial_block(
                coordinates[row].latitude,
                coordinates[row].longitude,
                split.block_degrees,
            )
            not in training_blocks
        ],
        dtype=np.int64,
    )


def variant_features(
    fix: FixPlan, generic: FixPlan, layers: LayerPlan, model_id: str
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[int]]:
    """(training-city array, town array) per variant, on the instructed kept layers."""
    plain_towns, town_manifest = read_instructed(
        fix, model_id, "towns-name-country", False
    )
    kept = town_manifest.kept_layers
    loaded = {
        "instructed": [
            read_instructed(fix, model_id, "cities-name-country", True),
            read_instructed(fix, model_id, "towns-name-country", True),
        ],
        "generic": [
            read_instructed(generic, model_id, "cities-name-country", True),
            read_instructed(generic, model_id, "towns-name-country", True),
        ],
    }
    if any(
        manifest.kept_layers != kept
        for pairs in loaded.values()
        for array, manifest in pairs
    ):
        raise ValueError("Town and city features keep different layers.")
    plain_cities = np.asarray(
        np.load(
            feature_path(layers, model_id, "cities", "name-country", "last"),
            mmap_mode="r",
        )[kept]
    )
    variants = {"plain": (plain_cities, plain_towns)}
    for variant, pairs in loaded.items():
        variants[variant] = (pairs[0][0], pairs[1][0])
    return variants, kept


def town_score(
    identity: tuple[str, FreshPart, SplitKind, int],
    method: FixMethod,
    score: ScoreMatrix,
    targets: tuple[np.ndarray, np.ndarray],
    fix: FixPlan,
    fusion_layer: int | None = None,
    blend_weight: float | None = None,
) -> TownScore:
    model_id, part, kind, seed = identity
    truth, codes = targets
    countries = country_reflections(
        score, truth, codes, fix.minimum_cities, fix.neighbors
    )
    return TownScore(
        model_id=model_id,
        part=part,
        split_kind=kind,
        seed=seed,
        method=method,
        blend_weight=blend_weight,
        fusion_layer=fusion_layer,
        towns=int(sum(country.cities for country in countries)),
        city_knn_gain=float(np.mean([country.knn_gain for country in countries])),
        city_rho=float(np.mean([country.rho for country in countries])),
        city_countries=countries,
    )


def run_fresh(config: Path, part: FreshPart) -> Path:
    plan = FreshPlan.model_validate_json(config.read_bytes())
    fix = FixPlan.model_validate_json(plan.fix_plan.read_bytes())
    generic = FixPlan.model_validate_json(plan.generic_plan.read_bytes())
    if [model.model_id for model in generic.models] != [
        model.model_id for model in fix.models
    ]:
        raise ValueError("The generic plan must list the fix plan's models.")
    output = plan.output_directory / part
    if part == "test":
        if (output / "scores.jsonl").exists():
            raise FileExistsError("Fresh test results already exist; test runs once.")
        if not (plan.output_directory / "check" / "scores.jsonl").exists():
            raise FileNotFoundError("Run the check part before the test part.")
    layers = LayerPlan.model_validate_json(fix.layers_plan.read_bytes())
    towns = town_items(TownSource())
    if towns.source_sha256 != plan.towns_sha256:
        raise ValueError("Fresh towns differ from the plan's checksum.")
    cities = dataset_items("cities", "name-country")
    city_latlon = np.asarray(
        [[point.latitude, point.longitude] for point in cities.coordinates]
    )
    city_truth = spherical_vectors(city_latlon).astype(np.float32)
    city_codes = np.asarray(cities.country_codes)
    town_truth = spherical_vectors(
        np.asarray([[point.latitude, point.longitude] for point in towns.coordinates])
    ).astype(np.float32)
    town_codes = np.asarray(towns.country_codes)
    part_rows = town_partition(len(towns.item_ids), plan)[part]
    variant_methods: dict[str, list[tuple[FixMethod, str]]] = {
        "plain": [
            ("cosine", "cosine"),
            ("fusion", "fusion"),
            ("probe", "probe"),
            ("fusion-probe", "fusion-probe"),
        ],
        "instructed": [
            ("instruction", "cosine"),
            ("instruction-fusion", "fusion"),
            ("instruction-probe", "probe"),
            ("instruction-fusion-probe", "fusion-probe"),
        ],
        "generic": [
            ("generic-instruction", "cosine"),
            ("generic-instruction-fusion-probe", "fusion-probe"),
        ],
    }
    rows_written: list[TownScore] = []
    for model in fix.models:
        variants, kept = variant_features(fix, generic, layers, model.model_id)
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
                train_rows = split_indices(split, "train")
                rows = scored_town_rows(kind, split, part_rows, towns.coordinates)
                targets = (town_truth[rows], town_codes[rows])
                identity = (model.model_id, part, kind, seed)
                plain_cosine = np.zeros((0, 0), dtype=np.float32)
                for variant, (city_array, town_array) in variants.items():
                    fitted = fit_fix(
                        city_array, train_rows, city_truth, city_codes, fix, seed
                    )
                    scores = component_scores(town_array, rows, fitted, fix)
                    rows_written += [
                        town_score(
                            identity,
                            method,
                            scores[component],
                            targets,
                            fix,
                            kept[fitted.layer] if "fusion" in component else None,
                        )
                        for method, component in variant_methods[variant]
                    ]
                    if variant == "plain":
                        plain_cosine = scores["cosine"]
                rows_written.append(
                    town_score(
                        identity,
                        "oracle",
                        (plain_cosine + fix.oracle_weight * targets[0] @ targets[0].T)
                        / (1 + fix.oracle_weight),
                        targets,
                        fix,
                        blend_weight=fix.oracle_weight,
                    )
                )
                plain_cities, plain_towns = variants["plain"]
                for method, groups in [
                    ("feuerstein-global", np.zeros(len(train_rows), dtype=int)),
                    ("feuerstein-country", city_codes[train_rows]),
                ]:
                    mapping = fit_feuerstein(
                        unit(plain_cities[-1][train_rows]),
                        city_latlon[train_rows],
                        groups,
                    )
                    positions = feuerstein_positions(
                        unit(plain_towns[-1][rows]), mapping
                    )
                    rows_written.append(
                        town_score(
                            identity, method, positions @ positions.T, targets, fix
                        )
                    )
                    rows_written += [
                        town_score(
                            identity,
                            method,
                            blend(plain_cosine, positions, weight),
                            targets,
                            fix,
                            blend_weight=weight,
                        )
                        for weight in plan.feuerstein_weights
                    ]
                print(
                    f"Scored {model.model_id} {part} {kind} {seed}: {len(rows)} towns",
                    flush=True,
                )
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output / "scores.jsonl",
        "".join(row.model_dump_json() + "\n" for row in rows_written).encode(),
    )
    report_path = output / "report.md"
    write_atomic(
        report_path, fresh_report(plan, fix, layers, part, rows_written).encode()
    )
    return report_path


def country_difference(
    first: list[CountryReflection],
    second: list[CountryReflection],
    layers: LayerPlan,
) -> tuple[float, float, float]:
    other = {row.country_code: row.knn_gain for row in second}
    differences = np.asarray([row.knn_gain - other[row.country_code] for row in first])
    low, high = bootstrap_interval(differences, layers)
    return float(differences.mean()), low, high


def fresh_report(
    plan: FreshPlan,
    fix: FixPlan,
    layers: LayerPlan,
    part: FreshPart,
    rows: list[TownScore],
) -> str:
    index = {
        (row.model_id, row.split_kind, row.seed, row.method, row.blend_weight): row
        for row in rows
    }
    method_keys = list(dict.fromkeys((row.method, row.blend_weight) for row in rows))
    lines = [
        f"# Fresh towns, {plan.experiment_id}, {part} part",
        "",
        "GeoNames towns of 5,000-14,999 people that are not in `cities15000`, scored"
        " with components fit on the original training cities. Random-item scores all"
        " towns of the part; spatial-block scores only towns outside that split's"
        f" training blocks. Means over seeds {layers.split_seeds}.",
        "",
        "| Model | Split | Method | Weight | Towns | Countries | City kNN | City rho"
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
                    f" | {runs[0].towns} | {len(runs[0].city_countries)}"
                    f" | {np.mean([run.city_knn_gain for run in runs]):.3f}"
                    f" | {np.mean([run.city_rho for run in runs]):.3f}"
                    f" | {' / '.join(str(run.fusion_layer) for run in runs)} |"
                )
    comparisons: list[tuple[str, str, FixMethod, FixMethod, float | None]] = [
        ("G1", "Fix vs cosine", "instruction-fusion-probe", "cosine", None),
        (
            "G2",
            "Geographic vs similarity instruction, both with fusion and probe",
            "instruction-fusion-probe",
            "generic-instruction-fusion-probe",
            None,
        ),
        (
            "G3",
            "Fix vs within-country seed-pair axes blended with cosine",
            "instruction-fusion-probe",
            "feuerstein-country",
            plan.feuerstein_claim_weight,
        ),
        (
            "descriptive",
            "Is layer fusion needed?",
            "instruction-fusion-probe",
            "instruction-probe",
            None,
        ),
        (
            "descriptive",
            "Is the probe needed?",
            "instruction-fusion-probe",
            "instruction-fusion",
            None,
        ),
    ]
    verdicts = []
    for label, title, first, second, weight in comparisons:
        lines += [
            "",
            f"## {label}: {title} ({first} minus {second}"
            f"{'' if weight is None else f' w={weight}'}, city kNN gain paired by"
            " country)",
            "",
            "| Model | Split | Seed | Difference [95% CI] |",
            "| --- | --- | --- | --- |",
        ]
        claim_lows = []
        for model in fix.models:
            for kind in fix.split_kinds:
                for seed in layers.split_seeds:
                    one = index[(model.model_id, kind, seed, first, None)]
                    two = index[(model.model_id, kind, seed, second, weight)]
                    mean, low, high = country_difference(
                        one.city_countries, two.city_countries, layers
                    )
                    if model.model_id in plan.claim_models:
                        claim_lows.append(low)
                    lines.append(
                        f"| {model.model_id} | {kind} | {seed}"
                        f" | {mean:+.3f} [{low:+.3f}, {high:+.3f}] |"
                    )
        if label != "descriptive":
            above = sum(low > 0 for low in claim_lows)
            verdicts.append(
                f"| {label} | {above} of {len(claim_lows)}"
                f" | {'Supported' if above == len(claim_lows) else 'Not supported'} |"
            )
    lines[4:4] = [
        "## Claims (" + ", ".join(plan.claim_models) + ")",
        "",
        "| Claim | Runs with CI above 0 | Verdict |",
        "| --- | --- | --- |",
        *verdicts,
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the coordinate-free fix on fresh held-out towns."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part", choices=["check", "test"], required=True)
    arguments = parser.parse_args()
    print(run_fresh(arguments.config, arguments.part), flush=True)


if __name__ == "__main__":
    main()
