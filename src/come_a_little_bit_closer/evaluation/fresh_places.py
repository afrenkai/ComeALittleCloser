import argparse
from pathlib import Path

import numpy as np

from come_a_little_bit_closer.evaluation.fix_baselines import (
    blend,
    feuerstein_positions,
    fit_feuerstein,
)
from come_a_little_bit_closer.evaluation.fix_schemas import (
    FixCollection,
    FixMethod,
    FixPlan,
    FreshClaim,
    FreshPart,
    FreshPlan,
    PlanPair,
    ScoredSet,
    ScoreMatrix,
    TownScore,
)
from come_a_little_bit_closer.evaluation.geo_fix import (
    component_scores,
    country_reflections,
    fit_fix,
    plain_cities,
    unit,
)
from come_a_little_bit_closer.evaluation.instructed_extract import (
    read_instructed,
    town_source,
)
from come_a_little_bit_closer.evaluation.layer_extract import (
    dataset_items,
    town_items,
)
from come_a_little_bit_closer.evaluation.layer_run import build_split, write_atomic
from come_a_little_bit_closer.evaluation.layer_schemas import (
    CountryReflection,
    DatasetItems,
    LayerPlan,
    SplitKind,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.location_schemas import (
    ProbeSettings,
    SplitManifest,
)
from come_a_little_bit_closer.evaluation.probes import bootstrap_interval
from come_a_little_bit_closer.evaluation.splits import (
    holdout_parts,
    spatial_block,
    split_indices,
)
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import Coordinates


def town_partition(
    count: int, plan: FreshPlan, cap: int | None = None
) -> dict[FreshPart, np.ndarray]:
    check, test = holdout_parts(count, plan.partition_seed, plan.check_fraction, cap)
    return {"check": check, "test": test}


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


def set_collection(scored_set: ScoredSet) -> FixCollection:
    collections: dict[ScoredSet, FixCollection] = {
        "cities": "cities-name-country",
        "towns": "towns-name-country",
        "small-towns": "small-towns-name-country",
    }
    return collections[scored_set]


def variant_features(
    fix: FixPlan,
    generic: FixPlan,
    layers: LayerPlan,
    model_id: str,
    sets: list[ScoredSet],
) -> tuple[dict[str, dict[str, np.ndarray]], list[int]]:
    """Per variant: training-city features and each scored set's features."""
    kept = read_instructed(fix, model_id, "cities-name-country", True)[1].kept_layers
    variants: dict[str, dict[str, np.ndarray]] = {
        "plain": {"cities": plain_cities(fix, layers, model_id, kept)},
        "instructed": {},
        "generic": {},
    }
    sources = {
        "plain": (fix, False),
        "instructed": (fix, True),
        "generic": (generic, True),
    }
    for variant, (plan, instructed) in sources.items():
        for scored_set in ["cities", *sets]:
            if scored_set in variants[variant]:
                continue
            array, manifest = read_instructed(
                plan, model_id, set_collection(scored_set), instructed
            )
            if manifest.kept_layers != kept:
                raise ValueError("Scored-set and city features keep different layers.")
            variants[variant][scored_set] = array
    return variants, kept


def town_score(
    identity: tuple[str, FreshPart, ScoredSet, SplitKind, int],
    method: FixMethod,
    score: ScoreMatrix,
    targets: tuple[np.ndarray, np.ndarray],
    fix: FixPlan,
    fusion_layer: int | None = None,
    blend_weight: float | None = None,
) -> TownScore:
    model_id, part, scored_set, kind, seed = identity
    truth, codes = targets
    countries = country_reflections(
        score, truth, codes, fix.minimum_cities, fix.neighbors
    )
    return TownScore(
        model_id=model_id,
        part=part,
        scored_set=scored_set,
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


def plan_pairs(plan: FreshPlan) -> list[PlanPair]:
    return [
        PlanPair(
            fix_plan=plan.fix_plan, generic_plan=plan.generic_plan, sets=plan.sets
        ),
        *plan.extra_pairs,
    ]


def scored_places(plan: FreshPlan) -> dict[ScoredSet, DatasetItems]:
    """Items of every scored set, with town checksums checked against the plan."""
    sets = {scored_set for pair in plan_pairs(plan) for scored_set in pair.sets}
    places = {"cities": dataset_items("cities", "name-country")}
    expected = {"towns": plan.towns_sha256, "small-towns": plan.small_towns_sha256}
    for scored_set in sets - {"cities"}:
        items = town_items(town_source(set_collection(scored_set)))
        if items.source_sha256 != expected[scored_set]:
            raise ValueError(f"{scored_set} differ from the plan's checksum.")
        places[scored_set] = items
    return places


def run_fresh(config: Path, part: FreshPart) -> Path:
    plan = FreshPlan.model_validate_json(config.read_bytes())
    output = plan.output_directory / part
    if part == "test":
        if (output / "scores.jsonl").exists():
            raise FileExistsError("Fresh test results already exist; test runs once.")
        if not (plan.output_directory / "check" / "scores.jsonl").exists():
            raise FileNotFoundError("Run the check part before the test part.")
    places = scored_places(plan)
    cities = places["cities"]
    city_latlon = np.asarray(
        [[point.latitude, point.longitude] for point in cities.coordinates]
    )
    city_truth = spherical_vectors(city_latlon).astype(np.float32)
    city_codes = np.asarray(cities.country_codes)
    truths = {
        scored_set: (
            spherical_vectors(
                np.asarray(
                    [[point.latitude, point.longitude] for point in items.coordinates]
                )
            ).astype(np.float32),
            np.asarray(items.country_codes),
        )
        for scored_set, items in places.items()
    }
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
    model_ids: list[str] = []
    split_settings = set()
    for pair in plan_pairs(plan):
        fix = FixPlan.model_validate_json(pair.fix_plan.read_bytes())
        generic = FixPlan.model_validate_json(pair.generic_plan.read_bytes())
        if [model.model_id for model in generic.models] != [
            model.model_id for model in fix.models
        ]:
            raise ValueError("The generic plan must list the fix plan's models.")
        layers = LayerPlan.model_validate_json(fix.layers_plan.read_bytes())
        split_settings.add(
            (
                tuple(layers.split_seeds),
                layers.block_degrees,
                layers.train_fraction,
                layers.validation_fraction,
            )
        )
        if len(split_settings) > 1:
            raise ValueError("Plan pairs must share split settings.")
        part_rows = {
            scored_set: town_partition(
                len(places[scored_set].item_ids),
                plan,
                plan.test_caps.get(scored_set),
            )[part]
            for scored_set in pair.sets
            if scored_set != "cities"
        }
        for model in fix.models:
            model_ids.append(model.model_id)
            features, kept = variant_features(
                fix, generic, layers, model.model_id, pair.sets
            )
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
                    rows = {
                        scored_set: (
                            split_indices(
                                split, "validation" if part == "check" else "test"
                            )
                            if scored_set == "cities"
                            else scored_town_rows(
                                kind,
                                split,
                                part_rows[scored_set],
                                places[scored_set].coordinates,
                            )
                        )
                        for scored_set in pair.sets
                    }
                    targets = {
                        scored_set: (
                            truths[scored_set][0][rows[scored_set]],
                            truths[scored_set][1][rows[scored_set]],
                        )
                        for scored_set in pair.sets
                    }
                    plain_cosine: dict[str, np.ndarray] = {}
                    for variant, arrays in features.items():
                        fitted = fit_fix(
                            arrays["cities"],
                            train_rows,
                            city_truth,
                            city_codes,
                            fix,
                            seed,
                        )
                        for scored_set in pair.sets:
                            identity = (model.model_id, part, scored_set, kind, seed)
                            scores = component_scores(
                                arrays[scored_set], rows[scored_set], fitted, fix
                            )
                            rows_written += [
                                town_score(
                                    identity,
                                    method,
                                    scores[component],
                                    targets[scored_set],
                                    fix,
                                    (
                                        kept[fitted.layer]
                                        if "fusion" in component
                                        else None
                                    ),
                                )
                                for method, component in variant_methods[variant]
                            ]
                            if variant == "plain":
                                plain_cosine[scored_set] = scores["cosine"]
                    plain = features["plain"]
                    for method, groups in [
                        ("feuerstein-global", np.zeros(len(train_rows), dtype=int)),
                        ("feuerstein-country", city_codes[train_rows]),
                    ]:
                        mapping = fit_feuerstein(
                            unit(plain["cities"][-1][train_rows]),
                            city_latlon[train_rows],
                            groups,
                        )
                        for scored_set in pair.sets:
                            identity = (model.model_id, part, scored_set, kind, seed)
                            positions = feuerstein_positions(
                                unit(plain[scored_set][-1][rows[scored_set]]), mapping
                            )
                            rows_written.append(
                                town_score(
                                    identity,
                                    method,
                                    positions @ positions.T,
                                    targets[scored_set],
                                    fix,
                                )
                            )
                            rows_written += [
                                town_score(
                                    identity,
                                    method,
                                    blend(plain_cosine[scored_set], positions, weight),
                                    targets[scored_set],
                                    fix,
                                    blend_weight=weight,
                                )
                                for weight in plan.feuerstein_weights
                            ]
                    for scored_set in pair.sets:
                        truth = targets[scored_set][0]
                        rows_written.append(
                            town_score(
                                (model.model_id, part, scored_set, kind, seed),
                                "oracle",
                                (
                                    plain_cosine[scored_set]
                                    + fix.oracle_weight * truth @ truth.T
                                )
                                / (1 + fix.oracle_weight),
                                targets[scored_set],
                                fix,
                                blend_weight=fix.oracle_weight,
                            )
                        )
                    print(
                        f"Scored {model.model_id} {part} {kind} {seed}: "
                        + ", ".join(
                            f"{scored_set} {len(rows[scored_set])}"
                            for scored_set in pair.sets
                        ),
                        flush=True,
                    )
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output / "scores.jsonl",
        "".join(row.model_dump_json() + "\n" for row in rows_written).encode(),
    )
    report_path = output / "report.md"
    write_atomic(
        report_path,
        fresh_report(plan, model_ids, part, rows_written).encode(),
    )
    return report_path


def country_difference(
    first: list[CountryReflection],
    second: list[CountryReflection],
    settings: ProbeSettings,
) -> tuple[float, float, float]:
    other = {row.country_code: row.knn_gain for row in second}
    differences = np.asarray([row.knn_gain - other[row.country_code] for row in first])
    low, high = bootstrap_interval(differences, settings)
    return float(differences.mean()), low, high


def plan_claims(plan: FreshPlan) -> list[FreshClaim]:
    if plan.claims:
        return plan.claims
    return [
        FreshClaim(
            label="G1",
            title="Fix vs cosine",
            first="instruction-fusion-probe",
            second="cosine",
            models=plan.claim_models,
            sets=plan.sets,
        ),
        FreshClaim(
            label="G2",
            title="Geographic vs similarity instruction, both with fusion and probe",
            first="instruction-fusion-probe",
            second="generic-instruction-fusion-probe",
            models=plan.claim_models,
            sets=plan.sets,
        ),
        FreshClaim(
            label="G3",
            title="Fix vs within-country seed-pair axes blended with cosine",
            first="instruction-fusion-probe",
            second="feuerstein-country",
            second_weight=plan.feuerstein_claim_weight,
            models=plan.claim_models,
            sets=plan.sets,
        ),
    ]


def fresh_report(
    plan: FreshPlan, model_ids: list[str], part: FreshPart, rows: list[TownScore]
) -> str:
    settings = LayerPlan.model_validate_json(
        FixPlan.model_validate_json(plan.fix_plan.read_bytes()).layers_plan.read_bytes()
    )
    seeds = sorted({row.seed for row in rows})
    kinds = list(dict.fromkeys(row.split_kind for row in rows))
    index = {
        (
            row.model_id,
            row.scored_set,
            row.split_kind,
            row.seed,
            row.method,
            row.blend_weight,
        ): row
        for row in rows
    }
    method_keys = list(dict.fromkeys((row.method, row.blend_weight) for row in rows))
    model_sets = list(dict.fromkeys((row.model_id, row.scored_set) for row in rows))
    lines = [
        f"# Held-out places, {plan.experiment_id}, {part} part",
        "",
        "Components are fit on the original training cities. Cities are the"
        " original validation (check) or test split; towns (5,000-14,999 people) and"
        " small towns (1,000-4,999) are new GeoNames places. Random-item scores all"
        " places of the part; spatial-block scores only places outside that split's"
        f" training blocks. Means over seeds {seeds}.",
        "",
        "| Model | Set | Split | Method | Weight | Places | Countries | City kNN"
        " | City rho | Fusion layer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_id, scored_set in model_sets:
        for kind in kinds:
            for method, weight in method_keys:
                runs = [
                    index[(model_id, scored_set, kind, seed, method, weight)]
                    for seed in seeds
                ]
                lines.append(
                    f"| {model_id} | {scored_set} | {kind} | {method}"
                    f" | {'' if weight is None else weight}"
                    f" | {runs[0].towns} | {len(runs[0].city_countries)}"
                    f" | {np.mean([run.city_knn_gain for run in runs]):.3f}"
                    f" | {np.mean([run.city_rho for run in runs]):.3f}"
                    f" | {' / '.join(str(run.fusion_layer) for run in runs)} |"
                )
    descriptive = [
        FreshClaim(
            label="descriptive",
            title="Is layer fusion needed?",
            first="instruction-fusion-probe",
            second="instruction-probe",
            models=model_ids,
            sets=["cities", "towns", "small-towns"],
        ),
        FreshClaim(
            label="descriptive",
            title="Is the probe needed?",
            first="instruction-fusion-probe",
            second="instruction-fusion",
            models=model_ids,
            sets=["cities", "towns", "small-towns"],
        ),
    ]
    verdicts = []
    for claim in plan_claims(plan) + descriptive:
        lines += [
            "",
            f"## {claim.label}: {claim.title} ({claim.first} minus {claim.second}"
            f"{'' if claim.second_weight is None else f' w={claim.second_weight}'},"
            " city kNN gain paired by country)",
            "",
            "| Model | Set | Split | Seed | Difference [95% CI] |",
            "| --- | --- | --- | --- | --- |",
        ]
        lows = []
        for model_id, scored_set in model_sets:
            if model_id not in claim.models or scored_set not in claim.sets:
                continue
            for kind in kinds:
                for seed in seeds:
                    one = index[(model_id, scored_set, kind, seed, claim.first, None)]
                    two = index[
                        (
                            model_id,
                            scored_set,
                            kind,
                            seed,
                            claim.second,
                            claim.second_weight,
                        )
                    ]
                    mean, low, high = country_difference(
                        one.city_countries, two.city_countries, settings
                    )
                    lows.append(low)
                    lines.append(
                        f"| {model_id} | {scored_set} | {kind} | {seed}"
                        f" | {mean:+.3f} [{low:+.3f}, {high:+.3f}] |"
                    )
        if claim.label != "descriptive":
            above = sum(low > 0 for low in lows)
            verdict = "Supported" if lows and above == len(lows) else "Not supported"
            verdicts.append(
                f"| {claim.label} | {', '.join(claim.models)} | {', '.join(claim.sets)}"
                f" | {above} of {len(lows)} | {verdict} |"
            )
    lines[4:4] = [
        "## Claims",
        "",
        "| Claim | Models | Sets | Runs with CI above 0 | Verdict |",
        "| --- | --- | --- | --- | --- |",
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
