import argparse
from pathlib import Path

import numpy as np

from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.evaluation.fix_schemas import (
    CountrySearch,
    FittedFix,
    FixCollection,
    FixPlan,
    FreshPart,
    ScoreMatrix,
    SearchPlan,
    SearchQuery,
    SearchScore,
    SearchSource,
    SmallSearchSource,
)
from come_a_little_bit_closer.evaluation.geo_fix import (
    centered_unit,
    fit_fix,
    plain_cities,
    unit,
)
from come_a_little_bit_closer.evaluation.instructed_extract import read_instructed
from come_a_little_bit_closer.evaluation.layer_extract import (
    dataset_items,
    town_items,
)
from come_a_little_bit_closer.evaluation.layer_reflected import kilometres
from come_a_little_bit_closer.evaluation.layer_run import build_split, write_atomic
from come_a_little_bit_closer.evaluation.layer_schemas import LayerPlan, SplitKind
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.location_schemas import (
    ProbeSettings,
    SplitManifest,
)
from come_a_little_bit_closer.evaluation.probes import bootstrap_interval
from come_a_little_bit_closer.evaluation.search_items import (
    catalog_stations,
    located_mask,
    search_queries,
)
from come_a_little_bit_closer.evaluation.splits import spatial_block, split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors


def cross_scores(
    queries: np.ndarray, documents: np.ndarray, fitted: FittedFix, plan: FixPlan
) -> dict[str, ScoreMatrix]:
    """Query-by-document cosine, layer fusion, distance probe and both.

    The probe distance is normalised by its mean over all query-document pairs.
    """
    query_output = unit(queries[-1])
    document_output = unit(documents[-1])
    cosine = query_output @ document_output.T
    query_middle = centered_unit(unit(queries[fitted.layer]), fitted.layer_center)
    document_middle = centered_unit(unit(documents[fitted.layer]), fitted.layer_center)
    fused = (cosine + plan.fusion_weight * query_middle @ document_middle.T) / (
        1 + plan.fusion_weight
    )
    query_points = (query_output - fitted.output_center) @ fitted.probe
    document_points = (document_output - fitted.output_center) @ fitted.probe
    squared = (
        (query_points**2).sum(1)[:, None]
        + (document_points**2).sum(1)[None, :]
        - 2 * query_points @ document_points.T
    )
    distance = plan.distance_weight * squared / squared.mean()
    return {
        "cosine": cosine,
        "fusion": fused,
        "probe": cosine - distance,
        "fusion-probe": fused - distance,
    }


def query_pools(
    queries: list[SearchQuery], stations: list[Station]
) -> list[np.ndarray]:
    """Stations of the query's genre in the query's country."""
    return [
        np.asarray(
            [
                index
                for index, station in enumerate(stations)
                if station.country_code == query.country_code
                and query.genre in station.genre_tags
            ],
            dtype=np.int64,
        )
        for query in queries
    ]


def search_gains(
    score: ScoreMatrix,
    pools: list[np.ndarray],
    query_truth: np.ndarray,
    station_truth: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """Per query: 1 - mean km to the top-ranked pool stations / mean km to the pool."""
    gains = []
    for row, pool in enumerate(pools):
        distances = kilometres(station_truth[pool] @ query_truth[row])
        top = np.argsort(-score[row, pool], kind="stable")[:neighbors]
        gains.append(1 - distances[top].mean() / distances.mean())
    return np.asarray(gains)


def search_score(
    identity: tuple[str, FreshPart, SplitKind, int],
    method: str,
    gains: np.ndarray,
    codes: np.ndarray,
) -> SearchScore:
    model_id, part, kind, seed = identity
    countries = [
        CountrySearch(
            country_code=str(code),
            queries=int((codes == code).sum()),
            gain=float(gains[codes == code].mean()),
        )
        for code in np.unique(codes)
    ]
    return SearchScore(
        model_id=model_id,
        part=part,
        split_kind=kind,
        seed=seed,
        method=method,
        queries=len(gains),
        search_gain=float(np.mean([country.gain for country in countries])),
        countries=countries,
    )


def scored_queries(
    kind: SplitKind, split: SplitManifest, queries: list[SearchQuery]
) -> np.ndarray:
    """All queries; for spatial-block, only those whose town lies outside training
    blocks."""
    if kind == "random-item":
        return np.arange(len(queries))
    training_blocks = {
        item.block_id for item in split.assignments if item.split == "train"
    }
    return np.asarray(
        [
            index
            for index, query in enumerate(queries)
            if spatial_block(query.latitude, query.longitude, split.block_degrees)
            not in training_blocks
        ],
        dtype=np.int64,
    )


def run_search(config: Path, part: FreshPart) -> Path:
    plan = SearchPlan.model_validate_json(config.read_bytes())
    output = plan.output_directory / part
    if part == "test":
        if (output / "scores.jsonl").exists():
            raise FileExistsError("Search test results already exist; test runs once.")
        if not (plan.output_directory / "check" / "scores.jsonl").exists():
            raise FileNotFoundError("Run the check part before the test part.")
    source = SmallSearchSource() if plan.origins == "small-towns" else SearchSource()
    if town_items(source.towns).source_sha256 != plan.towns_sha256:
        raise ValueError("Fresh towns differ from the plan's checksum.")
    stations = catalog_stations()
    queries = search_queries(stations, source, part)
    query_collections: dict[tuple[str, FreshPart], FixCollection] = {
        ("towns", "check"): "search-queries-check",
        ("towns", "test"): "search-queries-test",
        ("small-towns", "check"): "small-search-queries-check",
        ("small-towns", "test"): "small-search-queries-test",
    }
    collection = query_collections[(plan.origins, part)]
    pools = query_pools(queries, stations)
    if plan.exclude_mislocated or source.exclude_mislocated:
        located = located_mask(stations)
        pools = [pool[located[pool]] for pool in pools]
    station_truth = spherical_vectors(
        np.asarray(
            [
                [station.coordinates.latitude, station.coordinates.longitude]
                for station in stations
            ]
        )
    ).astype(np.float32)
    query_truth = spherical_vectors(
        np.asarray([[query.latitude, query.longitude] for query in queries])
    ).astype(np.float32)
    query_codes = np.asarray([query.country_code for query in queries])
    cities = dataset_items("cities", "name-country")
    city_truth = spherical_vectors(
        np.asarray([[point.latitude, point.longitude] for point in cities.coordinates])
    ).astype(np.float32)
    city_codes = np.asarray(cities.country_codes)
    rows_written: list[SearchScore] = []
    for pair in plan.pairs:
        fix = FixPlan.model_validate_json(pair.fix_plan.read_bytes())
        search = FixPlan.model_validate_json(pair.search_plan.read_bytes())
        layers = LayerPlan.model_validate_json(fix.layers_plan.read_bytes())
        searchable = {model.model_id for model in search.models}
        for model in fix.models:
            instructed_cities, manifest = read_instructed(
                fix, model.model_id, "cities-name-country", True
            )
            kept = manifest.kept_layers
            features = {
                "plain": (
                    plain_cities(fix, layers, model.model_id, kept),
                    read_instructed(fix, model.model_id, collection, False)[0],
                    read_instructed(fix, model.model_id, "search-documents", False)[0],
                ),
                "instructed": (
                    instructed_cities,
                    read_instructed(fix, model.model_id, collection, True)[0],
                    read_instructed(fix, model.model_id, "search-documents", True)[0],
                ),
            }
            standard_queries = (
                read_instructed(search, model.model_id, collection, True)[0]
                if model.model_id in searchable
                else None
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
                    rows = scored_queries(kind, split, queries)
                    identity = (model.model_id, part, kind, seed)
                    row_pools = [pools[row] for row in rows]

                    def gains_of(score: ScoreMatrix) -> np.ndarray:
                        return search_gains(
                            score,
                            row_pools,
                            query_truth[rows],
                            station_truth,
                            fix.neighbors,
                        )

                    methods: dict[str, np.ndarray] = {
                        "oracle": gains_of(query_truth[rows] @ station_truth.T)
                    }
                    for variant, (
                        city_array,
                        query_array,
                        document_array,
                    ) in features.items():
                        fitted = fit_fix(
                            city_array, train_rows, city_truth, city_codes, fix, seed
                        )
                        scores = cross_scores(
                            query_array[:, rows], document_array, fitted, fix
                        )
                        prefix = "plain" if variant == "plain" else "instruction"
                        methods[f"{prefix}-cosine"] = gains_of(scores["cosine"])
                        methods[f"{prefix}-fusion-probe"] = gains_of(
                            scores["fusion-probe"]
                        )
                        if variant == "plain" and standard_queries is not None:
                            standard = unit(standard_queries[-1][rows])
                            methods["standard"] = gains_of(
                                standard @ unit(document_array[-1]).T
                            )
                    rows_written += [
                        search_score(identity, method, gains, query_codes[rows])
                        for method, gains in methods.items()
                    ]
                    print(
                        f"Searched {model.model_id} {part} {kind} {seed}:"
                        f" {len(rows)} queries",
                        flush=True,
                    )
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output / "scores.jsonl",
        "".join(row.model_dump_json() + "\n" for row in rows_written).encode(),
    )
    report_path = output / "report.md"
    settings = LayerPlan.model_validate_json(
        FixPlan.model_validate_json(
            plan.pairs[0].fix_plan.read_bytes()
        ).layers_plan.read_bytes()
    )
    write_atomic(
        report_path, search_report(plan, part, rows_written, settings).encode()
    )
    return report_path


def country_gain_difference(
    first: SearchScore, second: SearchScore, settings: ProbeSettings
) -> tuple[float, float, float]:
    other = {row.country_code: row.gain for row in second.countries}
    differences = np.asarray(
        [row.gain - other[row.country_code] for row in first.countries]
    )
    low, high = bootstrap_interval(differences, settings)
    return float(differences.mean()), low, high


def search_report(
    plan: SearchPlan, part: FreshPart, rows: list[SearchScore], settings: ProbeSettings
) -> str:
    index = {(row.model_id, row.split_kind, row.seed, row.method): row for row in rows}
    models = list(dict.fromkeys(row.model_id for row in rows))
    kinds = list(dict.fromkeys(row.split_kind for row in rows))
    seeds = sorted({row.seed for row in rows})
    methods = list(dict.fromkeys(row.method for row in rows))
    lines = [
        f"# Station search, {plan.experiment_id}, {part} part",
        "",
        "Query: `{genre} radio stations near {town}, {country}`. Pool: stations of the"
        " genre in the town's country. Search gain: 1 - mean km to the 10 top-ranked"
        " pool stations / mean km to the pool, averaged per country, then over"
        f" countries. Means over seeds {seeds}.",
        *(
            [
                "",
                "Descriptive sensitivity: pools exclude stations whose nearest"
                " cities15000 place lies in another country than their catalog"
                " country.",
            ]
            if plan.exclude_mislocated
            else []
        ),
        "",
        "| Model | Split | Method | Queries | Countries | Search gain |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_id in models:
        for kind in kinds:
            for method in methods:
                runs = [
                    index[(model_id, kind, seed, method)]
                    for seed in seeds
                    if (model_id, kind, seed, method) in index
                ]
                if not runs:
                    continue
                lines.append(
                    f"| {model_id} | {kind} | {method} | {runs[0].queries}"
                    f" | {len(runs[0].countries)}"
                    f" | {np.mean([run.search_gain for run in runs]):.3f} |"
                )
    comparisons = [
        (plan.claim_label, "instruction-fusion-probe", "standard"),
        ("descriptive", "instruction-cosine", "standard"),
        ("descriptive", "instruction-fusion-probe", "instruction-cosine"),
        ("descriptive", "plain-fusion-probe", "plain-cosine"),
    ]
    verdicts = []
    for label, first, second in comparisons:
        lines += [
            "",
            f"## {label}: {first} minus {second} (search gain paired by country)",
            "",
            "| Model | Split | Seed | Difference [95% CI] |",
            "| --- | --- | --- | --- |",
        ]
        for model_id in models:
            lows = []
            for kind in kinds:
                for seed in seeds:
                    if (model_id, kind, seed, second) not in index:
                        continue
                    mean, low, high = country_gain_difference(
                        index[(model_id, kind, seed, first)],
                        index[(model_id, kind, seed, second)],
                        settings,
                    )
                    lows.append(low)
                    lines.append(
                        f"| {model_id} | {kind} | {seed}"
                        f" | {mean:+.3f} [{low:+.3f}, {high:+.3f}] |"
                    )
            if label == plan.claim_label and model_id in plan.claim_models:
                above = sum(low > 0 for low in lows)
                verdict = (
                    "Supported" if lows and above == len(lows) else "Not supported"
                )
                verdicts.append(
                    f"| {label} {model_id} | {above} of {len(lows)} | {verdict} |"
                )
    if verdicts:
        lines[4:4] = [
            "## Claims",
            "",
            "| Claim | Runs with CI above 0 | Verdict |",
            "| --- | --- | --- |",
            *verdicts,
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the station search task.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part", choices=["check", "test"], required=True)
    arguments = parser.parse_args()
    print(run_search(arguments.config, arguments.part), flush=True)


if __name__ == "__main__":
    main()
