import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from come_a_little_bit_closer.evaluation.layer_extract import (
    dataset_items,
    feature_path,
)
from come_a_little_bit_closer.evaluation.layer_run import (
    build_split,
    evaluated_indices,
    prediction_path,
    write_atomic,
)
from come_a_little_bit_closer.evaluation.layer_schemas import (
    CountryReflection,
    EvaluationMode,
    FinalCell,
    LayerPlan,
    NeighborCrossover,
    NeighborGeolocation,
    ReflectedGeography,
    ReflectedPlan,
    SplitKind,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.probes import block_means, bootstrap_interval
from come_a_little_bit_closer.evaluation.splits import split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation without tie correction; ties are negligible for float scores."""
    first_ranks = np.empty(len(first))
    first_ranks[np.argsort(first)] = np.arange(len(first))
    second_ranks = np.empty(len(second))
    second_ranks[np.argsort(second)] = np.arange(len(second))
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def kilometres(cosines: np.ndarray) -> np.ndarray:
    return 6371.0088 * np.arccos(np.clip(cosines, -1.0, 1.0))


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def country_reflection(
    similarity: np.ndarray, true_km: np.ndarray, neighbors: int
) -> tuple[float, float]:
    """Spearman of similarity against closeness, and the 10-nearest distance gain.

    The gain is 1 - (mean km to the most similar cities) / (mean km to all of them),
    so 0 is random within the country and the true nearest cities give the ceiling.
    """
    upper = np.triu_indices(len(true_km), 1)
    rho = spearman(similarity[upper], -true_km[upper])
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    nearest = np.argpartition(-masked, neighbors, axis=1)[:, :neighbors]
    nearest_km = np.take_along_axis(true_km, nearest, axis=1).mean()
    everyone_km = true_km.sum() / (len(true_km) * (len(true_km) - 1))
    return rho, float(1.0 - nearest_km / everyone_km)


def reflected_countries(
    vectors: np.ndarray,
    truth: np.ndarray,
    codes: np.ndarray,
    minimum: int,
    neighbors: int,
) -> list[CountryReflection]:
    summary = []
    for code in np.unique(codes):
        members = np.flatnonzero(codes == code)
        if len(members) < max(minimum, neighbors + 2):
            continue
        rho, gain = country_reflection(
            vectors[members] @ vectors[members].T,
            kilometres(truth[members] @ truth[members].T),
            neighbors,
        )
        summary.append(
            CountryReflection(
                country_code=str(code), cities=len(members), rho=rho, knn_gain=gain
            )
        )
    return summary


def neighbor_positions(
    queries: np.ndarray,
    database: np.ndarray,
    database_truth: np.ndarray,
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """Mean position of the most similar same-country database items (NaN if none)."""
    positions = np.full((len(queries), 3), np.nan)
    for code in np.unique(query_codes):
        rows = np.flatnonzero(query_codes == code)
        pool = np.flatnonzero(database_codes == code)
        if not len(pool):
            continue
        count = min(neighbors, len(pool))
        similarity = queries[rows] @ database[pool].T
        nearest = np.argsort(-similarity, axis=1)[:, :count]
        positions[rows] = normalize(database_truth[pool][nearest].mean(axis=1))
    return positions


def gain_interval(
    differences: np.ndarray, blocks: list[str], plan: LayerPlan
) -> tuple[float, float, float]:
    means = block_means(differences, blocks)
    low, high = bootstrap_interval(means, plan)
    return float(means.mean()), low, high


def run_reflected(config: Path, mode: EvaluationMode) -> Path:
    plan = ReflectedPlan.model_validate_json(config.read_bytes())
    layers = LayerPlan.model_validate_json(plan.layers_plan.read_bytes())
    items = dataset_items("cities", "name-country")
    truth = spherical_vectors(
        np.asarray([[point.latitude, point.longitude] for point in items.coordinates])
    )
    codes = np.asarray(items.country_codes)
    split_kinds: list[SplitKind] = ["random-item", "spatial-block"]
    splits = {
        (kind, seed): build_split(
            kind,
            items.item_ids,
            items.coordinates,
            layers,
            seed,
            items.source_sha256,
            plan_checksum(plan.layers_plan),
        )
        for kind in split_kinds
        for seed in layers.split_seeds
    }
    reflected: list[ReflectedGeography] = []
    ceiling = {}
    for seed in layers.split_seeds:
        rows = evaluated_indices(splits[("random-item", seed)], mode)
        countries = reflected_countries(
            truth[rows], truth[rows], codes[rows], plan.minimum_cities, plan.neighbors
        )
        ceiling[seed] = float(np.mean([country.knn_gain for country in countries]))
    geolocation: list[NeighborGeolocation] = []
    advantages: dict[tuple[str, str, SplitKind, int], tuple[np.ndarray, list[str]]] = {}
    for cell in plan.cells:
        path = feature_path(
            layers, cell.model_id, "cities", "name-country", cell.pooling
        )
        features = np.load(path, mmap_mode="r")
        layer = features.shape[0] - 1
        vectors = normalize(np.asarray(features[layer], dtype=np.float64))
        with np.load(
            prediction_path(
                layers,
                mode,
                FinalCell(
                    model_id=cell.model_id,
                    dataset="cities",
                    condition="name-country",
                    pooling=cell.pooling,
                    layer=layer,
                ),
            )
        ) as arrays:
            saved = {key: arrays[key] for key in arrays.files}
        for (kind, seed), split in splits.items():
            prefix = f"{kind}/{seed}/country-residual"
            rows = saved[f"{prefix}/rows"]
            if not np.array_equal(
                np.sort(rows), np.sort(evaluated_indices(split, mode))
            ):
                raise ValueError("Saved predictions do not match the evaluated rows.")
            train_rows = split_indices(split, "train")
            if kind == "random-item":
                center = vectors[train_rows].mean(axis=0)
                decoded = spherical_vectors(
                    saved[f"{prefix}/predicted_coordinates"].astype(np.float64)
                )
                for similarity, space in [
                    ("cosine", vectors[rows]),
                    ("centered-cosine", normalize(vectors[rows] - center)),
                    ("probe-decoded", decoded),
                ]:
                    countries = reflected_countries(
                        space,
                        truth[rows],
                        codes[rows],
                        plan.minimum_cities,
                        plan.neighbors,
                    )
                    reflected.append(
                        ReflectedGeography(
                            model_id=cell.model_id,
                            pooling=cell.pooling,
                            layer=layer,
                            mode=mode,
                            seed=seed,
                            similarity=similarity,
                            rho=float(np.mean([country.rho for country in countries])),
                            knn_gain=float(
                                np.mean([country.knn_gain for country in countries])
                            ),
                            countries=countries,
                        )
                    )
            positions = neighbor_positions(
                vectors[rows],
                vectors[train_rows],
                truth[train_rows],
                codes[rows],
                codes[train_rows],
                plan.neighbors,
            )
            country_errors = saved[f"{prefix}/country_errors_km"].astype(np.float64)
            probe_errors = saved[f"{prefix}/errors_km"].astype(np.float64)
            neighbor_errors = kilometres(np.sum(positions * truth[rows], axis=1))
            missing = np.isnan(neighbor_errors)
            neighbor_errors[missing] = country_errors[missing]
            blocks = [str(block) for block in saved[f"{prefix}/block_ids"]]
            neighbor = gain_interval(country_errors - neighbor_errors, blocks, layers)
            probe = gain_interval(country_errors - probe_errors, blocks, layers)
            difference = gain_interval(probe_errors - neighbor_errors, blocks, layers)
            order = np.argsort(rows)
            advantages[(cell.model_id, cell.pooling, kind, seed)] = (
                (probe_errors - neighbor_errors)[order],
                [blocks[position] for position in order],
            )
            geolocation.append(
                NeighborGeolocation(
                    model_id=cell.model_id,
                    pooling=cell.pooling,
                    layer=layer,
                    mode=mode,
                    split_kind=kind,
                    seed=seed,
                    neighbor_gain_km=neighbor[0],
                    neighbor_ci_low=neighbor[1],
                    neighbor_ci_high=neighbor[2],
                    probe_gain_km=probe[0],
                    probe_ci_low=probe[1],
                    probe_ci_high=probe[2],
                    difference_km=difference[0],
                    difference_ci_low=difference[1],
                    difference_ci_high=difference[2],
                )
            )
        print(f"Scored {cell.model_id} {cell.pooling} layer {layer}", flush=True)
    crossover = [
        NeighborCrossover(
            base_id=base_id,
            retrieval_id=retrieval_id,
            pooling=pooling,
            mode=mode,
            split_kind=kind,
            seed=seed,
            interaction_km=interaction[0],
            ci_low=interaction[1],
            ci_high=interaction[2],
        )
        for base_id, retrieval_id in plan.pairs
        for pooling in ["mean", "last"]
        if (base_id, pooling, "random-item", layers.split_seeds[0]) in advantages
        and (retrieval_id, pooling, "random-item", layers.split_seeds[0]) in advantages
        for kind in split_kinds
        for seed in layers.split_seeds
        for interaction in [
            gain_interval(
                advantages[(retrieval_id, pooling, kind, seed)][0]
                - advantages[(base_id, pooling, kind, seed)][0],
                advantages[(base_id, pooling, kind, seed)][1],
                layers,
            )
        ]
    ]
    output = plan.output_directory / mode
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output / "reflected.jsonl",
        "".join(row.model_dump_json() + "\n" for row in reflected).encode(),
    )
    write_atomic(
        output / "geolocation.jsonl",
        "".join(row.model_dump_json() + "\n" for row in geolocation).encode(),
    )
    write_atomic(
        output / "crossover.jsonl",
        "".join(row.model_dump_json() + "\n" for row in crossover).encode(),
    )
    report_path = output / "report.md"
    write_atomic(
        report_path,
        reflected_report(
            plan, layers, reflected, geolocation, crossover, ceiling
        ).encode(),
    )
    return report_path


def country_value(country: CountryReflection, measure: Literal["rho", "knn"]) -> float:
    return country.rho if measure == "rho" else country.knn_gain


def paired_text(
    retrieval: ReflectedGeography,
    base: ReflectedGeography,
    layers: LayerPlan,
    measure: Literal["rho", "knn"],
) -> str:
    base_values = {
        country.country_code: country_value(country, measure)
        for country in base.countries
    }
    differences = np.asarray(
        [
            country_value(country, measure) - base_values[country.country_code]
            for country in retrieval.countries
            if country.country_code in base_values
        ]
    )
    low, high = bootstrap_interval(differences, layers)
    return f"{differences.mean():+.3f} [{low:+.3f}, {high:+.3f}]"


def reflected_report(
    plan: ReflectedPlan,
    layers: LayerPlan,
    reflected: list[ReflectedGeography],
    geolocation: list[NeighborGeolocation],
    crossover: list[NeighborCrossover],
    ceiling: dict[int, float],
) -> str:
    mode = reflected[0].mode
    index = {
        (row.model_id, row.pooling, row.seed, row.similarity): row for row in reflected
    }
    seeds = layers.split_seeds
    lines = [
        f"# Reflected geography, {plan.experiment_id} {mode}",
        "",
        "Cities `name-country`, last layer. Per country with at least "
        f"{plan.minimum_cities} evaluated cities: rho = Spearman of similarity"
        " against closeness; kNN gain = 1 - (km to the"
        f" {plan.neighbors} most similar) / (km to all). Mean over countries; one"
        " value per seed.",
        "",
        "Ceiling (similarity = true closeness), kNN gain per seed: "
        + " / ".join(f"{ceiling[seed]:.3f}" for seed in seeds)
        + ".",
        "",
        "## Reflection by model",
        "",
        "| Model | Pooling | cosine rho | cosine kNN gain | centred cosine kNN gain"
        " | probe-decoded kNN gain |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in plan.cells:
        values = {
            similarity: [
                index[(cell.model_id, cell.pooling, seed, similarity)] for seed in seeds
            ]
            for similarity in ["cosine", "centered-cosine", "probe-decoded"]
        }
        lines.append(
            f"| {cell.model_id} | {cell.pooling} | "
            + " / ".join(f"{row.rho:.3f}" for row in values["cosine"])
            + " | "
            + " | ".join(
                " / ".join(f"{row.knn_gain:.3f}" for row in values[similarity])
                for similarity in ["cosine", "centered-cosine", "probe-decoded"]
            )
            + " |"
        )
    lines += [
        "",
        "## Retrieval minus base, same pooling (countries paired, bootstrap CI)",
        "",
        "| Pair | Pooling | Similarity | Seed | rho | kNN gain |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    poolings = {(cell.model_id, cell.pooling) for cell in plan.cells}
    for base_id, retrieval_id in plan.pairs:
        for pooling in ["mean", "last"]:
            if (base_id, pooling) not in poolings or (
                retrieval_id,
                pooling,
            ) not in poolings:
                continue
            for similarity in ["cosine", "centered-cosine"]:
                for seed in seeds:
                    retrieval = index[(retrieval_id, pooling, seed, similarity)]
                    base = index[(base_id, pooling, seed, similarity)]
                    lines.append(
                        f"| {base_id} → {retrieval_id} | {pooling} | {similarity}"
                        f" | {seed}"
                        f" | {paired_text(retrieval, base, layers, 'rho')}"
                        f" | {paired_text(retrieval, base, layers, 'knn')} |"
                    )
    lines += [
        "",
        "## Nearest-neighbour geolocation vs the linear probe",
        "",
        "Gain over the training country centroid, km, block mean [95% CI]."
        " Difference = nearest-neighbour minus probe, paired by place.",
        "",
        "| Model | Pooling | Split | Seed | Probe | Nearest neighbour | Difference |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in geolocation:
        lines.append(
            f"| {row.model_id} | {row.pooling} | {row.split_kind} | {row.seed}"
            f" | {row.probe_gain_km:.0f} [{row.probe_ci_low:.0f},"
            f" {row.probe_ci_high:.0f}]"
            f" | {row.neighbor_gain_km:.0f} [{row.neighbor_ci_low:.0f},"
            f" {row.neighbor_ci_high:.0f}]"
            f" | {row.difference_km:+.0f} [{row.difference_ci_low:+.0f},"
            f" {row.difference_ci_high:+.0f}] |"
        )
    lines += [
        "",
        "## Crossover: does retrieval training shift the balance toward neighbours?",
        "",
        "Retrieval minus base of (probe error - neighbour error), km, paired by place,"
        " block mean [95% CI]. Positive = retrieval training favours nearest"
        " neighbours relative to the probe.",
        "",
        "| Pair | Pooling | Split | Seed | Interaction |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in crossover:
        lines.append(
            f"| {row.base_id} → {row.retrieval_id} | {row.pooling} | {row.split_kind}"
            f" | {row.seed} | {row.interaction_km:+.0f} [{row.ci_low:+.0f},"
            f" {row.ci_high:+.0f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-country geography reflected in embedding similarity."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    arguments = parser.parse_args()
    print(run_reflected(arguments.config, arguments.mode), flush=True)


if __name__ == "__main__":
    main()
