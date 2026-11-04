import argparse
from pathlib import Path

import numpy as np
import torch

from come_a_little_bit_closer.evaluation.fix_schemas import (
    DistanceProbe,
    FittedFix,
    FixCollection,
    FixMethod,
    FixPlan,
    FixScore,
    ScoreMatrix,
    StsPairs,
    UnitMatrix,
)
from come_a_little_bit_closer.evaluation.instructed_extract import (
    read_instructed,
    read_sts,
)
from come_a_little_bit_closer.evaluation.layer_extract import (
    dataset_items,
    feature_path,
)
from come_a_little_bit_closer.evaluation.layer_reflected import (
    country_reflection,
    kilometres,
    spearman,
)
from come_a_little_bit_closer.evaluation.layer_run import (
    build_split,
    evaluated_indices,
    write_atomic,
)
from come_a_little_bit_closer.evaluation.layer_schemas import (
    CountryReflection,
    EvaluationMode,
    LayerPlan,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.probes import bootstrap_interval
from come_a_little_bit_closer.evaluation.splits import split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors


def unit(vectors: np.ndarray) -> UnitMatrix:
    values = np.asarray(vectors, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def centered_unit(vectors: np.ndarray, center: np.ndarray) -> UnitMatrix:
    return unit(np.asarray(vectors, dtype=np.float32) - center)


def top_neighbors(score: np.ndarray, neighbors: int) -> np.ndarray:
    masked = score.copy()
    np.fill_diagonal(masked, -np.inf)
    return np.argpartition(-masked, neighbors, axis=1)[:, :neighbors]


def country_reflections(
    score: np.ndarray,
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
            score[np.ix_(members, members)],
            kilometres(truth[members] @ truth[members].T),
            neighbors,
        )
        summary.append(
            CountryReflection(
                country_code=str(code), cities=len(members), rho=rho, knn_gain=gain
            )
        )
    return summary


def neighbor_km(score: np.ndarray, truth: np.ndarray, neighbors: int) -> float:
    nearest = top_neighbors(score, neighbors)
    cosines = np.einsum("ij,ikj->ik", truth, truth[nearest])
    return float(kilometres(cosines).mean())


def station_genres(texts: list[str]) -> list[set[str]]:
    return [
        {tag for tag in text.split("Genres: ", 1)[1].split(", ") if tag}
        for text in texts
    ]


def genre_agreement(score: np.ndarray, genres: list[set[str]], neighbors: int) -> float:
    """Share of each tagged item's neighbours that share at least one genre tag."""
    nearest = top_neighbors(score, neighbors)
    return float(
        np.mean(
            [
                np.mean([bool(genres[row] & genres[other]) for other in nearest[row]])
                for row in range(len(genres))
                if genres[row]
            ]
        )
    )


def train_distance_probe(
    vectors: UnitMatrix,
    positions: np.ndarray,
    codes: np.ndarray,
    plan: FixPlan,
    seed: int,
) -> DistanceProbe:
    """Rank-k map whose squared distances rank same-country neighbours by km.

    Triplets: an anchor, one of its nearest same-country training items, and a
    random same-country item; hinge loss on squared distances in the map's space.
    """
    generator = np.random.default_rng(plan.probe_seed + seed)
    torch.manual_seed(plan.probe_seed + seed)
    anchors, positives, pools = [], [], []
    for code in np.unique(codes):
        members = np.flatnonzero(codes == code)
        if len(members) <= plan.probe_positives:
            continue
        km = kilometres(positions[members] @ positions[members].T)
        np.fill_diagonal(km, np.inf)
        nearest = np.argsort(km, axis=1)[:, : plan.probe_positives]
        anchors.append(members)
        positives.append(members[nearest])
        pools.append(np.repeat(members[None, :], len(members), axis=0))
    anchor_index = np.concatenate(anchors)
    positive_table = np.concatenate(positives)
    pool_sizes = np.concatenate(
        [np.full(len(table), table.shape[1]) for table in pools]
    )
    pool_rows = [row for table in pools for row in table]
    inputs = torch.tensor(vectors)
    projection = torch.nn.Parameter(
        torch.randn(inputs.shape[1], plan.probe_rank) * 0.01
    )
    optimizer = torch.optim.Adam([projection], lr=plan.probe_learning_rate)
    for step in range(plan.probe_steps):
        batch = generator.integers(0, len(anchor_index), plan.probe_batch)
        positive = positive_table[
            batch, generator.integers(0, plan.probe_positives, len(batch))
        ]
        negative = np.asarray(
            [pool_rows[row][generator.integers(0, pool_sizes[row])] for row in batch]
        )
        anchor_points = inputs[anchor_index[batch]] @ projection
        near = ((anchor_points - inputs[positive] @ projection) ** 2).sum(1)
        far = ((anchor_points - inputs[negative] @ projection) ** 2).sum(1)
        loss = torch.relu(plan.probe_margin + near - far).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return projection.detach().numpy().astype(np.float32)


def probe_distances(vectors: UnitMatrix, probe: DistanceProbe) -> ScoreMatrix:
    points = vectors @ probe
    norms = (points**2).sum(1)
    squared = norms[:, None] + norms[None, :] - 2 * points @ points.T
    return (squared / squared.mean()).astype(np.float32)


def choose_fusion_layer(
    array: np.ndarray,
    sample: np.ndarray,
    truth: np.ndarray,
    codes: np.ndarray,
    plan: FixPlan,
) -> int:
    """Index of the layer whose centred cosine best reflects geography."""
    gains = []
    for index in range(array.shape[0]):
        vectors = unit(array[index][sample])
        centred = vectors - vectors.mean(axis=0)
        lengths = np.linalg.norm(centred, axis=1, keepdims=True)
        if np.median(lengths) < 1e-6:
            gains.append(-np.inf)
            continue
        centred = centred / np.maximum(lengths, 1e-12)
        countries = country_reflections(
            centred @ centred.T,
            truth[sample],
            codes[sample],
            plan.minimum_cities,
            plan.neighbors,
        )
        gains.append(float(np.mean([country.knn_gain for country in countries])))
    return int(np.argmax(gains))


def fit_fix(
    city_array: np.ndarray,
    train_rows: np.ndarray,
    truth: np.ndarray,
    codes: np.ndarray,
    plan: FixPlan,
    seed: int,
) -> FittedFix:
    """Choose the fusion layer and train the distance probe on training cities."""
    generator = np.random.default_rng(plan.probe_seed + seed)
    sample = np.sort(
        generator.choice(
            train_rows, min(plan.layer_sample, len(train_rows)), replace=False
        )
    )
    layer = choose_fusion_layer(city_array, sample, truth, codes, plan)
    layer_center = unit(city_array[layer][train_rows]).mean(axis=0)
    training_output = unit(city_array[-1][train_rows])
    output_center = training_output.mean(axis=0)
    probe = train_distance_probe(
        training_output - output_center,
        truth[train_rows],
        codes[train_rows],
        plan,
        seed,
    )
    return FittedFix(
        layer=layer,
        layer_center=layer_center,
        output_center=output_center,
        probe=probe,
    )


def set_scores(
    array: np.ndarray, rows: np.ndarray | None, fitted: FittedFix, plan: FixPlan
) -> tuple[ScoreMatrix, ScoreMatrix]:
    """Plain cosine and the fused, distance-corrected score for one set of texts."""
    output = unit(array[-1] if rows is None else array[-1][rows])
    middle_raw = array[fitted.layer] if rows is None else array[fitted.layer][rows]
    middle = centered_unit(unit(middle_raw), fitted.layer_center)
    cosine = output @ output.T
    fused = (cosine + plan.fusion_weight * middle @ middle.T) / (1 + plan.fusion_weight)
    corrected = fused - plan.distance_weight * probe_distances(
        output - fitted.output_center, fitted.probe
    )
    return cosine, corrected


def sts_spearman(score: np.ndarray, pairs: StsPairs) -> float:
    return spearman(score[pairs.first, pairs.second], pairs.scores)


def run_fix(config: Path, mode: EvaluationMode) -> Path:
    plan = FixPlan.model_validate_json(config.read_bytes())
    layers = LayerPlan.model_validate_json(plan.layers_plan.read_bytes())
    output = plan.output_directory / mode
    if mode == "final" and (output / "scores.jsonl").exists():
        raise FileExistsError("Final fix results already exist; final runs once.")
    sts_collection: FixCollection = "sts-dev" if mode == "dev" else "sts-test"
    sts = read_sts(
        plan.sts_files["dev" if mode == "dev" else "test"],
        plan.sts_sha256["dev" if mode == "dev" else "test"],
    )
    cities = dataset_items("cities", "name-country")
    stations = dataset_items("stations", "metadata")
    city_truth = spherical_vectors(
        np.asarray([[point.latitude, point.longitude] for point in cities.coordinates])
    ).astype(np.float32)
    station_truth = spherical_vectors(
        np.asarray(
            [[point.latitude, point.longitude] for point in stations.coordinates]
        )
    ).astype(np.float32)
    city_codes = np.asarray(cities.country_codes)
    station_codes = np.asarray(stations.country_codes)
    genres = station_genres(stations.texts)
    rows_written: list[FixScore] = []
    for model in plan.models:
        plain_sts, sts_manifest = read_instructed(
            plan, model.model_id, sts_collection, False
        )
        kept = sts_manifest.kept_layers
        plain = {
            "cities": np.asarray(
                np.load(
                    feature_path(
                        layers, model.model_id, "cities", "name-country", "last"
                    ),
                    mmap_mode="r",
                )[kept]
            ),
            "stations": np.asarray(
                np.load(
                    feature_path(
                        layers, model.model_id, "stations", "metadata", "last"
                    ),
                    mmap_mode="r",
                )[kept]
            ),
            "sts": plain_sts,
        }
        instructed_cities, city_manifest = read_instructed(
            plan, model.model_id, "cities-name-country", True
        )
        if city_manifest.kept_layers != kept:
            raise ValueError("Plain and instructed features keep different layers.")
        instructed = {
            "cities": instructed_cities,
            "stations": read_instructed(
                plan, model.model_id, "stations-metadata", True
            )[0],
            "sts": read_instructed(plan, model.model_id, sts_collection, True)[0],
        }
        for kind in plan.split_kinds:
            for seed in layers.split_seeds:
                split = build_split(
                    kind,
                    cities.item_ids,
                    cities.coordinates,
                    layers,
                    seed,
                    cities.source_sha256,
                    plan_checksum(plan.layers_plan),
                )
                rows = evaluated_indices(split, mode)
                train_rows = split_indices(split, "train")
                train_codes = city_codes[train_rows]
                fallback = unit(city_truth[train_rows].mean(axis=0, keepdims=True))[0]
                centroids = {
                    code: unit(
                        city_truth[train_rows][train_codes == code].mean(
                            axis=0, keepdims=True
                        )
                    )[0]
                    for code in np.unique(train_codes)
                }
                city_country = np.asarray(
                    [centroids.get(code, fallback) for code in city_codes[rows]]
                )
                station_country = np.asarray(
                    [centroids.get(code, fallback) for code in station_codes]
                )
                scored = {}
                fusion_layers = {}
                for variant, arrays in [("plain", plain), ("instructed", instructed)]:
                    fitted = fit_fix(
                        arrays["cities"], train_rows, city_truth, city_codes, plan, seed
                    )
                    scored[variant] = {
                        "cities": set_scores(arrays["cities"], rows, fitted, plan),
                        "stations": set_scores(arrays["stations"], None, fitted, plan),
                        "sts": set_scores(arrays["sts"], None, fitted, plan),
                    }
                    fusion_layers[variant] = fitted.layer
                city_cosine = scored["plain"]["cities"][0]
                station_cosine = scored["plain"]["stations"][0]
                methods: list[
                    tuple[
                        FixMethod,
                        ScoreMatrix,
                        ScoreMatrix,
                        ScoreMatrix | None,
                        int | None,
                    ]
                ] = [
                    (
                        "cosine",
                        city_cosine,
                        station_cosine,
                        scored["plain"]["sts"][0],
                        None,
                    ),
                    (
                        "country",
                        (
                            city_cosine
                            + plan.country_weight * city_country @ city_country.T
                        )
                        / (1 + plan.country_weight),
                        (
                            station_cosine
                            + plan.country_weight * station_country @ station_country.T
                        )
                        / (1 + plan.country_weight),
                        None,
                        None,
                    ),
                    (
                        "oracle",
                        (
                            city_cosine
                            + plan.oracle_weight * city_truth[rows] @ city_truth[rows].T
                        )
                        / (1 + plan.oracle_weight),
                        (
                            station_cosine
                            + plan.oracle_weight * station_truth @ station_truth.T
                        )
                        / (1 + plan.oracle_weight),
                        None,
                        None,
                    ),
                    (
                        "fusion-probe",
                        scored["plain"]["cities"][1],
                        scored["plain"]["stations"][1],
                        scored["plain"]["sts"][1],
                        kept[fusion_layers["plain"]],
                    ),
                    (
                        "instruction",
                        scored["instructed"]["cities"][0],
                        scored["instructed"]["stations"][0],
                        scored["instructed"]["sts"][0],
                        None,
                    ),
                    (
                        "instruction-fusion-probe",
                        scored["instructed"]["cities"][1],
                        scored["instructed"]["stations"][1],
                        scored["instructed"]["sts"][1],
                        kept[fusion_layers["instructed"]],
                    ),
                ]
                for (
                    method,
                    city_score,
                    station_score,
                    sts_score,
                    fusion_layer,
                ) in methods:
                    city_countries = country_reflections(
                        city_score,
                        city_truth[rows],
                        city_codes[rows],
                        plan.minimum_cities,
                        plan.neighbors,
                    )
                    station_countries = country_reflections(
                        station_score,
                        station_truth,
                        station_codes,
                        plan.minimum_cities,
                        plan.neighbors,
                    )
                    rows_written.append(
                        FixScore(
                            model_id=model.model_id,
                            mode=mode,
                            split_kind=kind,
                            seed=seed,
                            method=method,
                            fusion_layer=fusion_layer,
                            city_knn_gain=float(
                                np.mean([row.knn_gain for row in city_countries])
                            ),
                            city_rho=float(
                                np.mean([row.rho for row in city_countries])
                            ),
                            city_neighbor_km=neighbor_km(
                                city_score, city_truth[rows], plan.neighbors
                            ),
                            station_knn_gain=float(
                                np.mean([row.knn_gain for row in station_countries])
                            ),
                            station_rho=float(
                                np.mean([row.rho for row in station_countries])
                            ),
                            station_neighbor_km=neighbor_km(
                                station_score, station_truth, plan.neighbors
                            ),
                            genre_agreement=genre_agreement(
                                station_score, genres, plan.neighbors
                            ),
                            sts_spearman=(
                                None
                                if sts_score is None
                                else sts_spearman(sts_score, sts)
                            ),
                            city_countries=city_countries,
                        )
                    )
                print(f"Scored {model.model_id} {kind} {seed}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output / "scores.jsonl",
        "".join(row.model_dump_json() + "\n" for row in rows_written).encode(),
    )
    report_path = output / "report.md"
    write_atomic(report_path, fix_report(plan, layers, rows_written).encode())
    return report_path


def sts_text(runs: list[FixScore]) -> str:
    values = [run.sts_spearman for run in runs if run.sts_spearman is not None]
    return f"{np.mean(values):.3f}" if values else "n/a"


def fix_report(plan: FixPlan, layers: LayerPlan, rows: list[FixScore]) -> str:
    index = {(row.model_id, row.split_kind, row.seed, row.method): row for row in rows}
    methods: list[FixMethod] = [
        "cosine",
        "country",
        "oracle",
        "fusion-probe",
        "instruction",
        "instruction-fusion-probe",
    ]
    lines = [
        f"# Coordinate-free fixes, {plan.experiment_id} {rows[0].mode}",
        "",
        f'Instruction: "{plan.instruction}". Within-country kNN gain for cities'
        " (evaluated split) and stations (all), and station genre agreement; mean over"
        f" seeds {layers.split_seeds}.",
        "",
        "| Model | Split | Method | City kNN | Station kNN | Genre | STS rho"
        " | Fusion layer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in plan.models:
        for kind in plan.split_kinds:
            for method in methods:
                runs = [
                    index[(model.model_id, kind, seed, method)]
                    for seed in layers.split_seeds
                ]
                lines.append(
                    f"| {model.model_id} | {kind} | {method}"
                    f" | {np.mean([run.city_knn_gain for run in runs]):.3f}"
                    f" | {np.mean([run.station_knn_gain for run in runs]):.3f}"
                    f" | {np.mean([run.genre_agreement for run in runs]):.3f}"
                    f" | {sts_text(runs)}"
                    f" | {' / '.join(str(run.fusion_layer) for run in runs)} |"
                )
    lines += [
        "",
        "## Full fix minus cosine, city kNN gain paired by country (95% CI)",
        "",
        "| Model | Split | Seed | Difference | Genre change | STS change |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in plan.models:
        for kind in plan.split_kinds:
            for seed in layers.split_seeds:
                fixed = index[(model.model_id, kind, seed, "instruction-fusion-probe")]
                base = index[(model.model_id, kind, seed, "cosine")]
                base_gain = {
                    row.country_code: row.knn_gain for row in base.city_countries
                }
                differences = np.asarray(
                    [
                        row.knn_gain - base_gain[row.country_code]
                        for row in fixed.city_countries
                    ]
                )
                low, high = bootstrap_interval(differences, layers)
                lines.append(
                    f"| {model.model_id} | {kind} | {seed}"
                    f" | {differences.mean():+.3f} [{low:+.3f}, {high:+.3f}]"
                    f" | {fixed.genre_agreement - base.genre_agreement:+.3f}"
                    f" | {fixed.sts_spearman - base.sts_spearman:+.3f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score coordinate-free fixes for within-country geography."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    arguments = parser.parse_args()
    print(run_fix(arguments.config, arguments.mode), flush=True)


if __name__ == "__main__":
    main()
