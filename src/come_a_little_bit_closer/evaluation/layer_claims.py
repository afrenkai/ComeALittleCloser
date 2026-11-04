import argparse
from pathlib import Path

import numpy as np

from come_a_little_bit_closer.evaluation.layer_run import write_atomic
from come_a_little_bit_closer.evaluation.layer_schemas import (
    ClaimRoster,
    EvaluationMode,
    LayerBlockScores,
    LayerPlan,
    LayerProbeScore,
    Pooling,
    PopularityGain,
    ProbeTarget,
    SplitKind,
)
from come_a_little_bit_closer.evaluation.probes import bootstrap_interval, r_squared


def read_scores(
    plan: LayerPlan, mode: EvaluationMode, model_id: str
) -> list[LayerProbeScore]:
    path = plan.output_directory / mode / f"{model_id}.jsonl"
    return [
        LayerProbeScore.model_validate_json(line)
        for line in path.read_text().splitlines()
    ]


def read_blocks(
    plan: LayerPlan, mode: EvaluationMode, model_id: str
) -> list[LayerBlockScores]:
    path = plan.output_directory / mode / "blocks" / f"{model_id}.jsonl"
    return [
        LayerBlockScores.model_validate_json(line)
        for line in path.read_text().splitlines()
    ]


def frozen_layers(plan: LayerPlan, model_id: str, pooling: Pooling) -> tuple[int, int]:
    """Best and last frozen layer for cities name-country; equal when they coincide."""
    layers = [
        cell.layer
        for cell in plan.final_cells
        if (cell.model_id, cell.dataset, cell.condition, cell.pooling)
        == (model_id, "cities", "name-country", pooling)
    ]
    if not layers:
        raise ValueError(f"No frozen cities name-country {pooling} cell: {model_id}")
    return min(layers), max(layers)


def probe_runs(
    scores: list[LayerProbeScore],
    pooling: Pooling,
    layer: int,
    probe_target: ProbeTarget,
    split_kind: SplitKind,
) -> list[LayerProbeScore]:
    return sorted(
        (
            score
            for score in scores
            if score.control == "probe"
            and score.dataset == "cities"
            and score.condition == "name-country"
            and score.pooling == pooling
            and score.layer == layer
            and score.probe_target == probe_target
            and score.split_kind == split_kind
        ),
        key=lambda score: score.seed,
    )


def run_text(runs: list[LayerProbeScore]) -> str:
    return "; ".join(
        f"{run.gain_over_country_km:.0f} "
        f"[{run.gain_ci_low:.0f}, {run.gain_ci_high:.0f}]"
        for run in runs
    )


def every_seed_above_zero(runs: list[LayerProbeScore], plan: LayerPlan) -> bool:
    return len(runs) == len(plan.split_seeds) and all(
        run.gain_ci_low > 0 for run in runs
    )


def block_gains(
    rows: list[LayerBlockScores], layer: int, split_kind: SplitKind, seed: int
) -> dict[str, float]:
    matches = [
        row
        for row in rows
        if row.control == "probe"
        and row.dataset == "cities"
        and row.condition == "name-country"
        and row.pooling == "mean"
        and row.layer == layer
        and row.probe_target == "country-residual"
        and row.split_kind == split_kind
        and row.seed == seed
    ]
    if len(matches) != 1:
        raise ValueError("Expected one block row per paired cell.")
    return {
        block.block_id: block.country_mean_km - block.mean_km
        for block in matches[0].blocks
    }


def paired_difference(
    base: list[LayerBlockScores],
    retrieval: list[LayerBlockScores],
    base_layer: int,
    retrieval_layer: int,
    split_kind: SplitKind,
    seed: int,
    plan: LayerPlan,
) -> tuple[float, float, float]:
    """Retrieval minus base gain over blocks scored for both, block bootstrap."""
    base_gains = block_gains(base, base_layer, split_kind, seed)
    retrieval_gains = block_gains(retrieval, retrieval_layer, split_kind, seed)
    shared = sorted(set(base_gains) & set(retrieval_gains))
    differences = np.asarray(
        [retrieval_gains[block_id] - base_gains[block_id] for block_id in shared],
        dtype=np.float64,
    )
    low, high = bootstrap_interval(differences, plan)
    return float(differences.mean()), low, high


def clipped_r2(
    plan: LayerPlan,
    mode: EvaluationMode,
    model_id: str,
    layer: int,
    split_kind: SplitKind,
    seed: int,
) -> tuple[float, float]:
    path = (
        plan.output_directory
        / mode
        / "predictions"
        / model_id
        / f"cities-name-country-mean-{layer}.npz"
    )
    with np.load(path) as arrays:
        prefix = f"{split_kind}/{seed}/latlon"
        raw = arrays[f"{prefix}/raw_outputs"].astype(np.float64)
        truth = arrays["true_coordinates"][arrays[f"{prefix}/rows"]].astype(np.float64)
    latitude = np.clip(raw[:, 0], -90, 90)
    longitude = np.clip(raw[:, 1], -180, 180)
    return r_squared(latitude, truth[:, 0]), r_squared(longitude, truth[:, 1])


def smallest_quintile(
    plan: LayerPlan, mode: EvaluationMode, split_kind: SplitKind
) -> dict[tuple[str, str], PopularityGain]:
    path = (
        plan.output_directory
        / mode
        / "popularity"
        / f"{split_kind}-country-residual.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(f"Run evaluation.layer_popularity first: {path}")
    rows = [
        PopularityGain.model_validate_json(line)
        for line in path.read_text().splitlines()
    ]
    return {(row.model_id, row.cell): row for row in rows if row.quantile == 1}


def verdict(supported: bool) -> str:
    return "supported" if supported else "NOT supported"


def claims_report(plan: LayerPlan, mode: EvaluationMode, roster: ClaimRoster) -> str:
    splits: list[SplitKind] = ["spatial-block", "random-item"]
    scores = {
        model_id: read_scores(plan, mode, model_id) for model_id in plan.model_ids
    }
    lines = [
        f"# Frozen claims, {plan.experiment_id} {mode}",
        "",
        "Gain over the country centroid in km, one `gain [95% CI]` per seed"
        " (seeds in plan order). Criteria: `docs/FREEZE.md`.",
        "",
        "## H1: decoders from 4B up hold within-country geography",
        "",
        "| Model | Layer | spatial-block | random-item | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    h1_results = []
    for model_id in roster.h1_models:
        best, last = frozen_layers(plan, model_id, "mean")
        runs = {
            split: probe_runs(scores[model_id], "mean", best, "country-residual", split)
            for split in splits
        }
        supported = all(every_seed_above_zero(runs[split], plan) for split in splits)
        h1_results.append(supported)
        lines.append(
            f"| {model_id} | {best} | {run_text(runs['spatial-block'])} | "
            f"{run_text(runs['random-item'])} | {verdict(supported)} |"
        )
    lines += [
        "",
        f"H1 overall: {verdict(all(h1_results))}.",
        "",
        "## H2: retrieval training keeps it (paired retrieval minus base, per block)",
        "",
        "| Pair | Split | Retrieval gain CIs above 0 |"
        " Paired difference per seed | Direction |",
        "| --- | --- | --- | --- | --- |",
    ]
    h2_results = []
    pair_directions: dict[str, set[str]] = {}
    for role, pairs in [("H2", roster.h2_pairs), ("secondary", roster.secondary_pairs)]:
        for base_id, retrieval_id in pairs:
            base_best = frozen_layers(plan, base_id, "mean")[0]
            retrieval_best = frozen_layers(plan, retrieval_id, "mean")[0]
            base_blocks = read_blocks(plan, mode, base_id)
            retrieval_blocks = read_blocks(plan, mode, retrieval_id)
            for split in splits:
                runs = probe_runs(
                    scores[retrieval_id],
                    "mean",
                    retrieval_best,
                    "country-residual",
                    split,
                )
                kept = every_seed_above_zero(runs, plan)
                differences = [
                    paired_difference(
                        base_blocks,
                        retrieval_blocks,
                        base_best,
                        retrieval_best,
                        split,
                        seed,
                        plan,
                    )
                    for seed in plan.split_seeds
                ]
                if all(low > 0 for mean, low, high in differences):
                    direction = "improves"
                elif all(high < 0 for mean, low, high in differences):
                    direction = "removes some"
                else:
                    direction = "no claim"
                if role == "H2":
                    h2_results.append(kept)
                pair_directions.setdefault(f"{base_id} → {retrieval_id}", set()).add(
                    direction
                )
                lines.append(
                    f"| {base_id} → {retrieval_id} ({role}) | {split} | {kept} | "
                    + "; ".join(
                        f"{mean:.0f} [{low:.0f}, {high:.0f}]"
                        for mean, low, high in differences
                    )
                    + f" | {direction} |"
                )
    lines += ["", "Direction per pair (claimed only if both splits agree):", ""]
    lines += [
        f"- {pair}: {directions.pop() if len(directions) == 1 else 'no claim'}"
        for pair, directions in pair_directions.items()
    ]
    lines += [
        "",
        "H2 overall (retrieval keeps the gain in every H2 pair): "
        f"{verdict(all(h2_results))}.",
        "",
        "## H3: MLM encoders up to 3.5B show none",
        "",
        "| Model | Layer | spatial-block | random-item | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    h3_results = []
    for model_id in roster.h3_models:
        best = frozen_layers(plan, model_id, "mean")[0]
        runs = {
            split: probe_runs(scores[model_id], "mean", best, "country-residual", split)
            for split in splits
        }
        supported = all(
            run.gain_ci_low <= 0 or run.gain_over_country_km < roster.h3_gain_floor_km
            for split in splits
            for run in runs[split]
        ) and all(len(runs[split]) == len(plan.split_seeds) for split in splits)
        h3_results.append(supported)
        lines.append(
            f"| {model_id} | {best} | {run_text(runs['spatial-block'])} | "
            f"{run_text(runs['random-item'])} | {verdict(supported)} |"
        )
    lines += [
        "",
        f"H3 overall: {verdict(all(h3_results))}.",
        "",
        "## H4: the retrieval output vector carries geography beyond the country",
        "",
        "Last layer, `sphere` target, random-item.",
        "",
        "| Model | Pooling | Layer | Gain per seed | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    h4_results = []
    for role, model_ids, pooling in [
        ("H4", roster.h4_models, "last"),
        ("secondary", roster.h4_secondary_models, "mean"),
    ]:
        for model_id in model_ids:
            last = frozen_layers(plan, model_id, pooling)[1]
            runs = probe_runs(scores[model_id], pooling, last, "sphere", "random-item")
            supported = every_seed_above_zero(runs, plan)
            if role == "H4":
                h4_results.append(supported)
            lines.append(
                f"| {model_id} ({role}) | {pooling} | {last} | {run_text(runs)} | "
                f"{verdict(supported)} |"
            )
    lines += [
        "",
        f"H4 overall: {verdict(all(h4_results))}.",
        "",
        "## H5: lat/lon R² mostly reflects knowing the country (descriptive)",
        "",
        "Best layer, `latlon` target, mean pooling, random-item; seed means."
        " Clipped R² (primary) clips predictions to valid latitude and longitude.",
        "",
        "| Model | Layer | R² lat / lon clipped | R² lat / lon raw |"
        " Country centroid R² lat / lon |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model_id in roster.h1_models + roster.h3_models + roster.random_init_models:
        best = frozen_layers(plan, model_id, "mean")[0]
        runs = probe_runs(scores[model_id], "mean", best, "latlon", "random-item")
        clipped = np.asarray(
            [
                clipped_r2(plan, mode, model_id, best, "random-item", run.seed)
                for run in runs
            ]
        ).mean(axis=0)
        centroid = [
            score
            for score in scores[model_id]
            if score.control == "country-centroid"
            and score.dataset == "cities"
            and score.condition == "name-country"
            and score.split_kind == "random-item"
        ]
        lines.append(
            f"| {model_id} | {best} | {clipped[0]:.2f} / {clipped[1]:.2f} | "
            f"{np.mean([run.r2_latitude for run in runs]):.2f} / "
            f"{np.mean([run.r2_longitude for run in runs]):.2f} | "
            f"{np.mean([row.r2_latitude for row in centroid]):.2f} / "
            f"{np.mean([row.r2_longitude for row in centroid]):.2f} |"
        )
    lines += [
        "",
        "## H6: the within-country signal is not limited to famous places",
        "",
        "Smallest population quintile, H1 cells, seeds pooled.",
        "",
        "| Model | spatial-block | random-item | Verdict |",
        "| --- | --- | --- | --- |",
    ]
    quintiles = {split: smallest_quintile(plan, mode, split) for split in splits}
    h6_results = []
    for model_id in roster.h1_models:
        cell = f"cities-name-country-mean-{frozen_layers(plan, model_id, 'mean')[0]}"
        rows = {split: quintiles[split][(model_id, cell)] for split in splits}
        supported = all(rows[split].gain_ci_low > 0 for split in splits)
        h6_results.append(supported)
        lines.append(
            f"| {model_id} | "
            + " | ".join(
                f"{rows[split].gain_over_country_km:.0f} "
                f"[{rows[split].gain_ci_low:.0f}, {rows[split].gain_ci_high:.0f}]"
                for split in splits
            )
            + f" | {verdict(supported)} |"
        )
    lines += ["", f"H6 overall: {verdict(all(h6_results))}.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the frozen layer-probe claims.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["dev", "final"], default="final")
    arguments = parser.parse_args()
    plan = LayerPlan.model_validate_json(arguments.config.read_bytes())
    output_path = plan.output_directory / arguments.mode / "claims.md"
    write_atomic(
        output_path, claims_report(plan, arguments.mode, ClaimRoster()).encode()
    )
    print(output_path, flush=True)


if __name__ == "__main__":
    main()
