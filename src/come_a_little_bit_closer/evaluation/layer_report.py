import argparse
import math
import textwrap
from pathlib import Path

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from come_a_little_bit_closer.evaluation.layer_extract import dataset_items
from come_a_little_bit_closer.evaluation.layer_run import (
    build_split,
    fit_layer_probe,
    load_features,
)
from come_a_little_bit_closer.evaluation.layer_schemas import (
    DistortionPanel,
    EvaluationMode,
    LayerDataset,
    LayerPlan,
    LayerProbeScore,
    ScoreSelection,
    SeedMeanScore,
    SplitKind,
    StagePalette,
    TextCondition,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.location_schemas import ProbeTargets
from come_a_little_bit_closer.evaluation.probes import (
    angular_errors,
    coordinate_centroid,
    predict_ridge,
    unit_predictions,
)
from come_a_little_bit_closer.evaluation.splits import split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import (
    CoordinateMatrix,
    ScoreVector,
)
from come_a_little_bit_closer.models.layer_catalog import find_layer_model


def load_scores(plan: LayerPlan, mode: EvaluationMode) -> list[LayerProbeScore]:
    rows: list[LayerProbeScore] = []
    for path in sorted((plan.output_directory / mode).glob("*.jsonl")):
        rows.extend(
            LayerProbeScore.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line
        )
    return rows


def seed_means(
    rows: list[LayerProbeScore], selection: ScoreSelection
) -> list[SeedMeanScore]:
    groups: dict[tuple[str, str | None, int | None], list[LayerProbeScore]] = {}
    for row in rows:
        if (
            row.dataset == selection.dataset
            and row.condition == selection.condition
            and row.split_kind == selection.split_kind
            and row.control == selection.control
            and (selection.pooling is None or row.pooling == selection.pooling)
        ):
            groups.setdefault((row.model_id, row.pooling, row.layer), []).append(row)
    return [
        SeedMeanScore(
            model_id=members[0].model_id,
            family=members[0].family,
            stage=members[0].stage,
            stage_order=members[0].stage_order,
            pooling=members[0].pooling,
            layer=members[0].layer,
            relative_depth=members[0].relative_depth,
            block_mean_km=float(np.mean([row.block_mean_km for row in members])),
            gain_over_country_km=float(
                np.mean([row.gain_over_country_km for row in members])
            ),
            gain_ci_low=float(np.mean([row.gain_ci_low for row in members])),
            gain_ci_high=float(np.mean([row.gain_ci_high for row in members])),
            seeds=len(members),
        )
        for members in groups.values()
    ]


def best_layers(means: list[SeedMeanScore]) -> dict[str, SeedMeanScore]:
    best: dict[str, SeedMeanScore] = {}
    for score in means:
        incumbent = best.get(score.model_id)
        if incumbent is None or score.gain_over_country_km > (
            incumbent.gain_over_country_km
        ):
            best[score.model_id] = score
    return best


def ordered_families(rows: list[LayerProbeScore], palette: StagePalette) -> list[str]:
    present = {row.family for row in rows}
    known = [family for family in palette.family_order if family in present]
    return known + sorted(present - set(known))


def stage_label(score: SeedMeanScore, palette: StagePalette) -> str:
    return f"{palette.stage_names.get(score.stage, score.stage)} ({score.model_id})"


def stage_color(stage: str, palette: StagePalette) -> str:
    return palette.colors.get(stage, "#000000")


def sorted_stage_models(
    rows: list[LayerProbeScore], family: str
) -> list[tuple[int, str]]:
    return sorted(
        {(row.stage_order, row.model_id) for row in rows if row.family == family}
    )


def draw_zero_line(axes: Axes, label: bool) -> None:
    axes.axhline(0, color="#222222", linewidth=1.0, zorder=1)
    if label:
        axes.annotate(
            "country centroid",
            xy=(1.0, 0),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="#222222",
        )


def draw_depth_curves(
    rows: list[LayerProbeScore],
    families: list[str],
    split_kind: SplitKind,
    palette: StagePalette,
    path: Path,
) -> None:
    selection = ScoreSelection(
        dataset="cities",
        condition="name-country",
        split_kind=split_kind,
        control="probe",
        pooling="mean",
    )
    means = seed_means(rows, selection)
    columns = min(len(families), 2)
    grid_rows = math.ceil(len(families) / columns)
    figure = Figure(
        figsize=(6.2 * columns, 4.4 * grid_rows + 0.9),
        layout="constrained",
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    panels = figure.subplots(
        grid_rows, columns, sharey=True, sharex=True, squeeze=False
    ).ravel()
    for position, family in enumerate(families):
        axes = panels[position]
        axes.set_title(family, loc="left", fontweight="bold")
        draw_zero_line(axes, label=True)
        family_means = sorted(
            (score for score in means if score.family == family),
            key=lambda score: (score.stage_order, score.layer or 0),
        )
        for stage_order, model_id in sorted_stage_models(rows, family):
            curve = [score for score in family_means if score.model_id == model_id]
            if not curve:
                continue
            color = stage_color(curve[0].stage, palette)
            depth = np.asarray([score.relative_depth for score in curve], dtype=float)
            axes.fill_between(
                depth,
                [score.gain_ci_low for score in curve],
                [score.gain_ci_high for score in curve],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            axes.plot(
                depth,
                [score.gain_over_country_km for score in curve],
                color=color,
                linewidth=1.8,
                linestyle="--" if stage_order == 0 else "-",
                label=stage_label(curve[0], palette),
            )
        axes.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncols=1
        )
        axes.grid(axis="y", color="#dddddd", linewidth=0.6)
        axes.set_xlim(0, 1)
    for axes in panels[len(families) :]:
        axes.set_visible(False)
    for axes in panels[::columns]:
        axes.set_ylabel("Gain over country centroid (km)")
    for axes in panels[-columns:]:
        axes.set_xlabel("Relative depth (layer / last layer)")
    questions = {
        "spatial-block": "unseen 10-degree regions",
        "random-item": "random held-out cities",
    }
    figure.suptitle(
        "Where geography dies: probe gain over the country centroid by depth\n"
        f"Validation (dev) results, {split_kind} split ({questions[split_kind]}), "
        "cities with name-country text, mean pooling. Positive = better than knowing "
        "the country. Lines are seed means; bands are seed-mean 95% bootstrap CIs "
        "over blocks.",
        fontsize=11,
        ha="left",
        x=0.01,
        wrap=True,
    )
    figure.savefig(path, dpi=200)


def best_stage_scores(
    rows: list[LayerProbeScore], family: str, selection: ScoreSelection
) -> dict[str, SeedMeanScore]:
    best = best_layers(seed_means(rows, selection))
    return {
        model_id: score for model_id, score in best.items() if score.family == family
    }


def draw_trajectory(
    rows: list[LayerProbeScore], family: str, palette: StagePalette, path: Path
) -> None:
    split_kinds: list[SplitKind] = ["spatial-block", "random-item"]
    figure = Figure(figsize=(11, 5.4), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, 2, squeeze=False).ravel()
    for position, split_kind in enumerate(split_kinds):
        axes = panels[position]
        selection = ScoreSelection(
            dataset="cities",
            condition="name-country",
            split_kind=split_kind,
            control="probe",
            pooling="mean",
        )
        scores = sorted(
            best_stage_scores(rows, family, selection).values(),
            key=lambda score: score.stage_order,
        )
        draw_zero_line(axes, label=True)
        positions = np.arange(len(scores), dtype=float)
        gains = np.asarray([score.gain_over_country_km for score in scores])
        axes.plot(positions, gains, color="#999999", linewidth=1.2, zorder=2)
        for position, score in enumerate(scores):
            axes.errorbar(
                position,
                score.gain_over_country_km,
                yerr=[
                    [score.gain_over_country_km - score.gain_ci_low],
                    [score.gain_ci_high - score.gain_over_country_km],
                ],
                fmt="o",
                markersize=9,
                capsize=5,
                color=stage_color(score.stage, palette),
                zorder=3,
            )
            axes.annotate(
                f"layer {score.layer}",
                xy=(position, score.gain_ci_high),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
            )
        axes.set_xticks(positions)
        axes.set_xticklabels(
            [
                textwrap.fill(
                    palette.stage_names.get(score.stage, score.stage),
                    width=16,
                    break_on_hyphens=False,
                )
                + f"\n{score.model_id}"
                for score in scores
            ],
            fontsize=8,
        )
        axes.set_xlim(-0.5, len(scores) - 0.5)
        axes.margins(y=0.15)
        axes.grid(axis="y", color="#dddddd", linewidth=0.6)
        axes.set_title(f"{split_kind} split", loc="left", fontweight="bold")
        axes.set_ylabel("Best-layer gain over country centroid (km)")
        axes.set_xlabel("Training stage (in order)")
    figure.suptitle(
        f"{family} training-stage trajectory: gain at each stage's best layer\n"
        "Validation (dev) results, cities with name-country text, mean pooling. "
        "Layer chosen per stage by maximal seed-mean gain on the same validation "
        "items, so values are optimistic; bars are seed-mean 95% bootstrap CIs.",
        fontsize=11,
        ha="left",
        x=0.01,
        wrap=True,
    )
    figure.savefig(path, dpi=200)


@jaxtyped(typechecker=beartype)
def vectors_to_coordinates(vectors: ProbeTargets) -> CoordinateMatrix:
    return np.column_stack(
        (
            np.degrees(np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0))),
            np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])),
        )
    )


@jaxtyped(typechecker=beartype)
def unwrapped_longitudes(
    true_longitudes: ScoreVector, predicted_longitudes: ScoreVector
) -> ScoreVector:
    return true_longitudes + (predicted_longitudes - true_longitudes + 180) % 360 - 180


def distortion_panel(
    plan: LayerPlan,
    config_path: Path,
    score: SeedMeanScore,
    dataset: LayerDataset,
    condition: TextCondition,
) -> DistortionPanel:
    if score.pooling is None or score.layer is None:
        raise ValueError("Distortion maps need a fitted layer.")
    item_ids, _, places, _, source_sha256 = dataset_items(dataset, condition)
    targets = spherical_vectors(
        np.asarray(
            [[place.latitude, place.longitude] for place in places], dtype=np.float64
        )
    )
    manifest = build_split(
        "random-item",
        item_ids,
        places,
        plan,
        plan.split_seeds[0],
        source_sha256,
        plan_checksum(config_path),
    )
    features, _ = load_features(
        plan, score.model_id, dataset, condition, score.pooling, item_ids, source_sha256
    )
    layer_features = np.asarray(features[score.layer], dtype=np.float64)
    train = split_indices(manifest, "train")
    validation = split_indices(manifest, "validation")
    model = fit_layer_probe(layer_features, targets, manifest, plan, "probe")
    predictions = unit_predictions(
        predict_ridge(model, layer_features[validation]),
        coordinate_centroid(targets[train]),
    )
    return DistortionPanel(
        score=score,
        true_coordinates=vectors_to_coordinates(targets[validation]),
        predicted_coordinates=vectors_to_coordinates(predictions),
        errors_km=angular_errors(
            predictions, targets[validation], plan.geometry.earth_radius_km
        ),
    )


def draw_distortion(
    panels: list[DistortionPanel],
    family: str,
    plan: LayerPlan,
    palette: StagePalette,
    path: Path,
) -> None:
    figure = Figure(figsize=(10, 9.4), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axes_list = figure.subplots(len(panels), 1, sharex=True, squeeze=False).ravel()
    sample_size = min(palette.arrow_sample, len(panels[0].errors_km))
    chosen = np.sort(
        np.random.default_rng(plan.bootstrap_seed).choice(
            len(panels[0].errors_km), size=sample_size, replace=False
        )
    )
    for position, panel in enumerate(panels):
        axes = axes_list[position]
        color = stage_color(panel.score.stage, palette)
        truth = panel.true_coordinates
        predicted = panel.predicted_coordinates
        axes.scatter(
            truth[:, 1], truth[:, 0], s=2, color="#8a8a8a", alpha=0.45, linewidths=0
        )
        start_longitude = truth[chosen, 1]
        end_longitude = unwrapped_longitudes(
            start_longitude, predicted[chosen, 1].copy()
        )
        axes.quiver(
            start_longitude,
            truth[chosen, 0],
            end_longitude - start_longitude,
            predicted[chosen, 0] - truth[chosen, 0],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.001,
            headwidth=3.5,
            headlength=4.5,
            color=color,
            alpha=0.4,
        )
        axes.set_xlim(-180, 180)
        axes.set_ylim(-58, 82)
        axes.set_aspect("equal")
        axes.grid(color="#e6e6e6", linewidth=0.5)
        axes.set_axisbelow(True)
        axes.set_xticks(range(-180, 181, 60))
        axes.set_yticks(range(-60, 81, 30))
        axes.set_ylabel("Latitude (degrees)")
        axes.set_title(
            f"{palette.stage_names.get(panel.score.stage, panel.score.stage)} "
            f"({panel.score.model_id}), layer {panel.score.layer}: "
            f"median error {np.median(panel.errors_km):,.0f} km, "
            f"mean {np.mean(panel.errors_km):,.0f} km",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )
    axes_list[-1].set_xlabel("Longitude (degrees)")
    figure.suptitle(
        f"{family}: where the probe moves each city (true to predicted position)\n"
        "Validation (dev) results, random-item split, cities with name-country text, "
        "mean pooling, at each stage's best layer. "
        f"Grey dots: all {len(panels[0].errors_km):,} validation cities (true "
        f"positions); arrows: {sample_size:,} random validation cities, drawn from "
        "true to predicted position.",
        fontsize=11,
        ha="left",
        x=0.01,
        wrap=True,
    )
    figure.savefig(path, dpi=200)


def summary_lines(
    rows: list[LayerProbeScore], plan: LayerPlan, mode: EvaluationMode
) -> list[str]:
    lines = [
        f"# Layerwise probe summary ({mode} mode)",
        "",
        f"All numbers are **validation ({mode}-mode) results** from the "
        f"`{plan.experiment_id}` plan; they are not held-out test results. "
        "Best layer = maximal seed-mean gain over the country-centroid control "
        "(over pooling and layer), chosen on the same validation items, so gains "
        "are optimistic. CIs are seed-means of 95% block-bootstrap bounds.",
        "",
        "## Models and revisions",
        "",
        "| model | family | stage | checkpoint | revision |",
        "|---|---|---|---|---|",
    ]
    for model_id in sorted({row.model_id for row in rows}):
        spec = find_layer_model(model_id)
        lines.append(
            f"| {model_id} | {spec.family} | {spec.stage} | {spec.checkpoint} | "
            f"{spec.revision}{' (random init)' if spec.random_init else ''} |"
        )
    lines += [
        "",
        "## Best layer per model, dataset, condition, and split kind (validation)",
        "",
        "Errors are block-mean km (lower is better). Probe and permuted-labels "
        "errors are at the best layer; country centroid is the control.",
        "",
        "| model | dataset | condition | split | pooling | layer | rel. depth | "
        "gain over country (km) [CI] | probe km | permuted-labels km | "
        "country-centroid km |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    cells = sorted(
        {(row.dataset, row.condition, row.split_kind) for row in rows},
    )
    for dataset, condition, split_kind in cells:
        probe = seed_means(
            rows,
            ScoreSelection(
                dataset=dataset,
                condition=condition,
                split_kind=split_kind,
                control="probe",
            ),
        )
        permuted = {
            (score.model_id, score.pooling, score.layer): score
            for score in seed_means(
                rows,
                ScoreSelection(
                    dataset=dataset,
                    condition=condition,
                    split_kind=split_kind,
                    control="permuted-labels",
                ),
            )
        }
        country = {
            score.model_id: score
            for score in seed_means(
                rows,
                ScoreSelection(
                    dataset=dataset,
                    condition=condition,
                    split_kind=split_kind,
                    control="country-centroid",
                ),
            )
        }
        best = best_layers(probe)
        for score in sorted(
            best.values(), key=lambda item: (item.family, item.stage_order)
        ):
            control = permuted.get((score.model_id, score.pooling, score.layer))
            baseline = country.get(score.model_id)
            lines.append(
                f"| {score.model_id} | {dataset} | {condition} | {split_kind} | "
                f"{score.pooling} | {score.layer} | {score.relative_depth:.2f} | "
                f"{score.gain_over_country_km:,.0f} "
                f"[{score.gain_ci_low:,.0f}, {score.gain_ci_high:,.0f}] | "
                f"{score.block_mean_km:,.0f} | "
                f"{control.block_mean_km if control else float('nan'):,.0f} | "
                f"{baseline.block_mean_km if baseline else float('nan'):,.0f} |"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the layerwise probe report.")
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/layers-v1.json")
    )
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    arguments = parser.parse_args()
    plan = LayerPlan.model_validate_json(arguments.config.read_bytes())
    palette = StagePalette()
    rows = load_scores(plan, arguments.mode)
    if not rows:
        raise FileNotFoundError("No scored models found for this mode.")
    directory = plan.output_directory / arguments.mode / "report"
    directory.mkdir(parents=True, exist_ok=True)
    present = {row.model_id for row in rows}
    missing = [model_id for model_id in plan.model_ids if model_id not in present]
    print(f"Models present: {sorted(present)}", flush=True)
    print(f"Models missing (skipped): {missing}", flush=True)
    families = ordered_families(rows, palette)
    with rc_context(palette.font_settings):
        draw_depth_curves(
            rows,
            families,
            "spatial-block",
            palette,
            directory / "where-geography-dies.png",
        )
        draw_depth_curves(
            rows,
            families,
            "random-item",
            palette,
            directory / "where-geography-dies-random-item.png",
        )
        selection = ScoreSelection(
            dataset="cities",
            condition="name-country",
            split_kind="random-item",
            control="probe",
            pooling="mean",
        )
        for family in families:
            best = best_stage_scores(rows, family, selection)
            stages = sorted(best.values(), key=lambda score: score.stage_order)
            chosen = [score for score in stages if score.stage_order > 0]
            if len(chosen) < 2:
                print(f"Skipping distortion map for {family}: needs two stages.")
                continue
            draw_distortion(
                [
                    distortion_panel(
                        plan, arguments.config, score, "cities", "name-country"
                    )
                    for score in [chosen[0], chosen[-1]]
                ],
                family,
                plan,
                palette,
                directory / f"distortion-{family}.png",
            )
        if any(row.family == "xlmr-large" for row in rows):
            draw_trajectory(
                rows, "xlmr-large", palette, directory / "stage-trajectory.png"
            )
        else:
            print("Skipping stage-trajectory.png: xlmr-large is missing.")
    (directory / "summary.md").write_text(
        "\n".join(summary_lines(rows, plan, arguments.mode)) + "\n"
    )
    print(directory, flush=True)


if __name__ == "__main__":
    main()
