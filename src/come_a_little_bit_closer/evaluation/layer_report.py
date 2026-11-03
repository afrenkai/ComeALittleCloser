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
from matplotlib.layout_engine import ConstrainedLayoutEngine

from come_a_little_bit_closer.evaluation.layer_extract import dataset_items
from come_a_little_bit_closer.evaluation.layer_run import (
    build_split,
    layer_predictions,
    load_features,
)
from come_a_little_bit_closer.evaluation.layer_schemas import (
    DistortionPanel,
    EvaluationMode,
    LayerDataset,
    LayerPlan,
    LayerProbeScore,
    ProbeTarget,
    ScoreControl,
    ScoreSelection,
    SeedMeanScore,
    SeedRange,
    SplitKind,
    StagePalette,
    TextCondition,
)
from come_a_little_bit_closer.evaluation.location_prepare import plan_checksum
from come_a_little_bit_closer.evaluation.probes import (
    angular_errors,
    vectors_to_coordinates,
)
from come_a_little_bit_closer.evaluation.splits import split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import ScoreVector
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


def seed_range(values: list[float | None]) -> SeedRange | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return SeedRange(mean=float(np.mean(present)), low=min(present), high=max(present))


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
            and row.probe_target == selection.probe_target
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
            r2_latitude=seed_range([row.r2_latitude for row in members]),
            r2_longitude=seed_range([row.r2_longitude for row in members]),
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
    return f"{stage_name(score.stage, palette)} ({score.model_id})"


def stage_name(stage: str, palette: StagePalette) -> str:
    if stage not in palette.stage_names:
        raise ValueError(f"Stage {stage!r} has no display name in the palette.")
    return palette.stage_names[stage]


def stage_color(stage: str, palette: StagePalette) -> str:
    if stage not in palette.colors:
        raise ValueError(f"Stage {stage!r} has no color in the palette.")
    return palette.colors[stage]


def titled_figure(width: float, height: float, title: str, footnote: str) -> Figure:
    figure = Figure(
        figsize=(width, height),
        layout=ConstrainedLayoutEngine(rect=(0, 0.06, 1, 0.94)),
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    figure.suptitle(title, fontsize=12, fontweight="bold", ha="left", x=0.01)
    figure.text(
        0.01,
        0.01,
        footnote,
        fontsize=8.5,
        ha="left",
        va="bottom",
        wrap=True,
        color="#333333",
    )
    return figure


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
    probe_target: ProbeTarget = "sphere",
) -> None:
    selection = ScoreSelection(
        dataset="cities",
        condition="name-country",
        split_kind=split_kind,
        control="probe",
        pooling="mean",
        probe_target=probe_target,
    )
    means = seed_means(rows, selection)
    heading = {
        "sphere": "probe gain over the country centroid",
        "country-residual": "residual-probe gain over the country centroid",
    }
    questions = {
        "spatial-block": "unseen 10-degree regions",
        "random-item": "random held-out cities",
    }
    columns = min(len(families), 2)
    grid_rows = math.ceil(len(families) / columns)
    figure = titled_figure(
        6.2 * columns,
        4.4 * grid_rows + 1.1,
        f"Where geography dies: {heading[probe_target]} by depth "
        f"(validation, dev; {split_kind} split)",
        f"Cities, name-country text, mean pooling; {questions[split_kind]}. "
        + (
            "The probe predicts the offset from the training-only country " "centroid. "
            if probe_target == "country-residual"
            else ""
        )
        + "Positive = better than knowing the country. Lines are seed means; bands "
        "are seed-mean 95% block-bootstrap CIs.",
    )
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
    figure.savefig(path, dpi=200)


def draw_r2_curves(
    rows: list[LayerProbeScore],
    families: list[str],
    palette: StagePalette,
    path: Path,
) -> None:
    selection = ScoreSelection(
        dataset="cities",
        condition="name-country",
        split_kind="random-item",
        control="probe",
        pooling="mean",
        probe_target="latlon",
    )
    means = seed_means(rows, selection)
    figure = titled_figure(
        4.6 * len(families),
        8.2,
        "Linear probe R-squared by depth, comparable to Gurnee & Tegmark's R-squared "
        "(ridge onto lat/lon; validation, dev)",
        "Cities, name-country text, mean pooling, random-item split. Ridge probe "
        "onto [latitude, longitude] in degrees; R-squared = 1 - SSE/SST on "
        "validation cities. Longitude R-squared uses raw degrees and ignores "
        "wraparound. Lines are seed means; bands are the min-max across seeds.",
    )
    panels = figure.subplots(2, len(families), sharey="row", sharex=True, squeeze=False)
    for column, family in enumerate(families):
        family_means = sorted(
            (score for score in means if score.family == family),
            key=lambda score: (score.stage_order, score.layer or 0),
        )
        for row_index, axis_name in enumerate(["Latitude", "Longitude"]):
            axes = panels[row_index, column]
            if row_index == 0:
                axes.set_title(family, loc="left", fontweight="bold")
            for stage_order, model_id in sorted_stage_models(rows, family):
                curve = [score for score in family_means if score.model_id == model_id]
                ranges = [
                    score.r2_latitude if row_index == 0 else score.r2_longitude
                    for score in curve
                ]
                if not curve or any(value is None for value in ranges):
                    continue
                color = stage_color(curve[0].stage, palette)
                depth = np.asarray(
                    [score.relative_depth for score in curve], dtype=float
                )
                axes.fill_between(
                    depth,
                    [value.low for value in ranges if value is not None],
                    [value.high for value in ranges if value is not None],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
                axes.plot(
                    depth,
                    [value.mean for value in ranges if value is not None],
                    color=color,
                    linewidth=1.8,
                    linestyle="--" if stage_order == 0 else "-",
                    label=stage_label(curve[0], palette),
                )
            axes.grid(axis="y", color="#dddddd", linewidth=0.6)
            axes.set_xlim(0, 1)
            if column == 0:
                axes.set_ylabel(f"{axis_name} R-squared")
            if row_index == 1:
                axes.set_xlabel("Relative depth (layer / last layer)")
                axes.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    frameon=False,
                    ncols=1,
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
    figure = titled_figure(
        11,
        5.8,
        f"{family} training stages: gain at each stage's best layer "
        "(validation, dev)",
        "Cities, name-country text, mean pooling. Layer chosen per stage by maximal "
        "seed-mean gain on the same validation items, so values are optimistic. "
        "Bars are seed-mean 95% block-bootstrap CIs.",
    )
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
                    stage_name(score.stage, palette),
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
    figure.savefig(path, dpi=200)


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
    items = dataset_items(dataset, condition)
    coordinates = np.asarray(
        [[place.latitude, place.longitude] for place in items.coordinates],
        dtype=np.float64,
    )
    targets = spherical_vectors(coordinates)
    manifest = build_split(
        "random-item",
        items.item_ids,
        items.coordinates,
        plan,
        plan.split_seeds[0],
        items.source_sha256,
        plan_checksum(config_path),
    )
    features, feature_manifest = load_features(
        plan,
        score.model_id,
        dataset,
        condition,
        score.pooling,
        items.item_ids,
        items.source_sha256,
    )
    if score.layer >= feature_manifest.layers:
        raise ValueError("Scored layer is outside the extracted layers.")
    layer_features = np.asarray(features[score.layer], dtype=np.float64)
    validation = split_indices(manifest, "validation")
    predictions = layer_predictions(
        layer_features,
        targets,
        coordinates,
        items.country_codes,
        manifest,
        plan,
        validation,
        "probe",
        "sphere",
    )[0]
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
    figure = titled_figure(
        10,
        9.4,
        f"{family}: where the probe moves each city (validation, dev)",
        "Random-item split, cities, name-country text, mean pooling, each stage's "
        f"best layer. Grey dots: all {len(panels[0].errors_km):,} validation cities "
        f"(true positions); arrows: {palette.arrow_sample:,} random validation "
        "cities, from true to predicted position; arrows crossing +/-180 degrees "
        "are clipped at the frame.",
    )
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
            f"{stage_name(panel.score.stage, palette)} "
            f"({panel.score.model_id}), layer {panel.score.layer}: "
            f"median error {np.median(panel.errors_km):,.0f} km, "
            f"mean {np.mean(panel.errors_km):,.0f} km",
            loc="left",
            fontweight="bold",
            fontsize=11,
        )
    axes_list[-1].set_xlabel("Longitude (degrees)")
    figure.savefig(path, dpi=200)


def km_or_na(score: SeedMeanScore | None) -> str:
    return "n/a" if score is None else f"{score.block_mean_km:,.0f}"


def best_by_r2(means: list[SeedMeanScore]) -> dict[str, SeedMeanScore]:
    best: dict[str, SeedMeanScore] = {}
    for score in means:
        if score.r2_latitude is None or score.r2_longitude is None:
            continue
        incumbent = best.get(score.model_id)
        if incumbent is None or r2_average(score) > r2_average(incumbent):
            best[score.model_id] = score
    return best


def r2_average(score: SeedMeanScore) -> float:
    if score.r2_latitude is None or score.r2_longitude is None:
        raise ValueError("Score carries no R-squared values.")
    return (score.r2_latitude.mean + score.r2_longitude.mean) / 2


def selected_means(
    rows: list[LayerProbeScore],
    dataset: LayerDataset,
    condition: TextCondition,
    split_kind: SplitKind,
    control: ScoreControl,
    probe_target: ProbeTarget,
) -> list[SeedMeanScore]:
    return seed_means(
        rows,
        ScoreSelection(
            dataset=dataset,
            condition=condition,
            split_kind=split_kind,
            control=control,
            probe_target=probe_target,
        ),
    )


def credibility_lines(rows: list[LayerProbeScore]) -> list[str]:
    if not any(row.probe_target != "sphere" for row in rows):
        return []
    lines = [
        "",
        "## Probe credibility: country-residual gain and lat/lon R-squared "
        "(validation, dev)",
        "",
        "All values are validation (dev) results, seed means, with the best layer "
        "chosen on the same validation items (optimistic). Country-residual: the "
        "probe predicts the offset from the training-only country centroid; gain "
        "is over the country centroid (km), best layer by gain. Lat/lon R-squared "
        "(ridge onto latitude and longitude in degrees, comparable to Gurnee & "
        "Tegmark) is at the layer with the highest mean of the two R-squared "
        "values. Longitude R-squared uses raw degrees and ignores wraparound. "
        "Country-centroid R-squared is the reference: what knowing the country "
        "alone explains.",
        "",
        "| model | dataset | condition | split | residual layer | residual gain "
        "(km) [CI] | latlon layer | latlon R2 lat | latlon R2 lon | "
        "country-centroid R2 lat | country-centroid R2 lon |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    cells = sorted({(row.dataset, row.condition, row.split_kind) for row in rows})
    for dataset, condition, split_kind in cells:

        residual = best_layers(
            selected_means(
                rows, dataset, condition, split_kind, "probe", "country-residual"
            )
        )
        latlon = best_by_r2(
            selected_means(rows, dataset, condition, split_kind, "probe", "latlon")
        )
        country = {
            score.model_id: score
            for score in selected_means(
                rows, dataset, condition, split_kind, "country-centroid", "sphere"
            )
        }
        for model_id in sorted(set(residual) | set(latlon)):
            gain = residual.get(model_id)
            fit = latlon.get(model_id)
            baseline = country.get(model_id)
            lines.append(
                f"| {model_id} | {dataset} | {condition} | {split_kind} | "
                + (
                    f"{gain.layer} | {gain.gain_over_country_km:,.0f} "
                    f"[{gain.gain_ci_low:,.0f}, {gain.gain_ci_high:,.0f}]"
                    if gain
                    else "n/a | n/a"
                )
                + " | "
                + (
                    f"{fit.layer} | {r2_text(fit.r2_latitude)} | "
                    f"{r2_text(fit.r2_longitude)}"
                    if fit
                    else "n/a | n/a | n/a"
                )
                + f" | {r2_text(baseline.r2_latitude if baseline else None)} | "
                f"{r2_text(baseline.r2_longitude if baseline else None)} |"
            )
    return lines


def r2_text(value: SeedRange | None) -> str:
    return "n/a" if value is None else f"{value.mean:.3f}"


def summary_lines(rows: list[LayerProbeScore], plan: LayerPlan) -> list[str]:
    lines = [
        "# Layerwise probe summary (dev mode)",
        "",
        "All numbers are **validation (dev-mode) results** from the "
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
                f"{km_or_na(control)} | "
                f"{km_or_na(baseline)} |"
            )
    return lines + credibility_lines(rows)


def write_report(config_path: Path, mode: EvaluationMode) -> Path:
    if mode == "final":
        raise ValueError(
            "Final-mode reporting is not implemented; final results must be "
            "reported from the frozen cells explicitly."
        )
    plan = LayerPlan.model_validate_json(config_path.read_bytes())
    palette = StagePalette()
    rows = load_scores(plan, mode)
    if not rows:
        raise FileNotFoundError("No scored models found for this mode.")
    directory = plan.output_directory / mode / "report"
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
        if any(row.probe_target == "country-residual" for row in rows):
            draw_depth_curves(
                rows,
                families,
                "spatial-block",
                palette,
                directory / "where-geography-dies-residual.png",
                "country-residual",
            )
        else:
            print("Skipping where-geography-dies-residual.png: no residual rows.")
        if any(row.probe_target == "latlon" for row in rows):
            draw_r2_curves(rows, families, palette, directory / "r2-by-depth.png")
        else:
            print("Skipping r2-by-depth.png: no latlon rows.")
        selection = ScoreSelection(
            dataset="cities",
            condition="name-country",
            split_kind="random-item",
            control="probe",
            pooling="mean",
        )
        for family in families:
            best = best_stage_scores(rows, family, selection)
            by_stage = {score.stage: score for score in best.values()}
            if "base" not in by_stage or "retrieval" not in by_stage:
                print(
                    f"Skipping distortion map for {family}: needs base and retrieval."
                )
                continue
            draw_distortion(
                [
                    distortion_panel(plan, config_path, score, "cities", "name-country")
                    for score in [by_stage["base"], by_stage["retrieval"]]
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
    (directory / "summary.md").write_text("\n".join(summary_lines(rows, plan)) + "\n")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the layerwise probe report.")
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/layers-v1.json")
    )
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    arguments = parser.parse_args()
    print(write_report(arguments.config, arguments.mode), flush=True)


if __name__ == "__main__":
    main()
