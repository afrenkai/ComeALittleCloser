import argparse
import csv
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import BaseModel, TypeAdapter

from come_a_little_bit_closer.evaluation.location_schemas import (
    LocationPlan,
    ProbeScore,
    ResponsivenessScore,
    RetrievalScore,
    SplitManifest,
)
from come_a_little_bit_closer.evaluation.probes import bootstrap_interval
from come_a_little_bit_closer.evaluation.report_schemas import (
    ExecutionRecord,
    GenreCoverage,
    PairedRegretChange,
    ReportResults,
    ReportSummaries,
    ResponseSummary,
    RetrievalSummary,
    SplitSummary,
    TaxonomyRecord,
    TokenSummary,
)


def load_records[Record: BaseModel](
    path: Path, record_type: type[Record]
) -> list[Record]:
    return TypeAdapter(list[record_type]).validate_json(path.read_bytes())


def load_coverage(path: Path) -> GenreCoverage | None:
    if not path.exists():
        return None
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns:
            raise ValueError("Genre coverage CSV has no header.")
        rows = [{column: row[column] or "" for column in columns} for row in reader]
    return GenreCoverage(columns=columns, rows=rows)


def load_results(config_path: Path) -> ReportResults:
    plan = LocationPlan.model_validate_json(config_path.read_bytes())
    directory = plan.output_directory
    required = [
        directory / name
        for name in [
            "probes.json",
            "retrieval.json",
            "responsiveness.json",
            "token-audit.json",
            "execution.json",
            "frozen-plan.json",
        ]
    ] + [directory / f"split-{seed}.json" for seed in plan.split_seeds]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Report requires completed results: " + ", ".join(missing)
        )
    if (directory / "frozen-plan.json").read_bytes() != config_path.read_bytes():
        raise ValueError("Report configuration differs from the executed frozen plan.")
    results = ReportResults(
        plan=plan,
        execution=ExecutionRecord.model_validate_json(
            (directory / "execution.json").read_bytes()
        ),
        probes=load_records(directory / "probes.json", ProbeScore),
        retrieval=load_records(directory / "retrieval.json", RetrievalScore),
        responses=load_records(directory / "responsiveness.json", ResponsivenessScore),
        tokens=load_records(directory / "token-audit.json", TokenSummary),
        splits=[
            SplitManifest.model_validate_json(
                (directory / f"split-{seed}.json").read_bytes()
            )
            for seed in plan.split_seeds
        ],
        coverage=(
            load_coverage(config_path.parent / "genre-coverage.csv")
            if plan.genre_tags
            else None
        ),
        taxonomy=(
            TaxonomyRecord.model_validate_json(plan.taxonomy_manifest.read_bytes())
            if plan.taxonomy_manifest is not None
            else None
        ),
    )
    validate_results(results, hashlib.sha256(config_path.read_bytes()).hexdigest())
    return results


def require_complete[Key](actual: list[Key], expected: set[Key], label: str) -> None:
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Incomplete or duplicate {label} result coverage.")


def validate_results(results: ReportResults, plan_checksum: str) -> None:
    plan = results.plan
    execution = results.execution
    if not plan.split_seeds or len(plan.origins) < 2 or not plan.genres:
        raise ValueError(
            "A report requires split seeds, genres, and at least two origins."
        )
    if execution.plan_sha256 != plan_checksum:
        raise ValueError("Execution record has a different plan checksum.")
    if {item.original.model_id for item in execution.model_manifests} != set(
        plan.model_ids
    ):
        raise ValueError("Execution provenance does not cover the planned models.")
    for item in execution.model_manifests:
        if (
            item.original.source_sha256 != execution.source_sha256
            or item.metadata_only.source_sha256 != execution.source_sha256
            or item.original.revision != item.metadata_only.revision
        ):
            raise ValueError("Model provenance differs across report conditions.")
    for split in results.splits:
        if (
            split.plan_sha256 != plan_checksum
            or split.source_sha256 != execution.source_sha256
            or len(split.assignments) != execution.station_count
        ):
            raise ValueError("Split provenance differs from the execution record.")
        block_partitions: dict[str, str] = {}
        for assignment in split.assignments:
            if (
                block_partitions.setdefault(assignment.block_id, assignment.split)
                != assignment.split
            ):
                raise ValueError("A spatial block crosses report partitions.")
    conditions = ["metadata-only", "explicit-coordinates"]
    expected_probes = (
        {
            (model, condition, control, seed)
            for model in plan.model_ids
            for condition in conditions
            for control in ["real", "permuted-labels"]
            for seed in plan.split_seeds
        }
        | {
            (model, "dimension-control", "random-features", seed)
            for model in plan.model_ids
            for seed in plan.split_seeds
        }
        | {
            ("control", condition, control, seed)
            for condition, control in [
                ("training-only", "country-centroid"),
                ("training-only", "global-centroid"),
                ("true-coordinates", "spherical-positive"),
            ]
            for seed in plan.split_seeds
        }
    )
    require_complete(
        [
            (row.model_id, row.condition, row.control, row.seed)
            for row in results.probes
        ],
        expected_probes,
        "probe",
    )
    expected_retrieval = {
        (model, condition, query_condition, genre, origin.name)
        for model in plan.model_ids
        for condition in conditions
        for query_condition in ["origin", "no-origin"]
        for genre in plan.genres
        for origin in plan.origins
    }
    require_complete(
        [
            (
                row.model_id,
                row.condition,
                row.query_condition,
                row.genre,
                row.origin_name,
            )
            for row in results.retrieval
        ],
        expected_retrieval,
        "retrieval",
    )
    expected_responses = {
        (model, condition, query_condition, genre, first.name, second.name)
        for model in plan.model_ids
        for condition in conditions
        for query_condition in ["origin", "no-origin"]
        for genre in plan.genres
        for first, second in combinations(plan.origins, 2)
    }
    require_complete(
        [
            (
                row.model_id,
                row.condition,
                row.query_condition,
                row.genre,
                row.first_origin,
                row.second_origin,
            )
            for row in results.responses
        ],
        expected_responses,
        "responsiveness",
    )
    require_complete(
        [(row.model_id, row.condition) for row in results.tokens],
        {
            (model, condition)
            for model in plan.model_ids
            for condition in ["explicit-coordinates", "metadata-only", "queries"]
        },
        "token audit",
    )
    for row in results.tokens:
        expected_texts = (
            len(plan.genres) * (len(plan.origins) + 1)
            if row.condition == "queries"
            else execution.station_count
        )
        if row.texts != expected_texts or row.over_limit > row.texts:
            raise ValueError("Token-audit counts do not match the frozen experiment.")
    for row in results.responses:
        if (
            row.query_condition == "no-origin"
            and max(
                abs(row.neighbor_turnover),
                abs(row.movement_toward_origin_km),
                abs(row.query_cosine_distance),
            )
            > 1e-12
        ):
            raise ValueError(
                "The no-origin negative control is nonzero above tolerance."
            )
    for genre in plan.genres:
        pool_sizes = {row.candidates for row in results.retrieval if row.genre == genre}
        if len(pool_sizes) != 1 or min(pool_sizes) < max(
            plan.neighbors, plan.minimum_genre_candidates
        ):
            raise ValueError(
                "Genre candidate pools differ or violate the plan minimum."
            )

    if results.coverage is not None and plan.genre_tags:
        by_genre = {row["genre"]: row for row in results.coverage.rows}
        for genre in plan.genres:
            if genre not in by_genre:
                raise ValueError("Genre coverage omits an evaluated genre.")
            coverage = by_genre[genre]
            tags = TypeAdapter(list[str]).validate_json(coverage["station_tags"])
            if set(tags) != set(plan.genre_tags[genre]) or any(
                row.candidates != int(coverage["station_count"])
                for row in results.retrieval
                if row.genre == genre
            ):
                raise ValueError(
                    "Genre inventory differs from evaluated candidate pools."
                )


def summarize_results(results: ReportResults) -> ReportSummaries:
    plan = results.plan
    retrieval_summaries: list[RetrievalSummary] = []
    paired_summaries: list[PairedRegretChange] = []
    response_summaries: list[ResponseSummary] = []
    for model in plan.model_ids:
        for condition in ["metadata-only", "explicit-coordinates"]:
            for query_condition in ["origin", "no-origin"]:
                retrieval = [
                    row
                    for row in results.retrieval
                    if (row.model_id, row.condition, row.query_condition)
                    == (model, condition, query_condition)
                ]
                retrieval_summaries.append(
                    RetrievalSummary(
                        model_id=model,
                        condition=condition,
                        query_condition=query_condition,
                        genres=len(plan.genres),
                        origins=len(plan.origins),
                        cases=len(retrieval),
                        mean_regret_km=float(
                            np.mean([row.regret_km for row in retrieval])
                        ),
                        mean_retrieved_km=float(
                            np.mean([row.retrieved_mean_km for row in retrieval])
                        ),
                        mean_reference_km=float(
                            np.mean([row.reference_mean_km for row in retrieval])
                        ),
                        mean_geographic_overlap=float(
                            np.mean([row.geographic_overlap_at_k for row in retrieval])
                        ),
                        mean_genre_precision=float(
                            np.mean([row.genre_precision_at_k for row in retrieval])
                        ),
                    )
                )
                responses = [
                    row
                    for row in results.responses
                    if (row.model_id, row.condition, row.query_condition)
                    == (model, condition, query_condition)
                ]
                response_summaries.append(
                    ResponseSummary(
                        model_id=model,
                        condition=condition,
                        query_condition=query_condition,
                        genres=len(plan.genres),
                        origin_pairs=len(plan.origins) * (len(plan.origins) - 1) // 2,
                        cases=len(responses),
                        mean_turnover_percent=100
                        * float(np.mean([row.neighbor_turnover for row in responses])),
                        mean_movement_km=float(
                            np.mean(
                                [row.movement_toward_origin_km for row in responses]
                            )
                        ),
                        mean_query_cosine_distance=float(
                            np.mean([row.query_cosine_distance for row in responses])
                        ),
                    )
                )
            regrets = {
                (row.origin_name, row.genre, row.query_condition): row.regret_km
                for row in results.retrieval
                if row.model_id == model and row.condition == condition
            }
            reductions = np.asarray(
                [
                    np.mean(
                        [
                            regrets[(origin.name, genre, "no-origin")]
                            - regrets[(origin.name, genre, "origin")]
                            for genre in plan.genres
                        ]
                    )
                    for origin in plan.origins
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_interval(reductions, plan)
            paired_summaries.append(
                PairedRegretChange(
                    model_id=model,
                    condition=condition,
                    genres=len(plan.genres),
                    origins=len(plan.origins),
                    mean_reduction_km=float(reductions.mean()),
                    ci_low_km=low,
                    ci_high_km=high,
                )
            )
    splits = [
        SplitSummary(
            seed=manifest.seed,
            partition=partition,
            stations=sum(item.split == partition for item in manifest.assignments),
            blocks=len(
                {
                    item.block_id
                    for item in manifest.assignments
                    if item.split == partition
                }
            ),
        )
        for manifest in results.splits
        for partition in ["train", "validation", "test"]
    ]
    return ReportSummaries(
        retrieval=retrieval_summaries,
        paired_regret=paired_summaries,
        responses=response_summaries,
        splits=splits,
    )


def write_csv[Record: BaseModel](path: Path, records: list[Record]) -> None:
    if not records:
        raise ValueError("CSV summaries require at least one record.")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(type(records[0]).model_fields))
        writer.writeheader()
        writer.writerows(record.model_dump(mode="json") for record in records)


def style_panel(axis: Axes, title: str, label: str, models: list[str]) -> None:
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    axis.set_xlabel(label, fontsize=10)
    axis.set_yticks(range(len(models)), models)
    axis.set_ylim(len(models) - 0.45, -0.55)
    axis.grid(axis="x", color="#d6dde3", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def make_figure(results: ReportResults, summaries: ReportSummaries) -> Figure:
    plan = results.plan
    primary_seed = plan.split_seeds[0]
    figure = Figure(figsize=(13, 9), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    panels = [figure.add_subplot(position) for position in [221, 222, 223, 224]]
    for condition, color, offset in [
        ("metadata-only", "#0072B2", -0.16),
        ("explicit-coordinates", "#D55E00", 0.16),
    ]:
        for model_index, model in enumerate(plan.model_ids):
            position = model_index + offset
            probe = next(
                row
                for row in results.probes
                if (row.model_id, row.condition, row.control, row.seed)
                == (model, condition, "real", primary_seed)
            )
            panels[0].hlines(
                position,
                probe.block_mean_ci_low,
                probe.block_mean_ci_high,
                colors=color,
                linewidth=1.8,
            )
            panels[0].scatter(
                probe.block_mean_km,
                position,
                color=color,
                s=35,
                label=condition if model_index == 0 else None,
                zorder=3,
            )
            retrieval = next(
                row
                for row in summaries.retrieval
                if (row.model_id, row.condition, row.query_condition)
                == (model, condition, "origin")
            )
            response = next(
                row
                for row in summaries.responses
                if (row.model_id, row.condition, row.query_condition)
                == (model, condition, "origin")
            )
            for axis, value in [
                (panels[1], retrieval.mean_regret_km),
                (panels[2], response.mean_turnover_percent),
                (panels[3], response.mean_movement_km),
            ]:
                axis.scatter(value, position, color=color, s=35, zorder=3)
    country = next(
        row
        for row in results.probes
        if row.control == "country-centroid" and row.seed == primary_seed
    )
    panels[0].axvspan(
        country.block_mean_ci_low,
        country.block_mean_ci_high,
        color="#6e7781",
        alpha=0.14,
    )
    panels[0].axvline(
        country.block_mean_km,
        color="#4c5560",
        linestyle="--",
        linewidth=1.1,
        label="training country centroid",
    )
    panels[0].legend(fontsize=8, loc="best", frameon=False)
    panels[0].set_xlim(left=0)
    panels[1].set_xlim(left=0)
    panels[2].set_xlim(-1, 101)
    panels[3].axvline(0, color="#4c5560", linewidth=0.8)
    for axis, title, label in [
        (
            panels[0],
            f"a  Coordinate decoding · seed {primary_seed}",
            "Test-block mean angular error (km)",
        ),
        (
            panels[1],
            "b  Genre-matched retrieval · origin query",
            "Mean geographic regret (km)",
        ),
        (panels[2], "c  Origin-only neighbor changes", "Mean top-k turnover (%)"),
        (
            panels[3],
            "d  Alignment with assigned origin",
            "Mean directional movement (km)",
        ),
    ]:
        style_panel(axis, title, label, plan.model_ids)
    figure.suptitle(
        f"{plan.experiment_id} · {len(plan.genres)} genres · "
        f"{len(plan.origins)} origins · k={plan.neighbors}",
        fontsize=14,
    )
    figure.supxlabel(
        "Probe intervals: recorded 95% block bootstrap. "
        "Retrieval and shared-origin pair means are descriptive.",
        fontsize=9,
    )
    return figure


def markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    def escaped(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(escaped(column) for column in columns) + " |",
            "| " + " | ".join("---" for column in columns) + " |",
            *[
                "| " + " | ".join(escaped(value) for value in row) + " |"
                for row in rows
            ],
        ]
    )


def interval_text(mean: float, low: float, high: float) -> str:
    return f"{mean:,.1f} [{low:,.1f}, {high:,.1f}]"


def report_markdown(
    config_path: Path, results: ReportResults, summaries: ReportSummaries
) -> str:
    plan = results.plan
    execution = results.execution
    primary_seed = plan.split_seeds[0]
    stage = (
        "Frozen pilot" if plan.experiment_id == "location-v1" else "Expanded analysis"
    )
    lines = [
        f"# {plan.experiment_id}: geographic decoding and origin-conditioned retrieval",
        "",
        f"{stage}. Recorded execution: {execution.created_at_utc}.",
        "",
        f"{execution.station_count:,} stations; {len(plan.model_ids)} models; "
        f"{len(plan.genres)} genres; {len(plan.origins)} fixed origin sites; "
        f"k={plan.neighbors}.",
        "",
        "## Primary results",
        "",
        f"The primary split seed is {primary_seed}. "
        "Probe errors average station errors "
        "within each test block, then weight blocks equally. Brackets are recorded "
        "95% percentile bootstrap intervals over test blocks.",
        "",
        markdown_table(
            [
                "Model",
                "Document condition",
                "Error km [95% CI]",
                "Gain over country km [95% CI]",
                "Alpha",
            ],
            [
                [
                    row.model_id,
                    row.condition,
                    interval_text(
                        row.block_mean_km, row.block_mean_ci_low, row.block_mean_ci_high
                    ),
                    interval_text(
                        row.improvement_over_country_km,
                        row.improvement_ci_low,
                        row.improvement_ci_high,
                    ),
                    str(row.selected_alpha),
                ]
                for row in results.probes
                if row.control == "real" and row.seed == primary_seed
            ],
        ),
        "",
        "![Recorded experiment summary](location-summary.svg)",
        "",
        "The country-centroid control uses training observations only, with a "
        "training-global fallback for unseen or empty country codes. All controls, "
        "split seeds, station median/p90 "
        "errors, selected penalties, and paired country improvements appear in "
        "[probe-summary.csv](probe-summary.csv).",
        "",
        "## Retrieval and origin changes",
        "",
        "Cosine retrieval and the exact Haversine reference share the frozen "
        "genre-mapped candidate pool. Means weight every genre and origin equally. "
        "Genre precision uses the same eligibility mapping on unfiltered retrieval.",
        "",
        markdown_table(
            [
                "Model",
                "Documents",
                "Origin-query regret km",
                "Overlap@k",
                "Unfiltered tag precision@k",
            ],
            [
                [
                    row.model_id,
                    row.condition,
                    f"{row.mean_regret_km:,.1f}",
                    f"{row.mean_geographic_overlap:.3f}",
                    f"{row.mean_genre_precision:.3f}",
                ]
                for row in summaries.retrieval
                if row.query_condition == "origin"
            ],
        ),
        "",
        "Paired regret reduction is no-origin regret minus origin-query regret. "
        "Genres are averaged within each origin before resampling the origin sites; "
        f"{plan.bootstrap_replicates:,} bootstrap draws "
        f"use seed {plan.bootstrap_seed}. "
        "Positive values indicate lower regret when coordinates are supplied.",
        "",
        markdown_table(
            ["Model", "Documents", "Regret reduction km [95% origin CI]"],
            [
                [
                    row.model_id,
                    row.condition,
                    interval_text(row.mean_reduction_km, row.ci_low_km, row.ci_high_km),
                ]
                for row in summaries.paired_regret
            ],
        ),
        "",
        "Pairwise response summaries are descriptive. Origin pairs share sites; "
        "no pair-level confidence intervals or independence assumptions are used. "
        "Positive "
        "movement indicates closer alignment with each query's assigned origin.",
        "",
        markdown_table(
            [
                "Model",
                "Documents",
                "Turnover %",
                "Movement km",
                "Query cosine distance",
            ],
            [
                [
                    row.model_id,
                    row.condition,
                    f"{row.mean_turnover_percent:.2f}",
                    f"{row.mean_movement_km:,.1f}",
                    f"{row.mean_query_cosine_distance:.6f}",
                ]
                for row in summaries.responses
                if row.query_condition == "origin"
            ],
        ),
        "",
        "Every no-origin control has turnover, movement, and query self-distance "
        "zero within 1e-12. Both query conditions remain in "
        "[retrieval-summary.csv](retrieval-summary.csv) and "
        "[responsiveness-summary.csv](responsiveness-summary.csv). "
        "Paired origin intervals "
        "are in [paired-regret-change.csv](paired-regret-change.csv).",
        "",
        "## Spatial splits",
        "",
        f"Occupied {plan.block_degrees:g}° cells are assigned as indivisible blocks. "
        "Seed results are separate sensitivity checks; "
        "overlapping test sets are not pooled.",
        "",
        markdown_table(
            ["Seed", "Partition", "Stations", "Blocks"],
            [
                [str(row.seed), row.partition, str(row.stations), str(row.blocks)]
                for row in summaries.splits
            ],
        ),
        "",
        "## Token audit",
        "",
        "Lengths include native prompt prefixes before truncation. "
        "An over-limit explicit "
        "document is not guaranteed to retain its coordinate suffix.",
        "",
        markdown_table(
            ["Model", "Condition", "Texts", "Over limit", "Maximum tokens", "Limit"],
            [
                [
                    row.model_id,
                    row.condition,
                    str(row.texts),
                    str(row.over_limit),
                    str(row.maximum_tokens),
                    str(row.token_limit),
                ]
                for row in results.tokens
            ],
        ),
        "",
        "## Genre coverage",
        "",
        "Frozen retrieval genres: " + ", ".join(plan.genres) + ".",
        "",
    ]
    if results.coverage is not None:
        coverage = results.coverage
        selected = [
            row
            for row in coverage.rows
            if any(value in plan.genres for value in row.values())
        ]
        visible = selected if selected else coverage.rows
        lines.extend(
            [
                "The coverage inventory and alias mapping are preserved in "
                "[genre-coverage.csv](genre-coverage.csv). "
                "The table shows inventory rows matching the frozen genres.",
                "",
                markdown_table(
                    coverage.columns,
                    [[row[column] for column in coverage.columns] for row in visible],
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Pilot eligibility uses exact catalog tags. Candidate counts below "
                "come from this run; the expanded MusicBrainz inventory is separate.",
                "",
                markdown_table(
                    ["Exact pilot tag", "Candidate stations"],
                    [
                        [
                            genre,
                            str(
                                next(
                                    row.candidates
                                    for row in results.retrieval
                                    if row.genre == genre
                                )
                            ),
                        ]
                        for genre in plan.genres
                    ],
                ),
                "",
            ]
        )
    if results.taxonomy is not None:
        taxonomy = results.taxonomy
        lines.extend(
            [
                f"Taxonomy: [{taxonomy.source_url}]({taxonomy.source_url}); "
                f"retrieved {taxonomy.retrieved_at_utc}. "
                f"Selected {taxonomy.selected_genres} of "
                f"{taxonomy.reference_genres:,} canonical source genres, requiring "
                f"at least {taxonomy.minimum_stations} candidate stations. "
                f"Normalization: {taxonomy.normalization}.",
                "",
                f"Taxonomy source SHA-256: `{taxonomy.source_sha256}`. "
                "Exact taxonomy inventory: "
                "[taxonomy-manifest.json](taxonomy-manifest.json).",
                "",
            ]
        )
    elif not plan.genre_tags:
        lines.extend(
            [
                "This pilot uses its prespecified exact catalog tags. "
                "The later spelling-normalized taxonomy mapping did not determine "
                "pilot candidate eligibility.",
                "",
            ]
        )
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Configuration: `{config_path}`; experiment `{plan.experiment_id}`.",
            f"- Exact plan SHA-256: `{execution.plan_sha256}`.",
            f"- Source snapshot SHA-256: `{execution.source_sha256}`.",
            "- Exact executed plan: [frozen-plan.json](frozen-plan.json).",
            "- Full model settings, software versions, and source-file hashes: "
            "[execution.json](execution.json).",
            "- Raw per-station test errors: "
            "[probe-predictions.csv](probe-predictions.csv).",
            "- Raw ranked station IDs and distances: [retrieval.json](retrieval.json).",
            "- Raw pairwise changes: [responsiveness.json](responsiveness.json).",
            "",
            markdown_table(
                ["Model", "Checkpoint", "Pinned revision", "Dimensions", "Token limit"],
                [
                    [
                        item.original.model_id,
                        item.original.checkpoint,
                        item.original.revision,
                        str(item.original.dimensions),
                        str(item.original.max_sequence_length),
                    ]
                    for item in execution.model_manifests
                ],
            ),
            "",
            markdown_table(
                ["Software", "Version"],
                [
                    [package, version]
                    for package, version in sorted(execution.software.items())
                ],
            ),
            "",
            "## Interpretation limits",
            "",
            "- Metadata-only documents retain names, country codes, and genre tags. "
            "They are not location-free inputs. Explicit-coordinate decoding "
            "may reflect transcription.",
            "- Probe recoverability and retrieval sensitivity do not establish "
            "causal use, internal mechanisms, or mediation of geographic information.",
            "- No audio is encoded; CLAP uses its text tower. Genre labels are "
            "catalog metadata, not audio-derived or human relevance judgments.",
            "- Geographic blocks have no exclusion buffers. Station families are "
            "not grouped beyond exact-stream deduplication. Boundary stations "
            "and shared branding "
            "can reduce effective independence.",
            "- Intervals are conditional on the fixed corpus, prompts, genres, "
            "sites, and partitions. Site resampling does not establish "
            "generalization to arbitrary origins.",
            "- Model and condition comparisons have no multiplicity "
            "adjustment. A positive interval is not a confirmatory significance claim.",
            "- The pilot follows exploratory UI observations. An expanded plan "
            "is a subsequent "
            "analysis, not a retrospective preregistration of the pilot.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(config_path: Path) -> None:
    results = load_results(config_path)
    summaries = summarize_results(results)
    directory = results.plan.output_directory
    write_csv(directory / "probe-summary.csv", results.probes)
    write_csv(directory / "retrieval-summary.csv", summaries.retrieval)
    write_csv(directory / "paired-regret-change.csv", summaries.paired_regret)
    write_csv(directory / "responsiveness-summary.csv", summaries.responses)
    write_csv(directory / "split-summary.csv", summaries.splits)
    write_csv(directory / "token-summary.csv", results.tokens)
    if results.coverage is not None:
        with (directory / "genre-coverage.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=results.coverage.columns)
            writer.writeheader()
            writer.writerows(results.coverage.rows)
    if results.plan.taxonomy_manifest is not None:
        (directory / "taxonomy-manifest.json").write_bytes(
            results.plan.taxonomy_manifest.read_bytes()
        )
    figure = make_figure(results, summaries)
    for extension in ["svg", "png", "pdf"]:
        figure.savefig(directory / f"location-summary.{extension}", dpi=220)
    (directory / "REPORT.md").write_text(
        report_markdown(config_path, results, summaries)
    )
    print(f"Report and scientific figures: {directory / 'REPORT.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize completed local location experiments."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/location-v1.json")
    )
    arguments = parser.parse_args()
    build_report(arguments.config)


if __name__ == "__main__":
    main()
