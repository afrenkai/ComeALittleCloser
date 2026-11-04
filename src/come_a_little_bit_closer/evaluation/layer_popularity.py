import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from come_a_little_bit_closer.data.cities import load_places
from come_a_little_bit_closer.evaluation.layer_run import write_atomic
from come_a_little_bit_closer.evaluation.layer_schemas import (
    EvaluationMode,
    LayerPlan,
    PopularityCell,
    PopularityGain,
    ProbeTarget,
    SplitKind,
)
from come_a_little_bit_closer.evaluation.location_schemas import ProbeSettings
from come_a_little_bit_closer.evaluation.probes import block_means, bootstrap_interval


def popularity_gains(
    arrays: Mapping[str, np.ndarray],
    population: Mapping[str, int],
    identity: PopularityCell,
    settings: ProbeSettings,
    quantiles: int,
) -> list[PopularityGain]:
    """Gain over the country centroid per population quantile, seeds pooled.

    Each quantile's interval is a block bootstrap over the 10-degree blocks
    that hold its places, as in the main probe scores.
    """
    prefixes = sorted(
        key.removesuffix("/rows")
        for key in arrays
        if key.startswith(f"{identity.split_kind}/")
        and key.endswith(f"/{identity.probe_target}/rows")
    )
    if not prefixes:
        raise ValueError(f"No {identity.split_kind} {identity.probe_target} rows.")
    item_ids = arrays["item_ids"]
    gains = np.concatenate(
        [
            arrays[f"{prefix}/country_errors_km"] - arrays[f"{prefix}/errors_km"]
            for prefix in prefixes
        ]
    ).astype(np.float64)
    sizes = np.asarray(
        [
            population[str(item_ids[row])]
            for prefix in prefixes
            for row in arrays[f"{prefix}/rows"]
        ]
    )
    block_ids = [
        str(block_id)
        for prefix in prefixes
        for block_id in arrays[f"{prefix}/block_ids"]
    ]
    edges = np.quantile(sizes, np.linspace(0, 1, quantiles + 1)[1:-1])
    levels = np.searchsorted(edges, sizes, side="right")
    summary = []
    for level in range(quantiles):
        selected = np.flatnonzero(levels == level)
        means = block_means(
            gains[selected], [block_ids[position] for position in selected]
        )
        low, high = bootstrap_interval(means, settings)
        summary.append(
            PopularityGain(
                **identity.model_dump(),
                quantile=level + 1,
                population_min=int(sizes[selected].min()),
                population_max=int(sizes[selected].max()),
                items=len(selected),
                blocks=len(means),
                gain_over_country_km=float(means.mean()),
                gain_ci_low=low,
                gain_ci_high=high,
            )
        )
    return summary


def write_popularity(
    config: Path,
    mode: EvaluationMode,
    split_kind: SplitKind,
    probe_target: ProbeTarget,
    quantiles: int,
) -> Path:
    plan = LayerPlan.model_validate_json(config.read_bytes())
    places = load_places(
        Path("data/raw/cities15000.zip"), Path("data/raw/countryInfo.txt")
    )
    population = {place.place_id: place.population for place in places}
    rows: list[PopularityGain] = []
    for path in sorted(
        (plan.output_directory / mode / "predictions").glob("*/cities-*.npz")
    ):
        identity = PopularityCell(
            model_id=path.parent.name,
            cell=path.stem,
            mode=mode,
            split_kind=split_kind,
            probe_target=probe_target,
        )
        with np.load(path) as arrays:
            rows.extend(popularity_gains(arrays, population, identity, plan, quantiles))
    output_path = (
        plan.output_directory
        / mode
        / "popularity"
        / f"{split_kind}-{probe_target}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(
        output_path, "".join(row.model_dump_json() + "\n" for row in rows).encode()
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-country probe gain by city population quantile."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    parser.add_argument(
        "--split", choices=["random-item", "spatial-block"], default="random-item"
    )
    parser.add_argument(
        "--target",
        choices=["sphere", "latlon", "country-residual"],
        default="country-residual",
    )
    parser.add_argument("--quantiles", type=int, default=5)
    arguments = parser.parse_args()
    print(
        write_popularity(
            arguments.config,
            arguments.mode,
            arguments.split,
            arguments.target,
            arguments.quantiles,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
