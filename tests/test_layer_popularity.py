import numpy as np

from come_a_little_bit_closer.evaluation.layer_popularity import popularity_gains
from come_a_little_bit_closer.evaluation.layer_schemas import PopularityCell
from come_a_little_bit_closer.evaluation.location_schemas import ProbeSettings


def test_gain_lands_in_the_most_populous_quantile_across_seeds() -> None:
    item_ids = np.asarray([str(index) for index in range(20)])
    arrays: dict[str, np.ndarray] = {"item_ids": item_ids}
    for seed, rows in [(1, np.arange(0, 20, 2)), (2, np.arange(1, 20, 2))]:
        prefix = f"random-item/{seed}/country-residual"
        arrays[f"{prefix}/rows"] = rows
        arrays[f"{prefix}/block_ids"] = np.asarray(["a", "b"] * 5)
        arrays[f"{prefix}/country_errors_km"] = np.where(rows >= 16, 100.0, 50.0)
        arrays[f"{prefix}/errors_km"] = np.full(len(rows), 50.0)
    arrays["spatial-block/1/country-residual/rows"] = np.arange(3)
    population = {str(index): 1000 * (index + 1) for index in range(20)}
    identity = PopularityCell(
        model_id="model",
        cell="cities-name-country-mean-3",
        mode="dev",
        split_kind="random-item",
        probe_target="country-residual",
    )
    rows = popularity_gains(
        arrays,
        population,
        identity,
        ProbeSettings(split_seeds=[1], ridge_alphas=[1.0]),
        5,
    )
    assert [row.items for row in rows] == [4] * 5
    assert [row.gain_over_country_km for row in rows] == [0.0] * 4 + [50.0]
    assert rows[-1].population_min == 17000 and rows[-1].blocks == 2
    assert rows[-1].gain_ci_low == rows[-1].gain_ci_high == 50.0
