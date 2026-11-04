import numpy as np

from come_a_little_bit_closer.evaluation.layer_reflected import (
    country_reflection,
    kilometres,
    neighbor_positions,
    spearman,
)
from come_a_little_bit_closer.geometry.distances import spherical_vectors


def test_true_closeness_is_perfectly_reflected_and_random_is_not() -> None:
    generator = np.random.default_rng(0)
    points = spherical_vectors(
        np.column_stack([generator.uniform(30, 50, 40), generator.uniform(0, 20, 40)])
    )
    true_km = kilometres(points @ points.T)
    rho, gain = country_reflection(-true_km, true_km, 5)
    assert np.isclose(rho, 1.0) and gain > 0.5
    noise = generator.normal(size=(40, 40))
    noise_rho, noise_gain = country_reflection(noise + noise.T, true_km, 5)
    assert abs(noise_rho) < 0.1 and abs(noise_gain) < 0.25
    assert np.isclose(spearman(np.arange(5.0), -np.arange(5.0)), -1.0)


def test_neighbor_positions_stay_in_country_and_mark_unseen_countries() -> None:
    database = np.eye(3)
    database_truth = spherical_vectors(
        np.asarray([[10.0, 10.0], [20.0, 20.0], [0.0, 0.0]])
    )
    queries = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions = neighbor_positions(
        queries,
        database,
        database_truth,
        np.asarray(["AA", "CC"]),
        np.asarray(["AA", "AA", "BB"]),
        1,
    )
    assert np.allclose(positions[0], database_truth[0])
    assert np.isnan(positions[1]).all()
