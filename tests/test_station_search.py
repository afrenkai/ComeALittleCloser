import numpy as np

from come_a_little_bit_closer.evaluation.station_search import search_gains
from come_a_little_bit_closer.geometry.distances import spherical_vectors


def test_search_gain_rewards_near_stations_and_is_zero_for_whole_pool() -> None:
    stations = spherical_vectors(
        np.asarray([[45.0, 4.0 + step] for step in np.linspace(0, 5, 30)])
    ).astype(np.float32)
    query = spherical_vectors(np.asarray([[45.0, 4.0]])).astype(np.float32)
    pool = [np.arange(30)]
    closest = search_gains(query @ stations.T, pool, query, stations, 10)
    farthest = search_gains(-(query @ stations.T), pool, query, stations, 10)
    everything = search_gains(query @ stations.T, pool, query, stations, 30)
    assert closest[0] > 0.6 and farthest[0] < -0.4
    assert abs(everything[0]) < 1e-6
