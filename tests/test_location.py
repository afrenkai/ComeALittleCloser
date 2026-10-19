from pathlib import Path

import numpy as np
import pytest

from come_a_little_bit_closer.data.genres import GenreParser, normalized_genre
from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.evaluation.counterfactual import evaluate_counterfactuals
from come_a_little_bit_closer.evaluation.location_prepare import (
    experiment_queries,
    metadata_document,
)
from come_a_little_bit_closer.evaluation.location_schemas import (
    LocationPlan,
    QueryManifest,
)
from come_a_little_bit_closer.evaluation.probes import (
    angular_errors,
    block_means,
    fit_ridge,
    predict_ridge,
    unit_predictions,
)
from come_a_little_bit_closer.evaluation.splits import (
    spatial_block,
    spatial_split,
    split_indices,
)
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import Coordinates
from come_a_little_bit_closer.models.batch import station_document
from come_a_little_bit_closer.models.schemas import EmbeddingManifest


@pytest.fixture
def location_plan() -> LocationPlan:
    return LocationPlan.model_validate_json(
        Path("experiments/location-v1.json").read_bytes()
    )


def station_fixture(
    index: int, latitude: float, longitude: float, genre: str = "rock"
) -> Station:
    return Station(
        station_id=str(index),
        name=f"Station {index}",
        stream_url=f"https://example.org/{index}",
        country_code="US",
        genre_tags=[genre],
        coordinates=Coordinates(latitude=latitude, longitude=longitude),
    )


def test_spatial_partition_reproducible_and_wraps_dateline(
    location_plan: LocationPlan,
) -> None:
    stations = [
        station_fixture(index, -70 + index * 5, -170 + index * 11)
        for index in range(28)
    ]
    first = spatial_split(stations, location_plan, 1729, "source", "plan")
    assert first == spatial_split(stations, location_plan, 1729, "source", "plan")
    assert spatial_block(0, -180, 10) == spatial_block(0, 180, 10)
    assert spatial_block(90, 0, 10) == spatial_block(89.99, 0, 10)
    for block in {assignment.block_id for assignment in first.assignments}:
        assert (
            len(
                {
                    assignment.split
                    for assignment in first.assignments
                    if assignment.block_id == block
                }
            )
            == 1
        )
    assert {assignment.split for assignment in first.assignments} == {
        "train",
        "validation",
        "test",
    }


def test_probe_recovers_sphere_without_test_leakage(
    location_plan: LocationPlan,
) -> None:
    stations = [
        station_fixture(index, -70 + index * 5, -170 + index * 11)
        for index in range(28)
    ]
    coordinates = np.asarray(
        [
            [station.coordinates.latitude, station.coordinates.longitude]
            for station in stations
        ],
        dtype=np.float64,
    )
    targets = spherical_vectors(coordinates)
    manifest = spatial_split(stations, location_plan, 1729, "source", "plan")
    train = split_indices(manifest, "train")
    validation = split_indices(manifest, "validation")
    test = split_indices(manifest, "test")
    plan = location_plan.model_copy(update={"ridge_alphas": [1e-10]})
    blocks = [assignment.block_id for assignment in manifest.assignments]
    model = fit_ridge(targets, targets, train, validation, blocks, plan)
    predictions = unit_predictions(
        predict_ridge(model, targets), np.asarray([1.0, 0.0, 0.0])
    )
    assert angular_errors(predictions[test], targets[test], 6371.0088).max() < 0.01
    altered = targets.copy()
    altered[test] = -altered[test]
    second = fit_ridge(altered, altered, train, validation, blocks, plan)
    assert np.allclose(model.weights, second.weights)
    assert np.allclose(model.feature_mean, second.feature_mean)
    assert model.alpha == second.alpha


def test_metadata_ablation_removes_only_coordinate_fields() -> None:
    station = station_fixture(0, 12.34567, -76.54321)
    assert station_document(station).startswith(
        metadata_document(station) + "\nLatitude:"
    )
    assert "12.34567" not in metadata_document(station)
    assert "Country code: US" in metadata_document(station)


def test_block_mean_metric_weights_regions_equally() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(block_means(values, ["a", "a", "a", "b"]), [2.0, 4.0])


def test_genre_vocabulary_normalizes_spelling_without_merging_styles() -> None:
    parser = GenreParser()
    parser.feed(
        '<a href="/genre/00000000-0000-0000-0000-000000000001">hip-hop</a>'
        '<a href="/country/US">US</a>'
    )
    assert [genre.name for genre in parser.genres.values()] == ["hip-hop"]
    assert normalized_genre("Hip-Hop") == normalized_genre("hip hop")
    assert normalized_genre("R&B") == normalized_genre("r and b")
    assert normalized_genre("rock") != normalized_genre("pop rock")


def test_origin_insensitive_retrieval_has_zero_movement(
    location_plan: LocationPlan,
) -> None:
    plan = location_plan.model_copy(
        update={
            "genres": ["rock"],
            "neighbors": 2,
            "origins": location_plan.origins[:2],
        }
    )
    stations = [
        station_fixture(0, 0, 0),
        station_fixture(1, 0, 100),
        station_fixture(2, 0, 50, "jazz"),
    ]
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
    )
    queries = np.tile(np.asarray([1.0, 0.0, 0.0]), (3, 1))
    manifest = QueryManifest(
        encoder_manifest=EmbeddingManifest(
            model_id="fixture",
            checkpoint="fixture",
            revision="fixture",
            source_sha256="source",
            station_count=3,
            dimensions=3,
            created_at="fixture",
            max_sequence_length=256,
            sentence_transformers_version="fixture",
        ),
        plan_sha256="plan",
        queries=experiment_queries(plan),
    )
    retrieval, responses = evaluate_counterfactuals(
        stations, vectors, queries, manifest, plan, "fixture", "fixture"
    )
    assert all(row.regret_km >= 0 for row in retrieval)
    assert all(set(row.retrieved_ids) == {"0", "1"} for row in retrieval)
    assert all(row.genre_precision_at_k == 0.5 for row in retrieval)
    assert all(
        row.neighbor_turnover == 0 and row.movement_toward_origin_km == 0
        for row in responses
    )
