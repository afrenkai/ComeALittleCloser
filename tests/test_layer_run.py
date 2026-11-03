from pathlib import Path

import numpy as np
import pytest

from come_a_little_bit_closer.evaluation import layer_run
from come_a_little_bit_closer.evaluation.layer_extract import feature_path
from come_a_little_bit_closer.evaluation.layer_report import write_report
from come_a_little_bit_closer.evaluation.layer_run import (
    run_model,
    score_country_control,
    score_layer,
)
from come_a_little_bit_closer.evaluation.layer_schemas import (
    DatasetItems,
    FeatureManifest,
    FinalCell,
    LayerPlan,
    LayerProbeScore,
)
from come_a_little_bit_closer.evaluation.location_schemas import SplitManifest
from come_a_little_bit_closer.evaluation.probes import (
    angular_errors,
    block_means,
    coordinate_centroid,
)
from come_a_little_bit_closer.evaluation.splits import random_split, split_indices
from come_a_little_bit_closer.geometry.distances import spherical_vectors
from come_a_little_bit_closer.geometry.schemas import Coordinates
from come_a_little_bit_closer.models.layer_catalog import find_layer_model


def synthetic_places(count: int) -> tuple[list[str], list[Coordinates], list[str]]:
    generator = np.random.default_rng(0)
    centers = generator.uniform([-50, -150], [60, 150], (8, 2))
    country_index = generator.integers(0, 8, count)
    points = centers[country_index] + generator.normal(0, 4, (count, 2))
    coordinates = [
        Coordinates(latitude=float(np.clip(lat, -89, 89)), longitude=float(lon))
        for lat, lon in points
    ]
    return (
        [f"p{index}" for index in range(count)],
        coordinates,
        [f"C{index}" for index in country_index],
    )


def coordinate_matrix(coordinates: list[Coordinates]) -> np.ndarray:
    return spherical_vectors(
        np.asarray([[c.latitude, c.longitude] for c in coordinates], dtype=np.float64)
    )


def tiny_plan(directory: Path) -> LayerPlan:
    return LayerPlan(
        experiment_id="tiny",
        model_ids=["xlmr-base"],
        datasets={"cities": ["name"]},
        split_kinds=["random-item"],
        split_seeds=[1],
        ridge_alphas=[0.01, 1.0],
        bootstrap_replicates=50,
        feature_directory=directory / "features",
        output_directory=directory / "outputs",
    )


def test_dev_mode_never_reads_test_targets() -> None:
    item_ids, coordinates, country_codes = synthetic_places(400)
    plan = tiny_plan(Path("unused"))
    manifest = random_split(item_ids, coordinates, plan, 1, "sha", "plan")
    targets = coordinate_matrix(coordinates)
    test = split_indices(manifest, "test")
    features = targets.copy()
    features[test] = np.nan
    targets[test] = np.nan
    for control in ["probe", "permuted-labels"]:
        values = score_layer(
            features, targets, country_codes, manifest, plan, "dev", control
        )
        assert np.isfinite(values.block_mean_km) and np.isfinite(
            values.gain_over_country_km
        )
        assert values.evaluated_items == len(split_indices(manifest, "validation"))
    control_values = score_country_control(
        targets, country_codes, manifest, plan, "dev"
    )
    assert control_values.gain_over_country_km == 0


def test_coordinate_features_beat_country_control() -> None:
    item_ids, coordinates, country_codes = synthetic_places(400)
    plan = tiny_plan(Path("unused"))
    manifest = random_split(item_ids, coordinates, plan, 1, "sha", "plan")
    targets = coordinate_matrix(coordinates)
    features = targets + np.random.default_rng(1).normal(0, 0.01, targets.shape)
    probe = score_layer(
        features, targets, country_codes, manifest, plan, "dev", "probe"
    )
    permuted = score_layer(
        features, targets, country_codes, manifest, plan, "dev", "permuted-labels"
    )
    assert probe.gain_over_country_km > 0
    assert permuted.gain_over_country_km < probe.gain_over_country_km


def test_country_without_training_items_uses_global_fallback() -> None:
    item_ids, coordinates, country_codes = synthetic_places(100)
    plan = tiny_plan(Path("unused"))
    manifest = random_split(item_ids, coordinates, plan, 1, "sha", "plan")
    validation = split_indices(manifest, "validation")
    train = split_indices(manifest, "train")
    for index in validation:
        country_codes[int(index)] = "ZZ"
    targets = coordinate_matrix(coordinates)
    values = score_country_control(targets, country_codes, manifest, plan, "dev")
    fallback = coordinate_centroid(targets[train])
    errors = angular_errors(
        np.tile(fallback, (len(validation), 1)),
        targets[validation],
        plan.geometry.earth_radius_km,
    )
    blocks = [manifest.assignments[int(index)].block_id for index in validation]
    assert values.block_mean_km == pytest.approx(block_means(errors, blocks).mean())


def final_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pooling: str = "mean",
    layer: int = 1,
    item_suffix: str = "",
    source_sha256: str = "sha",
) -> Path:
    item_ids, coordinates, country_codes = synthetic_places(200)
    monkeypatch.setattr(
        layer_run,
        "dataset_items",
        lambda dataset, condition: DatasetItems(
            item_ids=item_ids,
            texts=[],
            coordinates=coordinates,
            country_codes=country_codes,
            source_sha256="sha",
        ),
    )
    plan = tiny_plan(tmp_path).model_copy(
        update={
            "final_cells": [
                FinalCell(
                    model_id="xlmr-base",
                    dataset="cities",
                    condition="name",
                    pooling=pooling,
                    layer=layer,
                )
            ]
        }
    )
    config_path = tmp_path / "plan.json"
    config_path.write_text(plan.model_dump_json())
    path = feature_path(plan, "xlmr-base", "cities", "name", "mean")
    path.parent.mkdir(parents=True)
    targets = coordinate_matrix(coordinates)
    np.save(path, np.stack([targets, targets * 2]).astype(np.float16))
    path.with_suffix(".json").write_text(
        FeatureManifest(
            model_id="xlmr-base",
            model_spec=find_layer_model("xlmr-base"),
            checkpoint="c",
            revision="r",
            random_init=False,
            dataset="cities",
            condition="name",
            pooling="mean",
            source_sha256=source_sha256,
            item_ids=[item_id + item_suffix for item_id in item_ids],
            layers=2,
            hidden=3,
            dtype="float16",
            max_length=256,
            truncated_items=0,
            transformers_version="x",
            created_at="now",
        ).model_dump_json()
    )
    return config_path


def test_final_mode_refuses_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = final_setup(tmp_path, monkeypatch)
    output = run_model(config_path, "xlmr-base", "final")
    rows = [
        LayerProbeScore.model_validate_json(line)
        for line in output.read_text().splitlines()
    ]
    assert len(rows) == 3 and {row.mode for row in rows} == {"final"}
    with pytest.raises(FileExistsError):
        run_model(config_path, "xlmr-base", "final")


@pytest.mark.parametrize(("pooling", "layer"), [("last", 1), ("mean", 2)])
def test_invalid_final_cells_raise_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pooling: str, layer: int
) -> None:
    config_path = final_setup(tmp_path, monkeypatch, pooling, layer)
    with pytest.raises(ValueError, match="Invalid final cells"):
        run_model(config_path, "xlmr-base", "final")
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("item_suffix", "source_sha256", "message"),
    [("x", "sha", "item IDs"), ("", "other", "source checksum")],
)
def test_feature_provenance_mismatch_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    item_suffix: str,
    source_sha256: str,
    message: str,
) -> None:
    config_path = final_setup(
        tmp_path, monkeypatch, item_suffix=item_suffix, source_sha256=source_sha256
    )
    with pytest.raises(ValueError, match=message):
        run_model(config_path, "xlmr-base", "final")


def test_dev_run_writes_plan_copy_and_split_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = final_setup(tmp_path, monkeypatch)
    output = run_model(config_path, "xlmr-base", "dev")
    assert (
        output.with_name("xlmr-base.plan.json").read_bytes() == config_path.read_bytes()
    )
    split_file = output.parent / "splits" / "cities-random-item-1.json"
    assert SplitManifest.model_validate_json(split_file.read_bytes()).seed == 1


def test_final_run_refuses_differing_split_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = final_setup(tmp_path, monkeypatch)
    split_file = tmp_path / "outputs" / "final" / "splits" / "cities-random-item-1.json"
    split_file.parent.mkdir(parents=True)
    split_file.write_text("{}")
    with pytest.raises(FileExistsError, match="differing final split"):
        run_model(config_path, "xlmr-base", "final")


def test_unequal_items_and_coordinates_raise() -> None:
    with pytest.raises(ValueError, match="equal length"):
        random_split(["a"], [], tiny_plan(Path("unused")), 1, "sha", "plan")


def test_plan_rejects_inconsistent_final_cells(tmp_path: Path) -> None:
    plan = tiny_plan(tmp_path)
    cell = FinalCell(
        model_id="xlmr-base",
        dataset="cities",
        condition="name",
        pooling="mean",
        layer=1,
    )
    with pytest.raises(ValueError, match="unique"):
        LayerPlan(**{**plan.model_dump(), "final_cells": [cell, cell]})
    with pytest.raises(ValueError, match="outside the plan"):
        LayerPlan(
            **{
                **plan.model_dump(),
                "final_cells": [cell.model_copy(update={"condition": "name-country"})],
            }
        )
    with pytest.raises(ValueError, match="Invalid conditions"):
        LayerPlan(**{**plan.model_dump(), "datasets": {"stations": ["name"]}})


def test_final_mode_report_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Final-mode reporting is not implemented"):
        write_report(tmp_path / "unused.json", "final")
