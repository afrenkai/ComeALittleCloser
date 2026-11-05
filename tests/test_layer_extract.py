import json
from pathlib import Path

import numpy as np
import pytest

from come_a_little_bit_closer.evaluation.layer_extract import (
    existing_features_match,
    feature_path,
    write_features,
)
from come_a_little_bit_closer.evaluation.layer_schemas import FeatureManifest, LayerPlan
from come_a_little_bit_closer.models.hidden_states import (
    LayerExtractor,
)
from come_a_little_bit_closer.models.layer_catalog import find_layer_model
from tests.test_hidden_states import save_tiny_bert, tiny_spec


def tiny_plan(directory: Path, max_length: int) -> LayerPlan:
    return LayerPlan(
        experiment_id="tiny",
        model_ids=["tiny-bert"],
        datasets={"cities": ["name"]},
        split_kinds=["random-item"],
        split_seeds=[1],
        ridge_alphas=[1.0],
        max_length=max_length,
        batch_size=2,
        feature_directory=directory / "features",
        output_directory=directory / "outputs",
    )


def test_layer_plan_parses() -> None:
    plan = LayerPlan.model_validate_json(
        Path("experiments/layers-v1.json").read_bytes()
    )
    assert len(plan.model_ids) == 12
    for model_id in plan.model_ids:
        assert find_layer_model(model_id).model_id == model_id


def test_long_text_is_truncated_and_counted(tmp_path: Path) -> None:
    save_tiny_bert(tmp_path / "model")
    plan = tiny_plan(tmp_path, max_length=8)
    extractor = LayerExtractor(
        tiny_spec(tmp_path / "model"), device="cpu", max_length=plan.max_length
    )
    long_text = " ".join(["rock"] * 40)
    paths = write_features(
        plan, extractor, "cities", "name", ["a", "b"], [long_text, "jazz"], "sha"
    )
    features = np.load(paths[0], mmap_mode="r")
    assert features.shape == (3, 2, 16)
    manifest = FeatureManifest.model_validate_json(
        paths[0].with_suffix(".json").read_text()
    )
    assert manifest.truncated_items == 1 and manifest.item_ids == ["a", "b"]
    assert extractor.count_truncated(["rock", long_text]) == 1
    decoder = LayerExtractor(
        tiny_spec(tmp_path / "model", architecture="decoder", end_token="[SEP]"),
        device="cpu",
        max_length=4,
    )
    assert decoder.count_truncated(["rock rock rock rock", "rock rock rock"]) == 1


def test_extract_rejects_mismatched_existing_features(tmp_path: Path) -> None:
    save_tiny_bert(tmp_path / "model")
    plan = tiny_plan(tmp_path, max_length=8)
    extractor = LayerExtractor(
        tiny_spec(tmp_path / "model"), device="cpu", max_length=plan.max_length
    )
    paths = write_features(plan, extractor, "cities", "name", ["a"], ["rock"], "sha")
    manifest_path = paths[0].with_suffix(".json")
    assert paths[0] == feature_path(plan, "tiny-bert", "cities", "name", "mean")
    expected = FeatureManifest.model_validate_json(manifest_path.read_text())
    assert not existing_features_match(tmp_path / "missing.json", expected)
    assert existing_features_match(
        manifest_path, expected.model_copy(update={"created_at": "later"})
    )
    changed = expected.model_copy(update={"revision": "other"})
    with pytest.raises(ValueError, match="does not match this run"):
        existing_features_match(manifest_path, changed)
    again = write_features(plan, extractor, "cities", "name", ["a"], ["rock"], "sha")
    assert again == paths
    with pytest.raises(ValueError, match="does not match this run"):
        write_features(plan, extractor, "cities", "name", ["b"], ["rock"], "sha")
    assert json.loads(manifest_path.read_text())["item_ids"] == ["a"]


def test_incomplete_array_is_rewritten_and_files_are_group_readable(
    tmp_path: Path,
) -> None:
    save_tiny_bert(tmp_path / "model")
    plan = tiny_plan(tmp_path, max_length=8)
    extractor = LayerExtractor(
        tiny_spec(tmp_path / "model"), device="cpu", max_length=plan.max_length
    )
    paths = write_features(plan, extractor, "cities", "name", ["a"], ["rock"], "sha")
    assert paths[0].stat().st_mode & 0o777 == 0o640
    assert paths[0].with_suffix(".json").stat().st_mode & 0o777 == 0o640
    paths[0].unlink()
    write_features(plan, extractor, "cities", "name", ["a"], ["rock"], "sha")
    assert np.load(paths[0], mmap_mode="r").shape == (3, 1, 16)
    assert list(paths[0].parent.glob("tmp*")) == []
