import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from come_a_little_bit_closer.evaluation.fix_schemas import FixPlan
from come_a_little_bit_closer.evaluation.geo_fix import (
    genre_agreement,
    probe_distances,
    train_distance_probe,
)
from come_a_little_bit_closer.evaluation.instructed_extract import read_sts
from come_a_little_bit_closer.evaluation.layer_reflected import kilometres, spearman
from come_a_little_bit_closer.geometry.distances import spherical_vectors


def small_plan(tmp_path: Path) -> FixPlan:
    return FixPlan(
        experiment_id="test",
        layers_plan=tmp_path / "layers.json",
        models=[],
        instruction="geo",
        instructed_directory=tmp_path,
        output_directory=tmp_path,
        sts_files={"dev": tmp_path / "dev.gz", "test": tmp_path / "test.gz"},
        sts_sha256={"dev": "x", "test": "x"},
        probe_rank=4,
        probe_steps=300,
        probe_batch=256,
        probe_learning_rate=1e-2,
    )


def test_distance_probe_recovers_distance_hidden_among_noise(tmp_path: Path) -> None:
    generator = np.random.default_rng(0)
    latlon = np.column_stack(
        [generator.uniform(40, 50, 400), generator.uniform(0, 10, 400)]
    )
    positions = spherical_vectors(latlon)
    signal = (latlon - latlon.mean(axis=0)) @ generator.normal(size=(2, 8)) * 0.3
    noise = 3 * generator.normal(size=(400, 120))
    vectors = np.hstack([signal, noise]).astype(np.float32)
    codes = np.asarray(["AA"] * 200 + ["BB"] * 200)
    probe = train_distance_probe(vectors, positions, codes, small_plan(tmp_path), 0)
    learned = probe_distances(vectors, probe)
    true_km = kilometres(positions @ positions.T)
    upper = np.triu_indices(400, 1)
    assert spearman(-learned[upper], -true_km[upper]) > 0.5
    raw = vectors @ vectors.T
    assert spearman(raw[upper], -true_km[upper]) < 0.3


def test_sts_pairs_share_sentences_and_reject_wrong_checksum(tmp_path: Path) -> None:
    records = [
        {"sentence1": "a", "sentence2": "b", "score": 1.0},
        {"sentence1": "b", "sentence2": "c", "score": 4.5},
    ]
    content = gzip.compress("\n".join(json.dumps(r) for r in records).encode())
    path = tmp_path / "sts.jsonl.gz"
    path.write_bytes(content)
    pairs = read_sts(path, hashlib.sha256(content).hexdigest())
    assert pairs.sentences == ["a", "b", "c"]
    assert pairs.first.tolist() == [0, 1] and pairs.second.tolist() == [1, 2]
    with pytest.raises(ValueError):
        read_sts(path, "0" * 64)


def test_genre_agreement_ignores_untagged_items() -> None:
    score = np.asarray([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=np.float32)
    genres = [{"rock"}, {"rock", "pop"}, set()]
    assert genre_agreement(score, genres, 1) == 1.0
