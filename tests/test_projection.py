import numpy as np
import pytest

from come_a_little_bit_closer.models.encoder import normalize_vectors
from come_a_little_bit_closer.projection.clustering import cluster_embeddings
from come_a_little_bit_closer.projection.pca import ProjectionFrame
from come_a_little_bit_closer.projection.schemas import ProjectionSettings


def test_precision_normalization_and_invalid_model_output() -> None:
    rounded_embeddings = np.array([[0.707, 0.707], [0.0001, 0.0]])
    normalized = normalize_vectors(rounded_embeddings)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-14)
    with pytest.raises(ValueError, match="nonzero"):
        normalize_vectors(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        normalize_vectors(np.array([[np.nan, 1.0]]))


def test_projection_centers_constant_corpus_without_nan() -> None:
    constant_embeddings = np.ones((3, 4))
    frame = ProjectionFrame(constant_embeddings)
    np.testing.assert_allclose(frame.transform(constant_embeddings), 0.0)
    assert frame.variance_ratio == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="finite"):
        frame.transform(np.array([[np.nan, 1.0, 1.0, 1.0]]))


def test_projection_retains_three_independent_axes() -> None:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ]
    )
    frame = ProjectionFrame(vectors)
    projected = frame.transform(vectors)
    assert projected.shape == (6, 3)
    np.testing.assert_allclose(frame.variance_ratio, [18 / 28, 8 / 28, 2 / 28])
    np.testing.assert_allclose(
        np.linalg.norm(projected, axis=1), np.linalg.norm(vectors, axis=1)
    )


def test_cosine_clusters_are_stable_and_use_native_vectors() -> None:
    vectors = np.array(
        [[1.0, 0.01, 0.0], [1.0, -0.01, 0.0], [0.0, 0.01, 1.0], [0.0, -0.01, 1.0]]
    )
    settings = ProjectionSettings(cluster_count=2)
    labels = cluster_embeddings(vectors, settings)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    np.testing.assert_array_equal(labels, cluster_embeddings(vectors * 7, settings))
    assert len(np.unique(cluster_embeddings(np.ones((3, 4)), settings))) == 1
