import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix, EmbeddingVector
from come_a_little_bit_closer.projection.schemas import (
    ProjectionBasis,
    ProjectionMatrix,
    ProjectionVector,
)


class ProjectionFrame:
    @jaxtyped(typechecker=beartype)
    def __init__(self, embeddings: EmbeddingMatrix) -> None:
        if (
            embeddings.shape[0] < 1
            or embeddings.shape[1] < 3
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(
                "Three-dimensional PCA requires finite vectors with three features."
            )
        self.mean: EmbeddingVector = embeddings.mean(axis=0)
        centered = embeddings - self.mean
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
        positions = np.argsort(eigenvalues)[-3:][::-1]
        self.basis: ProjectionBasis = eigenvectors[:, positions]
        for component in range(3):
            pivot = np.argmax(np.abs(self.basis[:, component]))
            if self.basis[pivot, component] < 0:
                self.basis[:, component] *= -1
        total_variance = float(np.maximum(eigenvalues, 0).sum())
        self.variance_ratio: list[float] = (
            (np.maximum(eigenvalues[positions], 0) / total_variance).tolist()
            if total_variance > 0
            else [0.0, 0.0, 0.0]
        )
        projected_corpus = centered @ self.basis
        self.center: ProjectionVector = projected_corpus.mean(axis=0)
        self.radius = float(
            np.linalg.norm(projected_corpus - self.center, axis=1).max()
        )

    @jaxtyped(typechecker=beartype)
    def transform(self, embeddings: EmbeddingMatrix) -> ProjectionMatrix:
        if not np.isfinite(embeddings).all():
            raise ValueError("Projection inputs must be finite.")
        return (embeddings - self.mean) @ self.basis
