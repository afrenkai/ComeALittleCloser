import os
import tempfile
from pathlib import Path

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix
from come_a_little_bit_closer.models.schemas import EmbeddingManifest


@jaxtyped(typechecker=beartype)
def validate_embeddings(
    station_ids: list[str],
    embeddings: EmbeddingMatrix,
    manifest: EmbeddingManifest,
) -> None:
    if embeddings.shape != (len(station_ids), manifest.dimensions):
        raise ValueError("Embedding matrix does not match the manifest dimensions.")
    if len(station_ids) != manifest.station_count:
        raise ValueError("Station identifiers do not match the manifest count.")
    if len(set(station_ids)) != len(station_ids) or not np.isfinite(embeddings).all():
        raise ValueError("Embedding artifacts require unique IDs and finite vectors.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("Embedding artifacts must contain unit-normalized vectors.")


@jaxtyped(typechecker=beartype)
def save_embeddings(
    artifact_path: Path,
    station_ids: list[str],
    embeddings: EmbeddingMatrix,
    manifest: EmbeddingManifest,
) -> None:
    validate_embeddings(station_ids, embeddings, manifest)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=artifact_path.parent, suffix=".npz", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        np.savez_compressed(
            temporary_file,
            station_ids=np.asarray(station_ids),
            embeddings=embeddings,
            manifest=np.asarray(manifest.model_dump_json()),
        )
    os.replace(temporary_path, artifact_path)


class EmbeddingArtifact:
    def __init__(self, artifact_path: Path) -> None:
        with np.load(artifact_path, allow_pickle=False) as archive:
            self.manifest = EmbeddingManifest.model_validate_json(
                str(archive["manifest"].item())
            )
            self.station_ids: list[str] = archive["station_ids"].tolist()
            self.embeddings: EmbeddingMatrix = np.asarray(
                archive["embeddings"], dtype=np.float64
            )
        self.positions = {
            station_id: position for position, station_id in enumerate(self.station_ids)
        }
        validate_embeddings(self.station_ids, self.embeddings, self.manifest)

    @jaxtyped(typechecker=beartype)
    def select(self, station_ids: list[str]) -> EmbeddingMatrix:
        return self.embeddings[
            [self.positions[station_id] for station_id in station_ids]
        ]
