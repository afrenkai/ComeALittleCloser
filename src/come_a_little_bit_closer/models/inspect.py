import argparse
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from come_a_little_bit_closer.models.artifacts import EmbeddingArtifact
from come_a_little_bit_closer.models.schemas import (
    ArtifactValidation,
    InferenceSettings,
)


def inspect_artifact(artifact_path: Path) -> ArtifactValidation:
    artifact = EmbeddingArtifact(artifact_path)
    return ArtifactValidation(
        artifact_path=artifact_path,
        manifest=artifact.manifest,
        maximum_norm_error=float(
            np.max(np.abs(np.linalg.norm(artifact.embeddings, axis=1) - 1))
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate cached embedding provenance."
    )
    parser.add_argument(
        "--directory", type=Path, default=InferenceSettings().embedding_directory
    )
    arguments = parser.parse_args()
    reports = [
        inspect_artifact(artifact_path)
        for artifact_path in sorted(arguments.directory.glob("*.npz"))
        if not artifact_path.name.startswith("tmp")
    ]
    print(TypeAdapter(list[ArtifactValidation]).dump_json(reports, indent=2).decode())


if __name__ == "__main__":
    main()
