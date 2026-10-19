from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import (
    Normalize,
    Pooling,
    Transformer,
)
from transformers import BertConfig, BertModel, BertTokenizerFast

from come_a_little_bit_closer.models.artifacts import EmbeddingArtifact, save_embeddings
from come_a_little_bit_closer.models.encoder import TextEncoder
from come_a_little_bit_closer.models.schemas import EmbeddingManifest, ModelSpec


def test_tiny_cpu_encoder_and_artifact_round_trip(tmp_path: Path) -> None:
    assert torch.version.cuda is None
    torch.manual_seed(1729)
    tokenizer = BertTokenizerFast(
        vocab={
            token: index
            for index, token in enumerate(
                [
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]",
                    "[MASK]",
                    "rock",
                    "jazz",
                    "station",
                    "near",
                ]
            )
        }
    )
    transformer_path = tmp_path / "transformer"
    tokenizer.save_pretrained(transformer_path)
    transformer = BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )
    transformer.save_pretrained(transformer_path)
    model_path = tmp_path / "sentence-model"
    SentenceTransformer(
        modules=[
            Transformer(str(transformer_path)),
            Pooling(16),
            Normalize(),
        ],
        device="cpu",
    ).save(str(model_path))
    spec = ModelSpec(
        model_id="tiny-cpu",
        label="Tiny random BERT fixture",
        checkpoint=str(model_path),
        description="Local randomly initialized test encoder; no semantic claim.",
    )
    encoder = TextEncoder(
        spec, revision="local-fixture", device="cpu", max_sequence_length=32
    )
    texts = ["rock station", "jazz station"]
    vectors = encoder.documents(texts, batch_size=2)
    query = encoder.query(texts[0])
    assert vectors.shape == (2, 16)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(query, vectors[0], atol=1e-6)
    assert not np.allclose(vectors[0], vectors[1])
    assert encoder.token_lengths(texts, document=True) == [4, 4]
    artifact_path = tmp_path / "tiny.npz"
    save_embeddings(
        artifact_path,
        ["rock", "jazz"],
        vectors,
        EmbeddingManifest(
            model_id=spec.model_id,
            checkpoint=spec.checkpoint,
            revision="local-fixture",
            source_sha256="fixture",
            station_count=2,
            dimensions=16,
            created_at="fixture",
            max_sequence_length=32,
            sentence_transformers_version="fixture",
        ),
    )
    restored = EmbeddingArtifact(artifact_path)
    np.testing.assert_allclose(restored.select(["jazz", "rock"]), vectors[[1, 0]])
