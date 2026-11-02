from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from come_a_little_bit_closer.models.hidden_states import (
    LayerExtractor,
    checked_float16,
    last_token_pool,
    mean_pool,
)
from come_a_little_bit_closer.models.layer_catalog import (
    find_layer_model,
    layer_model_specs,
)
from come_a_little_bit_closer.models.schemas import LayerModelSpec


def test_mean_pool_ignores_padding_and_special_tokens() -> None:
    states = torch.tensor(
        [
            [[100.0, 100.0], [2.0, 4.0], [4.0, 8.0], [100.0, 100.0]],
            [[1.0, 1.0], [3.0, 5.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )
    mask = torch.tensor([[False, True, True, False], [False, True, False, False]])
    pooled = mean_pool(states, mask)
    assert torch.equal(pooled, torch.tensor([[3.0, 6.0], [3.0, 5.0]]))
    with pytest.raises(ValueError):
        mean_pool(states, torch.zeros_like(mask))


def test_last_token_pool_left_and_right_padding() -> None:
    sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    padding = torch.zeros(2, 2)
    right = torch.cat([sequence, padding])[None]
    left = torch.cat([padding, sequence])[None]
    right_mask = torch.tensor([[True, True, True, False, False]])
    left_mask = torch.tensor([[False, False, True, True, True]])
    expected = torch.tensor([[5.0, 6.0]])
    assert torch.equal(last_token_pool(right, right_mask), expected)
    assert torch.equal(last_token_pool(left, left_mask), expected)


def test_checked_float16_rejects_overflow() -> None:
    assert checked_float16(np.array([1.5], dtype=np.float32)).dtype == np.float16
    with pytest.raises(ValueError):
        checked_float16(np.array([70000.0], dtype=np.float32))
    with pytest.raises(ValueError):
        checked_float16(np.array([np.nan], dtype=np.float32))


def test_layer_extractor_tiny_bert_shapes_and_order(tmp_path: Path) -> None:
    torch.manual_seed(1729)
    tokenizer = BertTokenizerFast(
        vocab={
            token: index
            for index, token in enumerate(
                ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "rock", "jazz"]
                + ["station", "near"]
            )
        }
    )
    tokenizer.save_pretrained(tmp_path)
    BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    ).save_pretrained(tmp_path)
    spec = LayerModelSpec(
        model_id="tiny-bert",
        family="tiny",
        stage="base",
        stage_order=1,
        checkpoint=str(tmp_path),
        revision="local-fixture",
        architecture="encoder",
    )
    texts = ["rock", "jazz station near rock"]
    one = LayerExtractor(spec, device="cpu", max_length=32).extract(texts, 1)
    two = LayerExtractor(spec, device="cpu", max_length=32).extract(texts, 2)
    assert list(one) == ["mean"]
    assert one["mean"].shape == (3, 2, 16)
    assert one["mean"].dtype == np.float16
    np.testing.assert_allclose(one["mean"], two["mean"], atol=2e-3)
    assert not np.allclose(one["mean"][:, 0], one["mean"][:, 1])


def test_registry_pins_every_revision() -> None:
    specs = layer_model_specs()
    ids = [spec.model_id for spec in specs]
    assert len(ids) == len(set(ids)) == 12
    assert all(
        len(spec.revision) == 40 and int(spec.revision, 16) >= 0 for spec in specs
    )
    assert find_layer_model("bge-m3").stage == "retrieval"
    for family in {spec.family for spec in specs if not spec.random_init}:
        members = [
            spec for spec in specs if spec.family == family and not spec.random_init
        ]
        top = max(spec.stage_order for spec in members)
        finals = [spec for spec in members if spec.stage_order == top]
        assert len(finals) == 1 and finals[0].stage == "retrieval"
    with pytest.raises(ValueError):
        find_layer_model("missing")
