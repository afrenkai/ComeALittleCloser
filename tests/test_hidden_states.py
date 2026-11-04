import re
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
from come_a_little_bit_closer.models.layer_schemas import LayerModelSpec


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


VOCAB = [
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


def save_tiny_bert(path: Path, dtype: torch.dtype = torch.float32) -> BertModel:
    torch.manual_seed(1729)
    BertTokenizerFast(
        vocab={token: index for index, token in enumerate(VOCAB)}, eos_token="[SEP]"
    ).save_pretrained(path)
    model = BertModel(
        BertConfig(
            vocab_size=len(VOCAB),
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            dtype=dtype,
        )
    )
    model.save_pretrained(path)
    return model


def tiny_spec(path: Path, **fields: object) -> LayerModelSpec:
    return LayerModelSpec(
        model_id="tiny-bert",
        family="tiny",
        stage="base",
        stage_order=1,
        checkpoint=str(path),
        revision="local-fixture",
        **{"architecture": "encoder", **fields},
    )


def test_layer_extractor_tiny_bert_shapes_order_and_special_tokens(
    tmp_path: Path,
) -> None:
    save_tiny_bert(tmp_path)
    extractor = LayerExtractor(tiny_spec(tmp_path), device="cpu", max_length=32)
    texts = ["jazz station near rock", "rock"]
    one = extractor.extract(texts, 1)["mean"]
    two = extractor.extract(texts, 2)["mean"]
    assert one.dtype == np.float16 and one.shape == (3, 2, 16)
    np.testing.assert_allclose(
        one.astype(np.float32), two.astype(np.float32), rtol=1e-2, atol=1e-2
    )
    for column, text in enumerate(texts):
        alone = extractor.extract([text], 1)["mean"][:, 0]
        np.testing.assert_allclose(
            one[:, column].astype(np.float32),
            alone.astype(np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
    encoded = extractor.tokenizer(texts[0], return_tensors="pt")
    with torch.no_grad():
        embeddings = extractor.model(
            **encoded, output_hidden_states=True
        ).hidden_states[0][0]
    np.testing.assert_allclose(
        one[0, 0].astype(np.float32),
        embeddings[1:-1].mean(dim=0).numpy(),
        rtol=1e-2,
        atol=1e-2,
    )
    with pytest.raises(ValueError):
        extractor.extract([], 1)


def test_decoder_appends_one_eos_excluded_from_mean(tmp_path: Path) -> None:
    save_tiny_bert(tmp_path)
    extractor = LayerExtractor(
        tiny_spec(tmp_path, architecture="decoder", end_token="[SEP]"),
        device="cpu",
        max_length=4,
    )
    batch = extractor.encode(["rock", "jazz station near rock"])
    assert batch.input_ids[0, :2].tolist() == [5, 3]
    assert batch.attention_mask.sum(dim=1).tolist() == [2, 4]
    assert batch.input_ids[1].tolist() == [6, 7, 8, 3]
    assert batch.content_mask.sum(dim=1).tolist() == [1, 3]
    features = extractor.extract(["rock"], 1)
    with torch.no_grad():
        states = extractor.model(
            input_ids=torch.tensor([[5, 3]]), output_hidden_states=True
        ).hidden_states
    for layer, state in enumerate(states):
        np.testing.assert_allclose(
            features["last"][layer, 0].astype(np.float32),
            state[0, 1].numpy(),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            features["mean"][layer, 0].astype(np.float32),
            state[0, 0].numpy(),
            rtol=1e-2,
            atol=1e-2,
        )


def test_random_init_is_float32_deterministic_and_preserves_global_rng(
    tmp_path: Path,
) -> None:
    save_tiny_bert(tmp_path, dtype=torch.bfloat16)
    spec = tiny_spec(tmp_path, random_init=True)
    state = torch.get_rng_state()
    first = LayerExtractor(spec, device="cpu", max_length=32)
    assert torch.equal(state, torch.get_rng_state())
    second = LayerExtractor(spec, device="cpu", max_length=32)
    assert {parameter.dtype for parameter in first.model.parameters()} == {
        torch.float32
    }
    texts = ["rock", "jazz station"]
    np.testing.assert_array_equal(
        first.extract(texts, 2)["mean"], second.extract(texts, 2)["mean"]
    )


def test_registry_pins_every_revision() -> None:
    specs = layer_model_specs()
    ids = [spec.model_id for spec in specs]
    assert len(ids) == len(set(ids)) == 18
    assert all(re.fullmatch("[0-9a-f]{40}", spec.revision) for spec in specs)
    assert find_layer_model("bge-m3").stage == "retrieval"
    assert find_layer_model("qwen3-8b-base").stage == "base"
    assert find_layer_model("qwen3-8b-embedding").family == "qwen3-8b"
    for family in {spec.family for spec in specs if not spec.random_init}:
        members = [
            spec for spec in specs if spec.family == family and not spec.random_init
        ]
        top = max(spec.stage_order for spec in members)
        finals = [spec for spec in members if spec.stage_order == top]
        assert len(finals) == 1 and finals[0].stage == "retrieval"
    with pytest.raises(ValueError):
        find_layer_model("missing")


def test_decoder_start_token_is_prepended_and_excluded_from_mean(
    tmp_path: Path,
) -> None:
    save_tiny_bert(tmp_path)
    extractor = LayerExtractor(
        tiny_spec(
            tmp_path, architecture="decoder", start_token="[CLS]", end_token="[SEP]"
        ),
        device="cpu",
        max_length=4,
    )
    batch = extractor.encode(["rock", "jazz station near rock"])
    assert batch.input_ids[0, :3].tolist() == [2, 5, 3]
    assert batch.input_ids[1].tolist() == [2, 6, 7, 3]
    assert batch.attention_mask.sum(dim=1).tolist() == [3, 4]
    assert batch.content_mask[:, 0].tolist() == [False, False]
    assert batch.content_mask.sum(dim=1).tolist() == [1, 2]
    assert extractor.count_truncated(["jazz station near", "rock station"]) == 1
