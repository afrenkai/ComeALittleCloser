import sys
from pathlib import Path

import pytest

from come_a_little_bit_closer.evaluation import location_prepare
from come_a_little_bit_closer.models import batch
from come_a_little_bit_closer.models.schemas import InferenceSettings


@pytest.mark.parametrize("flags, device", [([], "cpu"), (["--ampere"], "cuda")])
def test_embedding_cli_requires_explicit_ampere(
    flags: list[str], device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices: list[str] = []

    def capture(settings: InferenceSettings) -> None:
        devices.append(settings.device)

    monkeypatch.setattr(batch, "build_embeddings", capture)
    monkeypatch.setattr(sys, "argv", ["embed", *flags])
    batch.main()
    assert devices == [device]
    assert InferenceSettings().device == "cpu"


@pytest.mark.parametrize("flags, device", [([], "cpu"), (["--ampere"], "cuda")])
def test_preparation_cli_requires_explicit_ampere(
    flags: list[str], device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices: list[str] = []

    def capture(config_path: Path, model_id: str, selected_device: str) -> None:
        devices.append(selected_device)

    monkeypatch.setattr(location_prepare, "prepare_model", capture)
    monkeypatch.setattr(sys, "argv", ["prepare", "--model", "qwen3", *flags])
    location_prepare.main()
    assert devices == [device]
