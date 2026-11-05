import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from come_a_little_bit_closer.evaluation.layer_extract import (
    atomic_write,
    dataset_items,
)
from come_a_little_bit_closer.models.hidden_states import LayerExtractor
from come_a_little_bit_closer.models.layer_catalog import find_layer_model
from come_a_little_bit_closer.models.layer_schemas import ParityResult


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def card_reference(
    checkpoint: str, revision: str, texts: list[str], device: str
) -> np.ndarray:
    """Last-token embeddings as in the model card's Transformers example: left
    padding, the tokenizer's own special tokens, the final position."""
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, revision=revision, padding_side="left"
    )
    model = AutoModel.from_pretrained(
        checkpoint, revision=revision, dtype=torch.float32
    ).to(device)
    model.eval()
    batches = []
    for start in range(0, len(texts), 8):
        inputs = tokenizer(
            texts[start : start + 8],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            states = model(**inputs).last_hidden_state[:, -1, :]
        batches.append(states.float().cpu().numpy())
    return np.concatenate(batches)


def check_parity(
    model_id: str,
    pooling: str,
    prefix: str,
    device: str,
    sample: int,
    saved_modules: bool,
) -> ParityResult:
    """Compare our output-layer pooling with Sentence Transformers' embedding, or,
    when the saved modules do not load, the model card's Transformers example.

    Texts are city names, as is, and wrapped in a Qwen-style instruction.
    """
    spec = find_layer_model(model_id)
    cities = dataset_items("cities", "name-country").texts[:sample]
    texts = [prefix + text for text in cities] + [
        f"{prefix}Instruct: Given the name of a place, retrieve places that are"
        f" geographically closest to it\nQuery: {text}"
        for text in cities[: sample // 4]
    ]
    extractor = LayerExtractor(spec, device, 256)
    ours = unit_rows(extractor.extract(texts, 8)[pooling][-1].astype(np.float32))
    del extractor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    if saved_modules:
        reference = SentenceTransformer(
            spec.checkpoint,
            revision=spec.revision,
            device=None if spec.spread_over_gpus else device,
            model_kwargs={
                "dtype": torch.float32,
                "device_map": "auto" if spec.spread_over_gpus else None,
            },
        )
        theirs = unit_rows(
            reference.encode(texts, prompt="", batch_size=8, convert_to_numpy=True)
        )
    else:
        theirs = unit_rows(
            card_reference(spec.checkpoint, spec.revision, texts, device)
        )
    cosines = (ours * theirs).sum(axis=1)
    return ParityResult(
        model_id=model_id,
        pooling=pooling,
        prefix=prefix,
        texts=len(texts),
        minimum_cosine=float(cosines.min()),
        mean_cosine=float(cosines.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check our pooled features against the model's own embedding."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--pooling", choices=["last", "mean"], required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--sample", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("outputs/parity"))
    parser.add_argument("--ampere", action="store_true")
    parser.add_argument("--without-saved-modules", action="store_true")
    arguments = parser.parse_args()
    result = check_parity(
        arguments.model,
        arguments.pooling,
        arguments.prefix,
        "cuda" if arguments.ampere else "cpu",
        arguments.sample,
        not arguments.without_saved_modules,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    atomic_write(
        arguments.output / f"{arguments.model}.json",
        lambda handle: handle.write((result.model_dump_json(indent=2) + "\n").encode()),
    )
    print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
