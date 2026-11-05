import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from come_a_little_bit_closer.evaluation.layer_extract import (
    atomic_write,
    dataset_items,
)
from come_a_little_bit_closer.models.hidden_states import LayerExtractor
from come_a_little_bit_closer.models.layer_catalog import find_layer_model
from come_a_little_bit_closer.models.layer_schemas import ParityResult


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def check_parity(
    model_id: str, pooling: str, prefix: str, device: str, sample: int
) -> ParityResult:
    """Compare our output-layer pooling with Sentence Transformers' embedding.

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
    reference = SentenceTransformer(
        spec.checkpoint,
        revision=spec.revision,
        device=device,
        model_kwargs={"dtype": torch.float32},
    )
    theirs = unit_rows(
        reference.encode(texts, prompt="", batch_size=8, convert_to_numpy=True)
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
    arguments = parser.parse_args()
    result = check_parity(
        arguments.model,
        arguments.pooling,
        arguments.prefix,
        "cuda" if arguments.ampere else "cpu",
        arguments.sample,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    atomic_write(
        arguments.output / f"{arguments.model}.json",
        lambda handle: handle.write((result.model_dump_json(indent=2) + "\n").encode()),
    )
    print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
