import math

import numpy as np

from come_a_little_bit_closer.evaluation.location_schemas import (
    IndexVector,
    ProbeSettings,
    SplitManifest,
    SplitName,
    StationSplit,
)
from come_a_little_bit_closer.geometry.schemas import Coordinates


def spatial_block(latitude: float, longitude: float, degrees: float) -> str:
    latitude_index = min(
        math.ceil(180 / degrees) - 1, math.floor((latitude + 90) / degrees)
    )
    longitude_index = math.floor(((longitude + 180) % 360) / degrees)
    return f"{latitude_index}:{longitude_index}"


def spatial_split(
    item_ids: list[str],
    coordinates: list[Coordinates],
    settings: ProbeSettings,
    seed: int,
    source_sha256: str,
    plan_sha256: str,
) -> SplitManifest:
    if len(item_ids) != len(coordinates):
        raise ValueError("Item IDs and coordinates must have equal length.")
    if settings.train_fraction + settings.validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition.")
    station_blocks = [
        spatial_block(coordinate.latitude, coordinate.longitude, settings.block_degrees)
        for coordinate in coordinates
    ]
    blocks = sorted(set(station_blocks))
    if len(blocks) < 3:
        raise ValueError("Spatial splitting requires at least three occupied blocks.")
    shuffled = np.random.default_rng(seed).permutation(len(blocks))
    train_end = max(1, min(len(blocks) - 2, int(len(blocks) * settings.train_fraction)))
    validation_end = max(
        train_end + 1,
        min(
            len(blocks) - 1,
            int(len(blocks) * (settings.train_fraction + settings.validation_fraction)),
        ),
    )
    partitions: dict[str, SplitName] = {}
    for position, block_index in enumerate(shuffled):
        split: SplitName = (
            "train"
            if position < train_end
            else "validation" if position < validation_end else "test"
        )
        partitions[blocks[int(block_index)]] = split
    return SplitManifest(
        seed=seed,
        source_sha256=source_sha256,
        plan_sha256=plan_sha256,
        block_degrees=settings.block_degrees,
        assignments=[
            StationSplit(
                station_id=item_id,
                block_id=station_blocks[position],
                split=partitions[station_blocks[position]],
            )
            for position, item_id in enumerate(item_ids)
        ],
    )


def random_split(
    item_ids: list[str],
    coordinates: list[Coordinates],
    settings: ProbeSettings,
    seed: int,
    source_sha256: str,
    plan_sha256: str,
) -> SplitManifest:
    if len(item_ids) != len(coordinates):
        raise ValueError("Item IDs and coordinates must have equal length.")
    if settings.train_fraction + settings.validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition.")
    train_end = int(len(item_ids) * settings.train_fraction)
    validation_end = int(
        len(item_ids) * (settings.train_fraction + settings.validation_fraction)
    )
    names: list[SplitName] = ["test"] * len(item_ids)
    for rank, position in enumerate(
        np.random.default_rng(seed).permutation(len(item_ids))
    ):
        names[int(position)] = (
            "train"
            if rank < train_end
            else "validation" if rank < validation_end else "test"
        )
    return SplitManifest(
        seed=seed,
        source_sha256=source_sha256,
        plan_sha256=plan_sha256,
        block_degrees=settings.block_degrees,
        assignments=[
            StationSplit(
                station_id=item_id,
                block_id=spatial_block(
                    coordinates[position].latitude,
                    coordinates[position].longitude,
                    settings.block_degrees,
                ),
                split=names[position],
            )
            for position, item_id in enumerate(item_ids)
        ],
    )


def split_indices(manifest: SplitManifest, split: SplitName) -> IndexVector:
    return np.asarray(
        [
            index
            for index, item in enumerate(manifest.assignments)
            if item.split == split
        ],
        dtype=np.int64,
    )
