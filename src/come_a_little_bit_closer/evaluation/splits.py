import math

import numpy as np

from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.evaluation.location_schemas import (
    IndexVector,
    LocationPlan,
    SplitManifest,
    SplitName,
    StationSplit,
)


def spatial_block(latitude: float, longitude: float, degrees: float) -> str:
    latitude_index = min(
        math.ceil(180 / degrees) - 1, math.floor((latitude + 90) / degrees)
    )
    longitude_index = math.floor(((longitude + 180) % 360) / degrees)
    return f"{latitude_index}:{longitude_index}"


def spatial_split(
    stations: list[Station],
    plan: LocationPlan,
    seed: int,
    source_sha256: str,
    plan_sha256: str,
) -> SplitManifest:
    if plan.train_fraction + plan.validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition.")
    station_blocks = [
        spatial_block(
            station.coordinates.latitude,
            station.coordinates.longitude,
            plan.block_degrees,
        )
        for station in stations
    ]
    blocks = sorted(set(station_blocks))
    if len(blocks) < 3:
        raise ValueError("Spatial splitting requires at least three occupied blocks.")
    shuffled = np.random.default_rng(seed).permutation(len(blocks))
    train_end = max(1, min(len(blocks) - 2, int(len(blocks) * plan.train_fraction)))
    validation_end = max(
        train_end + 1,
        min(
            len(blocks) - 1,
            int(len(blocks) * (plan.train_fraction + plan.validation_fraction)),
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
        block_degrees=plan.block_degrees,
        assignments=[
            StationSplit(
                station_id=station.station_id,
                block_id=station_blocks[position],
                split=partitions[station_blocks[position]],
            )
            for position, station in enumerate(stations)
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
