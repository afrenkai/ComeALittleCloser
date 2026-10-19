import math
from pathlib import Path

from pydantic import TypeAdapter

from come_a_little_bit_closer.data.schemas import (
    DataSummary,
    RadioBrowserRecord,
    Station,
    StationDataset,
)
from come_a_little_bit_closer.geometry.schemas import Coordinates


def load_stations(source_path: Path) -> StationDataset:
    records = TypeAdapter(list[RadioBrowserRecord]).validate_json(
        source_path.read_bytes()
    )
    stations: list[Station] = []
    seen_streams: set[str] = set()
    valid_coordinate_records = 0
    duplicate_stream_records = 0
    empty_stream_records = 0
    for record in records:
        latitude = record.geo_lat
        longitude = record.geo_long
        if latitude is None or longitude is None:
            continue
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            continue
        if abs(latitude) > 90 or abs(longitude) > 180:
            continue
        valid_coordinate_records += 1
        stream_url = record.url_resolved.strip()
        if not stream_url:
            empty_stream_records += 1
            continue
        if stream_url in seen_streams:
            duplicate_stream_records += 1
            continue
        seen_streams.add(stream_url)
        stations.append(
            Station(
                station_id=record.stationuuid,
                name=record.name,
                stream_url=stream_url,
                country_code=record.countrycode.strip().upper(),
                genre_tags=sorted(
                    {
                        tag.strip().casefold()
                        for tag in record.tags.split(",")
                        if tag.strip()
                    }
                ),
                coordinates=Coordinates(latitude=latitude, longitude=longitude),
            )
        )
    summary = DataSummary(
        source_records=len(records),
        valid_coordinate_records=valid_coordinate_records,
        duplicate_stream_records=duplicate_stream_records,
        empty_stream_records=empty_stream_records,
        retained_stations=len(stations),
        countries=len(
            {station.country_code for station in stations if station.country_code}
        ),
        rock_stations=sum("rock" in station.genre_tags for station in stations),
    )
    return StationDataset(stations=stations, summary=summary)
