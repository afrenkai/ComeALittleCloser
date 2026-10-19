import json
from pathlib import Path

from come_a_little_bit_closer.data.stations import load_stations


def test_invalid_coordinates_and_duplicate_streams(tmp_path: Path) -> None:
    record = {
        "stationuuid": "first",
        "name": "Rock station",
        "url_resolved": "https://example.org/audio",
        "countrycode": "US",
        "tags": "Rock, rock, jazz",
        "geo_lat": 40.0,
        "geo_long": -73.0,
    }
    source_path = tmp_path / "stations.json"
    source_path.write_text(
        json.dumps(
            [record, record | {"stationuuid": "duplicate"}, record | {"geo_lat": 91}]
        )
    )
    dataset = load_stations(source_path)
    assert dataset.summary.source_records == 3
    assert dataset.summary.valid_coordinate_records == 2
    assert dataset.summary.duplicate_stream_records == 1
    assert dataset.summary.retained_stations == 1
    assert dataset.stations[0].genre_tags == ["jazz", "rock"]
