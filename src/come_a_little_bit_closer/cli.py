import argparse
from pathlib import Path

from come_a_little_bit_closer.data.stations import load_stations
from come_a_little_bit_closer.evaluation.baseline import compare_geographic_rankings
from come_a_little_bit_closer.evaluation.schemas import BaselineRequest
from come_a_little_bit_closer.geometry.schemas import Coordinates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Haversine and spherical-coordinate cosine neighborhoods."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw/radio-browser-geolocated.json")
    )
    parser.add_argument("--latitude", type=float, default=40.73061)
    parser.add_argument("--longitude", type=float, default=-73.935242)
    parser.add_argument("--genre", default="rock")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.json"))
    arguments = parser.parse_args()
    request = BaselineRequest(
        source_path=arguments.data,
        output_path=arguments.output,
        origin=Coordinates(latitude=arguments.latitude, longitude=arguments.longitude),
        genre=arguments.genre,
        limit=arguments.limit,
    )
    dataset = load_stations(request.source_path)
    report = compare_geographic_rankings(dataset, request)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    request.output_path.write_text(report.model_dump_json(indent=2) + "\n")
    print(report.model_dump_json(indent=2))
