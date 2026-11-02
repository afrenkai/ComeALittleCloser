import csv
import io
import zipfile
from pathlib import Path

from come_a_little_bit_closer.data.schemas import Place, PlaceCondition
from come_a_little_bit_closer.geometry.schemas import Coordinates


def load_country_names(country_info_path: Path) -> dict[str, str]:
    country_names: dict[str, str] = {}
    for line in country_info_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        country_names[columns[0]] = columns[4]
    return country_names


def load_places(archive_path: Path, country_info_path: Path) -> list[Place]:
    country_names = load_country_names(country_info_path)
    with zipfile.ZipFile(archive_path) as archive:
        text = archive.read("cities15000.txt").decode("utf-8")
    rows = csv.reader(
        io.StringIO(text, newline=""), delimiter="\t", quoting=csv.QUOTE_NONE
    )
    places = [
        Place(
            place_id=row[0],
            name=row[1],
            country_code=row[8],
            country_name=country_names.get(row[8], ""),
            population=int(row[14]),
            coordinates=Coordinates(latitude=float(row[4]), longitude=float(row[5])),
        )
        for row in rows
        if row
    ]
    return sorted(places, key=lambda place: int(place.place_id))


def place_text(place: Place, condition: PlaceCondition) -> str:
    if condition == "name-country" and place.country_name:
        return f"{place.name}, {place.country_name}"
    return place.name
