import csv
import io
import zipfile
from collections import Counter
from pathlib import Path

from come_a_little_bit_closer.data.schemas import Place, PlaceCondition, TownSource
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
        text = archive.read(f"{archive_path.stem}.txt").decode("utf-8")
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


def load_fresh_towns(source: TownSource) -> list[Place]:
    """Towns new to the reference extract, with a country name and a unique text.

    Towns whose name-country text equals a reference place's, or another town's,
    are dropped: identical texts get identical vectors.
    """
    reference = load_places(source.reference_archive, source.country_info)
    reference_ids = {place.place_id for place in reference}
    reference_texts = {place_text(place, "name-country") for place in reference}
    candidates = [
        place
        for place in load_places(source.archive, source.country_info)
        if place.place_id not in reference_ids
        and place.country_name
        and source.minimum_population <= place.population < source.maximum_population
        and place_text(place, "name-country") not in reference_texts
    ]
    counts = Counter(place_text(place, "name-country") for place in candidates)
    return [
        place for place in candidates if counts[place_text(place, "name-country")] == 1
    ]
