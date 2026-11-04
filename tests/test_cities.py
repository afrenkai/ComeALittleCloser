import zipfile
from pathlib import Path

from come_a_little_bit_closer.data.cities import load_places, place_text
from come_a_little_bit_closer.data.schemas import Place
from come_a_little_bit_closer.geometry.schemas import Coordinates


def geonames_row(
    place_id: str, name: str, latitude: str, longitude: str, code: str, population: str
) -> str:
    columns = [""] * 19
    columns[0] = place_id
    columns[1] = name
    columns[4] = latitude
    columns[5] = longitude
    columns[8] = code
    columns[14] = population
    return "\t".join(columns)


def test_load_places_reads_geonames_columns(tmp_path: Path) -> None:
    archive_path = tmp_path / "cities15000.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "cities15000.txt",
            "\n".join(
                [
                    geonames_row(
                        "2996944", "Lyon", "45.74846", "4.84671", "FR", "472317"
                    ),
                    geonames_row("100", "Nowhere", "1.5", "-2.5", "ZZ", "15000"),
                ]
            )
            + "\n",
        )
    country_info_path = tmp_path / "countryInfo.txt"
    country_info_path.write_text(
        "# comment\nFR\tFRA\t250\tFR\tFrance\n" "US\tUSA\t840\tUS\tUnited States\n"
    )
    places = load_places(archive_path, country_info_path)
    assert [place.place_id for place in places] == ["100", "2996944"]
    assert places[1].name == "Lyon"
    assert places[1].country_name == "France"
    assert places[1].population == 472317
    assert places[1].coordinates == Coordinates(latitude=45.74846, longitude=4.84671)
    assert places[0].country_name == ""


def test_place_text_conditions() -> None:
    place = Place(
        place_id="1",
        name="Lyon",
        country_code="FR",
        country_name="France",
        population=1,
        coordinates=Coordinates(latitude=0, longitude=0),
    )
    assert place_text(place, "name") == "Lyon"
    assert place_text(place, "name-country") == "Lyon, France"
    unnamed = place.model_copy(update={"country_name": ""})
    assert place_text(unnamed, "name-country") == "Lyon"
