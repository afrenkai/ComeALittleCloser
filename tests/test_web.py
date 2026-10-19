import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from come_a_little_bit_closer.geometry.schemas import EmbeddingMatrix, EmbeddingVector
from come_a_little_bit_closer.models.artifacts import save_embeddings
from come_a_little_bit_closer.models.schemas import (
    EmbeddingManifest,
    ModelEncoder,
    ModelSpec,
)
from come_a_little_bit_closer.storage.catalog import initialize_catalog, snapshot_sha256
from come_a_little_bit_closer.storage.schemas import DatabaseSettings
from come_a_little_bit_closer.web.app import create_app
from come_a_little_bit_closer.web.schemas import WebSettings


@pytest.fixture
def web_settings(tmp_path: Path) -> WebSettings:
    source_path = tmp_path / "stations.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "stationuuid": station_id,
                    "name": "Rock <script>alert(1)</script>",
                    "url_resolved": f"https://example.org/{station_id}",
                    "countrycode": "US",
                    "tags": genre,
                    "geo_lat": 0.0,
                    "geo_long": longitude,
                }
                for station_id, longitude, genre in [
                    ("across", -179.9, "rock,drum and bass"),
                    ("far", 150.0, "jazz"),
                ]
            ]
        )
    )
    return WebSettings(
        database=DatabaseSettings(
            source_path=source_path, database_path=tmp_path / "stations.duckdb"
        ),
        embedding_directory=tmp_path / "embeddings",
    )


@pytest.fixture
def query_values() -> dict[str, str]:
    return {
        "latitude": "0",
        "longitude": "179.9",
        "model": "spherical",
        "query": "radio",
    }


def test_api_geographic_control_and_input_boundaries(
    web_settings: WebSettings, query_values: dict[str, str]
) -> None:
    with TestClient(create_app(web_settings)) as client:
        response = client.get("/api/search", params=query_values | {"limit": "2"})
        assert response.status_code == 200
        result = response.json()
        assert result["geographic_neighbors"][0]["station_id"] == "across"
        assert result["overlap_at_k"] == 1
        assert result["is_learned"] is False
        assert client.get("/api/search").status_code == 422
        assert (
            client.get(
                "/api/search", params=query_values | {"latitude": "91"}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/api/search", params=query_values | {"longitude": "nan"}
            ).status_code
            == 422
        )
        assert (
            client.get("/api/search", params=query_values | {"limit": "0"}).status_code
            == 422
        )
        assert (
            client.get(
                "/api/search", params=query_values | {"model": "unknown"}
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/api/search", params=query_values | {"genre": "' OR 1=1"}
            ).json()["eligible_stations"]
            == 0
        )
        unavailable = client.get(
            "/api/search", params=query_values | {"model": "clap"}
        ).json()
        assert unavailable["model_available"] is False
        assert unavailable["embedding_neighbors"] == []
        assert client.get("/api/stats").json()["station_count"] == 2
        assert len(client.get("/api/stations").json()) == 2
        projection = result["projection"]
        assert projection["method"] == "PCA"
        assert len(projection["points"]) == 2
        assert len(projection["explained_variance_ratio"]) == 3
        assert "z" in projection["query"]
        assert projection["points"][0]["stream_url"].startswith("https://")


def test_html_fragments_escape_snapshot_data(
    web_settings: WebSettings, query_values: dict[str, str]
) -> None:
    with TestClient(create_app(web_settings)) as client:
        page = client.get("/")
        fragment = client.get(
            "/explore", params=query_values, headers={"HX-Request": "true"}
        )
        assert page.status_code == fragment.status_code == 200
        assert "hx-get" in page.text
        assert "query text" in page.text
        assert "Search stations" not in page.text
        assert "catalog-status" not in page.text
        assert 'id="station-audio"' in page.text
        assert 'value="nearest rock station"' not in page.text
        assert 'value="40.73061"' not in page.text
        assert "Rock &lt;script&gt;" not in page.text
        assert "<script>alert(1)</script>" not in fragment.text
        assert "&lt;script&gt;" in fragment.text
        assert "<!DOCTYPE" not in fragment.text.upper()


def test_catalog_rejects_different_source_snapshot(web_settings: WebSettings) -> None:
    initialize_catalog(web_settings.database)
    web_settings.database.source_path.write_text("[]")
    with pytest.raises(ValueError, match="differs"):
        initialize_catalog(web_settings.database)


class QueryEncoder:
    def documents(self, texts: list[str], batch_size: int) -> EmbeddingMatrix:
        raise RuntimeError("The fixture only supports query inference.")

    def query(self, text: str) -> EmbeddingVector:
        return np.array([1.0, 0.0, 0.0] if "drum" in text else [0.0, 1.0, 0.0])

    def token_lengths(self, texts: list[str], document: bool) -> list[int]:
        raise RuntimeError("The fixture only supports query inference.")


@pytest.fixture
def learned_cache(
    web_settings: WebSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def query_encoder(
        spec: ModelSpec, revision: str, device: str, max_sequence_length: int
    ) -> ModelEncoder:
        assert spec.model_id == "qwen3"
        return QueryEncoder()

    monkeypatch.setattr(
        "come_a_little_bit_closer.web.retrieval.create_encoder", query_encoder
    )
    monkeypatch.setattr(
        "come_a_little_bit_closer.web.retrieval.inference_installed", lambda: True
    )
    save_embeddings(
        web_settings.embedding_directory / "qwen3.npz",
        ["across", "far"],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        EmbeddingManifest(
            model_id="qwen3",
            checkpoint="fixture",
            revision="fixture",
            source_sha256=snapshot_sha256(web_settings.database.source_path),
            station_count=2,
            dimensions=3,
            created_at="fixture",
            max_sequence_length=256,
            sentence_transformers_version="fixture",
        ),
    )


@pytest.mark.usefixtures("learned_cache")
def test_query_changes_repopulate_neighbors_without_sticky_genre(
    web_settings: WebSettings, query_values: dict[str, str]
) -> None:
    with TestClient(create_app(web_settings)) as client:
        drum_result = client.get(
            "/api/search",
            params=query_values | {"model": "qwen3", "query": "drum and bass"},
        ).json()
        jazz_result = client.get(
            "/api/search", params=query_values | {"model": "qwen3", "query": "jazz"}
        ).json()
        assert drum_result["genre"] == "drum and bass"
        assert jazz_result["genre"] == "jazz"
        assert drum_result["embedding_neighbors"][0]["station_id"] == "across"
        assert jazz_result["embedding_neighbors"][0]["station_id"] == "far"
        assert drum_result["projection"]["query"] != jazz_result["projection"]["query"]
        assert (
            drum_result["projection"]["clusters"]
            == jazz_result["projection"]["clusters"]
        )
        assert (
            drum_result["projection"]["frame_radius"]
            == jazz_result["projection"]["frame_radius"]
        )


@pytest.mark.usefixtures("learned_cache")
def test_genre_constraint_excludes_nearer_wrong_genre_and_audits_raw_model(
    web_settings: WebSettings, query_values: dict[str, str]
) -> None:
    with TestClient(create_app(web_settings)) as client:
        result = client.get(
            "/api/search",
            params=query_values
            | {
                "query": "nearest rock station",
                "longitude": "150",
                "model": "qwen3",
                "limit": "1",
            },
        ).json()
        assert result["genre_selection"]["source"] == "query"
        assert result["genre"] == "rock"
        assert result["geographic_neighbors"][0]["station_id"] == "across"
        assert result["embedding_neighbors"][0]["station_id"] == "across"
        assert result["unfiltered_embedding_neighbors"][0]["station_id"] == "far"
        assert result["unfiltered_embedding_neighbors"][0]["distance_km"] == 0
        assert result["metrics"]["unfiltered_genre_precision_at_k"] == 0
        assert result["metrics"]["geographic_mean_distance_km"] > 0
        assert result["metrics"]["excess_distance_km"] == 0
        assert len(result["projection"]["points"]) == 1
        fragment = client.get(
            "/explore",
            params=query_values | {"query": "nearest rock station", "model": "qwen3"},
        ).text
        assert "From query: rock" in fragment
        assert "Unfiltered genre match" in fragment


def test_genre_boundaries_ambiguity_and_missing_candidates(
    web_settings: WebSettings, query_values: dict[str, str]
) -> None:
    with TestClient(create_app(web_settings)) as client:
        hard_rock = client.get(
            "/api/search", params=query_values | {"query": "nearest hard-rock station"}
        ).json()
        assert hard_rock["genre"] == "hard rock"
        assert hard_rock["eligible_stations"] == 0
        assert hard_rock["geographic_neighbors"] == []
        assert hard_rock["metrics"]["geographic_mean_distance_km"] is None
        assert (
            client.get(
                "/api/search",
                params=query_values | {"query": "nearest rock or jazz station"},
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/api/search",
                params=query_values
                | {"query": "nearest rock station", "genre": "jazz"},
            ).status_code
            == 400
        )
        explicit = client.get(
            "/api/search",
            params=query_values | {"query": "rock or jazz", "genre": "jazz"},
        ).json()
        assert explicit["genre_selection"]["source"] == "filter"
        assert explicit["geographic_neighbors"][0]["station_id"] == "far"
        unrelated = client.get(
            "/api/search", params=query_values | {"query": "stations around Houston"}
        ).json()
        assert unrelated["genre_selection"]["source"] == "none"
        assert unrelated["eligible_stations"] == 2
