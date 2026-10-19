from threading import Lock

import numpy as np

from come_a_little_bit_closer.data.genre_matching import GenreMatcher
from come_a_little_bit_closer.data.schemas import Station
from come_a_little_bit_closer.geometry.distances import (
    cosine_similarities,
    haversine_distances,
    spherical_vectors,
)
from come_a_little_bit_closer.geometry.schemas import (
    Coordinates,
    EmbeddingMatrix,
    EmbeddingVector,
    ScoreVector,
)
from come_a_little_bit_closer.models.artifacts import EmbeddingArtifact
from come_a_little_bit_closer.models.catalog import find_model, model_specs
from come_a_little_bit_closer.models.encoder import inference_installed
from come_a_little_bit_closer.models.factory import create_encoder
from come_a_little_bit_closer.models.schemas import ModelEncoder, ModelStatus
from come_a_little_bit_closer.projection.clustering import (
    cluster_embeddings,
    describe_clusters,
)
from come_a_little_bit_closer.projection.pca import ProjectionFrame
from come_a_little_bit_closer.projection.schemas import (
    EmbeddingProjection,
    ProjectionCluster,
    ProjectionPoint,
    ProjectionQuery,
)
from come_a_little_bit_closer.storage.catalog import StationCatalog
from come_a_little_bit_closer.web.metrics import comparison_metrics
from come_a_little_bit_closer.web.schemas import (
    Neighbor,
    SearchParameters,
    SearchResult,
    WebSettings,
)


class Retriever:
    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings
        self.catalog = StationCatalog(settings.database.database_path)
        self.stats = self.catalog.stats()
        self.genre_matcher = GenreMatcher(
            settings.genre_vocabulary_path, self.catalog.stations()
        )
        self.artifacts: dict[str, EmbeddingArtifact] = {}
        self.encoders: dict[str, ModelEncoder] = {}
        self.projections: dict[str, ProjectionFrame] = {}
        self.cluster_labels: dict[str, dict[str, int]] = {}
        self.cluster_descriptions: dict[str, list[ProjectionCluster]] = {}
        self.model_lock = Lock()

    def models(self) -> list[ModelStatus]:
        statuses = [
            ModelStatus(
                model_id="spherical",
                label="Spherical coordinates · control",
                available=True,
                description=(
                    "Unit-vector coordinate cosine. Its ranking agrees with Haversine "
                    "by construction; no learned encoder is used."
                ),
            )
        ]
        for spec in model_specs():
            artifact_path = self.settings.embedding_directory / f"{spec.model_id}.npz"
            available = (
                spec.text_supported
                and artifact_path.is_file()
                and inference_installed()
            )
            statuses.append(
                ModelStatus(
                    model_id=spec.model_id,
                    label=spec.label,
                    available=available,
                    description=spec.description,
                )
            )
        return statuses

    def model_vectors(
        self, parameters: SearchParameters, stations: list[Station]
    ) -> tuple[EmbeddingVector, EmbeddingMatrix]:
        spec = find_model(parameters.model)
        with self.model_lock:
            if spec.model_id not in self.artifacts:
                artifact = EmbeddingArtifact(
                    self.settings.embedding_directory / f"{spec.model_id}.npz"
                )
                if artifact.manifest.source_sha256 != self.stats.source_sha256:
                    raise ValueError(
                        "Embedding cache uses a different source snapshot."
                    )
                self.artifacts[spec.model_id] = artifact
            artifact = self.artifacts[spec.model_id]
            if spec.model_id not in self.encoders:
                self.encoders[spec.model_id] = create_encoder(
                    artifact.manifest.encoder_settings or spec,
                    artifact.manifest.revision,
                    self.settings.query_device,
                    artifact.manifest.max_sequence_length,
                )
            encoder = self.encoders[spec.model_id]
        query_text = (
            f"{parameters.query}\nOrigin latitude: {parameters.latitude:.5f}\n"
            f"Origin longitude: {parameters.longitude:.5f}"
        )
        return (
            encoder.query(query_text),
            artifact.select([station.station_id for station in stations]),
        )

    def projection_frame(self, model_id: str) -> ProjectionFrame:
        with self.model_lock:
            if model_id not in self.projections:
                corpus_stations = self.catalog.stations()
                if model_id == "spherical":
                    vectors = spherical_vectors(
                        np.asarray(
                            [
                                [
                                    station.coordinates.latitude,
                                    station.coordinates.longitude,
                                ]
                                for station in corpus_stations
                            ],
                            dtype=np.float64,
                        )
                    )
                else:
                    vectors = self.artifacts[model_id].select(
                        [station.station_id for station in corpus_stations]
                    )
                self.projections[model_id] = ProjectionFrame(vectors)
                labels = cluster_embeddings(vectors, self.settings.projection)
                self.cluster_labels[model_id] = {
                    station.station_id: int(labels[position])
                    for position, station in enumerate(corpus_stations)
                }
                self.cluster_descriptions[model_id] = describe_clusters(
                    corpus_stations, labels
                )
            return self.projections[model_id]

    def search(self, parameters: SearchParameters) -> SearchResult:
        matching_models = [
            model for model in self.models() if model.model_id == parameters.model
        ]
        if not matching_models:
            raise ValueError(f"Unknown embedding model: {parameters.model}")
        model = matching_models[0]
        selection = self.genre_matcher.resolve(parameters.query, parameters.genre)
        stations = self.catalog.stations()
        candidate_positions = [
            position
            for position, station in enumerate(stations)
            if selection.source == "none"
            or any(tag in selection.station_tags for tag in station.genre_tags)
        ]
        result = SearchResult(
            origin=Coordinates(
                latitude=parameters.latitude, longitude=parameters.longitude
            ),
            query=parameters.query,
            genre=selection.genre,
            genre_selection=selection,
            model_id=model.model_id,
            model_label=model.label,
            model_description=model.description,
            is_learned=model.model_id != "spherical",
            eligible_stations=len(candidate_positions),
            overlap_at_k=0,
            geographic_neighbors=[],
            embedding_neighbors=[],
            model_available=model.available,
        )
        if not stations:
            result.message = "The station catalog is empty."
            return result
        if not candidate_positions:
            result.message = (
                "No stations match the requested genre tag in this snapshot."
            )
        locations = np.asarray(
            [
                [station.coordinates.latitude, station.coordinates.longitude]
                for station in stations
            ],
            dtype=np.float64,
        )
        origin = np.asarray(
            [parameters.latitude, parameters.longitude], dtype=np.float64
        )
        distances = haversine_distances(origin, locations, self.settings.geometry)
        geographic_order = sorted(
            candidate_positions,
            key=lambda position: (distances[position], stations[position].station_id),
        )[: parameters.limit]
        similarities: ScoreVector | None = None
        if model.available:
            if parameters.model == "spherical":
                query_vector = spherical_vectors(origin[np.newaxis, :])[0]
                candidate_vectors = spherical_vectors(locations)
            else:
                query_vector, candidate_vectors = self.model_vectors(
                    parameters, stations
                )
            similarities = cosine_similarities(query_vector, candidate_vectors)
        result.geographic_neighbors = [
            make_neighbor(stations[position], position, distances, similarities)
            for position in geographic_order
        ]
        result.metrics = comparison_metrics(
            result.geographic_neighbors, [], [], selection
        )
        if similarities is None:
            result.message = "This model has no usable embedding cache yet. " + (
                model.description
            )
            return result
        embedding_order = sorted(
            candidate_positions,
            key=lambda position: (
                -similarities[position],
                stations[position].station_id,
            ),
        )[: parameters.limit]
        result.embedding_neighbors = [
            make_neighbor(stations[position], position, distances, similarities)
            for position in embedding_order
        ]
        unfiltered_order = sorted(
            range(len(stations)),
            key=lambda position: (
                -similarities[position],
                stations[position].station_id,
            ),
        )[: parameters.limit]
        result.unfiltered_embedding_neighbors = [
            make_neighbor(stations[position], position, distances, similarities)
            for position in unfiltered_order
        ]
        result.metrics = comparison_metrics(
            result.geographic_neighbors,
            result.embedding_neighbors,
            result.unfiltered_embedding_neighbors,
            selection,
        )
        result.overlap_at_k = (
            len(set(geographic_order) & set(embedding_order)) / len(geographic_order)
            if geographic_order
            else 0
        )
        frame = self.projection_frame(parameters.model)
        projected_candidates = frame.transform(candidate_vectors)
        projected_query = frame.transform(query_vector[np.newaxis, :])[0]
        geographic_ranks = {
            position: rank + 1 for rank, position in enumerate(geographic_order)
        }
        embedding_ranks = {
            position: rank + 1 for rank, position in enumerate(embedding_order)
        }
        candidate_indices = set(candidate_positions)
        result.projection = EmbeddingProjection(
            description=(
                f"PCA of {model.label}. Basis fitted to the full station snapshot; "
                "three-dimensional projection does not determine retrieval rank. "
                "Cluster colors come from native-space spherical k-means; "
                "tag labels describe clusters and are not ground truth."
            ),
            explained_variance_ratio=frame.variance_ratio,
            query=ProjectionQuery(
                x=projected_query[0], y=projected_query[1], z=projected_query[2]
            ),
            clusters=self.cluster_descriptions[parameters.model],
            frame_center=ProjectionQuery(
                x=frame.center[0], y=frame.center[1], z=frame.center[2]
            ),
            frame_radius=frame.radius,
            points=[
                ProjectionPoint(
                    station_id=station.station_id,
                    name=station.name,
                    country_code=station.country_code,
                    stream_url=station.stream_url,
                    latitude=station.coordinates.latitude,
                    longitude=station.coordinates.longitude,
                    x=projected_candidates[position, 0],
                    y=projected_candidates[position, 1],
                    z=projected_candidates[position, 2],
                    cluster_id=self.cluster_labels[parameters.model][
                        station.station_id
                    ],
                    distance_km=distances[position],
                    similarity=similarities[position],
                    geographic_rank=geographic_ranks.get(position),
                    embedding_rank=embedding_ranks.get(position),
                )
                for position, station in enumerate(stations)
                if position in candidate_indices
            ],
        )
        return result


def make_neighbor(
    station: Station,
    position: int,
    distances: ScoreVector,
    similarities: ScoreVector | None,
) -> Neighbor:
    return Neighbor(
        station_id=station.station_id,
        name=station.name,
        country_code=station.country_code,
        latitude=station.coordinates.latitude,
        longitude=station.coordinates.longitude,
        distance_km=float(distances[position]),
        similarity=float(similarities[position]) if similarities is not None else None,
        genre_tags=station.genre_tags,
        stream_url=station.stream_url,
    )
