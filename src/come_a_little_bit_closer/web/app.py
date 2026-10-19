from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from come_a_little_bit_closer.models.schemas import ModelStatus
from come_a_little_bit_closer.storage.catalog import initialize_catalog
from come_a_little_bit_closer.storage.schemas import CatalogStats
from come_a_little_bit_closer.web.retrieval import Retriever
from come_a_little_bit_closer.web.schemas import (
    GlobeStation,
    SearchParameters,
    SearchResult,
    WebSettings,
)


def create_app(settings: WebSettings | None = None) -> FastAPI:
    configuration = settings or WebSettings()
    initialize_catalog(configuration.database)
    retriever = Retriever(configuration)
    globe_stations = [
        GlobeStation(
            station_id=station.station_id,
            name=station.name,
            country_code=station.country_code,
            latitude=station.coordinates.latitude,
            longitude=station.coordinates.longitude,
            stream_url=station.stream_url,
        )
        for station in retriever.catalog.stations()
    ]
    web_directory = Path(__file__).parent
    templates = Jinja2Templates(directory=web_directory / "templates")
    app = FastAPI(title="Come A Little Bit Closer", version="0.1.0")
    app.mount("/static", StaticFiles(directory=web_directory / "static"), name="static")

    def retrieve(parameters: SearchParameters) -> SearchResult:
        try:
            return retriever.search(parameters)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "models": [model.model_dump() for model in retriever.models()],
                "initial_search": None,
            },
        )

    @app.get("/explore", response_class=HTMLResponse)
    def explore(
        request: Request, parameters: Annotated[SearchParameters, Query()]
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={"result": retrieve(parameters).model_dump()},
        )

    @app.get("/api/search", response_model=SearchResult)
    def api_search(parameters: Annotated[SearchParameters, Query()]) -> SearchResult:
        return retrieve(parameters)

    @app.get("/api/stats", response_model=CatalogStats)
    def api_stats() -> CatalogStats:
        return retriever.stats

    @app.get("/api/models", response_model=list[ModelStatus])
    def api_models() -> list[ModelStatus]:
        return retriever.models()

    @app.get("/api/stations", response_model=list[GlobeStation])
    def api_stations() -> list[GlobeStation]:
        return globe_stations

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
