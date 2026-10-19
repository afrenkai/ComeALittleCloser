from typing import Annotated, Literal

from pydantic import BaseModel, Field, FiniteFloat, JsonValue

type GeoPosition = tuple[FiniteFloat, FiniteFloat]


class PolygonGeometry(BaseModel):
    type: Literal["Polygon"]
    coordinates: list[list[GeoPosition]]


class MultiPolygonGeometry(BaseModel):
    type: Literal["MultiPolygon"]
    coordinates: list[list[list[GeoPosition]]]


class GeoFeature(BaseModel):
    type: Literal["Feature"]
    properties: dict[str, JsonValue]
    geometry: Annotated[
        PolygonGeometry | MultiPolygonGeometry, Field(discriminator="type")
    ]


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"]
    features: list[GeoFeature]
