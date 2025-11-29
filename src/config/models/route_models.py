from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """Input parameters for computing a low-surveillance route.

    The request captures the city context and the start / end locations
    expressed as latitude and longitude.

    :param city: Name of the city for which the route is computed.
    :param country: Optional ISO country code used for disambiguation.
    :param start_lat: Latitude of the starting point.
    :param start_lon: Longitude of the starting point.
    :param end_lat: Latitude of the ending point.
    :param end_lon: Longitude of the ending point.
    :param data_path: Optional override to the input data file if it differs
                      from the standard pipeline outputs.
    :param mode: Optional logical travel mode (for now primarily "walk").
    """

    city: str = Field(..., description="City name used for routing context.")
    country: Optional[str] = Field(
        default=None,
        description="Optional ISO country code for city disambiguation.",
    )
    start_lat: float = Field(..., description="Latitude of the starting point.")
    start_lon: float = Field(..., description="Longitude of the starting point.")
    end_lat: float = Field(..., description="Latitude of the ending point.")
    end_lon: float = Field(..., description="Longitude of the ending point.")
    data_path: Optional[Path] = Field(
        default=None,
        description=(
            "Optional path to an existing data file when not using the "
            "default pipeline location."
        ),
    )
    mode: Optional[str] = Field(
        default="walk", description="Logical travel mode, e.g. 'walk' or 'bike'."
    )


class RouteMetrics(BaseModel):
    """Summary metrics describing a computed route.

    These metrics are used both for scoring candidate routes and for reporting
    the characteristics of the final chosen route.

    :param length_m: Total length of the route in metres.
    :param exposure_score: Aggregate exposure score based on cameras near
                           the path.
    :param camera_count_near_route: Total number of cameras within the
                                    configured buffer radius of the route.
    :param baseline_length_m: Length in metres of the baseline shortest route
                              used as a reference.
    :param baseline_exposure_score: Exposure score of the baseline route.
    """

    length_m: float = Field(..., description="Total route length in metres.")
    exposure_score: float = Field(
        ..., description="Aggregate exposure score along the route."
    )
    camera_count_near_route: int = Field(
        ..., description="Number of cameras falling within the route buffer."
    )
    baseline_length_m: Optional[float] = Field(
        default=None,
        description=(
            "Length of the baseline shortest path used for comparison, "
            "expressed in metres."
        ),
    )
    baseline_exposure_score: Optional[float] = Field(
        default=None,
        description="Exposure score of the baseline shortest path, if computed.",
    )


class RouteResult(BaseModel):
    """Output artefacts and metrics for a low-surveillance route.

    :param city: City name for which the route was computed.
    :param route_geojson_path: Filesystem path to the GeoJSON representation
                               of the route.
    :param route_map_path: Filesystem path to the HTML map visualising the
                           route and nearby cameras.
    :param metrics: Computed :class:`RouteMetrics` instance.
    :param from_cache: Flag indicating whether the result was served entirely
                       from cache.
    """

    city: str = Field(..., description="City name used for routing context.")
    route_geojson_path: Path = Field(
        ..., description="Path to the GeoJSON file describing the route."
    )
    route_map_path: Path = Field(
        ..., description="Path to the HTML map with the rendered route."
    )
    metrics: RouteMetrics = Field(..., description="Summary statistics.")
    from_cache: bool = Field(
        default=False,
        description="Whether the route was served from a cached computation.",
    )
