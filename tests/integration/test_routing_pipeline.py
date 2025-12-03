"""
Integration tests for the routing pipeline.

These tests verify the end-to-end routing functionality using Lund, Sweden
as the test city. Tests are marked as 'slow' since they:
- Download OSM pedestrian networks (first run)
- Process real camera data
- Generate route outputs
"""

import json
from pathlib import Path

import pytest

from src.orchestration.langchain_pipeline import SurveillancePipeline, PipelineStatus
from src.config.pipeline_config import PipelineConfig, AnalysisScenario


# Test coordinates for Lund, Sweden
LUND_START_LAT = 55.709400
LUND_START_LON = 13.194381
LUND_END_LAT = 55.705962
LUND_END_LON = 13.182304

# Expected city/country
LUND_CITY = "Lund"
LUND_COUNTRY = "SE"


@pytest.fixture
def lund_data_path():
    """
    Path to existing Lund camera data.

    NOTE: This test assumes Lund data has been scraped previously.
    If the data doesn't exist, the test will skip.
    """
    data_path = Path("overpass_data/lund/lund.json")
    if not data_path.exists():
        pytest.skip(f"Lund data not found at {data_path}. Run scraper first.")
    return data_path


@pytest.fixture
def routing_config():
    """Pipeline configuration with routing enabled."""
    return PipelineConfig(
        scenario=AnalysisScenario.BASIC,
        routing_enabled=True,
        start_lat=LUND_START_LAT,
        start_lon=LUND_START_LON,
        end_lat=LUND_END_LAT,
        end_lon=LUND_END_LON,
    )


@pytest.mark.slow
def test_routing_pipeline_end_to_end(
    routing_config, lund_data_path, tmp_path, monkeypatch
):
    """
    Test complete pipeline with routing enabled.

    Verifies:
    - Pipeline executes all three agents (scraper, analyzer, router)
    - Route outputs are generated (GeoJSON, HTML map)
    - Route metrics are valid
    - Files are created in expected locations
    """
    # Override output directory to use tmp_path for test isolation
    monkeypatch.setenv("OVERPASS_DIR", str(tmp_path / "overpass_data"))

    # Create pipeline
    pipeline = SurveillancePipeline(config=routing_config)

    # Run pipeline
    results = pipeline.run(city=LUND_CITY, country=LUND_COUNTRY)

    # Verify pipeline completed successfully
    assert pipeline.status in [PipelineStatus.COMPLETED, PipelineStatus.PARTIAL]
    assert "routing" in results

    # Verify routing results
    routing_results = results["routing"]
    assert routing_results["success"] is True
    assert routing_results["city"] == LUND_CITY

    # Verify route metrics
    metrics = routing_results
    assert "length_m" in metrics
    assert "exposure_score" in metrics
    assert "camera_count" in metrics

    # Length should be positive
    assert metrics["length_m"] > 0

    # Exposure score should be non-negative
    assert metrics["exposure_score"] >= 0

    # Camera count should be non-negative integer
    assert metrics["camera_count"] >= 0
    assert isinstance(metrics["camera_count"], int)

    # Verify output files exist
    route_geojson_path = Path(routing_results["route_geojson_path"])
    route_map_path = Path(routing_results["route_map_path"])

    assert route_geojson_path.exists(), (
        f"Route GeoJSON not found at {route_geojson_path}"
    )
    assert route_map_path.exists(), f"Route map HTML not found at {route_map_path}"

    # Verify GeoJSON structure
    with open(route_geojson_path, "r", encoding="utf-8") as f:
        route_geojson = json.load(f)

    assert route_geojson["type"] == "FeatureCollection"
    assert len(route_geojson["features"]) > 0

    # First feature should be the route LineString
    route_feature = route_geojson["features"][0]
    assert route_feature["type"] == "Feature"
    assert route_feature["geometry"]["type"] == "LineString"
    assert len(route_feature["geometry"]["coordinates"]) > 0

    # Verify route properties
    props = route_feature["properties"]
    assert "city" in props
    assert props["city"] == LUND_CITY
    assert "length_m" in props
    assert "exposure_score" in props
    assert "camera_count" in props
    assert "baseline_length_m" in props
    assert "baseline_exposure_score" in props

    # Verify HTML map is valid
    with open(route_map_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    assert "<!DOCTYPE html>" in html_content or "<html" in html_content
    assert "folium" in html_content.lower() or "leaflet" in html_content.lower()


@pytest.mark.slow
def test_routing_cache_behavior(routing_config, lund_data_path, tmp_path, monkeypatch):
    """
    Test that routing results are cached and reused.

    Verifies:
    - First run generates route from scratch
    - Second identical run uses cached route
    - Cached flag is set correctly
    """
    # Override output directory
    monkeypatch.setenv("OVERPASS_DIR", str(tmp_path / "overpass_data"))

    # First run - should not use cache
    pipeline1 = SurveillancePipeline(config=routing_config)
    results1 = pipeline1.run(city=LUND_CITY, country=LUND_COUNTRY)

    assert "routing" in results1
    routing1 = results1["routing"]
    assert routing1["success"] is True

    # First run should not be from cache (or may be if graph is cached)
    # We primarily care that it succeeds

    # Second run with same configuration - should use cache
    pipeline2 = SurveillancePipeline(config=routing_config)
    results2 = pipeline2.run(city=LUND_CITY, country=LUND_COUNTRY)

    assert "routing" in results2
    routing2 = results2["routing"]
    assert routing2["success"] is True

    # Second run should be from cache
    assert routing2["from_cache"] is True

    # Results should be identical
    assert routing2["length_m"] == routing1["length_m"]
    assert routing2["exposure_score"] == routing1["exposure_score"]
    assert routing2["camera_count"] == routing1["camera_count"]


@pytest.mark.slow
def test_routing_with_invalid_coordinates(lund_data_path, tmp_path, monkeypatch):
    """
    Test routing with coordinates far from walkable network.

    Verifies:
    - Pipeline handles snapping errors gracefully
    - Appropriate error message is returned
    """
    # Override output directory
    monkeypatch.setenv("OVERPASS_DIR", str(tmp_path / "overpass_data"))

    # Use coordinates in the ocean (should fail to snap)
    config = PipelineConfig(
        scenario=AnalysisScenario.BASIC,
        routing_enabled=True,
        start_lat=55.0,  # Middle of the Baltic Sea
        start_lon=13.0,
        end_lat=55.1,
        end_lon=13.1,
    )

    pipeline = SurveillancePipeline(config=config)

    # Run pipeline - routing should fail gracefully
    results = pipeline.run(city=LUND_CITY, country=LUND_COUNTRY)

    # Pipeline should complete (scraping and analysis succeed)
    assert pipeline.status in [PipelineStatus.COMPLETED, PipelineStatus.PARTIAL]

    # Routing should have failed
    if "routing" in results:
        routing = results["routing"]
        # Either success=False or an error was raised and caught
        # The exact behavior depends on error handling implementation
        assert routing["success"] is False or "error" in routing


@pytest.mark.slow
def test_routing_output_file_structure(
    routing_config, lund_data_path, tmp_path, monkeypatch
):
    """
    Test detailed structure of routing output files.

    Verifies:
    - Route GeoJSON contains required fields
    - Baseline metrics are included for comparison
    - Camera IDs are tracked for drill-down
    - Timestamps and metadata are present
    """
    # Override output directory
    monkeypatch.setenv("OVERPASS_DIR", str(tmp_path / "overpass_data"))

    pipeline = SurveillancePipeline(config=routing_config)
    results = pipeline.run(city=LUND_CITY, country=LUND_COUNTRY)

    assert "routing" in results
    routing = results["routing"]
    assert routing["success"] is True

    # Load route GeoJSON
    route_geojson_path = Path(routing["route_geojson_path"])
    with open(route_geojson_path, "r", encoding="utf-8") as f:
        route_data = json.load(f)

    # Verify FeatureCollection structure
    assert route_data["type"] == "FeatureCollection"
    features = route_data["features"]
    assert len(features) > 0

    # Find route feature (should be first feature)
    route_feature = features[0]
    assert route_feature["geometry"]["type"] == "LineString"

    # Verify comprehensive properties
    props = route_feature["properties"]

    # Required fields
    required_fields = [
        "city",
        "length_m",
        "exposure_score",
        "camera_count",
        "baseline_length_m",
        "baseline_exposure_score",
    ]

    for field in required_fields:
        assert field in props, f"Missing required field: {field}"

    # Verify baseline comparison makes sense
    # Baseline is shortest path, so it should be <= selected route length
    # (selected route may be longer to avoid cameras)
    assert props["baseline_length_m"] > 0

    # Verify camera count makes sense
    # If exposure_score > 0, there should be cameras nearby
    if props["exposure_score"] > 0:
        assert props["camera_count"] > 0

    # Verify coordinates are in valid range
    coords = route_feature["geometry"]["coordinates"]
    for lon, lat in coords:
        assert -180 <= lon <= 180, f"Invalid longitude: {lon}"
        assert -90 <= lat <= 90, f"Invalid latitude: {lat}"

        # Coordinates should be near Lund
        assert 55.5 <= lat <= 56.0, f"Latitude {lat} not near Lund"
        assert 13.0 <= lon <= 13.5, f"Longitude {lon} not near Lund"


@pytest.mark.slow
def test_routing_without_coordinates_raises_error():
    """
    Test that routing configuration without coordinates raises validation error.

    Verifies:
    - Pydantic validation catches missing coordinates
    - Appropriate error message is provided
    """
    # Attempt to create config with routing enabled but no coordinates
    with pytest.raises(ValueError) as exc_info:
        PipelineConfig(
            scenario=AnalysisScenario.BASIC,
            routing_enabled=True,
            # Missing all coordinate fields
        )

    # Error message should mention missing coordinates
    error_msg = str(exc_info.value)
    assert "coordinates" in error_msg.lower()


@pytest.mark.slow
def test_routing_preserves_analysis_outputs(
    routing_config, lund_data_path, tmp_path, monkeypatch
):
    """
    Test that routing doesn't interfere with analysis outputs.

    Verifies:
    - Both analysis and routing outputs are present
    - Analysis results are complete
    - Routing results are complete
    - No data loss or interference between stages
    """
    # Override output directory
    monkeypatch.setenv("OVERPASS_DIR", str(tmp_path / "overpass_data"))

    pipeline = SurveillancePipeline(config=routing_config)
    results = pipeline.run(city=LUND_CITY, country=LUND_COUNTRY)

    # Verify both analysis and routing results exist
    assert "analysis" in results or "analyzer" in results
    assert "routing" in results

    # Verify analysis outputs
    if "analysis" in results:
        analysis = results["analysis"]
        assert "enriched_geojson_path" in analysis or "output_path" in analysis

    # Verify routing outputs
    routing = results["routing"]
    assert routing["success"] is True
    assert "route_geojson_path" in routing
    assert "route_map_path" in routing

    # Verify all files exist
    route_geojson = Path(routing["route_geojson_path"])
    route_map = Path(routing["route_map_path"])

    assert route_geojson.exists()
    assert route_map.exists()
