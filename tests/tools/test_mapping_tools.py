import json

import pytest

from src.config.settings import HeatmapSettings
from src.tools.mapping_tools import to_heatmap, to_hotspots


def test_to_heatmap_creates_html(geojson_file, tmp_path):
    """Test that to_heatmap creates an HTML file with expected content"""
    output_path = tmp_path / "test_heatmap.html"
    result = to_heatmap(geojson_file, output_path)

    assert result == output_path
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "L.heatLayer" in content
    assert "leaflet_heat.min.js" in content


def test_to_heatmap_with_settings(geojson_file, tmp_path):
    """Test that to_heatmap respects custom settings"""
    output_path = tmp_path / "test_heatmap_settings.html"
    settings = HeatmapSettings(radius=30, blur=20)
    result = to_heatmap(geojson_file, output_path, settings=settings)

    assert result == output_path
    content = output_path.read_text(encoding="utf-8")
    # Check if settings values are in the generated JSON configuration
    assert '"radius": 30' in content
    assert '"blur": 20' in content


def test_to_heatmap_empty_geojson(tmp_path):
    """Test that to_heatmap raises error for GeoJSON with no points"""
    empty_geojson = tmp_path / "empty.geojson"
    empty_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="No point features in GeoJSON for heatmap"):
        to_heatmap(empty_geojson, tmp_path / "output.html")


def test_to_hotspots_basic(geojson_file, tmp_path):
    """Test basic functionality of to_hotspots"""
    output_path = tmp_path / "hotspots.geojson"
    # Use parameters that will work with our sample points
    result = to_hotspots(geojson_file, output_path, eps=0.0005, min_samples=1)

    assert result == output_path
    assert output_path.exists()

    content = json.loads(output_path.read_text(encoding="utf-8"))
    assert content["type"] == "FeatureCollection"
    assert len(content["features"]) > 0


def test_to_hotspots_empty_input(tmp_path):
    """Test to_hotspots with empty input"""
    input_path = tmp_path / "empty.geojson"
    input_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )

    output_path = tmp_path / "empty_hotspots.geojson"
    result = to_hotspots(input_path, output_path)

    assert result == output_path
    content = json.loads(output_path.read_text(encoding="utf-8"))
    assert content["type"] == "FeatureCollection"
    assert len(content["features"]) == 0


def test_to_hotspots_clustering(geojson_file, tmp_path):
    """Test that to_hotspots correctly clusters nearby points"""
    output_path = tmp_path / "clustered_hotspots.geojson"
    result = to_hotspots(
        geojson_file,
        output_path,
        eps=0.001,  # Small epsilon to ensure clustering of very close points
        min_samples=1,  # Small number to ensure clustering with our test data
    )

    content = json.loads(result.read_text(encoding="utf-8"))
    assert len(content["features"]) > 0

    # Check cluster properties
    feature = content["features"][0]
    assert "cluster_id" in feature["properties"]
    assert "count" in feature["properties"]
    assert isinstance(feature["properties"]["cluster_id"], int)
    assert isinstance(feature["properties"]["count"], int)


def test_to_hotspots_custom_parameters(geojson_file, tmp_path):
    """Test to_hotspots with custom clustering parameters"""
    output_path = tmp_path / "custom_hotspots.geojson"
    result = to_hotspots(
        geojson_file,
        output_path,
        eps=0.0005,
        min_samples=10,  # High value to ensure no clusters are formed
    )

    assert result == output_path
    content = json.loads(output_path.read_text(encoding="utf-8"))
    # With high min_samples, we expect no clusters
    assert len(content["features"]) == 0
