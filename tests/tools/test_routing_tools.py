"""Tests for routing_tools module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from src.config.settings import RouteSettings
from src.tools.routing_tools import (
    load_camera_points,
    build_pedestrian_graph,
    snap_to_graph,
    compute_shortest_path,
)


# Fixtures


@pytest.fixture
def sample_camera_geojson(tmp_path):
    """Create a temporary GeoJSON file with sample camera points."""
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.4050, 52.5200]},
                "properties": {"id": 1},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.4100, 52.5210]},
                "properties": {"id": 2},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.4150, 52.5220]},
                "properties": {"id": 3},
            },
        ],
    }

    geojson_path = tmp_path / "cameras.geojson"
    geojson_path.write_text(json.dumps(geojson_data), encoding="utf-8")
    return geojson_path


@pytest.fixture
def empty_camera_geojson(tmp_path):
    """Create a temporary GeoJSON file with no features."""
    geojson_data = {"type": "FeatureCollection", "features": []}

    geojson_path = tmp_path / "empty_cameras.geojson"
    geojson_path.write_text(json.dumps(geojson_data), encoding="utf-8")
    return geojson_path


@pytest.fixture
def mixed_geometry_geojson(tmp_path):
    """Create a GeoJSON with mixed geometry types (Points and LineStrings)."""
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.4050, 52.5200]},
                "properties": {"id": 1},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[13.4050, 52.5200], [13.4100, 52.5210]],
                },
                "properties": {"id": 2},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [13.4150, 52.5220]},
                "properties": {"id": 3},
            },
        ],
    }

    geojson_path = tmp_path / "mixed_cameras.geojson"
    geojson_path.write_text(json.dumps(geojson_data), encoding="utf-8")
    return geojson_path


@pytest.fixture
def synthetic_graph():
    """Create a small synthetic graph for testing without OSM data."""
    # Create a 3x3 grid graph with tuple node IDs
    grid = nx.grid_2d_graph(3, 3)

    # Convert to MultiDiGraph and relabel with integer node IDs
    # Map (i, j) -> i * 3 + j to get scalar IDs from 0-8
    mapping = {node: node[0] * 3 + node[1] for node in grid.nodes()}
    G = nx.relabel_nodes(grid, mapping)
    G = nx.MultiDiGraph(G)

    # Add graph-level attributes that osmnx expects
    G.graph["crs"] = "EPSG:4326"

    # Add node attributes (x, y coordinates) like OSMnx does
    # Recreate original grid positions from scalar IDs
    for node_id in G.nodes():
        i = node_id // 3  # row
        j = node_id % 3  # column
        G.nodes[node_id]["x"] = j * 0.01  # longitude-like
        G.nodes[node_id]["y"] = i * 0.01  # latitude-like

    # Add edge attributes (length) like OSMnx does
    for u, v, key in G.edges(keys=True):
        # Simple Euclidean distance in "degrees"
        dx = G.nodes[v]["x"] - G.nodes[u]["x"]
        dy = G.nodes[v]["y"] - G.nodes[u]["y"]
        G.edges[u, v, key]["length"] = (dx**2 + dy**2) ** 0.5

    return G


@pytest.fixture
def route_settings():
    """Create default RouteSettings for testing."""
    return RouteSettings()


# Tests for load_camera_points


def test_load_camera_points_success(sample_camera_geojson):
    """Test loading camera points from valid GeoJSON."""
    coords = load_camera_points(sample_camera_geojson)

    assert len(coords) == 3
    # Check format is (lat, lon) - GeoJSON stores as [lon, lat]
    assert coords[0] == (52.5200, 13.4050)
    assert coords[1] == (52.5210, 13.4100)
    assert coords[2] == (52.5220, 13.4150)


def test_load_camera_points_empty(empty_camera_geojson):
    """Test that empty GeoJSON raises ValueError."""
    with pytest.raises(ValueError, match="No point features found"):
        load_camera_points(empty_camera_geojson)


def test_load_camera_points_missing_file():
    """Test that missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="GeoJSON file not found"):
        load_camera_points(Path("/nonexistent/path.geojson"))


def test_load_camera_points_filters_non_point_features(mixed_geometry_geojson):
    """Test that non-Point features are filtered out."""
    coords = load_camera_points(mixed_geometry_geojson)

    # Should only have 2 points (LineString filtered out)
    assert len(coords) == 2
    assert coords[0] == (52.5200, 13.4050)
    assert coords[1] == (52.5220, 13.4150)


# Tests for build_pedestrian_graph


@patch("src.tools.routing_tools.ox.graph_from_place")
@patch("src.tools.routing_tools.ox.save_graphml")
@patch("src.tools.routing_tools.ox.load_graphml")
def test_build_pedestrian_graph_cache_miss(
    mock_load, mock_save, mock_from_place, route_settings, tmp_path
):
    """Test graph building when cache doesn't exist."""
    # Setup mock
    mock_graph = MagicMock(spec=nx.MultiDiGraph)
    mock_from_place.return_value = mock_graph

    cache_dir = tmp_path / ".graph_cache"

    result = build_pedestrian_graph("Berlin", "DE", route_settings, cache_dir=cache_dir)

    # Verify osmnx was called
    mock_from_place.assert_called_once_with(
        "Berlin, DE", network_type=route_settings.network_type
    )

    # Verify graph was saved to cache
    mock_save.assert_called_once()
    assert result == mock_graph


@patch("src.tools.routing_tools.ox.load_graphml")
def test_build_pedestrian_graph_cache_hit(mock_load, route_settings, tmp_path):
    """Test graph loading from cache when it exists."""
    # Create cache directory and file
    cache_dir = tmp_path / ".graph_cache"
    cache_dir.mkdir()

    # Create a dummy cache file (hash will match if inputs are same)
    import hashlib

    cache_key = hashlib.sha256(
        f"Berlin_DE_{route_settings.network_type}".encode()
    ).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.graphml"
    cache_file.touch()

    # Setup mock
    mock_graph = MagicMock(spec=nx.MultiDiGraph)
    mock_load.return_value = mock_graph

    result = build_pedestrian_graph("Berlin", "DE", route_settings, cache_dir=cache_dir)

    # Verify graph was loaded from cache
    mock_load.assert_called_once()
    assert result == mock_graph


@patch("src.tools.routing_tools.ox.graph_from_place")
def test_build_pedestrian_graph_invalid_location(
    mock_from_place, route_settings, tmp_path
):
    """Test that invalid location raises ValueError."""
    mock_from_place.side_effect = Exception("Location not found")

    with pytest.raises(ValueError, match="Failed to build graph"):
        build_pedestrian_graph(
            "NonexistentCity", None, route_settings, cache_dir=tmp_path
        )


# Tests for snap_to_graph


def test_snap_to_graph_success(synthetic_graph, route_settings):
    """Test snapping a coordinate to the nearest node."""
    # Point very close to node 0 which has coords (0.0, 0.0)
    lat, lon = 0.0001, 0.0001  # Very close to origin

    node = snap_to_graph(synthetic_graph, lat, lon, route_settings)

    # Should snap to node 0 (top-left corner)
    assert node == 0


def test_snap_to_graph_threshold_exceeded(synthetic_graph, route_settings):
    """Test that distant coordinates raise ValueError."""
    # Point very far from the graph (graph is at ~0.0-0.02 range)
    lat, lon = 10.0, 10.0

    with pytest.raises(ValueError, match="Cannot snap.*to walkable network"):
        snap_to_graph(synthetic_graph, lat, lon, route_settings)


def test_snap_to_graph_custom_threshold(synthetic_graph):
    """Test snap with custom threshold settings."""
    # Create settings with very small threshold
    settings = RouteSettings(snap_distance_threshold_m=1.0)

    # This should fail even for moderately distant points
    lat, lon = 0.1, 0.1  # About 11km away

    with pytest.raises(ValueError, match="Cannot snap"):
        snap_to_graph(synthetic_graph, lat, lon, settings)


# Tests for compute_shortest_path


def test_compute_shortest_path_success(synthetic_graph):
    """Test computing shortest path in connected graph."""
    # Path from corner to corner (node 0 to node 8)
    # Node 0 = (0,0), Node 8 = (2,2) in grid coordinates
    path = compute_shortest_path(synthetic_graph, 0, 8)

    # Should return a list of nodes
    assert isinstance(path, list)
    assert len(path) >= 3  # At minimum: start, some intermediates, end
    assert path[0] == 0
    assert path[-1] == 8


def test_compute_shortest_path_same_node(synthetic_graph):
    """Test path computation when start and end are the same."""
    # Node 4 = center of grid at (1, 1)
    path = compute_shortest_path(synthetic_graph, 4, 4)

    # Path to self should be just that node
    assert path == [4]


def test_compute_shortest_path_no_path():
    """Test that disconnected graph raises ValueError."""
    # Create disconnected graph
    G = nx.MultiDiGraph()
    G.add_node(1)
    G.add_node(2)
    # No edges between nodes - disconnected

    with pytest.raises(ValueError, match="No walkable path exists"):
        compute_shortest_path(G, 1, 2)


def test_compute_shortest_path_adjacent_nodes(synthetic_graph):
    """Test path between adjacent nodes."""
    # Node 0 = (0,0), Node 1 = (0,1) - adjacent horizontally
    path = compute_shortest_path(synthetic_graph, 0, 1)

    # Direct neighbors should have path of length 2
    assert len(path) == 2
    assert path == [0, 1]
