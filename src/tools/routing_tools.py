"""
Routing tools for computing low-surveillance walking routes.

This module provides composable functions for building pedestrian networks,
snapping coordinates to graph nodes, computing paths, and scoring routes based
on surveillance camera exposure.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point

from src.config.logger import logger
from src.config.models.route_models import RouteMetrics
from src.config.settings import RouteSettings


def load_camera_points(geojson_path: Path) -> List[Tuple[float, float]]:
    """
    Extract camera coordinates from enriched GeoJSON FeatureCollection.

    :param geojson_path: Path to the enriched camera GeoJSON file.
    :return: List of (latitude, longitude) tuples for each camera point.
    :raises FileNotFoundError: If geojson_path does not exist.
    :raises ValueError: If GeoJSON contains no point features.
    """
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    data = json.loads(geojson_path.read_text(encoding="utf-8"))

    # Extract coordinates from Point features only
    # GeoJSON format: [longitude, latitude], we return (latitude, longitude)
    coords = [
        (feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0])
        for feat in data.get("features", [])
        if feat.get("geometry", {}).get("type", "").lower() == "point"
    ]

    if not coords:
        raise ValueError(f"No point features found in GeoJSON: {geojson_path}")

    logger.info(f"Loaded {len(coords)} camera points from {geojson_path.name}")
    return coords


def build_pedestrian_graph(
    city: str,
    country: Optional[str],
    settings: RouteSettings,
    cache_dir: Path = Path("overpass_data/.graph_cache"),
) -> nx.MultiDiGraph:
    """
    Build or load a cached pedestrian network graph for a city.

    Uses OSMnx to download OpenStreetMap data and construct a routable graph.
    Results are cached to disk to avoid repeated downloads for the same city.

    :param city: City name.
    :param country: Optional ISO country code for disambiguation.
    :param settings: RouteSettings instance containing network_type.
    :param cache_dir: Directory for caching OSM graphs.
    :return: NetworkX MultiDiGraph representing the pedestrian network.
    :raises ValueError: If osmnx cannot find the specified location.
    """
    # Create cache key from city, country, and network type
    location_str = f"{city}, {country}" if country else city
    cache_key_input = f"{city}_{country}_{settings.network_type}"
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_key}.graphml"

    # Check cache first
    if cache_file.exists():
        logger.info(f"Loading cached graph for {location_str} from {cache_file.name}")
        return ox.load_graphml(cache_file)

    # Cache miss - download from OSM
    logger.info(
        f"Downloading pedestrian network for {location_str} "
        f"(network_type={settings.network_type})"
    )

    try:
        G = ox.graph_from_place(location_str, network_type=settings.network_type)
    except Exception as e:
        raise ValueError(f"Failed to build graph for '{location_str}': {str(e)}") from e

    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, cache_file)
    logger.info(f"Cached graph to {cache_file}")

    return G


def snap_to_graph(
    G: nx.MultiDiGraph, lat: float, lon: float, settings: RouteSettings
) -> int:
    """
    Snap a latitude/longitude coordinate to the nearest graph node.

    :param G: NetworkX graph representing the street network.
    :param lat: Latitude of the point to snap.
    :param lon: Longitude of the point to snap.
    :param settings: RouteSettings instance containing snap_distance_threshold_m.
    :return: Node ID of the nearest node in the graph.
    :raises ValueError: If the nearest node is farther than the threshold distance.
    """
    # osmnx expects (longitude, latitude) order
    nearest_node = ox.distance.nearest_nodes(G, lon, lat, return_dist=False)

    # Calculate actual distance to verify it's within threshold
    node_data = G.nodes[nearest_node]
    node_lat = node_data["y"]
    node_lon = node_data["x"]

    # Simple haversine-like distance approximation (good enough for short distances)
    # More accurate would be to use osmnx.distance.great_circle, but this is simpler
    from math import radians, sin, cos, sqrt, atan2

    R = 6371000  # Earth radius in meters

    lat1, lon1 = radians(lat), radians(lon)
    lat2, lon2 = radians(node_lat), radians(node_lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_m = R * c

    if distance_m > settings.snap_distance_threshold_m:
        raise ValueError(
            f"Cannot snap ({lat}, {lon}) to walkable network: "
            f"nearest node is {distance_m:.1f}m away "
            f"(threshold: {settings.snap_distance_threshold_m}m)"
        )

    logger.debug(
        f"Snapped ({lat:.6f}, {lon:.6f}) to node {nearest_node} "
        f"at distance {distance_m:.1f}m"
    )

    return nearest_node


def compute_shortest_path(G: nx.MultiDiGraph, src: int, dst: int) -> List[int]:
    """
    Compute the shortest path between two nodes by distance.

    :param G: NetworkX graph representing the street network.
    :param src: Source node ID.
    :param dst: Destination node ID.
    :return: List of node IDs representing the shortest path.
    :raises ValueError: If no path exists between the nodes.
    """
    try:
        path = nx.shortest_path(G, src, dst, weight="length")
    except nx.NetworkXNoPath as e:
        raise ValueError(
            f"No walkable path exists between nodes {src} and {dst}"
        ) from e

    logger.debug(f"Computed shortest path: {len(path)} nodes from {src} to {dst}")
    return path


def generate_candidate_paths(
    G: nx.MultiDiGraph, src: int, dst: int, k: int
) -> List[List[int]]:
    """
    Generate up to k candidate paths between nodes.

    Uses NetworkX k-shortest simple paths algorithm to find alternative routes.
    If fewer than k simple paths exist, returns all available paths.
    Always returns at least one path (the shortest) if any path exists.

    :param G: NetworkX graph representing the street network.
    :param src: Source node ID.
    :param dst: Destination node ID.
    :param k: Maximum number of candidate paths to generate.
    :return: List of paths, where each path is a list of node IDs.
    :raises ValueError: If no path exists between the nodes.
    """
    try:
        # nx.shortest_simple_paths returns a generator
        path_generator = nx.shortest_simple_paths(G, src, dst, weight="length")

        # Collect up to k paths
        paths = []
        for path in path_generator:
            paths.append(path)
            if len(paths) >= k:
                break

        if not paths:
            raise ValueError(f"No paths found between nodes {src} and {dst}")

        logger.debug(
            f"Generated {len(paths)} candidate path(s) between {src} and {dst} "
            f"(requested {k})"
        )
        return paths

    except nx.NetworkXNoPath as e:
        raise ValueError(
            f"No walkable path exists between nodes {src} and {dst}"
        ) from e


def compute_exposure_for_path(
    G: nx.MultiDiGraph,
    path_nodes: List[int],
    cameras: List[Tuple[float, float]],
    settings: RouteSettings,
) -> RouteMetrics:
    """
    Compute exposure metrics for a path based on nearby surveillance cameras.

    The function samples the path geometry at regular intervals, buffers it by
    the configured radius, and counts cameras within the buffer using spatial
    indexing for performance.

    :param G: NetworkX graph representing the street network.
    :param path_nodes: List of node IDs representing the path.
    :param cameras: List of (latitude, longitude) tuples for camera positions.
    :param settings: RouteSettings instance containing buffer_radius_m.
    :return: RouteMetrics instance with exposure score and camera count.
    """
    # Handle edge case: zero cameras
    if len(cameras) == 0:
        logger.warning("No cameras in dataset - route has zero exposure")
        # Calculate path length
        path_length_m = _calculate_path_length(G, path_nodes)
        return RouteMetrics(
            length_m=path_length_m,
            exposure_score=0.0,
            camera_count_near_route=0,
        )

    # Build path geometry from node coordinates
    path_coords = []
    for node in path_nodes:
        node_data = G.nodes[node]
        path_coords.append((node_data["x"], node_data["y"]))  # (lon, lat)

    # Create LineString for the path
    path_line = LineString(path_coords)

    # Calculate total path length in meters
    path_length_m = _calculate_path_length(G, path_nodes)

    # Convert buffer radius from meters to degrees (approximate)
    # At equator: 1 degree ≈ 111km, so buffer_m / 111000 gives degrees
    buffer_radius_deg = settings.buffer_radius_m / 111000.0

    # Create buffered polygon around the path
    buffered_path = path_line.buffer(buffer_radius_deg)

    # Build GeoDataFrame for cameras with spatial index
    camera_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in cameras],
        crs="EPSG:4326",
    )

    # Create GeoDataFrame for buffered path
    path_gdf = gpd.GeoDataFrame([{"geometry": buffered_path}], crs="EPSG:4326")

    # Spatial join to find cameras within buffer
    cameras_in_buffer = gpd.sjoin(camera_gdf, path_gdf, predicate="within", how="inner")

    camera_count = len(cameras_in_buffer)

    # Compute exposure score
    # Simple model: cameras per km of route
    if path_length_m > 0:
        exposure_score = (camera_count / path_length_m) * 1000.0  # per km
    else:
        exposure_score = 0.0

    logger.debug(
        f"Path exposure: {camera_count} cameras along {path_length_m:.1f}m "
        f"(score: {exposure_score:.2f} cameras/km)"
    )

    return RouteMetrics(
        length_m=path_length_m,
        exposure_score=exposure_score,
        camera_count_near_route=camera_count,
    )


def _calculate_path_length(G: nx.MultiDiGraph, path_nodes: List[int]) -> float:
    """
    Calculate total path length in meters from edge weights.

    :param G: NetworkX graph representing the street network.
    :param path_nodes: List of node IDs representing the path.
    :return: Total path length in meters.
    """
    if len(path_nodes) < 2:
        return 0.0

    total_length = 0.0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]

        # Get edge data (there may be multiple edges between nodes)
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            # Take the first edge's length (key=0)
            length = edge_data[0].get("length", 0.0)
            total_length += length

    return total_length
