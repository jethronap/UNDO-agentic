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

import networkx as nx
import osmnx as ox

from src.config.logger import logger
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
