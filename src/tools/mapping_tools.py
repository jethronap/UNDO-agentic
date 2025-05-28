from pathlib import Path
from typing import Union, List, Dict, Tuple

import folium
from folium.plugins import HeatMap
import json
import numpy as np
from sklearn.cluster import DBSCAN

from src.config.settings import HeatmapSettings


def to_heatmap(
    geojson_path: Path, output_html: Path, settings: HeatmapSettings = HeatmapSettings()
) -> Path:
    """
    Read a GeoJSON FeatureCollection of Point features, build a folium HeatMap, and save to an HTML file.
    :param geojson_path: The Path object to the geojson file
    :param output_html: The Path object the output html file
    :param settings: The Heatmap settings
    :return: The html filepath
    """
    data = json.loads(geojson_path.read_text(encoding="utf-8"))

    coords = [
        (feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0])
        for feat in data.get("features", [])
        if feat["geometry"]["type"].lower() == "point"
    ]
    if not coords:
        raise RuntimeError("No point features in GeoJSON for heatmap")

    # center map at mean lat/lon
    avg_lat = sum(lat for lat, _ in coords) / len(coords)
    avg_lon = sum(lon for _, lon in coords) / len(coords)

    m = folium.Map(location=(avg_lat, avg_lon), zoom_start=13)
    HeatMap(coords, radius=settings.radius, blur=settings.blur).add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    return output_html


def to_hotspots(
    geojson_path: Union[str, Path],
    output_file: Union[str, Path],
    *,
    eps: float = 0.0001,
    min_samples: int = 5,
) -> Path:
    """
    Read a GeoJSON of points, cluster them with DBSCAN, and write
    a new GeoJSON of cluster centroids with a `count` property.

    :param geojson_path: Path to enriched .geojson of cameras.
    :param output_file: Path where to write the hotspots GeoJSON.
    :param eps: DBSCAN epsilon in degrees (~0.005° ≈ 500m).
    :param min_samples: Minimum points per cluster.
    :return: Path to the written hotspots geojson
    """
    gj = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    coords = []
    for feat in gj.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        coords.append((lat, lon))

    if not coords:
        # nothing to cluster; emit an empty collection
        out = {"type": "FeatureCollection", "features": []}
        Path(output_file).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return Path(output_file)

    X = np.radians(np.array(coords))
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="haversine").fit(X)

    labels = clustering.labels_
    clusters: Dict[int, List[Tuple[float, float]]] = {}

    for lbl, (lat, lon) in zip(labels, coords):
        if lbl == -1:
            continue
        clusters.setdefault(lbl, []).append((lat, lon))

    features = []
    for lbl, pts in clusters.items():
        mean_lat = sum(p[0] for p in pts) / len(pts)
        mean_lon = sum(p[1] for p in pts) / len(pts)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [mean_lon, mean_lat]},
                # cast lbl to native int
                "properties": {"cluster_id": int(lbl), "count": len(pts)},
            }
        )

    out = {"type": "FeatureCollection", "features": features}
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path
