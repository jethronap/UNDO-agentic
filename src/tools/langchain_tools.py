"""
LangChain Tools Wrapper Module.

This module wraps existing utility functions as LangChain Tool objects
"""

from pathlib import Path
from typing import List, Optional
import json

from langchain_core.tools import tool

from src.config.logger import logger
from src.config.models.tools import (
    LoadOverpassInput,
    SaveEnrichedInput,
    SaveOverpassInput,
    ToGeoJSONInput,
    ToHeatmapInput,
    ToHotspotsInput,
    ComputeStatisticsInput,
    PrivatePublicPieInput,
    PlotZoneSensitivityInput,
    PlotSensitivityReasonsInput,
    PlotHotspotsInput,
)
from src.tools.io_tools import (
    load_overpass_elements as _load_overpass_elements,
    save_enriched_elements as _save_enriched_elements,
    save_overpass_dump as _save_overpass_dump,
    to_geojson as _to_geojson,
)
from src.tools.mapping_tools import (
    to_heatmap as _to_heatmap,
    to_hotspots as _to_hotspots,
)
from src.tools.stat_tools import compute_statistics as _compute_statistics
from src.tools.chart_tools import (
    private_public_pie as _private_public_pie,
    plot_zone_sensitivity as _plot_zone_sensitivity,
    plot_sensitivity_reasons as _plot_sensitivity_reasons,
    plot_hotspots as _plot_hotspots,
)

# ============================================================================
# LangChain Tool Wrappers
# ============================================================================


@tool(args_schema=LoadOverpassInput)
def load_overpass_elements(path: str) -> str:
    """
    This tool loads raw surveillance data previously fetched from
    Overpass API and saved to disk.

    :param path: Path to the Overpass JSON dump file to load
    :return: JSON string containing loaded elements and count statistics
    """

    try:
        logger.debug(f"[Tool] Loading Overpass elements from: {path}")
        elements = _load_overpass_elements(path)
        logger.info(f"[Tool] Loaded {len(elements)} elements from {path}")
        return json.dumps({"elements": elements, "count": len(elements)})
    except FileNotFoundError:
        error_msg = f"File not found: {path}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})
    except Exception as e:
        error_msg = f"Failed to load Overpass elements: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=SaveEnrichedInput)
def save_enriched_elements(elements: str, path: str) -> str:
    """
    This tool saves enriched surveillance data (elements that have been
    analyzed and augmented with additional metadata) to disk.

    :param elements: JSON string containing enriched surveillance elements
    :param path: Path where the enriched JSON file should be saved
    :return: JSON string containing output path and success status
    """
    try:
        logger.debug(f"[Tool] Saving enriched elements for: {path}")
        # Parse JSON string back to list
        elements_data = json.loads(elements)
        if isinstance(elements_data, dict) and "elements" in elements_data:
            elements_list = elements_data["elements"]
        else:
            elements_list = elements_data

        output_path = _save_enriched_elements(elements_list, path)
        logger.info(f"[Tool] Saved enriched data to: {output_path}")
        return json.dumps({"output_path": output_path, "success": True})
    except Exception as e:
        error_msg = f"Failed to save enriched elements: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=SaveOverpassInput)
def save_overpass_dump(data: str, city: str, dest: str) -> str:
    """
    This tool saves raw Overpass API response data to disk in JSON format.

    :param data: JSON string containing the raw Overpass API response
    :param city: Name of the city the data was collected for
    :param dest: Destination directory where the file should be saved
    :return: JSON string containing output path and success status
    """
    try:
        logger.debug(f"[Tool] Saving Overpass dump for city: {city}")
        # Parse JSON string back to dict
        data_dict = json.loads(data)
        output_path = _save_overpass_dump(data_dict, city, dest)
        logger.info(f"[Tool] Saved Overpass dump to: {output_path}")
        return json.dumps({"output_path": str(output_path), "success": True})
    except Exception as e:
        error_msg = f"Failed to save Overpass dump: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=ToGeoJSONInput)
def to_geojson(enriched_file: str, output_file: Optional[str] = None) -> str:
    """
    Convert enriched surveillance data to GeoJSON format.

    This tool converts enriched surveillance camera data into GeoJSON format
    for mapping and spatial analysis. Each camera becomes a GeoJSON Feature
    with properties from the enriched data.

    :param enriched_file: Path to the JSON file containing enriched surveillance elements
    :param output_file: Optional path where the GeoJSON should be saved. If not provided,
                       the GeoJSON is only returned in memory
    :return: JSON string containing feature count and success status. If output_file
             was provided, also includes the output path
    """
    try:
        logger.debug(f"[Tool] Converting to GeoJSON: {enriched_file}")
        geojson_data = _to_geojson(enriched_file, output_file)
        feature_count = len(geojson_data.get("features", []))
        logger.info(f"[Tool] Created GeoJSON with {feature_count} features")

        result = {
            "feature_count": feature_count,
            "success": True,
        }
        if output_file:
            result["output_path"] = output_file

        return json.dumps(result)
    except Exception as e:
        error_msg = f"Failed to create GeoJSON: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=ToHeatmapInput)
def to_heatmap(geojson_path: str, output_html: str) -> str:
    """
    Create an interactive heatmap visualization from surveillance data.

    This tool generates a Folium-based heatmap showing the density of surveillance
    cameras across the geographic area. Areas with higher camera concentration
    appear "hotter" on the map.

    :param geojson_path: Path to the GeoJSON file containing camera locations
    :param output_html: Path where the output HTML map should be saved
    :return: JSON string containing the output path and success status
    """
    try:
        logger.debug(f"[Tool] Creating heatmap from: {geojson_path}")
        output_path = _to_heatmap(Path(geojson_path), Path(output_html))
        logger.info(f"[Tool] Created heatmap at: {output_path}")
        return json.dumps({"output_path": str(output_path), "success": True})
    except Exception as e:
        error_msg = f"Failed to create heatmap: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=ToHotspotsInput)
def to_hotspots(
    geojson_path: str, output_file: str, eps: float = 0.0001, min_samples: int = 5
) -> str:
    """
    Generate surveillance camera hotspots using DBSCAN clustering.

    This tool analyzes the spatial distribution of surveillance cameras to identify
    areas with high camera density (hotspots) using the DBSCAN clustering algorithm.
    The results are saved as a GeoJSON file containing cluster centroids and metadata.

    :param geojson_path: Path to input GeoJSON file containing camera locations
    :param output_file: Path where the output GeoJSON file with hotspots should be saved
    :param eps: DBSCAN epsilon parameter - maximum distance between points in a cluster
    :param min_samples: DBSCAN minimum samples parameter - minimum points to form a cluster
    :return: JSON string containing the output path and success status
    """
    try:
        logger.debug(f"[Tool] Computing hotspots from: {geojson_path}")
        output_path = _to_hotspots(
            geojson_path, output_file, eps=eps, min_samples=min_samples
        )
        logger.info(f"[Tool] Created hotspots file at: {output_path}")
        return json.dumps({"output_path": str(output_path), "success": True})
    except Exception as e:
        error_msg = f"Failed to compute hotspots: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=ComputeStatisticsInput)
def compute_statistics(elements: str) -> str:
    """
    Compute statistical summaries of surveillance camera data.

    This tool analyzes a collection of surveillance cameras and computes various
    statistical measures including totals, distributions across zones, sensitivity
    counts, and type/operator breakdowns.

    :param elements: JSON string containing surveillance camera elements to analyze,
                    either as a list or dict with "elements" key
    :return: JSON string containing computed statistics including total counts,
             sensitivity breakdowns, zone distributions, and type/operator counts
    """
    try:
        logger.debug("[Tool] Computing statistics")
        # Parse JSON string back to list
        elements_data = json.loads(elements)
        if isinstance(elements_data, dict) and "elements" in elements_data:
            elements_list = elements_data["elements"]
        else:
            elements_list = elements_data

        stats = _compute_statistics(elements_list)

        # Convert Counter objects to dicts for JSON serialization
        stats_serializable = {
            "total": stats["total"],
            "sensitive_count": stats["sensitive_count"],
            "public_count": stats["public_count"],
            "private_count": stats["private_count"],
            "zone_counts": dict(stats["zone_counts"]),
            "zone_sensitivity_counts": dict(stats["zone_sensitivity_counts"]),
            "camera_type_counts": dict(stats["camera_type_counts"]),
            "operator_counts": dict(stats["operator_counts"]),
        }

        logger.info(
            f"[Tool] Computed statistics: {stats_serializable['total']} total cameras"
        )
        return json.dumps(stats_serializable)
    except Exception as e:
        error_msg = f"Failed to compute statistics: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=PrivatePublicPieInput)
def private_public_pie(stats: str, output_dir: str) -> str:
    """
    Create a pie chart showing distribution of private vs public cameras.

    This tool generates a pie chart visualization showing the proportion of
    surveillance cameras that are privately operated versus publicly operated.
    The chart is saved as an image file in the specified output directory.

    :param stats: JSON string containing statistics including 'public_count' and
                 'private_count' values from compute_statistics()
    :param output_dir: Directory path where the output chart image should be saved
    :return: JSON string containing the output file path and success status
    """
    try:
        logger.debug("[Tool] Creating privacy distribution pie chart")
        # Parse JSON string back to dict
        stats_dict = json.loads(stats)
        output_path = _private_public_pie(stats_dict, Path(output_dir))
        logger.info(f"[Tool] Created pie chart at: {output_path}")
        return json.dumps({"output_path": str(output_path), "success": True})
    except Exception as e:
        error_msg = f"Failed to create pie chart: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=PlotZoneSensitivityInput)
def plot_zone_sensitivity(stats: str, output_dir: str, top_n: int = 10) -> str:
    """
    Create a bar chart showing camera sensitivity by zone.

    This tool generates a visualization showing how many cameras in each zone
    were flagged as sensitive vs non-sensitive. The chart displays the top N
    zones by total camera count.

    :param stats: JSON string containing statistics including 'zone_sensitivity_counts'
                 from compute_statistics()
    :param output_dir: Directory path where the output chart image should be saved
    :param top_n: Number of top zones (by camera count) to include in visualization
    :return: JSON string containing the output file path and success status
    """
    try:
        logger.debug("[Tool] Creating zone sensitivity chart")
        # Parse JSON string back to dict
        stats_dict = json.loads(stats)
        output_path = _plot_zone_sensitivity(stats_dict, Path(output_dir), top_n=top_n)
        logger.info(f"[Tool] Created zone sensitivity chart at: {output_path}")
        return json.dumps({"output_path": str(output_path), "success": True})
    except Exception as e:
        error_msg = f"Failed to create zone sensitivity chart: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=PlotSensitivityReasonsInput)
def plot_sensitivity_reasons(
    enriched_file: str, output_file: str, top_n: int = 5
) -> str:
    """
    Create a bar chart showing the most common reasons for camera sensitivity.

    This tool analyzes the sensitivity reason tags assigned to cameras during
    enrichment and generates a visualization showing the distribution of these
    reasons. The chart displays the top N most frequent sensitivity reasons.

    :param enriched_file: Path to JSON file containing enriched camera data
    :param output_file: Path where the output chart image should be saved
    :param top_n: Number of top sensitivity reasons to include in visualization
    :return: JSON string containing the output file path and success status
    """
    try:
        logger.debug(f"[Tool] Creating sensitivity reasons chart from: {enriched_file}")
        output_path = _plot_sensitivity_reasons(enriched_file, output_file, top_n=top_n)
        logger.info(f"[Tool] Created sensitivity reasons chart at: {output_path}")
        return json.dumps({"output_path": str(output_path), "success": True})
    except Exception as e:
        error_msg = f"Failed to create sensitivity reasons chart: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


@tool(args_schema=PlotHotspotsInput)
def plot_hotspots(hotspots_file: str, output_file: str) -> str:
    """
    Create a visualization of surveillance camera hotspots.

    This tool generates a visualization showing the identified camera hotspots
    (clusters) and their characteristics like size, density and geographic
    distribution. The plot helps identify areas with high concentrations of
    surveillance coverage.

    :param hotspots_file: Path to GeoJSON file containing hotspot cluster data
                         generated by to_hotspots()
    :param output_file: Path where the output visualization should be saved
    :return: JSON string containing the output file path and success status
    """
    try:
        logger.debug(f"[Tool] Creating hotspots visualization from: {hotspots_file}")
        output_path = _plot_hotspots(hotspots_file, output_file)
        logger.info(f"[Tool] Created hotspots visualization at: {output_path}")
        return json.dumps({"output_path": str(output_path), "success": True})
    except Exception as e:
        error_msg = f"Failed to create hotspots visualization: {str(e)}"
        logger.error(f"[Tool] {error_msg}")
        return json.dumps({"error": error_msg, "success": False})


# ============================================================================
# Tool Registry
# ============================================================================


def get_all_tools() -> List:
    """
    Get a list of all available LangChain tools.

    Returns:
        List of LangChain tool objects that agents can use.
    """
    return [
        load_overpass_elements,
        save_enriched_elements,
        save_overpass_dump,
        to_geojson,
        to_heatmap,
        to_hotspots,
        compute_statistics,
        private_public_pie,
        plot_zone_sensitivity,
        plot_sensitivity_reasons,
        plot_hotspots,
    ]


def get_io_tools() -> List:
    """Get tools related to input/output operations."""
    return [
        load_overpass_elements,
        save_enriched_elements,
        save_overpass_dump,
        to_geojson,
    ]


def get_analysis_tools() -> List:
    """Get tools related to data analysis."""
    return [
        compute_statistics,
        to_hotspots,
    ]


def get_visualization_tools() -> List:
    """Get tools related to data visualization."""
    return [
        to_heatmap,
        private_public_pie,
        plot_zone_sensitivity,
        plot_sensitivity_reasons,
        plot_hotspots,
    ]
