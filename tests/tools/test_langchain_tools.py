"""Tests for LangChain tools wrapper module."""

import json
from pathlib import Path
from unittest.mock import patch

from src.tools.langchain_tools import (
    load_overpass_elements,
    save_enriched_elements,
    to_geojson,
    to_heatmap,
    to_hotspots,
    compute_statistics,
    private_public_pie,
    plot_zone_sensitivity,
    plot_sensitivity_reasons,
    plot_hotspots,
    get_all_tools,
    get_io_tools,
    get_analysis_tools,
    get_visualization_tools,
)


class TestToolRegistries:
    """Test tool registry functions."""

    def test_get_all_tools(self):
        """Test that get_all_tools returns all expected tools."""
        tools = get_all_tools()
        assert len(tools) == 11
        tool_names = [t.name for t in tools]
        assert "load_overpass_elements" in tool_names
        assert "compute_statistics" in tool_names
        assert "plot_hotspots" in tool_names

    def test_get_io_tools(self):
        """Test that get_io_tools returns only I/O tools."""
        tools = get_io_tools()
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "load_overpass_elements" in tool_names
        assert "save_enriched_elements" in tool_names
        assert "save_overpass_dump" in tool_names
        assert "to_geojson" in tool_names

    def test_get_analysis_tools(self):
        """Test that get_analysis_tools returns only analysis tools."""
        tools = get_analysis_tools()
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "compute_statistics" in tool_names
        assert "to_hotspots" in tool_names

    def test_get_visualization_tools(self):
        """Test that get_visualization_tools returns only visualization tools."""
        tools = get_visualization_tools()
        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "to_heatmap" in tool_names
        assert "plot_hotspots" in tool_names


class TestLoadOverpassElements:
    """Test load_overpass_elements tool."""

    @patch("src.tools.langchain_tools._load_overpass_elements")
    def test_successful_load(self, mock_load):
        """Test successful loading of Overpass elements."""
        mock_elements = [{"id": 1, "tags": {}}, {"id": 2, "tags": {}}]
        mock_load.return_value = mock_elements

        result_str = load_overpass_elements.invoke({"path": "test.json"})
        result = json.loads(result_str)

        assert result["count"] == 2
        assert "elements" in result
        mock_load.assert_called_once()

    @patch("src.tools.langchain_tools._load_overpass_elements")
    def test_file_not_found(self, mock_load):
        """Test handling of missing file."""
        mock_load.side_effect = FileNotFoundError("File not found")

        result_str = load_overpass_elements.invoke({"path": "missing.json"})
        result = json.loads(result_str)

        assert result["success"] is False
        assert "error" in result


class TestSaveEnrichedElements:
    """Test save_enriched_elements tool."""

    @patch("src.tools.langchain_tools._save_enriched_elements")
    def test_successful_save(self, mock_save):
        """Test successful saving of enriched elements."""
        mock_save.return_value = "/path/to/enriched.json"
        elements = json.dumps([{"id": 1, "analysis": {}}])

        result_str = save_enriched_elements.invoke(
            {"elements": elements, "path": "source.json"}
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" in result
        mock_save.assert_called_once()

    @patch("src.tools.langchain_tools._save_enriched_elements")
    def test_save_with_dict_wrapper(self, mock_save):
        """Test saving when elements are wrapped in dict."""
        mock_save.return_value = "/path/to/enriched.json"
        elements = json.dumps({"elements": [{"id": 1, "analysis": {}}]})

        result_str = save_enriched_elements.invoke(
            {"elements": elements, "path": "source.json"}
        )
        result = json.loads(result_str)

        assert result["success"] is True


class TestComputeStatistics:
    """Test compute_statistics tool."""

    @patch("src.tools.langchain_tools._compute_statistics")
    def test_successful_computation(self, mock_compute):
        """Test successful statistics computation."""
        from collections import Counter

        mock_compute.return_value = {
            "total": 100,
            "sensitive_count": 30,
            "public_count": 60,
            "private_count": 40,
            "zone_counts": Counter({"town": 50, "building": 30}),
            "zone_sensitivity_counts": Counter({"town": 20}),
            "camera_type_counts": Counter({"dome": 40, "fixed": 60}),
            "operator_counts": Counter({"police": 20}),
        }

        elements = json.dumps(
            [{"id": 1, "analysis": {"sensitive": True, "public": True}}]
        )

        result_str = compute_statistics.invoke({"elements": elements})
        result = json.loads(result_str)

        assert result["total"] == 100
        assert result["sensitive_count"] == 30
        assert isinstance(result["zone_counts"], dict)
        assert "town" in result["zone_counts"]

    @patch("src.tools.langchain_tools._compute_statistics")
    def test_handles_dict_wrapper(self, mock_compute):
        """Test that compute_statistics handles dict-wrapped elements."""
        from collections import Counter

        mock_compute.return_value = {
            "total": 10,
            "sensitive_count": 5,
            "public_count": 5,
            "private_count": 5,
            "zone_counts": Counter({"town": 10}),
            "zone_sensitivity_counts": Counter(),
            "camera_type_counts": Counter(),
            "operator_counts": Counter(),
        }

        elements = json.dumps({"elements": [{"id": 1, "analysis": {}}]})

        result_str = compute_statistics.invoke({"elements": elements})
        result = json.loads(result_str)

        assert "total" in result


class TestToGeoJSON:
    """Test to_geojson tool."""

    @patch("src.tools.langchain_tools._to_geojson")
    def test_successful_conversion(self, mock_convert):
        """Test successful GeoJSON conversion."""
        mock_convert.return_value = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature"}] * 5,
        }

        result_str = to_geojson.invoke(
            {"enriched_file": "enriched.json", "output_file": "output.geojson"}
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["feature_count"] == 5
        assert "output_path" in result

    @patch("src.tools.langchain_tools._to_geojson")
    def test_without_output_file(self, mock_convert):
        """Test GeoJSON conversion without specifying output file."""
        mock_convert.return_value = {"type": "FeatureCollection", "features": []}

        result_str = to_geojson.invoke({"enriched_file": "enriched.json"})
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" not in result


class TestToHeatmap:
    """Test to_heatmap tool."""

    @patch("src.tools.langchain_tools._to_heatmap")
    def test_successful_heatmap_creation(self, mock_heatmap):
        """Test successful heatmap creation."""
        mock_heatmap.return_value = Path("/path/to/heatmap.html")

        result_str = to_heatmap.invoke(
            {"geojson_path": "data.geojson", "output_html": "heatmap.html"}
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" in result


class TestToHotspots:
    """Test to_hotspots tool."""

    @patch("src.tools.langchain_tools._to_hotspots")
    def test_successful_hotspots_computation(self, mock_hotspots):
        """Test successful hotspots computation."""
        mock_hotspots.return_value = Path("/path/to/hotspots.geojson")

        result_str = to_hotspots.invoke(
            {
                "geojson_path": "data.geojson",
                "output_file": "hotspots.geojson",
                "eps": 0.0001,
                "min_samples": 5,
            }
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" in result
        mock_hotspots.assert_called_once()


class TestPrivatePublicPie:
    """Test private_public_pie tool."""

    @patch("src.tools.langchain_tools._private_public_pie")
    def test_successful_chart_creation(self, mock_chart):
        """Test successful pie chart creation."""
        mock_chart.return_value = Path("/path/to/chart.png")

        stats = json.dumps({"total": 100, "public_count": 60, "private_count": 40})

        result_str = private_public_pie.invoke(
            {"stats": stats, "output_dir": "/output"}
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" in result


class TestPlotZoneSensitivity:
    """Test plot_zone_sensitivity tool."""

    @patch("src.tools.langchain_tools._plot_zone_sensitivity")
    def test_successful_plot_creation(self, mock_plot):
        """Test successful zone sensitivity plot creation."""
        mock_plot.return_value = Path("/path/to/zone_chart.png")

        stats = json.dumps(
            {
                "zone_counts": {"town": 50, "building": 30},
                "zone_sensitivity_counts": {"town": 20},
            }
        )

        result_str = plot_zone_sensitivity.invoke(
            {"stats": stats, "output_dir": "/output", "top_n": 10}
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" in result


class TestPlotSensitivityReasons:
    """Test plot_sensitivity_reasons tool."""

    @patch("src.tools.langchain_tools._plot_sensitivity_reasons")
    def test_successful_reasons_plot(self, mock_plot):
        """Test successful sensitivity reasons plot creation."""
        mock_plot.return_value = Path("/path/to/reasons.png")

        result_str = plot_sensitivity_reasons.invoke(
            {"enriched_file": "enriched.json", "output_file": "reasons.png", "top_n": 5}
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" in result


class TestPlotHotspots:
    """Test plot_hotspots tool."""

    @patch("src.tools.langchain_tools._plot_hotspots")
    def test_successful_hotspots_plot(self, mock_plot):
        """Test successful hotspots visualization creation."""
        mock_plot.return_value = Path("/path/to/hotspots.png")

        result_str = plot_hotspots.invoke(
            {"hotspots_file": "hotspots.geojson", "output_file": "hotspots.png"}
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert "output_path" in result


class TestToolSchemas:
    """Test that tools have proper schemas."""

    def test_load_overpass_elements_has_schema(self):
        """Test that load_overpass_elements has input schema."""
        assert load_overpass_elements.args_schema is not None
        schema = load_overpass_elements.args_schema.model_json_schema()
        assert "path" in schema["properties"]

    def test_compute_statistics_has_schema(self):
        """Test that compute_statistics has input schema."""
        assert compute_statistics.args_schema is not None
        schema = compute_statistics.args_schema.model_json_schema()
        assert "elements" in schema["properties"]

    def test_to_hotspots_has_schema(self):
        """Test that to_hotspots has input schema with defaults."""
        assert to_hotspots.args_schema is not None
        schema = to_hotspots.args_schema.model_json_schema()
        assert "eps" in schema["properties"]
        assert "min_samples" in schema["properties"]
        # Check defaults are present
        assert schema["properties"]["eps"]["default"] == 0.0001
        assert schema["properties"]["min_samples"]["default"] == 5


class TestToolDescriptions:
    """Test that tools have proper descriptions."""

    def test_all_tools_have_descriptions(self):
        """Test that all tools have non-empty descriptions."""
        tools = get_all_tools()
        for tool in tools:
            assert tool.description
            assert len(tool.description) > 10  # Meaningful description

    def test_tool_descriptions_are_helpful(self):
        """Test that tool descriptions provide useful information."""
        # Test a few key tools
        assert "surveillance" in load_overpass_elements.description.lower()
        assert "statistics" in compute_statistics.description.lower()
        assert "hotspots" in to_hotspots.description.lower()
