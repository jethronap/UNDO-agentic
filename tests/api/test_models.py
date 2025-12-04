"""
Tests for API request and response models.

These tests verify that Pydantic models correctly validate input data
and provide appropriate error messages for invalid inputs.
"""

import pytest
from pydantic import ValidationError

from src.api.models.requests import (
    ScrapeRequest,
    AnalyzeRequest,
    RouteComputeRequest,
    PipelineRequest,
)
from src.api.models.responses import (
    TaskStatus,
    TaskResponse,
    RouteMetricsResponse,
)
from src.config.pipeline_config import AnalysisScenario


def test_scrape_request_valid():
    """Test ScrapeRequest with valid data."""
    request = ScrapeRequest(city="Berlin", country="DE")
    assert request.city == "Berlin"
    assert request.country == "DE"


def test_scrape_request_without_country():
    """Test ScrapeRequest without optional country field."""
    request = ScrapeRequest(city="Berlin")
    assert request.city == "Berlin"
    assert request.country is None


def test_analyze_request_valid():
    """Test AnalyzeRequest with valid data."""
    request = AnalyzeRequest(
        data_path="overpass_data/berlin/berlin.json", scenario=AnalysisScenario.FULL
    )
    assert request.data_path == "overpass_data/berlin/berlin.json"
    assert request.scenario == AnalysisScenario.FULL


def test_route_compute_request_valid():
    """Test RouteComputeRequest with valid coordinates."""
    request = RouteComputeRequest(
        city="Berlin",
        country="DE",
        start_lat=52.52,
        start_lon=13.40,
        end_lat=52.50,
        end_lon=13.42,
    )
    assert request.city == "Berlin"
    assert request.start_lat == 52.52
    assert request.end_lat == 52.50


def test_route_compute_request_invalid_coordinates():
    """Test RouteComputeRequest rejects invalid coordinates."""
    with pytest.raises(ValidationError) as exc_info:
        RouteComputeRequest(
            city="Berlin",
            start_lat=100.0,  # Invalid: > 90
            start_lon=13.40,
            end_lat=52.50,
            end_lon=13.42,
        )
    assert "start_lat" in str(exc_info.value)


def test_pipeline_request_basic():
    """Test PipelineRequest without routing."""
    request = PipelineRequest(
        city="Berlin", country="DE", scenario=AnalysisScenario.BASIC
    )
    assert request.city == "Berlin"
    assert request.scenario == AnalysisScenario.BASIC
    assert request.routing_config is None


def test_pipeline_request_with_routing():
    """Test PipelineRequest with routing configuration."""
    routing_config = RouteComputeRequest(
        city="Berlin", start_lat=52.52, start_lon=13.40, end_lat=52.50, end_lon=13.42
    )
    request = PipelineRequest(
        city="Berlin", scenario=AnalysisScenario.FULL, routing_config=routing_config
    )
    assert request.routing_config is not None
    assert request.routing_config.start_lat == 52.52


def test_task_status_enum():
    """Test TaskStatus enum values."""
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.RUNNING.value == "running"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"
    assert TaskStatus.CANCELLED.value == "cancelled"


def test_task_response_valid():
    """Test TaskResponse model."""
    response = TaskResponse(
        task_id="test-123", status=TaskStatus.PENDING, message="Task started"
    )
    assert response.task_id == "test-123"
    assert response.status == TaskStatus.PENDING
    assert response.message == "Task started"


def test_route_metrics_response_valid():
    """Test RouteMetricsResponse model."""
    metrics = RouteMetricsResponse(
        length_m=1250.5,
        exposure_score=12.3,
        camera_count=8,
        baseline_length_m=950.2,
        baseline_exposure_score=18.7,
    )
    assert metrics.length_m == 1250.5
    assert metrics.camera_count == 8
