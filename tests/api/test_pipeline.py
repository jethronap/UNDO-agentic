"""
Tests for pipeline execution endpoints.

These tests verify the pipeline API endpoints work correctly,
including task creation, status retrieval, and lifecycle management.
"""

import time

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_pipeline_run_returns_task_id():
    """Test that POST /api/v1/pipeline/run returns a task_id."""
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "task_id" in data
    assert "status" in data
    assert "message" in data
    assert data["status"] == "pending"
    assert "TestCity" in data["message"]


def test_pipeline_run_with_country():
    """Test pipeline run with country code."""
    response = client.post(
        "/api/v1/pipeline/run",
        json={"city": "Berlin", "country": "DE", "scenario": "full"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"


def test_pipeline_run_with_routing():
    """Test pipeline run with routing configuration."""
    response = client.post(
        "/api/v1/pipeline/run",
        json={
            "city": "Berlin",
            "scenario": "basic",
            "routing_config": {
                "city": "Berlin",
                "start_lat": 52.52,
                "start_lon": 13.40,
                "end_lat": 52.50,
                "end_lon": 13.42,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"


def test_get_pipeline_status():
    """Test GET /api/v1/pipeline/{task_id} returns task status."""
    # Create a task
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )
    task_id = response.json()["task_id"]

    # Get task status
    response = client.get(f"/api/v1/pipeline/{task_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == task_id
    assert data["type"] == "pipeline"
    assert "status" in data
    assert "progress" in data
    assert "created_at" in data
    assert "metadata" in data


def test_get_nonexistent_pipeline():
    """Test getting status of non-existent pipeline."""
    response = client.get("/api/v1/pipeline/nonexistent-task-id")
    assert response.status_code == 404


def test_cancel_pipeline():
    """Test POST /api/v1/pipeline/{task_id}/cancel."""
    # Create a task
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )
    task_id = response.json()["task_id"]

    # Cancel the task (before it completes)
    # Note: In real scenario, task might complete before cancellation
    response = client.post(f"/api/v1/pipeline/{task_id}/cancel")

    # Could be 200 (successfully cancelled) or 400 (already completed)
    assert response.status_code in [200, 400]


def test_cancel_completed_pipeline():
    """Test that cancelling a completed pipeline returns error."""
    # Create and wait for task to complete
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )
    task_id = response.json()["task_id"]

    # Wait for task to complete
    time.sleep(1)

    # Try to cancel completed task
    response = client.post(f"/api/v1/pipeline/{task_id}/cancel")

    # Should fail because task is already completed/failed
    assert response.status_code == 400


def test_cancel_nonexistent_pipeline():
    """Test cancelling non-existent pipeline."""
    response = client.post("/api/v1/pipeline/nonexistent-task-id/cancel")
    assert response.status_code == 404


def test_delete_pipeline():
    """Test DELETE /api/v1/pipeline/{task_id}."""
    # Create a task
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )
    task_id = response.json()["task_id"]

    # Delete the task
    response = client.delete(f"/api/v1/pipeline/{task_id}")
    assert response.status_code == 200

    # Verify task is deleted
    response = client.get(f"/api/v1/pipeline/{task_id}")
    assert response.status_code == 404


def test_delete_nonexistent_pipeline():
    """Test deleting non-existent pipeline."""
    response = client.delete("/api/v1/pipeline/nonexistent-task-id")
    assert response.status_code == 404


def test_pipeline_invalid_request():
    """Test pipeline run with invalid request data."""
    response = client.post(
        "/api/v1/pipeline/run",
        json={
            "city": "Berlin",
            "scenario": "invalid_scenario",  # Invalid scenario
        },
    )

    # Should return 422 Unprocessable Entity due to validation error
    assert response.status_code == 422


def test_pipeline_task_metadata():
    """Test that task metadata contains request information."""
    response = client.post(
        "/api/v1/pipeline/run",
        json={"city": "Berlin", "country": "DE", "scenario": "full"},
    )
    task_id = response.json()["task_id"]

    # Get task and check metadata
    response = client.get(f"/api/v1/pipeline/{task_id}")
    data = response.json()

    assert data["metadata"]["city"] == "Berlin"
    assert data["metadata"]["country"] == "DE"
    assert data["metadata"]["scenario"] == "full"
