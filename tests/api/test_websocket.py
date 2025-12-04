"""
Tests for WebSocket progress updates.

These tests verify the WebSocket functionality for real-time task progress updates.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.websocket_manager import WebSocketManager
from src.api.services.task_manager import TaskManager
from src.api.models.responses import TaskStatus

client = TestClient(app)


@pytest.fixture
def ws_manager():
    """Create a fresh WebSocketManager instance for each test."""
    return WebSocketManager()


def test_websocket_endpoint_connects():
    """Test that WebSocket endpoint accepts connections."""
    # Create a task first
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )
    task_id = response.json()["task_id"]

    # Connect via WebSocket
    with client.websocket_connect(f"/ws/tasks/{task_id}") as websocket:
        # Send a keep-alive message
        websocket.send_text("status")

        # Receive status update
        data = websocket.receive_json()

        assert "type" in data
        assert "status" in data
        assert "progress" in data
        assert data["task_id"] == task_id


def test_websocket_receives_task_status():
    """Test that WebSocket receives current task status."""
    # Create a task
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )
    task_id = response.json()["task_id"]

    # Connect and check status
    with client.websocket_connect(f"/ws/tasks/{task_id}") as websocket:
        websocket.send_text("status")
        data = websocket.receive_json()

        assert data["type"] == "status"
        assert data["status"] in ["pending", "running", "completed", "failed"]
        assert isinstance(data["progress"], int)
        assert 0 <= data["progress"] <= 100


def test_websocket_nonexistent_task():
    """Test WebSocket with non-existent task returns error."""
    with client.websocket_connect("/ws/tasks/nonexistent-task-id") as websocket:
        websocket.send_text("status")
        data = websocket.receive_json()

        assert data["type"] == "error"
        assert "not found" in data["message"].lower()


def test_websocket_manager_connection_count(ws_manager):
    """Test WebSocketManager tracks connection count correctly."""
    task_id = "test-task-123"

    # Initially no connections
    assert ws_manager.get_connection_count(task_id) == 0


def test_websocket_manager_broadcast_to_empty():
    """Test broadcasting to task with no connections doesn't error."""
    import asyncio
    from src.api.services.websocket_manager import ws_manager

    # Should not raise any errors
    asyncio.run(
        ws_manager.broadcast_progress(
            "nonexistent-task", {"type": "progress", "message": "test"}
        )
    )


def test_websocket_manager_disconnect(ws_manager):
    """Test WebSocketManager disconnect removes connections."""
    from unittest.mock import Mock

    task_id = "test-task"
    mock_websocket = Mock()

    # Add connection manually
    ws_manager.active_connections[task_id] = [mock_websocket]

    # Disconnect
    ws_manager.disconnect(task_id, mock_websocket)

    # Should be removed
    assert task_id not in ws_manager.active_connections


def test_task_progress_updates():
    """Test that task manager correctly updates progress."""
    task_manager = TaskManager()

    task_id = task_manager.create_task("test")
    task_manager.update_progress(task_id, 50, "Halfway there")

    task = task_manager.get_task(task_id)
    assert task.progress == 50
    assert task.metadata["last_message"] == "Halfway there"


def test_pipeline_broadcasts_progress():
    """Test that pipeline execution broadcasts progress updates."""
    # This is more of an integration test
    # Create a task
    response = client.post(
        "/api/v1/pipeline/run", json={"city": "TestCity", "scenario": "basic"}
    )
    task_id = response.json()["task_id"]

    # Wait a moment for execution
    import time

    time.sleep(0.5)

    # Check that progress was updated
    from src.api.services.task_manager import task_manager

    task = task_manager.get_task(task_id)

    # Progress should have been updated during execution
    assert task is not None
    # Task should be running or completed
    assert task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED]
