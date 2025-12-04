"""
WebSocket connection manager for real-time progress updates.

This module provides WebSocket connection management for broadcasting
task progress updates to connected clients.
"""

from typing import Dict, List
from fastapi import WebSocket

from src.config.logger import logger


class WebSocketManager:
    """
    Manages WebSocket connections for real-time task updates.

    Maintains a mapping of task_id to list of connected WebSocket clients,
    allowing progress updates to be broadcast to all interested clients.
    """

    def __init__(self):
        """Initialize the WebSocket manager with empty connection storage."""
        # Map task_id -> list of WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, task_id: str, websocket: WebSocket) -> None:
        """
        Accept and store a WebSocket connection for a specific task.

        :param task_id: Task identifier to monitor
        :param websocket: WebSocket connection to add
        """
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logger.debug(f"WebSocket connected for task {task_id}")

    def disconnect(self, task_id: str, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.

        :param task_id: Task identifier
        :param websocket: WebSocket connection to remove
        """
        if task_id in self.active_connections:
            try:
                self.active_connections[task_id].remove(websocket)
                logger.debug(f"WebSocket disconnected for task {task_id}")

                # Clean up empty connection lists
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]
            except ValueError:
                # Connection already removed
                pass

    async def broadcast_progress(self, task_id: str, data: dict) -> None:
        """
        Broadcast progress update to all connected clients for a task.

        Automatically removes dead connections that fail to receive updates.

        :param task_id: Task identifier
        :param data: Progress data to send (will be JSON serialized)
        """
        if task_id not in self.active_connections:
            return

        dead_connections = []

        for connection in self.active_connections[task_id]:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket: {e}")
                dead_connections.append(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(task_id, conn)

    def get_connection_count(self, task_id: str) -> int:
        """
        Get number of active connections for a task.

        :param task_id: Task identifier
        :return: Number of active connections
        """
        return len(self.active_connections.get(task_id, []))


# Global WebSocket manager instance
ws_manager = WebSocketManager()
