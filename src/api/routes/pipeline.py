"""
Pipeline execution endpoints.

This module provides endpoints for running the complete surveillance analysis pipeline,
including scraping, analysis, and optional routing.
"""

from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.models.requests import PipelineRequest
from src.api.models.responses import TaskResponse, TaskStatus
from src.api.services.task_manager import task_manager
from src.api.services.websocket_manager import ws_manager
from src.orchestration.langchain_pipeline import create_pipeline
from src.config.logger import logger

router = APIRouter(prefix="/pipeline")


async def execute_pipeline_task(task_id: str, request: PipelineRequest) -> None:
    """
    Execute pipeline in background with real-time progress updates.

    This function runs the SurveillancePipeline asynchronously and broadcasts
    progress updates via WebSocket to connected clients.

    :param task_id: Task identifier
    :param request: Pipeline request parameters
    """
    try:
        task_manager.mark_running(task_id)
        logger.info(f"Starting pipeline task {task_id} for {request.city}")

        # Broadcast initialization
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "progress",
                "stage": "initializing",
                "progress": 0,
                "message": f"Initializing pipeline for {request.city}",
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Build configuration from request
        config_kwargs = {}

        # Add routing config if provided
        if request.routing_config:
            config_kwargs.update(
                {
                    "routing_enabled": True,
                    "start_lat": request.routing_config.start_lat,
                    "start_lon": request.routing_config.start_lon,
                    "end_lat": request.routing_config.end_lat,
                    "end_lon": request.routing_config.end_lon,
                }
            )

        # Broadcast scraping stage
        task_manager.update_progress(task_id, 20, "Scraping surveillance data...")
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "progress",
                "stage": "scraping",
                "progress": 20,
                "message": "Downloading surveillance data from OpenStreetMap",
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Create and run pipeline
        pipeline = create_pipeline(request.scenario, **config_kwargs)

        run_kwargs = {}
        if request.country:
            run_kwargs["country"] = request.country

        # Broadcast analysis stage
        task_manager.update_progress(task_id, 50, "Analyzing data...")
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "progress",
                "stage": "analyzing",
                "progress": 50,
                "message": "Analyzing surveillance infrastructure",
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Execute the pipeline (this is the same call the CLI uses)
        results = pipeline.run(request.city, **run_kwargs)

        # Broadcast completion
        task_manager.mark_completed(task_id, results)
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "completed",
                "stage": "completed",
                "progress": 100,
                "message": "Pipeline completed successfully",
                "timestamp": datetime.now().isoformat(),
            },
        )

        logger.info(f"Pipeline task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"Pipeline task {task_id} failed: {e}")
        task_manager.mark_failed(task_id, str(e))

        # Broadcast failure
        await ws_manager.broadcast_progress(
            task_id,
            {
                "type": "failed",
                "stage": "failed",
                "progress": 0,
                "message": f"Pipeline failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            },
        )


@router.post("/run", response_model=TaskResponse)
async def run_pipeline(
    request: PipelineRequest, background_tasks: BackgroundTasks
) -> TaskResponse:
    """
    Start a complete pipeline execution.

    The pipeline will run in the background. Use the returned task_id
    to check status via GET /api/v1/pipeline/{task_id}.

    :param request: Pipeline configuration
    :param background_tasks: FastAPI background tasks handler
    :return: Task creation response with task_id
    """
    # Create task
    task_id = task_manager.create_task(
        "pipeline",
        metadata={
            "city": request.city,
            "country": request.country,
            "scenario": request.scenario.value,
        },
    )

    # Schedule background execution
    background_tasks.add_task(execute_pipeline_task, task_id, request)

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Pipeline started for {request.city}",
    )


@router.get("/{task_id}")
async def get_pipeline_status(task_id: str):
    """
    Get pipeline task status and results.

    :param task_id: Task identifier
    :return: Task status with results (if completed)
    :raises HTTPException: 404 if task not found
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task.to_dict()


@router.post("/{task_id}/cancel")
async def cancel_pipeline(task_id: str):
    """
    Cancel a running pipeline task.

    Note: This only marks the task as cancelled. Actual cancellation
    of running operations is not implemented yet.

    :param task_id: Task identifier
    :return: Cancellation confirmation
    :raises HTTPException: 404 if task not found
    """
    task = task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel task in {task.status.value} state"
        )

    task_manager.mark_cancelled(task_id)

    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Task cancelled successfully",
    }


@router.delete("/{task_id}")
async def delete_pipeline_task(task_id: str):
    """
    Delete a pipeline task and its results.

    :param task_id: Task identifier
    :return: Deletion confirmation
    :raises HTTPException: 404 if task not found
    """
    if not task_manager.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    return {"task_id": task_id, "message": "Task deleted successfully"}
