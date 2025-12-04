"""
FastAPI web interface for the Agentic Counter-Surveillance system.

This package provides a RESTful API that exposes all CLI functionality via HTTP endpoints.
The API is completely independent of the CLI and reuses the existing SurveillancePipeline.
"""

from src.api.main import app

__all__ = ["app"]
