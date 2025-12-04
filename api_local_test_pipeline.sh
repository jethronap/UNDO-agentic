#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "Running health endpoint tests"
pytest tests/api/test_health.py
echo "Done..."
echo "==============================================="

echo "Running api models tests"
pytest tests/api/test_models.py
echo "Done..."
echo "==============================================="

echo "Running pipeline endpoint tests"
pytest tests/api/test_pipeline.py
echo "Done..."
echo "==============================================="

echo "Running task manager tests"
pytest tests/api/test_task_manager.py
echo "Done..."
echo "==============================================="