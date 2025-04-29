#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "Running base agent tests"
pytest tests/test_base_agent.py
echo "Done..."
echo "==============================================="

echo "Running dummy agent tests"
pytest tests/test_dummy_agent.py
echo "Done..."
echo "==============================================="

echo "Running llm wrapper tests"
pytest tests/test_llm_wrapper.py
echo "Done..."
echo "==============================================="

echo "Running memory store tests"
pytest tests/test_memory_store.py
echo "Done..."
echo "==============================================="

echo "Running ollama client tests"
pytest tests/test_ollama_client.py
echo "Done..."
echo "==============================================="