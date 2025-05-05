#!/bin/bash
PYTHONPATH="${PYTHONPATH}:$(realpath "./src")"
export PYTHONPATH

# Navigate to the script's directory (project root)
cd "$(dirname "$0")" || exit

echo "Running base agent tests"
pytest tests/agents/test_base_agent.py
echo "Done..."
echo "==============================================="

echo "Running dummy agent tests"
pytest tests/agents/test_dummy_agent.py
echo "Done..."
echo "==============================================="

echo "Running llm wrapper tests"
pytest tests/tools/test_llm_wrapper.py
echo "Done..."
echo "==============================================="

echo "Running ollama client tests"
pytest tests/tools/test_ollama_client.py
echo "Done..."
echo "==============================================="

echo "Running memory store tests"
pytest tests/memory/test_memory_store.py
echo "Done..."
echo "==============================================="

echo "Running with_retry decorator tests"
pytest tests/utils/test_retry§.py
echo "Done..."
echo "==============================================="