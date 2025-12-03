"""
Fixtures for integration tests.

Integration tests use real agents and LLMs, so we don't patch the client
like we do in unit tests.
"""

import pytest


# Override the patch_client autouse fixture from parent conftest
@pytest.fixture(autouse=True)
def patch_client():
    """Don't patch LLM client for integration tests - use real LLM."""
    # This fixture intentionally does nothing - it just overrides the parent
    # conftest's autouse fixture to prevent it from patching the client
    pass


# Override the patch_prompt_template autouse fixture
@pytest.fixture(autouse=True)
def patch_prompt_template():
    """Don't patch prompt templates for integration tests - use real prompts."""
    # This fixture intentionally does nothing
    pass
