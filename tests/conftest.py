import pytest
from src.config.settings import OllamaSettings, DatabaseSettings


@pytest.fixture
def ollama_settings():
    return OllamaSettings(
        base_url="http://test-server/api/generate",
        timeout_seconds=0.1,
        model="test-model",
    )


@pytest.fixture
def db_settings(tmp_path):
    return DatabaseSettings(
        url=f"sqlite:///{tmp_path / 'test_memory.db'}",
        echo=False,
    )
