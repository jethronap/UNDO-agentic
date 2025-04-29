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


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    # Replace OllamaClient used inside LocalLLM with our stub
    monkeypatch.setattr("src.tools.llm_wrapper.OllamaClient", StubClient)


class StubClient:
    def __init__(self, settings):
        pass

    def __call__(self, prompt, **kwargs):
        return self._response
