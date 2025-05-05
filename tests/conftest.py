import pytest
import requests

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


@pytest.fixture
def fake_sleep(monkeypatch):
    """
    Replace time.sleep so tests run instantly and record delays.
    """
    delays = []

    def _fake_sleep(seconds: float):
        delays.append(seconds)

    monkeypatch.setattr("time.sleep", _fake_sleep)
    return delays


def http_error(status: int) -> requests.HTTPError:
    r = requests.Response()
    r.status_code = status
    return requests.HTTPError(response=r)
