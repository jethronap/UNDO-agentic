import json
from types import SimpleNamespace

import pytest
import requests

from src.config.settings import OllamaSettings, DatabaseSettings, OverpassSettings


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


@pytest.fixture
def ovp_settings(tmp_path):
    return OverpassSettings(
        endpoint="https://overpass.test/api",
        headers={"User-Agent": "pytest"},
        query_timeout=10,
        timeout=2,
        dir=str(tmp_path / "ovp"),
        max_attempts=2,
        base_delay=0.0,
    )


class _DummyResp(SimpleNamespace):
    def raise_for_status(self):
        if 400 <= self.status_code:
            raise requests.HTTPError(response=self)

    def json(self):
        return json.loads(self._text) if isinstance(self._text, str) else self._text

    text = property(lambda self: self._text)


@pytest.fixture
def patch_requests(monkeypatch):
    """
    Fixture that lets each test prepare responses:

        store["get"]  = _DummyResp(...)
        store["post"] = _DummyResp(...)
    """
    store = {"get": None, "post": None}

    def fake_get(*_, **__):
        return store["get"]

    def fake_post(*_, **__):
        return store["post"]

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    return store


class MemoryStoreFake:
    def __init__(self):
        self.rows = []

    def store(self, agent_id: str, step: str, content: str):
        row = SimpleNamespace(agent_id=agent_id, step=step, content=content)
        self.rows.append(row)
        return row

    def load(self, agent_id: str):
        return [r for r in self.rows if r.agent_id == agent_id]


@pytest.fixture
def mem_fake():
    return MemoryStoreFake()
