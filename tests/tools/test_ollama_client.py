import pytest
import requests

from src.tools.ollama_client import OllamaClient


class DummyResponse:
    def __init__(self, status_code, data=None, exc=None):
        self.status_code = status_code
        self._data = data
        self._exc = exc

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"Status {self.status_code}")

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def test_ollama_success(monkeypatch, ollama_settings):
    client = OllamaClient(ollama_settings)
    dummy = {"foo": "bar"}

    def fake_post(url, json, timeout):
        assert url == ollama_settings.base_url
        return DummyResponse(200, data=dummy)

    monkeypatch.setattr(requests, "post", fake_post)

    result = client("hello")
    assert result == dummy


def test_ollama_http_error(monkeypatch, ollama_settings):
    client = OllamaClient(ollama_settings)

    def fake_post(url, json, timeout):
        return DummyResponse(500)

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError):
        client("hello")


def test_ollama_invalid_json(monkeypatch, ollama_settings):
    client = OllamaClient(ollama_settings)

    def fake_post(url, json, timeout):
        return DummyResponse(200, data=ValueError("bad json"))

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError):
        client("hello")
