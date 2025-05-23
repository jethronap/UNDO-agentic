import requests
import pytest
from src.utils.decorators import with_retry
from src.config.settings import OverpassSettings
from tests.conftest import http_error


def test_with_retry_no_retry(fake_sleep):
    def fn():
        return {"ok": True}

    wrapped = with_retry(fn, OverpassSettings(max_attempts=3, base_delay=0.1))
    assert wrapped() == {"ok": True}
    assert fake_sleep == []  # never slept


def test_with_retry_recovers(fake_sleep):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError()
        return {"ok": True}

    wrapped = with_retry(fn, OverpassSettings(max_attempts=3, base_delay=0.1))
    assert wrapped() == {"ok": True}
    assert calls["n"] == 2
    # slept once with base_delay
    assert fake_sleep == [0.1]


def test_with_retry_fatal_http(fake_sleep):
    def fn():
        raise http_error(400)

    wrapped = with_retry(fn, OverpassSettings(max_attempts=3, base_delay=0.1))
    with pytest.raises(requests.HTTPError):
        wrapped()
    assert fake_sleep == []  # immediate raise


def test_with_retry_exhausts_attempts(fake_sleep):
    def fn():
        raise requests.Timeout()

    settings = OverpassSettings(max_attempts=3, base_delay=0.1)
    wrapped = with_retry(fn, settings)

    with pytest.raises(RuntimeError):
        wrapped()

    # slept twice: 0.1, 0.2   (no sleep after final failure)
    assert fake_sleep == [0.1, 0.2]
