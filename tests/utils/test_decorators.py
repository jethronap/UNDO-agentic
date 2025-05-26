import requests
import pytest
import time
from src.utils.decorators import with_retry, log_action
from src.config.settings import OverpassSettings
from tests.conftest import http_error, DummyAgent


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


def test_simple_logs_start_and_end(swap_logger):
    d = DummyAgent()
    result = d.simple(21, context={"foo": "bar"})
    assert result == 42

    # first info call is at entry
    assert swap_logger.infos[0] == "MyAgent.simple"
    # debug should show first arg and context keys
    assert "args[0]=21" in swap_logger.debugs[0]
    assert "context_keys=['foo']" in swap_logger.debugs[1]
    # final info shows elapsed time and no hint
    end_msg = swap_logger.infos[-1]
    assert end_msg.startswith("MyAgent.simple: in ")
    assert "(count=" not in end_msg and ":" not in end_msg


def test_hint_for_list_and_file(swap_logger):
    d = DummyAgent()

    # list result hint
    lst = d.make_list(3)
    assert lst == [0, 1, 2]
    end = swap_logger.infos[-1]
    assert "(count=3)" in end

    # file result hint
    out = d.save_file("foo/bar")
    assert out == "foo/bar.json"
    end = swap_logger.infos[-1]
    assert ": 'foo/bar.json'" in end


def test_error_bubbles_and_logs_exception(swap_logger):
    d = DummyAgent()
    with pytest.raises(ValueError):
        d.blows_up()
    # we expect an info at start and no normal end-info,
    # but we did not catch exceptions in decorator, so no exception() call.
    # If you want exception() behavior, you'd have to catch in the decorator.
    assert swap_logger.infos[0] == "MyAgent.blows_up"
    # no successful end info for blows_up
    assert not any("blows_up: in" in msg for msg in swap_logger.infos[1:])


def test_timing_accuracy(swap_logger):
    # make sure we really time the function
    class Slow:
        name = "SlowAgent"

        @log_action
        def wait(self, sec):
            time.sleep(sec)
            return "done"

    s = Slow()
    # start = time.time()
    r = s.wait(0.05)
    assert r == "done"
    elapsed_logged = swap_logger.infos[-1]
    # parse the elapsed
    part = elapsed_logged.split("in ")[1]
    # e.g. "0.06s"
    secs = float(part.rstrip("s").split()[0])
    assert secs >= 0.05
