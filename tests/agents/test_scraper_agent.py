import json

import pytest

from src.utils.db import query_hash, payload_hash
from src.agents.scraper_agent import ScraperAgent


PAYLOAD_OK = {"elements": [{"id": 1}, {"id": 2}]}
PAYLOAD_EMPTY = {"elements": []}


def make_agent(mem_fake, run_stub, save_stub):
    """
    Helper to build an agent with injected stubs
    """
    tools = {"run_query": run_stub, "save_json": save_stub}
    return ScraperAgent(name="ScraperAgent", memory=mem_fake, tools=tools)


def test_first_run_saves(mem_fake, tmp_path, monkeypatch):
    saved_path = tmp_path / "lund.json"

    def run_stub(query):
        return PAYLOAD_OK

    def save_stub(data, city, overpass_dir):
        assert data is PAYLOAD_OK
        assert city == "Lund"
        return saved_path

    agent = make_agent(mem_fake, run_stub, save_stub)
    ctx = agent.achieve_goal({"city": "Lund"})

    # context flags
    assert ctx["cache_hit"] is False
    assert ctx["elements_count"] == 2
    assert ctx["save_json"] == str(saved_path)

    # memory rows contain cache entry
    cache_rows = [r for r in mem_fake.rows if r.step == "cache"]
    assert len(cache_rows) == 1
    qh = query_hash(ctx["query"])
    ph = payload_hash(PAYLOAD_OK)
    assert f"{qh}|{saved_path}|{ph}" in cache_rows[0].content


def test_second_run_hits_cache(mem_fake, tmp_path, monkeypatch):
    cached_path = tmp_path / "lund.json"
    cached_path.write_text(json.dumps(PAYLOAD_OK))

    # the exact query string we'll pretend the agent will use
    q = "FAKE QUERY TEXT"

    # pre‑seed cache row
    mem_fake.store(
        "ScraperAgent",
        "cache",
        f"{query_hash(q)}|{cached_path}|{payload_hash(PAYLOAD_OK)}",
    )

    # stub tools that must NOT be called
    def run_stub(_):
        pytest.fail("run_query should not be called on cache hit")

    def save_stub(*_):
        pytest.fail("save_json should not be called on cache hit")

    # make build_query return the same fake string
    monkeypatch.setattr("src.agents.scraper_agent.build_query", lambda *a, **k: q)

    agent = make_agent(mem_fake, run_stub, save_stub)

    ctx = agent.achieve_goal({"city": "Lund"})

    assert ctx["cache_hit"] is True
    assert ctx["run_query"] == PAYLOAD_OK
    assert ctx["save_json"] == str(cached_path)


def test_empty_result_skips_save(mem_fake, monkeypatch):
    def run_stub(query):
        return PAYLOAD_EMPTY

    def save_stub(*_):
        pytest.fail("save_json must not be called for empty result")

    agent = make_agent(mem_fake, run_stub, save_stub)

    ctx = agent.achieve_goal({"city": "Nowhere"})

    assert ctx["elements_count"] == 0
    assert ctx["save_json"] == "NO_DATA"

    empty_rows = [r for r in mem_fake.rows if r.step == "empty"]
    assert empty_rows, "empty marker row should be stored"
