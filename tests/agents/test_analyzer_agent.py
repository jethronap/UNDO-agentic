import json
from pathlib import Path
import pytest

from src.agents.analyzer_agent import AnalyzerAgent
from src.utils.db import payload_hash
from tests.conftest import make_raw_dump, set_stub_response


def test_first_run_creates_files(tmp_path, mem_fake):
    """
    • LLM is called once.
    • enriched + geojson are written.
    • stats are returned.
    """
    raw_path, raw = make_raw_dump(tmp_path)

    # what the LLM should spit back for each element
    set_stub_response(
        {
            "public": False,
            "sensitive": False,
            "camera_type": None,
            "mount_type": None,
        }
    )

    agent = AnalyzerAgent("AnalyzerAgent", memory=mem_fake)
    ctx = agent.achieve_goal({"path": str(raw_path)})

    # -------- assertions ----------
    enr = Path(ctx["output_path"])
    gj = Path(ctx["geojson_path"])

    assert enr.exists() and gj.exists()
    # enriched has our analysis field
    loaded = json.loads(enr.read_text())["elements"][0]["analysis"]
    assert loaded["public"] is False

    # GeoJSON features == 1
    assert len(json.loads(gj.read_text())["features"]) == 1

    # stats present & sensible
    stats = ctx["stats"]
    assert stats.get("total") == 1
    assert stats.get("sensitive_count") == 0

    # memory got a cache entry
    assert any(r.step == "enriched_cache" for r in mem_fake.rows)


def test_second_run_hits_cache(tmp_path, mem_fake, monkeypatch):
    """
    Pre-seed cache & existing files – LLM must **not** be invoked.
    """
    raw_path, raw = make_raw_dump(tmp_path)

    # Pretend previous run already produced these
    enriched_p = raw_path.with_name("lund_enriched.json")
    geojson_p = enriched_p.with_suffix(".geojson")
    enriched_p.write_text(json.dumps({"elements": []}), encoding="utf-8")
    geojson_p.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), "utf-8"
    )

    # cache row
    raw_hash = payload_hash(raw)
    mem_fake.store(
        "AnalyzerAgent",
        "enriched_cache",
        f"{raw_hash}|{enriched_p}|{geojson_p}",
    )

    # make LLM blow up if ever called
    from src.llm.langchain_llm import LangChainLLM

    monkeypatch.setattr(
        LangChainLLM,
        "generate_response",
        lambda *_a, **_kw: pytest.fail("LLM should not be called"),
    )

    agent = AnalyzerAgent("AnalyzerAgent", memory=mem_fake)
    ctx = agent.achieve_goal({"path": str(raw_path)})

    assert ctx["output_path"] == str(enriched_p)
    assert ctx["geojson_path"] == str(geojson_p)
