import re
import json
from pathlib import Path

from sqlalchemy.engine import Engine
from src.utils.db import get_engine, summarize, query_hash, payload_hash


def test_get_engine_sqlite(db_settings, tmp_path):
    engine: Engine = get_engine(db_settings)
    # must point to the temp sqlite file asked for
    assert engine.url.get_backend_name() == "sqlite"
    assert Path(engine.url.database).name == "test_memory.db"


def test_summarize_dict_elements():
    data = {"elements": [{"id": 1}, {"id": 2}, {"id": 3}]}
    out = summarize(data)
    # contains element count and an 8‑char hex digest
    assert "elements=3" in out
    assert re.search(r"sha256=[0-9a-f]{8}", out)


def test_summarize_long_string():
    long = "x" * 250
    out = summarize(long, max_len=50)
    # truncated and ends with ellipsis
    assert out.endswith("…")
    assert len(out) == 51  # 50 chars + ellipsis


def test_query_hash_stable_unique():
    q1 = "SELECT * FROM foo"
    q2 = "SELECT * FROM bar"

    h1a = query_hash(q1)
    h1b = query_hash(q1)
    h2 = query_hash(q2)

    assert h1a == h1b  # stable
    assert h1a != h2  # unique
    assert re.fullmatch(r"[0-9a-f]{8}", h1a)  # 8‑char hex


def test_payload_hash_changes_with_data():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "b": 3}  # slight change

    h1a = payload_hash(d1)
    h1b = payload_hash(json.loads(json.dumps(d1)))  # same semantics
    h2 = payload_hash(d2)

    assert h1a == h1b  # deterministic
    assert h1a != h2  # reflects data change
    assert len(h1a) == 64  # full SHA‑256 hex
