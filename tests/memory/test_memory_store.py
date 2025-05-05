from src.memory.store import MemoryStore
from src.memory.models import Memory


def test_store_and_load_roundtrip(db_settings):
    store = MemoryStore(db_settings)
    # store 2 memories for AgentA
    m1 = store.store("AgentA", "step1", "data1")
    m2 = store.store("AgentA", "step2", "data2")
    assert isinstance(m1, Memory) and isinstance(m2, Memory)

    records = store.load("AgentA")
    assert len(records) == 2
    # ensure content matches
    contents = {r.content for r in records}
    assert contents == {"data1", "data2"}


def test_isolated_databases(tmp_path):
    # two separate stores shouldn't see each other's data
    from src.config.settings import DatabaseSettings

    s1 = DatabaseSettings(url=f"sqlite:///{tmp_path / 'a.db'}")
    s2 = DatabaseSettings(url=f"sqlite:///{tmp_path / 'b.db'}")
    st1 = MemoryStore(s1)
    st2 = MemoryStore(s2)
    st1.store("X", "s", "d")
    assert len(st1.load("X")) == 1
    assert st2.load("X") == []
