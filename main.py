from src.config.logger import logger

from src.config.settings import DatabaseSettings
from src.memory.store import MemoryStore


def main():
    """
    Quick test of the SQLModel-based MemoryStore:
     1. Stores a test memory.
     2. Loads and prints all memories for the test agent.
    """
    # 1. Load DB settings and init store
    db_settings = DatabaseSettings()
    memory = MemoryStore(db_settings)

    # 2. Store a test memory
    logger.info("Storing a test memory...")
    test_mem = memory.store(
        agent_id="TestAgent",
        step="unit_test",
        content="This is only a test of the memory system.",
    )
    logger.success(
        f"  → Stored memory: id={test_mem.id}, "
        f"agent_id={test_mem.agent_id}, step={test_mem.step}"
    )

    # 3. Load memories back
    logger.info("Loading memories for TestAgent...")
    records = memory.load(agent_id="TestAgent")
    logger.success(f"  → Loaded {len(records)} record(s):")
    for rec in records:
        print(
            f"    • [{rec.id}] {rec.timestamp.isoformat()} "
            f"{rec.agent_id}/{rec.step} → {rec.content}"
        )


if __name__ == "__main__":
    main()
