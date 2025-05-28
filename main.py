from src.agents.analyzer_agent import AnalyzerAgent

from src.config.settings import DatabaseSettings
from src.memory.store import MemoryStore


# from src.utils.overpass import nominatim_relation_id


def main():
    """
    Quick test of the SQLModel-based MemoryStore:
     1. Stores a test memory.
     2. Loads and prints all memories for the test agent.
    """
    # 1. Load DB settings and init store
    db_settings = DatabaseSettings()
    memory = MemoryStore(db_settings)

    # # 2. Store a test memory
    # logger.info("Storing a test memory...")
    # test_mem = memory.store(
    #     agent_id="TestAgent",
    #     step="unit_test",
    #     content="This is only a test of the memory system.",
    # )
    # logger.success(
    #     f" Stored memory: id={test_mem.id}, "
    #     f"agent_id={test_mem.agent_id}, step={test_mem.step}"
    # )
    #
    # # 3. Load memories back
    # logger.info("Loading memories for TestAgent...")
    # records = memory.load(agent_id="TestAgent")
    # logger.success(f": Loaded {len(records)} record(s):")
    # for rec in records:
    #     print(
    #         f"    • [{rec.id}] {rec.timestamp.isoformat()} "
    #         f"{rec.agent_id}/{rec.step}: {rec.content}"
    #     )
    # agent = ScraperAgent(name="ScraperAgent", memory=memory)
    # result_context = agent.achieve_goal({"city": "Hamburg"})
    # logger.success(f"JSON saved to: {result_context['save_json']}")
    # print(nominatim_city("Athens", country="GR"))
    # print(nominatim_city("Athens", country="US"))
    # ctx1 = agent.achieve_goal({"city": "Berlin"})
    # logger.debug(ctx1["cache_hit"])  # False

    # 2nd run — served entirely from cache
    # ctx2 = agent.achieve_goal({"city": "Berlin"})
    # logger.debug(ctx2["cache_hit"])  # True
    # logger.debug(ctx2["run_query"] == ctx1["run_query"])  # True

    # # First run -download + save
    # ctx = agent.achieve_goal({"city": "Lund"})
    # logger.debug(ctx["elements_count"])  # 487
    #
    # # Second run — cached
    # ctx2 = agent.achieve_goal({"city": "Lund"})
    # logger.debug(ctx2["cache_hit"])  # True
    # logger.debug(ctx2["save_json"])  # existing file path
    #
    # # City with no matches
    # agent.achieve_goal({"city": "Smallville"})
    # # prints: WARNING: 0 surveillance objects found for Smallville
    analyzer = AnalyzerAgent("AnalyzerAgent", memory)

    ctx = analyzer.achieve_goal(
        {
            "path": "overpass_data/malmö/malmö.json",
            "generate_geojson": True,
            "generate_heatmap": True,
            "generate_chart": True,
            "plot_zone_sensitivity": True,
            "plot_sensitivity_reasons": True,
        }
    )
    print("Context: ", ctx)
    print("Enriched file:", ctx["output_path"])
    print("GeoJSON:", ctx["geojson_path"])
    # print("Heatmap:", ctx["heatmap_path"])
    print("Summary stats:", ctx["stats"])
    print("Chart:", ctx["chart_path"])
    print("Zone‐sensitivity chart at", ctx["chart_zone_sens"])
    print("Sensitivity reasons:", ctx["sensitivity_reasons_chart"])


if __name__ == "__main__":
    main()
