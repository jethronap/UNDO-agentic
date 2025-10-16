from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

from src.agents.surveillance_data_collector import SurveillanceDataCollector
from src.agents.langchain_analyzer import SurveillanceAnalyzerAgent
from src.config.logger import logger
from src.config.pipeline_config import PipelineConfig, AnalysisScenario
from src.config.settings import DatabaseSettings, LangChainSettings
from src.memory.store import MemoryStore


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SCRAPING = "scraping"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Completed with some errors


class SurveillancePipeline:
    """
    Multi-agent pipeline for end-to-end surveillance data analysis.

    Orchestrates:
    1. SurveillanceScraperAgent - Downloads data from OpenStreetMap
    2. SurveillanceAnalyzerAgent - Enriches and visualizes data

    Features:
    - Configurable analysis scenarios
    - Progress tracking and status reporting
    - Error recovery between agents
    - Shared memory for caching
    - Optional LangSmith observability
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        langchain_settings: Optional[LangChainSettings] = None,
    ):
        """
        Initialize the surveillance pipeline.

        :param config: Pipeline configuration (uses default if None)
        :param langchain_settings: LangChain settings for agents
        """
        self.config = config or PipelineConfig()
        self.settings = langchain_settings or LangChainSettings()

        # Create shared memory for both agents
        db_settings = DatabaseSettings()
        self.memory = MemoryStore(settings=db_settings)

        # Initialize agents
        self.scraper = SurveillanceDataCollector(
            name="ScraperAgent",
            memory=self.memory,
            settings=self.settings,
        )

        self.analyzer = SurveillanceAnalyzerAgent(
            name="AnalyzerAgent",
            memory=self.memory,
            settings=self.settings,
        )

        # Pipeline state
        self.status = PipelineStatus.PENDING
        self.current_step = None
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.errors = []

        logger.info(
            f"Initialized SurveillancePipeline with {self.config.scenario.value} scenario"
        )

    def run(self, city: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete pipeline for a city.

        :param city: City name to analyze
        :param kwargs: Additional arguments (country, output_dir override)
        :return: Dictionary with complete pipeline results
        """
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now()
        self.current_step = "initialization"

        logger.info(f"Starting pipeline for {city}")

        # Extract parameters
        country = kwargs.get("country", self.config.country_code)
        output_dir = kwargs.get("output_dir", self.config.output_dir)

        results = {
            "city": city,
            "country": country,
            "scenario": self.config.scenario.value,
            "start_time": self.start_time.isoformat(),
        }

        try:
            # Step 1: Scraping
            if self.config.scrape_enabled:
                scrape_result = self._run_scraper(city, country, output_dir)
                results["scrape"] = scrape_result

                if not scrape_result.get("success"):
                    if self.config.stop_on_error:
                        return self._finalize_results(results, PipelineStatus.FAILED)
                    else:
                        self.errors.append(
                            f"Scraping failed: {scrape_result.get('error')}"
                        )
                        return self._finalize_results(results, PipelineStatus.FAILED)

                # Get scraped data path for analysis
                data_path = scrape_result.get("filepath") or scrape_result.get(
                    "cached_path"
                )
                if not data_path:
                    error_msg = "No data path found from scraper"
                    logger.error(error_msg)
                    results["error"] = error_msg
                    return self._finalize_results(results, PipelineStatus.FAILED)
            else:
                # User provides data path directly
                data_path = kwargs.get("data_path")
                if not data_path:
                    error_msg = "Scraping disabled but no data_path provided"
                    logger.error(error_msg)
                    results["error"] = error_msg
                    return self._finalize_results(results, PipelineStatus.FAILED)
                results["scrape"] = {"skipped": True, "reason": "scraping disabled"}

            # Step 2: Analysis
            if self.config.analyze_enabled:
                analyze_result = self._run_analyzer(data_path)
                results["analyze"] = analyze_result

                if not analyze_result.get("success"):
                    if self.config.stop_on_error:
                        return self._finalize_results(results, PipelineStatus.FAILED)
                    else:
                        self.errors.append(
                            f"Analysis failed: {analyze_result.get('error')}"
                        )
                        return self._finalize_results(results, PipelineStatus.PARTIAL)

                # Check for visualization errors (partial success)
                if analyze_result.get("visualization_errors"):
                    self.errors.extend(analyze_result["visualization_errors"])
                    return self._finalize_results(results, PipelineStatus.PARTIAL)
            else:
                results["analyze"] = {"skipped": True, "reason": "analysis disabled"}

            # Success!
            return self._finalize_results(results, PipelineStatus.COMPLETED)

        except Exception as e:
            logger.error(f"Pipeline failed with exception: {e}")
            results["error"] = str(e)
            return self._finalize_results(results, PipelineStatus.FAILED)

    def _run_scraper(
        self,
        city: str,
        country: Optional[str],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Execute the scraper agent.

        :param city: City name
        :param country: Optional country code
        :param output_dir: Output directory
        :return: Scraper results
        """
        self.status = PipelineStatus.SCRAPING
        self.current_step = "scraping"

        logger.info(f"Scraping data for {city}")

        scrape_input = {
            "city": city,
            "overpass_dir": output_dir,
        }
        if country:
            scrape_input["country"] = country

        try:
            result = self.scraper.scrape(scrape_input)

            if result.get("success"):
                logger.info(
                    f"Scraping completed: {result.get('elements_count', 0)} elements"
                )
            else:
                logger.error(f"Scraping failed: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Scraper exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "city": city,
            }

    def _run_analyzer(self, data_path: str) -> Dict[str, Any]:
        """
        Execute the analyzer agent.

        :param data_path: Path to scraped data
        :return: Analyzer results
        """
        self.status = PipelineStatus.ANALYZING
        self.current_step = "analyzing"

        logger.info(f"Analyzing data from {data_path}")

        analyze_input = {
            "path": data_path,
            **self.config.to_analyzer_options(),
        }

        try:
            result = self.analyzer.analyze(analyze_input)

            if result.get("success"):
                logger.info(
                    f"Analysis completed: {result.get('element_count', 0)} elements enriched"
                )
            else:
                logger.error(f"Analysis failed: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Analyzer exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "path": data_path,
            }

    def _finalize_results(
        self,
        results: Dict[str, Any],
        status: PipelineStatus,
    ) -> Dict[str, Any]:
        """
        Finalize pipeline execution and add metadata.

        :param results: Current results dictionary
        :param status: Final pipeline status
        :return: Complete results with metadata
        """
        self.status = status
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        results.update(
            {
                "status": status.value,
                "end_time": self.end_time.isoformat(),
                "duration_seconds": duration,
                "success": status == PipelineStatus.COMPLETED,
                "partial_success": status == PipelineStatus.PARTIAL,
            }
        )

        if self.errors:
            results["errors"] = self.errors

        logger.info(f"Pipeline completed with status: {status.value} ({duration:.2f}s)")
        return results

    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.

        :return: Status dictionary
        """
        return {
            "status": self.status.value,
            "current_step": self.current_step,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "errors": self.errors,
        }


def create_pipeline(
    scenario: AnalysisScenario = AnalysisScenario.BASIC,
    **config_kwargs,
) -> SurveillancePipeline:
    """
    Factory function for creating configured pipelines.

    :param scenario: Analysis scenario to use
    :param config_kwargs: Additional configuration overrides
    :return: Configured SurveillancePipeline instance
    """
    config = PipelineConfig.from_scenario(scenario)

    # Apply any overrides
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    pipeline = SurveillancePipeline(config=config)
    logger.info(f"Created pipeline with {scenario.value} scenario")
    return pipeline
