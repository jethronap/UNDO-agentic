from pathlib import Path
from typing import Dict, Any, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.config.logger import logger
from src.config.settings import LangChainSettings
from src.llm.surveillance_llm import create_surveillance_llm
from src.memory.store import MemoryStore
from src.tools.surveillance_data_collector_tools import (
    create_surveillance_data_collector_tools,
)


class SurveillanceDataCollector:
    """
    Surveillance data collector agent for gathering camera data from OpenStreetMap.

    This agent collects surveillance camera data using the Overpass API with
    intelligent caching and robust error handling.

    Features:
    - ReAct-based tool execution with reliable parsing
    - Direct parameter passing for tool inputs
    - Intelligent caching to avoid redundant API calls
    - Comprehensive error handling and recovery
    """

    def __init__(
        self,
        name: str,
        memory: MemoryStore,
        settings: Optional[LangChainSettings] = None,
    ):
        """
        Initialize the surveillance data collector agent.

        :param name: Agent name for identification and caching
        :param memory: Memory store for caching query results
        :param settings: Optional LangChain settings for LLM configuration
        """
        self.name = name
        self.memory = memory
        self.settings = settings or LangChainSettings()

        # Create LLM
        self.llm = create_surveillance_llm(self.settings)

        # Create surveillance data collection tools with memory access
        self.tools = create_surveillance_data_collector_tools(memory)

        # Load simplified prompt template
        self.prompt = self._load_prompt_template()

        # Create ReAct agent with simplified setup
        self.agent = create_react_agent(
            llm=self.llm.llm,  # Use the underlying Ollama LLM
            tools=self.tools,
            prompt=self.prompt,
        )

        # Create agent executor with tight error handling
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.settings.agent_verbose,
            max_iterations=self.settings.agent_max_iterations,
            max_execution_time=self.settings.agent_max_execution_time,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        logger.info(f"Initialized {self.name} with {len(self.tools)} simplified tools")

    @staticmethod
    def _load_prompt_template() -> PromptTemplate:
        """
        Load the simplified scraper agent prompt template.

        :return: Configured PromptTemplate for the agent
        """
        # Use tested hardcoded template (file loading causes ReAct parsing issues)
        template = """You are a surveillance data collector. Collect camera data from cities efficiently.

Available tools: {tools}
Tool names: {tool_names}

FORMAT: Use this EXACT format:

Thought: [your reasoning]
Action: [exact tool name]
Action Input: {{"param": "value"}}
Observation: [result appears automatically]

Repeat Thought/Action/Action Input/Observation until complete, then:

Thought: I have completed the task
Final Answer: [brief 1-sentence summary with element count and filepath]

WORKFLOW:
1. Build query → 2. Check cache
   → If cache HIT: STOP. Task complete (data already saved at filepath)
   → If cache MISS: 3. Download → 4. Save

RULES:
- Use exact JSON format: {{"param": "value"}}
- CRITICAL: If cache hits (cache_hit: true), STOP immediately. Do NOT call save_overpass_data.
- Cache hit means data is already saved - filepath is in the cache response
- Only download and save if cache miss (cache_hit: false)
- Keep Final Answer brief (one sentence)

Question: {input}
{agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        )

    def scrape(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape surveillance data for a city using the simplified agent.

        :param input_data: Dict with 'city', optional 'country', 'overpass_dir'
        :return: Dict with scraping results including data, cache status, filepath
        """
        city = input_data["city"]
        country = input_data.get("country")
        overpass_dir = input_data.get("overpass_dir", "overpass_data")

        # Prepare directory structure
        city_dir = Path(overpass_dir) / city.lower().replace(" ", "_")
        city_dir.mkdir(parents=True, exist_ok=True)

        # Build simple input for agent
        country_info = f" in {country}" if country else ""
        agent_input = {
            "input": f"Collect surveillance camera data for {city}{country_info}. Save to directory: {city_dir}"
        }

        try:
            logger.info(f"Starting scrape for {city}{country_info}")

            # Execute agent
            result = self.executor.invoke(agent_input)

            # Extract results
            final_answer = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Parse results from intermediate steps to build response
            response = {
                "city": city,
                "city_dir": str(city_dir),
                "agent_output": final_answer,
                "success": True,
            }

            # Try to extract key information from steps
            for step in intermediate_steps:
                action, observation = step
                tool_name = action.tool

                if tool_name == "check_query_cache":
                    try:
                        import json

                        cache_result = (
                            json.loads(observation)
                            if isinstance(observation, str)
                            else observation
                        )
                        if cache_result.get("cache_hit"):
                            response["cache_hit"] = True
                            response["cached_path"] = cache_result.get("filepath")
                            response["elements_count"] = cache_result.get(
                                "elements_count", 0
                            )
                            if "data" in cache_result:
                                response["data"] = cache_result["data"]
                    except (json.JSONDecodeError, AttributeError):
                        logger.warning(f"Could not parse cache result: {observation}")

                elif tool_name == "save_overpass_data":
                    try:
                        import json

                        save_result = (
                            json.loads(observation)
                            if isinstance(observation, str)
                            else observation
                        )
                        response["cache_hit"] = False
                        response["empty"] = save_result.get("empty", False)
                        if not save_result.get("empty"):
                            response["filepath"] = save_result.get("filepath")
                            response["elements_count"] = save_result.get(
                                "elements_count", 0
                            )
                    except (json.JSONDecodeError, AttributeError):
                        logger.warning(f"Could not parse save result: {observation}")

            logger.info(f"Scrape completed for {city}")
            return response

        except Exception as e:
            error_msg = str(e)

            # Detect connection errors to LLM service
            if "Connection refused" in error_msg or "ConnectionError" in str(type(e)):
                logger.error(
                    f"Scraping failed for {city}: Cannot connect to LLM service (Ollama). "
                    f"Error: {error_msg}. "
                    f"Please ensure Ollama is running and accessible."
                )
            else:
                logger.error(f"Scraping failed for {city}: {error_msg}")

            return {
                "city": city,
                "country": country,
                "city_dir": str(city_dir),
                "success": False,
                "error": error_msg,
                "agent_output": f"Error: {error_msg}",
            }

    def achieve_goal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compatibility method with existing Agent interface.

        :param input_data: Dict with scraping parameters
        :return: Dict with scraping results
        """
        return self.scrape(input_data)
