from pathlib import Path
from typing import Dict, Any, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from src.config.logger import logger
from src.config.settings import LangChainSettings
from src.llm.surveillance_llm import create_surveillance_llm
from src.memory.store import MemoryStore
from src.tools.scraper_tools import create_scraper_tools


class SurveillanceScraperAgent:
    """
    LangChain-based agent for scraping surveillance data from OpenStreetMap.

    This agent uses ReAct pattern (Reason + Act) to intelligently collect surveillance
    camera data while optimizing for cache efficiency and handling errors gracefully.

    Features:
    - Reasoning about when to use cache vs API
    - Automatic cache integrity checking
    - Network error handling and retry logic
    - Empty result detection and caching
    - Progress logging and status reporting
    """

    def __init__(
        self,
        name: str,
        memory: MemoryStore,
        settings: Optional[LangChainSettings] = None,
    ):
        """
        Initialize the LangChain scraper agent.

        :param name: Agent name for identification and caching
        :param memory: Memory store for caching query results
        :param settings: Optional LangChain settings for LLM configuration
        """
        self.name = name
        self.memory = memory
        self.settings = settings or LangChainSettings()

        # Create LLM
        self.llm = create_surveillance_llm(self.settings)

        # Create tools with memory access
        self.tools = create_scraper_tools(memory)

        # Load prompt template
        self.prompt = self._load_prompt_template()

        # Create ReAct agent
        self.agent = create_react_agent(
            llm=self.llm.llm,  # Use the underlying Ollama LLM
            tools=self.tools,
            prompt=self.prompt,
        )

        # Create agent executor with error handling
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.settings.agent_verbose,
            max_iterations=self.settings.agent_max_iterations,
            max_execution_time=self.settings.agent_max_execution_time,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        logger.info(f"Initialized {self.name} with {len(self.tools)} tools")

    @staticmethod
    def _load_prompt_template() -> PromptTemplate:
        """
        Load the scraper agent prompt template.

        :return: Configured PromptTemplate for the agent
        """
        prompt_path = Path("src/prompts/scraper_agent_prompt.md")

        if not prompt_path.exists():
            logger.warning(f"Prompt file not found: {prompt_path}, using default")
            template = """You are a surveillance data scraper agent. Use tools to collect data efficiently.
            
                        Available tools: {tools}
                        Tool names: {tool_names}
                        
                        Use this format:
                        Thought: Consider what to do
                        Action: tool_name
                        Action Input: {{"param": "value"}}
                        Observation: result
                        ... (repeat Thought/Action/Observation as needed)
                        Thought: I now know the final answer
                        Final Answer: summary of results
                        
                        Question: {input}
                        {agent_scratchpad}"""
        else:
            # Load from file and wrap in ReAct format
            system_prompt = prompt_path.read_text(encoding="utf-8")
            template = f"""{system_prompt}

                        ## ReAct Format
                        
                        Use the following format to reason and act:
                        
                        Thought: Consider what action to take based on the mission and strategy
                        Action: tool_name
                        Action Input: {{"param": "value"}}
                        Observation: The result from the tool
                        ... (repeat Thought/Action/Observation as needed)
                        Thought: I have completed the task
                        Final Answer: Summary of what was accomplished
                        
                        Available tools: {{tools}}
                        Tool names: {{tool_names}}
                        
                        ## Begin Task
                        
                        Question: {{input}}
                        {{agent_scratchpad}}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        )

    def scrape(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape surveillance data for a city using the LangChain agent.

        :param input_data: Dict with 'city', optional 'country', 'overpass_dir'
        :return: Dict with scraping results including data, cache status, filepath
        """
        city = input_data["city"]
        country = input_data.get("country")
        overpass_dir = input_data.get("overpass_dir", "overpass_data")

        # Prepare directory structure
        city_dir = Path(overpass_dir) / city.lower().replace(" ", "_")
        city_dir.mkdir(parents=True, exist_ok=True)

        # Build input for agent
        country_info = f" in {country}" if country else ""
        agent_input = {
            "input": f"""Collect surveillance camera data for {city}{country_info}.

            City: {city}
            Country: {country if country else "Not specified"}
            Output Directory: {city_dir}
            Agent Name: {self.name}
            
            Follow the decision strategy: build query → check cache → query API (if needed) → save data.
            Be efficient and report your progress clearly."""
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
                "country": country,
                "city_dir": str(city_dir),
                "agent_output": final_answer,
                "success": True,
            }

            # Try to extract key information from steps
            for step in intermediate_steps:
                action, observation = step
                tool_name = action.tool

                if tool_name == "check_query_cache" and isinstance(observation, dict):
                    if observation.get("cache_hit"):
                        response["cache_hit"] = True
                        response["cached_path"] = observation.get("filepath")
                        response["elements_count"] = observation.get(
                            "elements_count", 0
                        )
                        response["data"] = observation.get("data")

                elif tool_name == "save_overpass_data" and isinstance(
                    observation, dict
                ):
                    response["cache_hit"] = False
                    response["empty"] = observation.get("empty", False)
                    if not observation.get("empty"):
                        response["filepath"] = observation.get("filepath")
                        response["elements_count"] = observation.get(
                            "elements_count", 0
                        )

            logger.info(f"Scrape completed for {city}")
            return response

        except Exception as e:
            logger.error(f"Scraping failed for {city}: {str(e)}")
            return {
                "city": city,
                "country": country,
                "city_dir": str(city_dir),
                "success": False,
                "error": str(e),
                "agent_output": f"Error: {str(e)}",
            }

    def achieve_goal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compatibility method with existing Agent interface.

        :param input_data: Dict with scraping parameters
        :return: Dict with scraping results
        """
        return self.scrape(input_data)
