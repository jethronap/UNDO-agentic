import json
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_core.tools import Tool, tool

from src.config.logger import logger
from src.config.models.tools import (
    BuildQueryInput,
    RunQueryInput,
    CheckCacheInput,
    SaveDataInput,
)
from src.config.settings import OverpassSettings
from src.memory.store import MemoryStore
from src.utils.db import query_hash, payload_hash
from src.utils.overpass import build_query, run_query as execute_overpass_query
from src.tools.io_tools import save_overpass_dump


@tool("build_overpass_query", args_schema=BuildQueryInput, return_direct=False)
def build_overpass_query_tool(city: str, country: Optional[str] = None) -> str:
    """
    Build an Overpass QL query to find surveillance cameras in a city.

    This tool constructs a properly formatted Overpass query that searches for
    all man_made=surveillance features within the specified city boundaries.
    It uses Nominatim to resolve the city name to OSM area ID.

    :param city: Name of the city to query
    :param country: Optional 2-letter ISO country code to disambiguate city
    :return: Formatted Overpass QL query string
    """
    try:
        logger.info(
            f"Building Overpass query for {city}" + (f", {country}" if country else "")
        )
        settings = OverpassSettings()
        query = build_query(city, country=country, settings=settings)
        logger.debug(f"Built query: {query[:100]}...")
        return query
    except Exception as e:
        error_msg = f"Failed to build query for {city}: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


@tool("run_overpass_query", args_schema=RunQueryInput, return_direct=False)
def run_overpass_query_tool(query: str) -> Dict[str, Any]:
    """
    Execute an Overpass API query and return the results.

    This tool submits the query to the Overpass API and returns the parsed
    JSON response containing surveillance camera data. It includes automatic
    retry logic for transient failures.

    :param query: The Overpass QL query string to execute
    :return: Dictionary containing Overpass API response with 'elements' list
    """
    try:
        logger.info("Executing Overpass query")
        settings = OverpassSettings()
        data = execute_overpass_query(query, settings=settings)
        element_count = len(data.get("elements", []))
        logger.info(f"Query returned {element_count} surveillance elements")
        return data
    except Exception as e:
        error_msg = f"Failed to execute Overpass query: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_check_cache_tool(memory: MemoryStore) -> Tool:
    """
    Factory function to create a cache checking tool with memory access.

    :param memory: MemoryStore instance for cache lookup
    :return: Configured LangChain Tool for cache checking
    """

    @tool("check_query_cache", args_schema=CheckCacheInput, return_direct=False)
    def check_query_cache_tool(query: str, agent_name: str) -> Dict[str, Any]:
        """
        Check if query results exist in cache and return cached data if valid.

        This tool looks up previous query results in the agent's memory to avoid
        redundant API calls. It verifies cache integrity using hash validation.

        :param query: The Overpass query to check in cache
        :param agent_name: Name of the agent to check cache for
        :return: Dictionary with cache_hit status and data if found
        """
        try:
            q_hash = query_hash(query)
            logger.info(f"Checking cache for query hash: {q_hash[:16]}...")

            # Look for cache entry in memory
            for mem in memory.load(agent_name):
                if mem.step == "cache" and mem.content.startswith(q_hash):
                    _, filepath_str, p_hash = mem.content.split("|")
                    filepath = Path(filepath_str)

                    if not filepath.exists():
                        logger.warning(f"Cache file missing: {filepath}")
                        continue

                    # Load and verify cached data
                    with filepath.open(encoding="utf-8") as f:
                        data = json.load(f)

                    # Verify integrity
                    if payload_hash(data) == p_hash:
                        element_count = len(data.get("elements", []))
                        logger.info(
                            f"Cache hit! Loaded {element_count} elements from {filepath}"
                        )
                        return {
                            "cache_hit": True,
                            "data": data,
                            "filepath": str(filepath),
                            "elements_count": element_count,
                        }
                    else:
                        logger.warning(f"Cache integrity check failed for {filepath}")

            logger.info("No valid cache entry found")
            return {"cache_hit": False, "data": None}

        except Exception as e:
            error_msg = f"Error checking cache: {str(e)}"
            logger.error(error_msg)
            return {"cache_hit": False, "data": None, "error": error_msg}

    return check_query_cache_tool


def create_save_data_tool(memory: MemoryStore) -> Tool:
    """
    Factory function to create a data saving tool with memory access.

    :param memory: MemoryStore instance for cache storage
    :return: Configured LangChain Tool for data saving
    """

    @tool("save_overpass_data", args_schema=SaveDataInput, return_direct=False)
    def save_overpass_data_tool(
        data: str,
        city: str,
        output_dir: str,
        query: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        Save Overpass query results to disk and cache the location.

        This tool persists the downloaded surveillance data and creates a cache
        entry for future lookups. It handles empty results appropriately.

        :param data: JSON string of Overpass API response data to save
        :param city: The city name for the data
        :param output_dir: Directory path to save the data
        :param query: The original query for cache tracking
        :param agent_name: Name of the agent saving data
        :return: Dictionary with save status and file path
        """
        try:
            # Parse JSON string if provided as string
            if isinstance(data, str):
                data_dict = json.loads(data)
            else:
                data_dict = data

            element_count = len(data_dict.get("elements", []))

            # Handle empty results
            if element_count == 0:
                logger.warning(f"Empty result set for {city} - recording in memory")
                memory.store(
                    agent_name,
                    "empty",
                    f"{city}|{query_hash(query)}",
                )
                return {
                    "saved": False,
                    "empty": True,
                    "message": f"No surveillance data found for {city}",
                }

            # Save non-empty data
            city_key = city.lower().replace(" ", "_")
            output_path = Path(output_dir) / f"{city_key}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            saved_path = save_overpass_dump(data_dict, city, output_path)

            # Cache the result
            q_hash = query_hash(query)
            p_hash = payload_hash(data_dict)
            memory.store(agent_name, "cache", f"{q_hash}|{saved_path}|{p_hash}")

            logger.info(f"Saved {element_count} elements to {saved_path}")
            return {
                "saved": True,
                "empty": False,
                "filepath": str(saved_path),
                "elements_count": element_count,
            }

        except Exception as e:
            error_msg = f"Failed to save data: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    return save_overpass_data_tool


def create_scraper_tools(memory: MemoryStore) -> list:
    """
    Create all tools needed for the surveillance scraper agent.

    :param memory: MemoryStore instance for caching
    :return: List of configured LangChain Tools
    """
    return [
        build_overpass_query_tool,
        run_overpass_query_tool,
        create_check_cache_tool(memory),
        create_save_data_tool(memory),
    ]
