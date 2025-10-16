import json
import re
from pathlib import Path
from typing import Dict, Any, Union

from langchain_core.tools import Tool, tool

from src.config.logger import logger
from src.config.settings import OverpassSettings
from src.memory.store import MemoryStore
from src.utils.db import query_hash, payload_hash
from src.utils.overpass import build_query, run_query as execute_overpass_query
from src.tools.io_tools import save_overpass_dump


def parse_tool_input(raw_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Robustly parse tool input from various formats that the model might produce.

    :param raw_input: Raw input from the model (could be string or dict)
    :return: Parsed parameters dictionary
    """
    if isinstance(raw_input, dict):
        return raw_input

    if isinstance(raw_input, str):
        # Try direct JSON parsing first
        try:
            return json.loads(raw_input)
        except json.JSONDecodeError:
            pass

        # Handle common model mistakes:
        # 1. Missing quotes around keys
        try:
            # Add quotes around unquoted keys
            fixed = re.sub(r"(\w+):", r'"\1":', raw_input)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 2. Extract JSON from mixed content
        json_match = re.search(r"\{.*\}", raw_input, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 3. Handle simple key:value format
        if ":" in raw_input and "{" not in raw_input:
            try:
                key, value = raw_input.split(":", 1)
                key = key.strip().strip("\"'")
                value = value.strip().strip("\"'")
                return {key: value}
            except ValueError:
                pass

    # Fallback: return empty dict and let the tool handle the error
    logger.warning(f"Could not parse tool input: {raw_input}")
    return {}


# Make tool input more flexible by accepting raw input
@tool("build_overpass_query", return_direct=False)
def build_overpass_query_tool(tool_input: Union[str, Dict[str, Any]]) -> str:
    """
    Build an Overpass QL query to find surveillance cameras in a city.

    This tool constructs a properly formatted Overpass query that searches for
    all man_made=surveillance features within the specified city boundaries.
    It uses Nominatim to resolve the city name to OSM area ID.

    Expected input: {"city": "CityName", "country": "CountryCode"} (country optional)
    :param tool_input: Tool parameters (JSON string or dict)
    :return: Formatted Overpass QL query string
    """
    try:
        # Parse the input robustly
        params = parse_tool_input(tool_input)
        logger.info(f"build_overpass_query_tool received: {tool_input}")
        logger.info(f"Parsed to: {params}")

        city = params.get("city")
        country = params.get("country")

        if not city:
            return 'Error: City parameter is required. Use format: {"city": "CityName"}'

        logger.info(
            f"Building Overpass query for {city}" + (f", {country}" if country else "")
        )
        settings = OverpassSettings()
        query = build_query(city, country=country, settings=settings)
        logger.debug(f"Built query: {query[:100]}...")
        return query
    except Exception as e:
        error_msg = f"Failed to build query: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


@tool("run_overpass_query", return_direct=False)
def run_overpass_query_tool(tool_input: Union[str, Dict[str, Any]]) -> str:
    """
    Execute an Overpass API query and return the results as JSON string.

    This tool submits the query to the Overpass API and returns the parsed
    JSON response containing surveillance camera data as a string.

    Expected input: {"query": "OverpassQL query string"}
    :param tool_input: Tool parameters (JSON string or dict)
    :return: JSON string of Overpass API response
    """
    try:
        # Parse the input robustly
        params = parse_tool_input(tool_input)
        logger.info(f"run_overpass_query_tool received: {tool_input}")
        logger.info(f"Parsed to: {params}")

        query = params.get("query")

        if not query:
            return '{"error": "Query parameter is required. Use format: {"query": "QueryString"}"}'

        logger.info("Executing Overpass query")
        settings = OverpassSettings()
        data = execute_overpass_query(query, settings=settings)
        element_count = len(data.get("elements", []))
        logger.info(f"Query returned {element_count} surveillance elements")

        # Store full data for later saving but return only summary to avoid overwhelming the model
        import tempfile

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, temp_file, indent=2)
        temp_file.close()

        # Return summary instead of full data
        summary = {
            "success": True,
            "elements_count": element_count,
            "temp_file": temp_file.name,
            "message": f"Successfully downloaded {element_count} surveillance cameras. Data stored temporarily.",
        }
        return json.dumps(summary)
    except Exception as e:
        error_msg = f"Failed to execute Overpass query: {str(e)}"
        logger.error(error_msg)
        return f'{{"error": "{error_msg}"}}'


def create_check_cache_tool(memory: MemoryStore) -> Tool:
    """
    Factory function to create a cache checking tool with memory access.

    :param memory: MemoryStore instance for cache lookup
    :return: Configured LangChain Tool for cache checking
    """

    @tool("check_query_cache", return_direct=False)
    def check_query_cache_tool(tool_input: Union[str, Dict[str, Any]]) -> str:
        """
        Check if query results exist in cache and return cached data if valid.

        This tool looks up previous query results in the agent's memory to avoid
        redundant API calls. It verifies cache integrity using hash validation.

        Expected input: {"query": "OverpassQL query", "agent_name": "ScraperAgent"}
        :param tool_input: Tool parameters (JSON string or dict)
        :return: JSON string with cache_hit status and data if found
        """
        try:
            # Parse the input robustly
            params = parse_tool_input(tool_input)
            logger.info(f"check_query_cache_tool received: {tool_input}")
            logger.info(f"Parsed to: {params}")

            query = params.get("query")
            agent_name = params.get("agent_name")

            if not query or not agent_name:
                error_msg = "Both query and agent_name are required. Use format: {'query': 'QueryString', 'agent_name': 'ScraperAgent'}"
                return json.dumps({"error": error_msg})

            logger.info(
                f"Checking cache for query: {query[:50]}... (agent: {agent_name})"
            )

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
                        result = {
                            "cache_hit": True,
                            "data": data,
                            "filepath": str(filepath),
                            "elements_count": element_count,
                        }
                        return json.dumps(result)
                    else:
                        logger.warning(f"Cache integrity check failed for {filepath}")

            logger.info("No valid cache entry found")
            return json.dumps({"cache_hit": False, "data": None})

        except Exception as e:
            error_msg = f"Error checking cache: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"cache_hit": False, "data": None, "error": error_msg})

    return check_query_cache_tool


def create_save_data_tool(memory: MemoryStore) -> Tool:
    """
    Factory function to create a data saving tool with memory access.

    :param memory: MemoryStore instance for cache storage
    :return: Configured LangChain Tool for data saving
    """

    @tool("save_overpass_data", return_direct=False)
    def save_overpass_data_tool(tool_input: Union[str, Dict[str, Any]]) -> str:
        """
        Save Overpass query results to disk and cache the location.

        This tool persists the downloaded surveillance data and creates a cache
        entry for future lookups. It handles empty results appropriately.

        Expected input: {"temp_file": "path", "city": "CityName", "output_dir": "dir", "query": "query", "agent_name": "agent"}
                    OR: {"data_json": "json_string", "city": "CityName", ...}
        :param tool_input: Tool parameters (JSON string or dict)
        :return: JSON string with save status and file path
        """
        try:
            # Parse the input robustly
            params = parse_tool_input(tool_input)
            logger.info(f"save_overpass_data_tool received: {tool_input}")
            logger.info(f"Parsed to: {params}")

            city = params.get("city")
            output_dir = params.get("output_dir")
            query = params.get("query")
            agent_name = params.get("agent_name")

            if not all([city, output_dir, query, agent_name]):
                return json.dumps(
                    {
                        "error": "All parameters (city, output_dir, query, agent_name) are required"
                    }
                )

            logger.info(f"Saving data for {city} to {output_dir}")

            # Handle both temp file reference and direct data
            temp_file_path = params.get("temp_file")
            data_json = params.get("data_json")

            if temp_file_path:
                # Load data from temp file
                import os

                try:
                    with open(temp_file_path, "r") as f:
                        data = json.load(f)
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    logger.info(f"Loaded data from temp file: {temp_file_path}")
                except Exception as e:
                    return json.dumps(
                        {"error": f"Failed to load temp file {temp_file_path}: {e}"}
                    )
            elif data_json:
                # Parse the JSON data directly
                try:
                    data = json.loads(data_json)
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"Invalid JSON data: {e}"})
            else:
                return json.dumps(
                    {"error": "Either temp_file or data_json parameter is required"}
                )

            element_count = len(data.get("elements", []))

            # Handle empty results
            if element_count == 0:
                logger.warning(f"Empty result set for {city} - recording in memory")
                memory.store(
                    agent_name,
                    "empty",
                    f"{city}|{query_hash(query)}",
                )
                result = {
                    "saved": False,
                    "empty": True,
                    "message": f"No surveillance data found for {city}",
                }
                return json.dumps(result)

            # Save non-empty data
            city_key = city.lower().replace(" ", "_")
            output_path = Path(output_dir) / f"{city_key}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            saved_path = save_overpass_dump(data, city, output_path)

            # Cache the result
            q_hash = query_hash(query)
            p_hash = payload_hash(data)
            memory.store(agent_name, "cache", f"{q_hash}|{saved_path}|{p_hash}")

            logger.info(f"Saved {element_count} elements to {saved_path}")
            result = {
                "saved": True,
                "empty": False,
                "filepath": str(saved_path),
                "elements_count": element_count,
            }
            return json.dumps(result)

        except Exception as e:
            error_msg = f"Failed to save data: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

    return save_overpass_data_tool


def create_surveillance_data_collector_tools(memory: MemoryStore) -> list:
    """
    Create all tools needed for the surveillance data collector agent.

    :param memory: MemoryStore instance for caching
    :return: List of configured LangChain tools
    """
    return [
        build_overpass_query_tool,
        run_overpass_query_tool,
        create_check_cache_tool(memory),
        create_save_data_tool(memory),
    ]
