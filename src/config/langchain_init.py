"""
LangChain initialization helper module.

This module provides factory functions and utilities for initializing
LangChain components with proper configuration from settings.
"""

import os
from typing import Optional

from src.config.settings import LangChainSettings
from src.config.logger import logger


def setup_langchain_environment(settings: Optional[LangChainSettings] = None) -> None:
    """
    Set up environment variables required by LangChain.

    This function configures the environment for LangChain operations,
    including LangSmith tracing if enabled.

    :param settings: LangChain settings. If None, creates default settings.
    """
    if settings is None:
        settings = LangChainSettings()

    # Configure LangSmith tracing (commented out until Issue #1 adds langsmith dependency)
    # if settings.tracing_enabled and settings.api_key:
    #     os.environ["LANGCHAIN_TRACING_V2"] = "true"
    #     os.environ["LANGCHAIN_API_KEY"] = settings.api_key
    #     os.environ["LANGCHAIN_ENDPOINT"] = settings.endpoint
    #     os.environ["LANGCHAIN_PROJECT"] = settings.project_name
    #     logger.info(f"LangSmith tracing enabled for project: {settings.project_name}")
    # else:
    #     os.environ["LANGCHAIN_TRACING_V2"] = "false"
    #     logger.debug("LangSmith tracing disabled")

    # Set Ollama configuration
    os.environ["OLLAMA_BASE_URL"] = settings.ollama_base_url
    logger.debug(f"Ollama base URL set to: {settings.ollama_base_url}")


def create_ollama_llm(settings: Optional[LangChainSettings] = None):
    """
    Create and configure an Ollama LLM instance for LangChain.

    This function will be fully implemented in Issue #5 when we add
    the langchain-community dependency and replace custom LLM wrapper.

    :param settings: LangChain settings. If None, creates default settings.
    :return: Configured Ollama LLM instance (placeholder for now)
    :raises ImportError: If langchain-community is not installed
    """
    if settings is None:
        settings = LangChainSettings()

    # Placeholder implementation until Issue #5

    # try:
    #     from langchain_community.llms import Ollama
    #
    #     llm = Ollama(
    #         model=settings.ollama_model,
    #         base_url=settings.ollama_base_url,
    #         temperature=settings.ollama_temperature,
    #         timeout=settings.ollama_timeout,
    #     )
    #     logger.info(f"Created Ollama LLM with model: {settings.ollama_model}")
    #     return llm
    # except ImportError as e:
    #     logger.error("langchain-community not installed. Run: uv add langchain-community")
    #     raise ImportError(
    #         "langchain-community required for Ollama integration. "
    #         "Install with: uv add langchain-community"
    #     ) from e

    logger.warning(
        "create_ollama_llm is a placeholder. Will be implemented in Issue #5"
    )
    return None


def get_langchain_config(settings: Optional[LangChainSettings] = None) -> dict:
    """
    Get LangChain configuration as a dictionary.

    Useful for passing configuration to agents and chains.

    :param settings: LangChain settings. If None, creates default settings.
    :return: Dictionary with LangChain configuration
    """
    if settings is None:
        settings = LangChainSettings()

    return {
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
        "ollama_timeout": settings.ollama_timeout,
        "ollama_temperature": settings.ollama_temperature,
        "agent_max_iterations": settings.agent_max_iterations,
        "agent_max_execution_time": settings.agent_max_execution_time,
        "agent_verbose": settings.agent_verbose,
        "memory_enabled": settings.memory_enabled,
        "memory_max_tokens": settings.memory_max_tokens,
        "tool_timeout": settings.tool_timeout,
    }


def validate_langchain_setup() -> bool:
    """
    Validate that LangChain environment is properly configured.

    Checks for required settings and dependencies.

    :return: True if setup is valid, False otherwise
    """
    try:
        settings = LangChainSettings()
        logger.debug("LangChain settings loaded successfully")

        # Check if Ollama base URL is accessible (basic validation)
        if not settings.ollama_base_url.startswith(("http://", "https://")):
            logger.error(f"Invalid Ollama base URL: {settings.ollama_base_url}")
            return False

        logger.info("LangChain setup validation passed")
        return True

    except Exception as e:
        logger.error(f"LangChain setup validation failed: {e}")
        return False


# Convenience function for quick initialization
def init_langchain(settings: Optional[LangChainSettings] = None) -> dict:
    """
    Initialize LangChain with all necessary setup.

    This is a convenience function that:
    1. Sets up environment variables
    2. Validates configuration
    3. Returns configuration dictionary

    :param settings: LangChain settings. If None, creates default settings.
    :return: Configuration dictionary
    :raises RuntimeError: If validation fails
    """
    if settings is None:
        settings = LangChainSettings()

    logger.info("Initializing LangChain...")

    # Set up environment
    setup_langchain_environment(settings)

    # Validate setup
    if not validate_langchain_setup():
        raise RuntimeError("LangChain setup validation failed")

    # Get config
    config = get_langchain_config(settings)

    logger.success("LangChain initialized successfully")
    return config
