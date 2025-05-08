from __future__ import annotations

from src.config.logger import logger
from typing import Any, Dict

from src.config.settings import OllamaSettings
from src.tools.ollama_client import OllamaClient


class LocalLLM:
    """
    High-level wrapper for local LLM generating text responses.
    """

    def __init__(self, settings: OllamaSettings = OllamaSettings()) -> None:
        """
        Initialize the LocalLLM with Ollama client and settings.

        :param settings: Pydantic settings for connecting to the Ollama server.
        :return: None
        :raise: Exception if initializing the Ollama client fails.
        """
        try:
            self.client: OllamaClient = OllamaClient(settings)
        except Exception as e:
            logger.error(f"Failed to initialize OllamaClient: {e}")
            raise

    def generate_response(self, prompt: str, **kwargs: Any) -> Any:
        """
        Generate a response from the LLM given a prompt.

        :param prompt: The text prompt to send to the LLM.
        :param kwargs: Additional parameters to pass to the OllamaClient.
        :return: The generated response from the LLM.
        :raise: RuntimeError if the LLM request fails or invalid response.
        """
        try:
            logger.debug(f"Sending prompt to LLM: {prompt!r}")
            response: Dict[str, Any] = self.client(prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM request error: {e}")
            raise RuntimeError(f"LLM request error: {e}") from e

        # Extract generated text from response
        text = ""
        if isinstance(response, Dict):
            if "response" in response and isinstance(response["response"], str):
                text = response["response"].strip()
                logger.debug("Extracted 'response' field from Ollama response")
            else:
                logger.warning(f"Unexpected response format: {response}")
        else:
            logger.warning(f"Invalid response type: {type(response)}")

        if not text:
            logger.warning("Received empty response from LLM")
        else:
            logger.debug(f"LLM returned text: {text!r}")

        return text
