import requests
from typing import Any, Dict
from loguru import logger

from src.config.settings import OllamaSettings


class OllamaClient:
    """
    Client for sending prompts to an Ollama server and retrieving responses.
    """

    def __init__(self, settings: OllamaSettings) -> None:
        """
        Initialize Ollama client.
        :param settings: The Ollama settings.
        """
        self.base_url = settings.base_url
        self.model = settings.model
        self.timeout = settings.timeout_seconds
        self.stream = settings.stream

    def __call__(self, prompt, **kwargs: Any) -> Dict[str, Any]:
        """
        Send a prompt to Ollama and return the JSON response.

        :param prompt: The text prompt to send.
        :raises RuntimeError: on HTTP or parse errors.
        :return: Parsed JSON from the Ollama server.
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": self.model,
            "stream": self.stream,
        }
        try:
            logger.debug(f"Sending prompt to Ollama: {prompt!r}")
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Ollama request failed:{e}")
            raise RuntimeError(f"Ollama request error: {e}") from e

        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Invalid JSON from Ollama: {e}")
            raise RuntimeError("Failed to parse Ollama response as JSON") from e

        logger.debug(f"Received response from Ollama: {data}")
        return data
