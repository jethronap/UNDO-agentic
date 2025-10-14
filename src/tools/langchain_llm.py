from typing import Any, Optional

from langchain_community.llms import Ollama

from src.config.logger import logger
from src.config.settings import LangChainSettings


class LangChainLLM:
    """
    LangChain-based wrapper for local LLM generating text responses.

    This replaces the custom LocalLLM implementation with LangChain's Ollama integration
    while maintaining the same interface for backward compatibility.
    """

    def __init__(self, settings: Optional[LangChainSettings] = None) -> None:
        """
        Initialize the LangChainLLM with LangChain's Ollama client.

        :param settings: Pydantic settings for connecting to the Ollama server.
        :return: None
        :raise: Exception if initializing the Ollama client fails.
        """
        try:
            self.settings = settings or LangChainSettings()
            self.llm = Ollama(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=self.settings.ollama_temperature,
                timeout=self.settings.ollama_timeout,
            )
            logger.debug(
                f"Initialized LangChain Ollama client with model: {self.settings.ollama_model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Ollama client: {e}")
            raise

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response from the LLM given a prompt.

        :param prompt: The text prompt to send to the LLM.
        :param kwargs: Additional parameters to pass to the LangChain LLM.
        :return: The generated response text from the LLM.
        :raise: RuntimeError if the LLM request fails or invalid response.
        """
        try:
            logger.debug(f"Sending prompt to LangChain LLM: {prompt!r}")

            # Handle additional kwargs if provided
            if kwargs:
                # Create a temporary instance with modified settings for this call
                temp_llm = Ollama(
                    base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    temperature=kwargs.get(
                        "temperature", self.settings.ollama_temperature
                    ),
                    timeout=kwargs.get("timeout", self.settings.ollama_timeout),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["temperature", "timeout"]
                    },
                )
                response = temp_llm.invoke(prompt)
            else:
                response = self.llm.invoke(prompt)

        except Exception as e:
            logger.error(f"LangChain LLM request error: {e}")
            raise RuntimeError(f"LangChain LLM request error: {e}") from e

        # LangChain's Ollama returns the response as a string directly
        if isinstance(response, str):
            text = response.strip()
            logger.debug("Received text response from LangChain LLM")
        else:
            logger.warning(
                f"Unexpected response type from LangChain LLM: {type(response)}"
            )
            text = str(response).strip() if response else ""

        if not text:
            logger.warning("Received empty response from LangChain LLM")
        else:
            logger.debug(f"LangChain LLM returned text: {text!r}")

        return text

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """
        Generate responses for multiple prompts in batch.

        :param prompts: List of text prompts to send to the LLM.
        :param kwargs: Additional parameters to pass to the LangChain LLM.
        :return: List of generated response texts from the LLM.
        :raise: RuntimeError if the LLM batch request fails.
        """
        try:
            logger.debug(f"Sending {len(prompts)} prompts to LangChain LLM in batch")

            if kwargs:
                temp_llm = Ollama(
                    base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    temperature=kwargs.get(
                        "temperature", self.settings.ollama_temperature
                    ),
                    timeout=kwargs.get("timeout", self.settings.ollama_timeout),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["temperature", "timeout"]
                    },
                )
                responses = temp_llm.batch(prompts)
            else:
                responses = self.llm.batch(prompts)

        except Exception as e:
            logger.error(f"LangChain LLM batch request error: {e}")
            raise RuntimeError(f"LangChain LLM batch request error: {e}") from e

        # Process batch responses
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, str):
                text = response.strip()
            else:
                logger.warning(
                    f"Unexpected response type for prompt {i}: {type(response)}"
                )
                text = str(response).strip() if response else ""
            results.append(text)

        logger.debug(f"LangChain LLM returned {len(results)} batch responses")
        return results
