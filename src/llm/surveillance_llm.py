import json
from typing import Any, Dict, Optional

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.logger import logger
from src.config.models.surveillance_metadata import SurveillanceMetadata
from src.config.settings import LangChainSettings
from src.prompts.prompt_template import PROMPT_v1


class SurveillanceLLM:
    """
    Enhanced LangChain-based LLM for surveillance data analysis.

    Features:
    - LangChain PromptTemplate integration
    - Pydantic output parsing with validation
    - Built-in retry logic with exponential backoff
    - Factory pattern for consistent initialization
    - Support for structured JSON outputs
    """

    def __init__(self, settings: Optional[LangChainSettings] = None) -> None:
        """
        Initialize the SurveillanceLLM with enhanced LangChain features.

        :param settings: Pydantic settings for LangChain configuration.
        :return: None
        :raise: Exception if initialization fails.
        """
        try:
            self.settings = settings or LangChainSettings()

            # Initialize LangChain LLM with new package
            self.llm = OllamaLLM(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=self.settings.ollama_temperature,
                # timeout=self.settings.ollama_timeout,
            )

            # Initialize prompt template and output parser (lazy)
            self.prompt_template = None
            self.output_parser = None
            self.chain = None

            logger.debug(
                f"Initialized SurveillanceLLM with model: {self.settings.ollama_model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize SurveillanceLLM: {e}")
            raise

    def _ensure_chain_initialized(self) -> None:
        """Lazily initialize the chain if not already created."""
        if self.chain is None:
            self.prompt_template = self._create_prompt_template()
            self.output_parser = PydanticOutputParser(
                pydantic_object=SurveillanceMetadata
            )
            self.chain = self.prompt_template | self.llm | self.output_parser
            logger.debug("Initialized LangChain prompt/parser chain")

    @staticmethod
    def _create_prompt_template() -> PromptTemplate:
        """Create LangChain PromptTemplate for surveillance analysis."""
        template = PROMPT_v1

        return PromptTemplate(
            input_variables=["tags", "format_instructions"],
            template=template,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def analyze_surveillance_element(
        self, element: Dict[str, Any]
    ) -> SurveillanceMetadata:
        """
        Analyze a surveillance element using LangChain with retry logic.

        :param element: OSM element dictionary with tags.
        :return: Parsed and validated SurveillanceMetadata.
        :raise: Exception if analysis fails after retries.
        """
        try:
            # Ensure chain is initialized
            self._ensure_chain_initialized()

            tags = element.get("tags", {})
            tags_json = json.dumps(tags, ensure_ascii=False, indent=2)

            logger.debug(f"Analyzing surveillance element with tags: {tags}")

            # Get format instructions from parser
            format_instructions = self.output_parser.get_format_instructions()

            # Use the chain to process the input
            result = self.chain.invoke(
                {"tags": tags_json, "format_instructions": format_instructions}
            )

            # Create the complete metadata object
            metadata = SurveillanceMetadata.from_raw(element, result.model_dump())

            logger.debug(
                f"Successfully analyzed element {element.get('id', 'unknown')}"
            )
            return metadata

        except ValidationError as e:
            logger.warning(
                f"Validation error for element {element.get('id', 'unknown')}: {e}"
            )
            # Fallback: create metadata with validation errors
            return SurveillanceMetadata.from_raw(element, {"schema_errors": str(e)})
        except Exception as e:
            logger.error(
                f"Failed to analyze element {element.get('id', 'unknown')}: {e}"
            )
            raise

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response from the LLM (backward compatibility method).

        :param prompt: The text prompt to send to the LLM.
        :param kwargs: Additional parameters.
        :return: The generated response text from the LLM.
        :raise: RuntimeError if the LLM request fails.
        """
        try:
            logger.debug(f"Generating response for prompt: {prompt!r}")

            # Handle kwargs by creating temporary LLM if needed
            if kwargs:
                temp_llm = OllamaLLM(
                    base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    temperature=kwargs.get(
                        "temperature", self.settings.ollama_temperature
                    ),
                    # timeout=kwargs.get("timeout", self.settings.ollama_timeout),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["temperature", "timeout"]
                    },
                )
                response = temp_llm.invoke(prompt)
            else:
                response = self.llm.invoke(prompt)

            logger.debug("Successfully generated LLM response")
            return str(response).strip()

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise RuntimeError(f"LLM generation error: {e}") from e

    def generate_batch(self, prompts: list, **kwargs: Any) -> list:
        """
        Generate responses for multiple prompts in batch.

        :param prompts: List of text prompts to send to the LLM.
        :param kwargs: Additional parameters.
        :return: List of generated response texts.
        :raise: RuntimeError if batch generation fails.
        """
        try:
            logger.debug(f"Generating batch responses for {len(prompts)} prompts")

            if kwargs:
                temp_llm = OllamaLLM(
                    base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    temperature=kwargs.get(
                        "temperature", self.settings.ollama_temperature
                    ),
                    # timeout=kwargs.get("timeout", self.settings.ollama_timeout),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["temperature", "timeout"]
                    },
                )
                responses = temp_llm.batch(prompts)
            else:
                responses = self.llm.batch(prompts)

            results = [str(response).strip() for response in responses]
            logger.debug(f"Successfully generated {len(results)} batch responses")
            return results

        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            raise RuntimeError(f"Batch generation error: {e}") from e


def create_surveillance_llm(
    settings: Optional[LangChainSettings] = None,
) -> SurveillanceLLM:
    """
    Factory function for creating SurveillanceLLM instances with consistent configuration.

    :param settings: Optional LangChain settings. If None, uses default settings.
    :return: Configured SurveillanceLLM instance.
    :raise: Exception if LLM creation fails.
    """
    try:
        llm = SurveillanceLLM(settings)
        logger.info(f"Created SurveillanceLLM with model: {llm.settings.ollama_model}")
        return llm
    except Exception as e:
        logger.error(f"Failed to create SurveillanceLLM: {e}")
        raise


# Backward compatibility alias
LangChainLLM = SurveillanceLLM
