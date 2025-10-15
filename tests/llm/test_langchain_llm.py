import pytest
from unittest.mock import Mock, patch

from src.llm.surveillance_llm import LangChainLLM
from src.config.settings import LangChainSettings


class TestLangChainLLM:
    """Test suite for LangChainLLM wrapper."""

    @pytest.fixture
    def mock_settings(self) -> LangChainSettings:
        """Create mock LangChain settings for testing."""
        return LangChainSettings(
            ollama_base_url="http://localhost:11434",
            ollama_model="test-model",
            ollama_temperature=0.7,
            ollama_timeout=30.0,
        )

    @pytest.fixture
    def mock_ollama_client(self) -> Mock:
        """Create a mock LangChain Ollama client."""
        mock_client = Mock()
        mock_client.invoke.return_value = "Mock LLM response"
        mock_client.batch.return_value = ["Response 1", "Response 2"]
        return mock_client

    @patch("src.llm.surveillance_llm.Ollama")
    def test_init_success(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test successful initialization of LangChainLLM."""
        mock_client = Mock()
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)

        assert llm.settings == mock_settings
        assert llm.llm == mock_client
        mock_ollama_class.assert_called_once_with(
            base_url="http://localhost:11434",
            model="test-model",
            temperature=0.7,
            timeout=30.0,
        )

    @patch("src.llm.surveillance_llm.Ollama")
    def test_init_default_settings(self, mock_ollama_class: Mock) -> None:
        """Test initialization with default settings."""
        mock_client = Mock()
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM()

        assert llm.settings is not None
        assert isinstance(llm.settings, LangChainSettings)
        assert llm.llm == mock_client

    @patch("src.llm.surveillance_llm.Ollama")
    def test_init_failure(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test initialization failure handling."""
        mock_ollama_class.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            LangChainLLM(mock_settings)

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_response_success(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test successful response generation."""
        mock_client = Mock()
        mock_client.invoke.return_value = "  Mock response text  "
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)
        result = llm.generate_response("Test prompt")

        assert result == "Mock response text"
        mock_client.invoke.assert_called_once_with("Test prompt")

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_response_with_kwargs(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test response generation with additional kwargs."""
        mock_client = Mock()
        mock_temp_client = Mock()
        mock_temp_client.invoke.return_value = "Temp response"
        mock_ollama_class.side_effect = [mock_client, mock_temp_client]

        llm = LangChainLLM(mock_settings)
        result = llm.generate_response(
            "Test prompt", temperature=0.9, custom_param="test"
        )

        assert result == "Temp response"
        # Verify that a temporary client was created with modified settings
        assert mock_ollama_class.call_count == 2
        temp_call_args = mock_ollama_class.call_args_list[1]
        assert temp_call_args[1]["temperature"] == 0.9
        assert temp_call_args[1]["custom_param"] == "test"

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_response_empty_response(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test handling of empty response."""
        mock_client = Mock()
        mock_client.invoke.return_value = "  "
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)
        result = llm.generate_response("Test prompt")

        assert result == ""

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_response_non_string_response(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test handling of non-string response."""
        mock_client = Mock()
        mock_client.invoke.return_value = {"response": "dict response"}
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)
        result = llm.generate_response("Test prompt")

        assert result == "{'response': 'dict response'}"

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_response_exception(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test exception handling during response generation."""
        mock_client = Mock()
        mock_client.invoke.side_effect = Exception("LLM error")
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)

        with pytest.raises(RuntimeError, match="LLM generation error: LLM error"):
            llm.generate_response("Test prompt")

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_batch_success(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test successful batch response generation."""
        mock_client = Mock()
        mock_client.batch.return_value = ["Response 1", "  Response 2  ", "Response 3"]
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = llm.generate_batch(prompts)

        assert results == ["Response 1", "Response 2", "Response 3"]
        mock_client.batch.assert_called_once_with(prompts)

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_batch_with_kwargs(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test batch generation with additional kwargs."""
        mock_client = Mock()
        mock_temp_client = Mock()
        mock_temp_client.batch.return_value = ["Batch response 1", "Batch response 2"]
        mock_ollama_class.side_effect = [mock_client, mock_temp_client]

        llm = LangChainLLM(mock_settings)
        prompts = ["Prompt 1", "Prompt 2"]
        results = llm.generate_batch(prompts, temperature=0.3)

        assert results == ["Batch response 1", "Batch response 2"]
        # Verify that a temporary client was created
        assert mock_ollama_class.call_count == 2
        temp_call_args = mock_ollama_class.call_args_list[1]
        assert temp_call_args[1]["temperature"] == 0.3

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_batch_non_string_responses(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test batch generation with non-string responses."""
        mock_client = Mock()
        mock_client.batch.return_value = ["String response", {"key": "value"}, 12345]
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = llm.generate_batch(prompts)

        assert results == ["String response", "{'key': 'value'}", "12345"]

    @patch("src.llm.surveillance_llm.Ollama")
    def test_generate_batch_exception(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test exception handling during batch generation."""
        mock_client = Mock()
        mock_client.batch.side_effect = Exception("Batch error")
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)
        prompts = ["Prompt 1", "Prompt 2"]

        with pytest.raises(RuntimeError, match="Batch generation error: Batch error"):
            llm.generate_batch(prompts)

    @patch("src.llm.surveillance_llm.Ollama")
    def test_backward_compatibility_interface(
        self, mock_ollama_class: Mock, mock_settings: LangChainSettings
    ) -> None:
        """Test that the interface matches the original LocalLLM."""
        mock_client = Mock()
        mock_client.invoke.return_value = "Compatible response"
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)

        # Test that the generate_response method exists and works
        assert hasattr(llm, "generate_response")
        result = llm.generate_response("test prompt", expect_json=True)
        assert result == "Compatible response"

        # Verify the kwargs are passed through (even if not used by LangChain)
        mock_client.invoke.assert_called_with("test prompt")

    @patch("src.llm.surveillance_llm.Ollama")
    def test_settings_validation(self, mock_ollama_class: Mock) -> None:
        """Test that settings are properly validated."""
        mock_client = Mock()
        mock_ollama_class.return_value = mock_client

        # Test with invalid temperature (should be handled by Pydantic)
        with pytest.raises(ValueError):
            LangChainSettings(ollama_temperature=2.0)  # > 1.0 should fail

        with pytest.raises(ValueError):
            LangChainSettings(ollama_temperature=-0.1)  # < 0.0 should fail

    @patch("src.llm.surveillance_llm.logger")
    @patch("src.llm.surveillance_llm.Ollama")
    def test_logging_behavior(
        self,
        mock_ollama_class: Mock,
        mock_logger: Mock,
        mock_settings: LangChainSettings,
    ) -> None:
        """Test that logging behavior works correctly."""
        mock_client = Mock()
        mock_client.invoke.return_value = "Logged response"
        mock_ollama_class.return_value = mock_client

        llm = LangChainLLM(mock_settings)
        llm.generate_response("Test prompt")

        # Verify debug logs were called
        debug_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if call[0][0].startswith("Initialized SurveillanceLLM")
            or call[0][0].startswith("Generating response for prompt")
            or call[0][0].startswith("Successfully generated")
        ]
        assert len(debug_calls) >= 2  # At least initialization and response logging
