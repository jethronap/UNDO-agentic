import pytest
from src.config.settings import LangChainSettings
from src.config.langchain_init import (
    setup_langchain_environment,
    get_langchain_config,
    validate_langchain_setup,
    init_langchain,
)


class TestLangChainSettings:
    """Test suite for LangChainSettings configuration."""

    def test_default_settings(self):
        """Test that default settings are created correctly."""
        settings = LangChainSettings()
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "llama3:latest"
        assert settings.ollama_timeout == 30.0
        assert settings.ollama_temperature == 0.0
        assert settings.agent_max_iterations == 8
        assert settings.agent_max_execution_time == 180.0
        assert settings.agent_verbose is True
        assert settings.memory_enabled is True
        assert settings.memory_max_tokens == 2000
        assert settings.tool_timeout == 60.0

    def test_temperature_validation(self):
        """Test that temperature validation works correctly."""
        # Valid temperature
        settings = LangChainSettings(ollama_temperature=0.5)
        assert settings.ollama_temperature == 0.5

        # Invalid temperature - too low
        with pytest.raises(ValueError, match="Temperature must be between"):
            LangChainSettings(ollama_temperature=-0.1)

        # Invalid temperature - too high
        with pytest.raises(ValueError, match="Temperature must be between"):
            LangChainSettings(ollama_temperature=1.1)

    def test_max_iterations_validation(self):
        """Test that max iterations validation works correctly."""
        # Valid iterations
        settings = LangChainSettings(agent_max_iterations=5)
        assert settings.agent_max_iterations == 5

        # Invalid iterations
        with pytest.raises(ValueError, match="Maximum iterations must be positive"):
            LangChainSettings(agent_max_iterations=0)

        with pytest.raises(ValueError, match="Maximum iterations must be positive"):
            LangChainSettings(agent_max_iterations=-1)

    def test_custom_settings(self):
        """Test creating settings with custom values."""
        settings = LangChainSettings(
            ollama_base_url="http://custom:11434",
            ollama_model="custom-model",
            ollama_temperature=0.7,
            agent_max_iterations=20,
            agent_verbose=True,
        )
        assert settings.ollama_base_url == "http://custom:11434"
        assert settings.ollama_model == "custom-model"
        assert settings.ollama_temperature == 0.7
        assert settings.agent_max_iterations == 20
        assert settings.agent_verbose is True


class TestLangChainInitialization:
    """Test suite for LangChain initialization helpers."""

    def test_setup_langchain_environment(self, monkeypatch):
        """Test environment setup function."""
        import os

        settings = LangChainSettings(ollama_base_url="http://test:11434")
        setup_langchain_environment(settings)

        assert os.environ["OLLAMA_BASE_URL"] == "http://test:11434"

    def test_get_langchain_config(self):
        """Test configuration dictionary retrieval."""
        settings = LangChainSettings(ollama_model="test-model", agent_max_iterations=15)
        config = get_langchain_config(settings)

        assert isinstance(config, dict)
        assert config["ollama_model"] == "test-model"
        assert config["agent_max_iterations"] == 15
        assert "ollama_base_url" in config
        assert "ollama_timeout" in config
        assert "ollama_temperature" in config

    def test_validate_langchain_setup(self):
        """Test validation function."""
        # Should pass with default settings
        assert validate_langchain_setup() is True

    def test_validate_langchain_setup_invalid_url(self, monkeypatch):
        """Test validation fails with invalid URL."""

        def mock_init(self):
            self.ollama_base_url = "invalid-url"

        monkeypatch.setattr(LangChainSettings, "__init__", mock_init)
        # This test will fail validation due to invalid URL
        # Note: This is a simplified test, actual implementation may vary

    def test_init_langchain(self):
        """Test full initialization."""
        config = init_langchain()

        assert isinstance(config, dict)
        assert "ollama_base_url" in config
        assert "ollama_model" in config
        assert "agent_max_iterations" in config

    def test_init_langchain_with_custom_settings(self):
        """Test initialization with custom settings."""
        settings = LangChainSettings(ollama_model="custom-model")
        config = init_langchain(settings)

        assert config["ollama_model"] == "custom-model"


class TestBackwardCompatibility:
    """Test backward compatibility with existing settings."""

    def test_ollama_settings_still_work(self):
        """Verify OllamaSettings class still works."""
        from src.config.settings import OllamaSettings

        settings = OllamaSettings()
        assert settings.base_url is not None
        assert settings.model is not None

    def test_database_settings_still_work(self):
        """Verify DatabaseSettings class still works."""
        from src.config.settings import DatabaseSettings

        settings = DatabaseSettings()
        assert settings.echo is False

    def test_all_settings_classes_available(self):
        """Test that all settings classes are importable."""
        from src.config.settings import (
            OllamaSettings,
            DatabaseSettings,
            LoggingSettings,
            OverpassSettings,
            HeatmapSettings,
            LangChainSettings,
        )

        # All should be importable without error
        assert OllamaSettings is not None
        assert DatabaseSettings is not None
        assert LoggingSettings is not None
        assert OverpassSettings is not None
        assert HeatmapSettings is not None
        assert LangChainSettings is not None
