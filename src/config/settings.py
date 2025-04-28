from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """
    Base settings for Ollama interaction
    """

    base_url: str
    timeout_seconds: float
    stream: bool = False
    model: str

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OLLAMA_")
