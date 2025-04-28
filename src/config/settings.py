from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """
    Base settings for Ollama interaction
    """

    base_url: str
    timeout_seconds: float
    # No chunked streaming by default.
    stream: bool = False
    model: str

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OLLAMA_")


class DatabaseSettings(BaseSettings):
    """
    Configuration for the SQLModel-based SQLite database.
    """

    url: str
    echo: bool = False  # SQLAlchemy echo for debugging SQL

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SQLITE_DB_", extra="allow"
    )
