from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """
    Base settings for Ollama interaction
    """

    base_url: str = Field(
        default="http://localhost:11434/api/generate", description="The Ollama base url"
    )
    timeout_seconds: float = Field(
        default=30.0, description="Timeout for calling Ollama"
    )
    stream: bool = Field(default=False, description="Flag to denote chunked streaming.")
    model: str = Field(default="gemma2", description="Ollama model name.")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OLLAMA_", extra="allow"
    )


class DatabaseSettings(BaseSettings):
    """
    Configuration for the SQLModel-based SQLite database.
    """

    url: str = Field(description="The url of the database")
    echo: bool = Field(default=False, description="SQLAlchemy echo for debugging SQL")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SQLITE_DB_", extra="allow"
    )


class LoggingSettings(BaseSettings):
    """
    Configuration for Loguru logging sinks.
    """

    level: str = Field(default="DEBUG", description="The log level")
    console: bool = Field(default=True, description="Show logs in console")
    enable_file: bool = Field(
        default=False, description="Flag to denote persistence of logs"
    )
    filepath: Optional[Path] = Field(
        default=None, description="Optional file path for logs"
    )
    rotation: str = Field(default="10 MB", description="Roll log after this size")
    retention: str = Field(
        default="7 days", description="Keep logs for this amount of time"
    )
    compression: str = Field(default="zip", description="Compress old logs")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOG_", extra="allow")
