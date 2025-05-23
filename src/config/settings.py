from pathlib import Path
from typing import Optional, Union, Dict, Any

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
    model: str = Field(default="qwen2.5", description="Ollama model name.")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OLLAMA_", extra="allow"
    )


class DatabaseSettings(BaseSettings):
    """
    Configuration for the SQLModel-based SQLite database.
    """

    url: str = Field(default=None, description="The url of the database")
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
        default="logs/agents.log", description="Optional file path for logs"
    )
    rotation: str = Field(default="10 MB", description="Roll log after this size")
    retention: str = Field(
        default="7 days", description="Keep logs for this amount of time"
    )
    compression: str = Field(default="zip", description="Compress old logs")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOG_", extra="allow")


class OverpassSettings(BaseSettings):
    """
    Configuration for Overpass pipeline.
    """

    endpoint: str = Field(
        default="https://overpass-api.de/api/interpreter",
        description="The Overpass API endpoint",
    )
    headers: Dict[str, Any] = Field(
        default={"User-Agent": "ACS-Agent/0.1 (contact@email)"},
        description="The headers used for making request to Overpass",
    )
    dir: Union[Path, str] = Field(
        default=Path("overpass_data"),
        description="The Path to the Overpass data directory",
    )
    query_timeout: int = Field(
        default=25, description="The timeout for the Overpass query"
    )
    timeout: int = Field(
        default=60, description="The timeout for the request made to Overpass API"
    )
    retry_http: set[int] = Field(
        default={429, 500, 502, 503, 504},
        description="The HTTP statuses to retry after hitting",
    )
    max_attempts: int = Field(default=4, description="The maximum number of retries")
    base_delay: float = Field(
        default=2.0, description="The number of delay between retries in seconds"
    )


class PromptsSettings(BaseSettings):
    """
    Configuration for the prompts
    """

    template_dir: Path = Field(
        default=Path("src/prompts"), description="The directory holding the prompts"
    )
    template_file: str = Field(
        default="prompt.md", description="The name of the template file"
    )


class HeatmapSettings(BaseSettings):
    """
    Configuration for the heatmaps
    """

    radius: int = Field(default=15, description="The radius of points")
    blur: int = Field(default=10, description="The blur of points")
