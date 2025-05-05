import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine

from src.config.settings import DatabaseSettings
from sqlmodel import create_engine


def get_engine(settings: DatabaseSettings) -> Engine:
    """
    Create and return a SQLAlchemy engine based on settings.
    :param settings: The SQLite db settings
    :return: SQLAlchemy Engine
    """
    engine = create_engine(
        settings.url,
        echo=settings.echo,
        connect_args={"check_same_thread": False},  # for SQLite multithreading
    )
    return engine


def summarize(result: Any, *, max_len: int = 200) -> str:
    """
      Summarizes the input `result` into a short string.
      :param result: Object to summarize.
      :param max_len: The length of characters of the summary
      :return: A string summary of the result. If it's a dict with an "elements" key, it returns a
    count of the elements and a SHA-256 hash. Otherwise, it returns a truncated string
    representation of the result.
    """

    if isinstance(result, dict) and "elements" in result:
        count = len(result["elements"])
        h = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:8]
        return f"[{datetime.now(timezone.utc)}] elements={count} sha256={h}"
    return (str(result)[:max_len] + "…") if len(str(result)) > max_len else str(result)
