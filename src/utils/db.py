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
