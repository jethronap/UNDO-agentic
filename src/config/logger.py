import sys
from pathlib import Path

from loguru import logger
from src.config.settings import LoggingSettings


_log_cfg = LoggingSettings()

logger.remove()

if _log_cfg.console:
    logger.add(
        sys.stderr,
        level=_log_cfg.level,
        format="<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "{message}",
        enqueue=True,  # Offloads I/O to a background thread
    )


if _log_cfg.enable_file:
    log_path = Path(_log_cfg.filepath)
    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path,
        level=_log_cfg.level,
        rotation=_log_cfg.rotation,
        retention=_log_cfg.retention,
        compression=_log_cfg.compression,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
