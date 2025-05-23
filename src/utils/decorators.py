import time
from functools import wraps
from typing import Callable, Dict, Any

import requests
from loguru import logger

from src.config.settings import OverpassSettings


def log_action(fn):
    """
    Decorator for logging:
      - INFO at start/end (with elapsed time)
      - DEBUG of args/kwargs and a short repr(result)
      - exception() on error
    """

    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        name = getattr(self, "name", fn.__qualname__)
        action = fn.__name__
        # Log entry
        logger.info(f"[{name}]: {action}()")
        logger.debug(f"[{name}]: args={args}, kwargs={kwargs}")
        start = time.time()
        try:
            result = fn(self, *args, **kwargs)
        except Exception:
            logger.exception(f"[{name}]: {action}() failed")
            raise
        elapsed = time.time() - start
        # Log exit
        summary = repr(result)
        if len(summary) > 100:
            summary = summary[:100] + "…"
        logger.debug(f"[{name}]: {action} returned {summary!r}")
        logger.info(f"[{name}]:  {action}() in {elapsed:.2f}s")
        return result

    return wrapped


def with_retry(
    fn: Callable[..., Dict[str, Any]], settings: OverpassSettings = OverpassSettings()
):
    """
    Decorator that retries fn(*args, **kwargs) on non-permanent Overpass errors
    """

    def wrapped(*args, **kwargs):
        attempt = 0
        while True:
            attempt += 1
            try:
                response = fn(*args, **kwargs)
                return response
            except requests.HTTPError as e:
                if e.response.status_code not in settings.retry_http:
                    raise
            except (requests.ConnectionError, requests.Timeout):
                pass
            if attempt >= settings.max_attempts:
                raise
            delay = settings.base_delay * 2 ** (attempt - 1)
            time.sleep(min(delay, 60))

    return wrapped
