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
        agent = getattr(self, "name", fn.__qualname__)
        action = fn.__name__

        logger.info(f"{agent}.{action}")

        # DEBUG: key arguments
        # if first arg is a string show it
        if args:
            first = args[0]
            logger.debug(f"{agent}.{action} args[0]={first!r}")
        # if there's a context dict, log its size
        ctx = kwargs.get("context") or (args[1] if len(args) > 1 else None)
        if isinstance(ctx, dict):
            logger.debug(f"{agent}.{action} context_keys={list(ctx.keys())}")

        start = time.time()
        result = fn(self, *args, **kwargs)
        elapsed = time.time() - start

        # build a small result hint
        hint = ""
        if isinstance(result, (list, tuple)):
            hint = f" (count={len(result)})"
        elif isinstance(result, str) and result.endswith(
            (".json", ".geojson", ".html")
        ):
            hint = f": {result!r}"

        logger.info(f"{agent}.{action}: in {elapsed:.2f}s{hint}")
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
