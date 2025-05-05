import time
import requests
from typing import Callable, Dict, Any

from src.config.settings import OverpassSettings


def with_retry(
    fn: Callable[..., Dict[str, Any]], settings: OverpassSettings = OverpassSettings()
):
    """
    Decorator that retries fn(*args, **kwargs) on transient Overpass errors
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
