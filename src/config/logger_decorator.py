import time
from src.config.logger import logger


def log_action(fn):
    def wrapped(self, *args, **kwargs):
        action = fn.__name__
        logger.info(f"[{self.name}] Starting {action}")
        start = time.time()

        try:
            result = fn(self, *args, **kwargs)
            duration = time.time() - start
            logger.info(f"[{self.name}] Finished {action} in {duration:.2f}s")
            return result
        except Exception:
            logger.exception(f"[{self.name}] Error during {action}")
            raise

    return wrapped
