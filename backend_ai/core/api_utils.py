import time
import random
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger("shortify_api_guard")

def rate_limit_guard(max_retries: int = 5, initial_backoff: float = 2.0, factor: float = 2.0):
    """
    Decorator that intercepts transient errors (like 429 Rate Limits or network timeouts)
    and applies exponential backoff with randomized jitter before retrying.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    # Check if error is a rate limit or transient network failure
                    is_transient = any(phrase in err_msg for phrase in [
                        "429", "rate limit", "quota exceeded", "resource exhausted",
                        "timeout", "too many requests", "service unavailable", "503"
                    ])
                    
                    if not is_transient or retries >= max_retries:
                        # Re-raise immediately if it's a structural code error or we exceeded retries
                        logger.error(f"[API Guard] Non-retryable error or retries exhausted: {e}")
                        raise e
                        
                    retries += 1
                    # Exponential backoff with random jitter (e.g. 0.8x to 1.2x)
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = backoff * jitter
                    
                    logger.warning(
                        f"[API Guard] Transient failure detected during {func.__name__}: {e}. "
                        f"Retrying in {sleep_time:.2f} seconds (Attempt {retries}/{max_retries})..."
                    )
                    
                    time.sleep(sleep_time)
                    backoff *= factor # double the wait for next loop
                    
        return wrapper
    return decorator
