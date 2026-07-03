import os
import time
import random
import logging
from functools import wraps
from typing import Callable, Any
from google import genai

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


class GeminiClientManager:
    def __init__(self):
        self.api_keys = []
        keys_set = set()
        
        # Load main key
        main_key = os.getenv("GEMINI_API_KEY")
        if main_key and main_key.strip():
            val = main_key.strip()
            keys_set.add(val)
            self.api_keys.append(val)
            
        # Load additional keys
        for i in range(1, 11):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key and key.strip():
                val = key.strip()
                if val not in keys_set:
                    keys_set.add(val)
                    self.api_keys.append(val)
                    
        # Check any other keys matching GEMINI_API_KEY_
        for env_key, env_val in os.environ.items():
            if env_key.startswith("GEMINI_API_KEY_") and env_val and env_val.strip():
                val = env_val.strip()
                if val not in keys_set:
                    keys_set.add(val)
                    self.api_keys.append(val)
                    
        self.current_index = 0
        self.clients = [genai.Client(api_key=k) for k in self.api_keys]
        logger.info(f"[GeminiManager] Initialized with {len(self.api_keys)} API keys.")
        
    def get_client(self) -> genai.Client:
        if not self.clients:
            # Fallback to default client
            logger.warning("[GeminiManager] No API keys configured in manager, falling back to default Client init")
            return genai.Client()
        return self.clients[self.current_index]
        
    def rotate_key(self):
        if len(self.api_keys) <= 1:
            return
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        logger.info(f"[GeminiManager] Rotating API key from index {old_index} to {self.current_index}")


class RotatableGeminiClient:
    def __init__(self, manager: GeminiClientManager):
        self._manager = manager

    @property
    def models(self):
        return RotatableService(self._manager, "models")

    @property
    def files(self):
        return RotatableService(self._manager, "files")


class RotatableService:
    def __init__(self, manager: GeminiClientManager, service_name: str):
        self._manager = manager
        self._service_name = service_name

    def __getattr__(self, name):
        return RotatableMethod(self._manager, self._service_name, name)


class RotatableMethod:
    def __init__(self, manager: GeminiClientManager, service_name: str, method_name: str):
        self._manager = manager
        self._service_name = service_name
        self._method_name = method_name

    def __call__(self, *args, **kwargs):
        max_attempts = max(len(self._manager.api_keys), 1)
        max_attempts = max(max_attempts * 2, 3)
        
        last_error = None
        for attempt in range(max_attempts):
            client = self._manager.get_client()
            service = getattr(client, self._service_name)
            method = getattr(service, self._method_name)
            
            try:
                return method(*args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(phrase in err_msg for phrase in [
                    "429", "rate limit", "quota exceeded", "resource exhausted",
                    "too many requests"
                ])
                if is_transient and len(self._manager.api_keys) > 1:
                    logger.warning(
                        f"[GeminiManager] Transient error/quota limit hit on key index {self._manager.current_index} during "
                        f"{self._service_name}.{self._method_name}: {e}. Rotating key..."
                    )
                    self._manager.rotate_key()
                    last_error = e
                    continue
                else:
                    raise e
        raise last_error


_gemini_manager = None

def get_gemini_client() -> RotatableGeminiClient:
    global _gemini_manager
    if _gemini_manager is None:
        _gemini_manager = GeminiClientManager()
    return RotatableGeminiClient(_gemini_manager)

