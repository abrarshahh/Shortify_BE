import os
import time
import random
import logging
from functools import wraps
from typing import Callable, Any, Optional
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
        self.file_key_map = {}
        self.key_cooldowns = [0.0] * len(self.api_keys)
        logger.info(f"[GeminiManager] Initialized with {len(self.api_keys)} API keys.")
        
    def get_client(self) -> genai.Client:
        if not self.clients:
            logger.warning("[GeminiManager] No API keys configured in manager, falling back to default Client init")
            return genai.Client()
            
        now = time.time()
        for offset in range(len(self.api_keys)):
            idx = (self.current_index + offset) % len(self.api_keys)
            if self.key_cooldowns[idx] < now:
                self.current_index = idx
                return self.clients[self.current_index]
        return self.clients[self.current_index]
        
    def rotate_key(self):
        if len(self.api_keys) <= 1:
            return
        old_index = self.current_index
        now = time.time()
        for offset in range(1, len(self.api_keys) + 1):
            idx = (old_index + offset) % len(self.api_keys)
            if self.key_cooldowns[idx] < now:
                self.current_index = idx
                logger.info(f"[GeminiManager] Rotating API key from index {old_index} to {self.current_index} (not in cooldown)")
                return
        self.current_index = (old_index + 1) % len(self.api_keys)
        logger.info(f"[GeminiManager] All keys in cooldown. Rotating API key from index {old_index} to {self.current_index}")


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
        def _find_file_name(arg) -> Optional[str]:
            if isinstance(arg, str) and (arg.startswith("files/") or ("files/" not in arg and len(arg) == 12 and arg.isalnum())):
                return arg if arg.startswith("files/") else f"files/{arg}"
            if hasattr(arg, "name") and isinstance(arg.name, str) and arg.name.startswith("files/"):
                return arg.name
            if isinstance(arg, dict):
                for k, v in arg.items():
                    res = _find_file_name(v)
                    if res:
                        return res
            if isinstance(arg, (list, tuple)):
                for item in arg:
                    res = _find_file_name(item)
                    if res:
                        return res
            return None

        file_name = None
        for arg in args:
            file_name = _find_file_name(arg)
            if file_name:
                break
        if not file_name:
            for k, v in kwargs.items():
                file_name = _find_file_name(v)
                if file_name:
                    break

        # Rotate key on a new file upload to load-balance across API keys
        if self._service_name == "files" and self._method_name == "upload" and len(self._manager.api_keys) > 1:
            self._manager.rotate_key()

        pinned_index = None
        if file_name and file_name in self._manager.file_key_map:
            pinned_index = self._manager.file_key_map[file_name]
            logger.info(f"[GeminiManager] Pinned API key index {pinned_index} for file {file_name}")

        max_attempts = max(len(self._manager.api_keys), 1)
        max_attempts = max(max_attempts * 2, 3)
        
        last_error = None
        for attempt in range(max_attempts):
            if pinned_index is not None:
                client = self._manager.clients[pinned_index]
            else:
                client = self._manager.get_client()

            service = getattr(client, self._service_name)
            method = getattr(service, self._method_name)
            
            try:
                result = method(*args, **kwargs)
                
                # If this was an upload, record the key that was used
                if hasattr(result, "name") and isinstance(result.name, str) and result.name.startswith("files/"):
                    used_index = pinned_index if pinned_index is not None else self._manager.current_index
                    self._manager.file_key_map[result.name] = used_index
                    logger.info(f"[GeminiManager] Recorded file {result.name} owned by key index {used_index}")
                    
                # If this was a delete, clean it up from mapping
                if self._service_name == "files" and self._method_name == "delete" and file_name:
                    self._manager.file_key_map.pop(file_name, None)
                    logger.info(f"[GeminiManager] Removed file {file_name} from key map after deletion")
                    
                return result
            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(phrase in err_msg for phrase in [
                    "429", "rate limit", "quota exceeded", "resource exhausted",
                    "too many requests"
                ])
                if is_transient:
                    import time
                    import re
                    sleep_time = 5.0
                    match = re.search(r"please retry in ([\d\.]+)s", err_msg)
                    if not match:
                        match = re.search(r"retrydelay\':\s*\'(\d+)s\'", err_msg)
                    if match:
                        try:
                            sleep_time = float(match.group(1)) + 1.5
                        except Exception:
                            pass
                            
                    used_index = pinned_index if pinned_index is not None else self._manager.current_index
                    self._manager.key_cooldowns[used_index] = time.time() + sleep_time
                    logger.info(f"[GeminiManager] Key index {used_index} marked in cooldown for {sleep_time:.1f}s")

                    if pinned_index is not None:
                        logger.warning(
                            f"[GeminiManager] Quota limit hit on PINNED key index {pinned_index} during "
                            f"{self._service_name}.{self._method_name}: {e}. Sleeping {sleep_time:.1f}s before retry..."
                        )
                        time.sleep(sleep_time)
                        last_error = e
                        continue
                    elif len(self._manager.api_keys) > 1:
                        logger.warning(
                            f"[GeminiManager] Transient error/quota limit hit on key index {self._manager.current_index} during "
                            f"{self._service_name}.{self._method_name}: {e}. Rotating key..."
                        )
                        self._manager.rotate_key()
                        last_error = e
                        continue
                raise e
        raise last_error


_gemini_manager = None

def get_gemini_client() -> RotatableGeminiClient:
    global _gemini_manager
    if _gemini_manager is None:
        _gemini_manager = GeminiClientManager()
    return RotatableGeminiClient(_gemini_manager)

