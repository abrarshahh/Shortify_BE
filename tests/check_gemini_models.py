import os
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv


def _pick_gemini_key() -> str:
    # Support both common names
    return (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


def fetch_gemini_models(api_key: str) -> List[Dict[str, object]]:
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": api_key}
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    models = [m for m in data.get("models", []) if m.get("name")]
    return sorted(models, key=lambda m: str(m.get("name", "")))


def _extract_rate_limits(headers: httpx.Headers) -> Dict[str, Optional[str]]:
    # Gemini may or may not expose these headers depending on endpoint/account.
    return {
        "requests_limit": headers.get("x-ratelimit-limit-requests"),
        "requests_remaining": headers.get("x-ratelimit-remaining-requests"),
        "tokens_limit": headers.get("x-ratelimit-limit-tokens"),
        "tokens_remaining": headers.get("x-ratelimit-remaining-tokens"),
        "reset_requests": headers.get("x-ratelimit-reset-requests"),
        "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
    }


def probe_gemini_model(client: httpx.Client, api_key: str, model_name: str) -> Tuple[bool, Dict[str, Optional[str]], str]:
    """
    Probe model with a tiny generateContent request.
    Returns: (usable_now, rate_limits, status_reason)
    """
    # model_name is usually like "models/gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
    params = {"key": api_key}
    payload = {
        "contents": [{"parts": [{"text": "ping"}]}],
        "generationConfig": {"maxOutputTokens": 1, "temperature": 0},
    }
    try:
        resp = client.post(url, params=params, json=payload)
    except Exception as exc:
        return False, {}, f"request_error:{exc}"

    limits = _extract_rate_limits(resp.headers)

    if resp.status_code == 200:
        return True, limits, "ok"

    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    msg = str(body).lower()
    if resp.status_code in (401, 403):
        return False, limits, "unauthorized_or_forbidden"
    if resp.status_code == 429:
        return False, limits, "rate_limited_or_quota_exceeded"
    if "billing" in msg or "payment" in msg or "insufficient" in msg:
        return False, limits, "paid_or_quota_required"
    if "not found" in msg or "unsupported" in msg or resp.status_code == 404:
        return False, limits, "not_available_for_key"
    return False, limits, f"http_{resp.status_code}"


def main() -> int:
    load_dotenv()
    api_key = _pick_gemini_key()
    if not api_key:
        print("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment/.env")
        return 1

    try:
        models = fetch_gemini_models(api_key)
    except Exception as exc:
        print(f"Failed to fetch Gemini models: {exc}")
        return 1

    if not models:
        print("No models returned by Gemini for this key.")
        return 0

    max_checks = int(os.getenv("MAX_MODEL_CHECKS", "200"))

    # Keep only generation-capable models
    generation_models = []
    for model in models:
        methods = model.get("supportedGenerationMethods") or []
        if "generateContent" in methods and model.get("name"):
            generation_models.append(model)
    generation_models = generation_models[:max_checks]

    free_usable = []
    with httpx.Client(timeout=30) as client:
        for m in generation_models:
            model_name = str(m.get("name"))
            usable, limits, reason = probe_gemini_model(client, api_key, model_name)
            if usable:
                free_usable.append((model_name, limits))
            # else:
            #     print(f"skip {model_name}: {reason}")

    if not free_usable:
        print("No currently-usable Gemini models detected for this key.")
        return 0

    print("Gemini models usable now (free-tier behavior) with rate-limit headers:")
    for model_name, limits in free_usable:
        req_lim = limits.get("requests_limit") or "unknown"
        tok_lim = limits.get("tokens_limit") or "unknown"
        req_rem = limits.get("requests_remaining") or "unknown"
        tok_rem = limits.get("tokens_remaining") or "unknown"
        print(
            f"- {model_name} | rpm_limit={req_lim} rpm_remaining={req_rem} "
            f"tpm_limit={tok_lim} tpm_remaining={tok_rem}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

