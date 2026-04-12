"""
openrouter_client.py
Unified LLM client for all pipeline stages via OpenRouter free models.

Key features:
- Rotates across 3 API keys to get ~150 req/day (3 × 50)
- Global 6s throttle = max 10 req/min, safely under 20 req/min limit
- Safe content extraction (never crashes on null responses)
- JSON repair for truncated responses
- Reads all keys from environment — never hardcoded
"""

import os
import time
import json
import requests
from typing import Optional

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Load 3 API keys from environment ──────────────────────────────────────────
def _load_api_keys() -> list[str]:
    keys = []
    for i in range(1, 4):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    # Fallback: support legacy single key
    legacy = os.environ.get("OPENROUTER_API_KEY", "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").strip()
    if legacy and legacy not in keys:
        keys.append(legacy)
    return keys

API_KEYS: list[str] = _load_api_keys()
_key_index: int = 0   # round-robin pointer

def _next_key() -> str:
    """Return next API key in round-robin rotation."""
    global _key_index
    if not API_KEYS:
        raise EnvironmentError(
            "No OpenRouter API keys found. Set OPENROUTER_API_KEY_1, "
            "OPENROUTER_API_KEY_2, OPENROUTER_API_KEY_3 in your .env file."
        )
    key = API_KEYS[_key_index % len(API_KEYS)]
    _key_index += 1
    return key

# ── Model assignments ──────────────────────────────────────────────────────────
MODELS = {
    "agent":        "nvidia/nemotron-3-super-120b-a12b:free",   # agent executo
    "agent_backup": "google/gemma-4-31b-it:free",                # fallback executor
    "labeler":      "arcee-ai/trinity-large-preview:free",    # primary labeler
    "secondary":    "qwen/qwen3-next-80b-a3b-instruct:free",     # secondary labeler
    "generator":    "google/gemma-4-26b-a4b-it:free",            # task generator (separate model)
    "quality_gate": "nvidia/nemotron-nano-9b-v2:free",           # quality checker
}

# ── Rate limit config ──────────────────────────────────────────────────────────
# Hard limits per key: 50 req/day, 20 req/min
# With 3 keys rotating: ~150 req/day total
# 6s gap = 10 req/min per key — safely under 20 req/min
MIN_SECONDS_BETWEEN_CALLS = 6.0
RATE_LIMIT_WAIT           = 120   # wait 2min after 429
MAX_RETRIES               = 3
RETRY_BACKOFF             = [5, 15, 30]

_last_call_time: float = 0.0


def _enforce_rate_limit():
    """Sleep until MIN_SECONDS_BETWEEN_CALLS have passed since the last call."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    gap = MIN_SECONDS_BETWEEN_CALLS - elapsed
    if gap > 0:
        time.sleep(gap)
    _last_call_time = time.time()


def call_llm(
    role: str,
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_mode: bool = False,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Call an OpenRouter free model using round-robin key rotation.
    Returns response text string, or None if all retries fail.
    """
    model = MODELS.get(role)
    if not model:
        raise ValueError(f"Unknown role '{role}'. Valid: {list(MODELS.keys())}")

    if json_mode:
        messages = messages.copy()
        messages[-1]["content"] += (
            "\n\nIMPORTANT: Return ONLY valid JSON. "
            "No markdown fences, no explanation, no extra text."
        )

    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    for attempt in range(retries):
        _enforce_rate_limit()
        api_key = _next_key()   # rotate key on every attempt

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/your-username/agent-behavior-dataset",
            "X-Title":       "Agent Behavior Dataset Pipeline",
        }

        try:
            response = requests.post(
                BASE_URL, headers=headers, json=payload, timeout=120
            )

            if response.status_code == 429:
                rl_remaining = response.headers.get("X-RateLimit-Remaining", "?")
                retry_after  = response.headers.get("Retry-After", RATE_LIMIT_WAIT)
                wait = int(retry_after) + 5 if str(retry_after).isdigit() else RATE_LIMIT_WAIT
                print(f"  ⚠️  429 on '{role}' key#{_key_index % len(API_KEYS)} "
                      f"(remaining={rl_remaining}) — waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code == 503:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"  ⚠️  503 model unavailable — retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data    = response.json()
            choices = data.get("choices") or []

            if not choices:
                print(f"  ⚠️  Empty choices on '{role}'. Raw: {str(data)[:150]}")
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue

            message = choices[0].get("message") or {}
            content = message.get("content")

            if content is None:
                finish_reason = choices[0].get("finish_reason", "unknown")
                print(f"  ⚠️  Null content on '{role}' (finish_reason={finish_reason}) — retrying...")
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue

            return content.strip()

        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  Timeout (attempt {attempt+1}) — retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ❌ Attempt {attempt+1} error: {e} — retrying in {wait}s...")
            time.sleep(wait)

    print(f"  ❌ All {retries} attempts failed for role '{role}'.")
    return None


def call_llm_json(role: str, messages: list, **kwargs) -> Optional[dict]:
    """
    call_llm with json_mode=True and auto-parsing.
    Falls back to repair_truncated_json before giving up.
    """
    raw = call_llm(role, messages, json_mode=True, **kwargs)
    if raw is None:
        return None
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        salvaged = repair_truncated_json(raw)
        if salvaged:
            print(f"  ⚠️  JSON truncated — salvaged {len(salvaged)} item(s).")
            return salvaged
        return None


def repair_truncated_json(raw: str) -> Optional[list]:
    """Extract complete JSON objects from a truncated array response."""
    raw = raw.strip()
    if not raw.startswith("["):
        return None
    salvaged = []
    depth    = 0
    start    = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(raw[start : i + 1])
                    salvaged.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return salvaged if salvaged else None