"""
openrouter_client.py

Key rotation strategy:
- 3 API keys loaded from .env at CALL TIME (not import time)
- Each key has its own exhausted flag
- On 429: mark current key as exhausted, immediately switch to next key
- Only waits if ALL keys are exhausted (then waits for reset window)
- No hardcoded keys anywhere in this file
"""

import os
import time
import json
import requests
from typing import Optional

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Model assignments (update these to match your OpenRouter account) ──────────
MODELS = {
    "agent":        "nvidia/nemotron-3-super-120b-a12b:free",
    "agent_backup": "openai/gpt-oss-120b:free",
    "labeler":      "arcee-ai/trinity-large-preview:free",
    "secondary":    "qwen/qwen3-next-80b-a3b-instruct:free",
    "generator":    "minimax/minimax-m2.5:free",
    "quality_gate": "z-ai/glm-4.5-air:free",
}

# ── Rate limit ─────────────────────────────────────────────────────────────────
MIN_SECONDS_BETWEEN_CALLS = 6.0   # 10 req/min max per key
MAX_RETRIES               = 6     # enough to try all 3 keys twice
RETRY_BACKOFF             = [5, 10, 20]

_last_call_time: float = 0.0

# Per-key exhaustion tracking: {key_prefix: exhausted_until_timestamp}
_key_exhausted_until: dict = {}


def _load_keys() -> list[str]:
    """Load API keys from environment at call time — never at import time."""
    keys = []
    for i in range(1, 4):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "").strip()
        if k and k != "sk-or-v1-...":
            keys.append(k)
    if not keys:
        raise EnvironmentError(
            "\n❌ No OpenRouter API keys found in environment.\n"
            "   Make sure your .env file has:\n"
            "   OPENROUTER_API_KEY_1=sk-or-v1-...\n"
            "   OPENROUTER_API_KEY_2=sk-or-v1-...\n"
            "   OPENROUTER_API_KEY_3=sk-or-v1-...\n"
            "   And that main.py loaded .env before importing this module."
        )
    return keys


def _key_id(key: str) -> str:
    """Short identifier for logging — never logs the full key."""
    return key[:12] + "..."


def _get_available_key(keys: list[str]) -> Optional[str]:
    """Return the first key that is not currently exhausted."""
    now = time.time()
    for key in keys:
        exhausted_until = _key_exhausted_until.get(_key_id(key), 0)
        if now >= exhausted_until:
            return key
    return None


def _mark_key_exhausted(key: str, wait_seconds: int = 3700):
    """Mark a key as exhausted for wait_seconds (default ~1 hour for daily limit)."""
    _key_exhausted_until[_key_id(key)] = time.time() + wait_seconds
    print(f"  🔒 Key {_key_id(key)} marked exhausted for {wait_seconds//60} min")


def _enforce_rate_limit():
    """Enforce minimum gap between calls to stay under 20 req/min."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    gap     = MIN_SECONDS_BETWEEN_CALLS - elapsed
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
    Call an OpenRouter free model with smart key rotation.

    On 429 → immediately marks current key exhausted, switches to next key.
    Only blocks if all 3 keys are exhausted simultaneously.
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

    keys = _load_keys()   # loaded fresh every call — picks up .env changes

    for attempt in range(retries):
        _enforce_rate_limit()

        # Pick next available (non-exhausted) key
        key = _get_available_key(keys)
        if key is None:
            # All keys exhausted — find the soonest reset and wait for it
            now   = time.time()
            waits = [
                max(0, _key_exhausted_until.get(_key_id(k), 0) - now)
                for k in keys
            ]
            min_wait = min(waits)
            print(f"  ⏳ All {len(keys)} keys exhausted. Waiting {int(min_wait//60)}min {int(min_wait%60)}s for reset...")
            time.sleep(min_wait + 5)
            key = _get_available_key(keys)
            if key is None:
                print("  ❌ Still no available key after wait. Aborting.")
                return None

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/your-username/agent-behavior-dataset",
            "X-Title":       "Agent Behavior Dataset Pipeline",
        }

        try:
            response = requests.post(
                BASE_URL, headers=headers, json=payload, timeout=120
            )

            if response.status_code == 429:
                # Check if it's a per-minute limit or daily limit
                remaining   = response.headers.get("X-RateLimit-Remaining", "0")
                retry_after = response.headers.get("Retry-After", "0")

                if retry_after and int(retry_after) > 300:
                    # Daily limit hit — mark key exhausted for ~1 hour
                    print(f"  🔒 Daily limit hit on {_key_id(key)} — switching key...")
                    _mark_key_exhausted(key, wait_seconds=3700)
                else:
                    # Per-minute limit — mark exhausted for 65s only
                    print(f"  ⚠️  Per-minute limit on {_key_id(key)} (remaining={remaining}) — switching key...")
                    _mark_key_exhausted(key, wait_seconds=65)
                continue   # immediately retry with next available key

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
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue

            message = choices[0].get("message") or {}
            content = message.get("content")

            if content is None:
                finish_reason = choices[0].get("finish_reason", "unknown")
                print(f"  ⚠️  Null content on '{role}' (finish_reason={finish_reason}) — retrying...")
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue

            # Success — log which key was used
            key_num = keys.index(key) + 1 if key in keys else "?"
            print(f"  ✓  [{role}] responded via key#{key_num} ({model.split('/')[1][:20]})")
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
    """call_llm with json_mode=True and auto-parsing. Repairs truncated JSON."""
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
        salvaged = _repair_truncated_json(raw)
        if salvaged:
            print(f"  ⚠️  JSON truncated — salvaged {len(salvaged)} item(s).")
            return salvaged
        return None


def _repair_truncated_json(raw: str) -> Optional[list]:
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
                    obj = json.loads(raw[start: i + 1])
                    salvaged.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return salvaged if salvaged else None