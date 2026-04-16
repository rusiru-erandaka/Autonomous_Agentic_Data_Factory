"""
openrouter_client.py

IMPORTANT ABOUT KEY ROTATION:
- Per-MINUTE limit (20 req/min): Shared across ALL keys on the same account.
  Switching keys does NOT help — you must wait ~60s.
- Per-DAY limit (50 req/day free): Per-key. Switching keys DOES help here.
- Strategy: on 429, first wait 65s (per-minute reset), then retry.
  If still 429, mark key as daily-exhausted and switch to next key.
"""

import os
import time
import json
import requests
from typing import Optional

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "agent":        "nvidia/nemotron-3-super-120b-a12b:free",
    "agent_backup": "openai/gpt-oss-120b:free",
    "labeler":      "arcee-ai/trinity-large-preview:free",
    "secondary":    "qwen/qwen3-next-80b-a3b-instruct:free",
    "generator":    "minimax/minimax-m2.5:free",
    "quality_gate": "z-ai/glm-4.5-air:free",
}

# ── Rate limit constants ───────────────────────────────────────────────────────
MIN_SECONDS_BETWEEN_CALLS = 6.0   # 10 req/min max
PER_MINUTE_WAIT           = 65    # wait after per-minute 429
PER_DAY_EXHAUSTED_WAIT    = 3700  # ~1hr after daily limit hit
MAX_RETRIES               = 8
RETRY_BACKOFF             = [5, 10, 20]

_last_call_time: float = 0.0

# Daily-exhausted keys: {key_prefix: exhausted_until_timestamp}
# Per-minute 429s are handled by sleeping, NOT by marking exhausted
_daily_exhausted_until: dict = {}


def _load_keys() -> list[str]:
    """Load API keys from environment at call time."""
    keys = []
    for i in range(1, 4):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "").strip()
        if k and not k.startswith("sk-or-v1-replace"):
            keys.append(k)
    if not keys:
        raise EnvironmentError(
            "\n❌ No OpenRouter API keys found.\n"
            "   Set OPENROUTER_API_KEY_1, _2, _3 in your .env file."
        )
    return keys


def _key_id(key: str) -> str:
    """Short ID for logging — never logs the full key."""
    return key[:16] + "..."


def _is_daily_exhausted(key: str) -> bool:
    return time.time() < _daily_exhausted_until.get(_key_id(key), 0)


def _mark_daily_exhausted(key: str):
    _daily_exhausted_until[_key_id(key)] = time.time() + PER_DAY_EXHAUSTED_WAIT
    print(f"  🔒 Key {_key_id(key)} daily limit reached — marked exhausted for ~1hr")


def _get_available_key(keys: list[str]) -> Optional[str]:
    """Return first key not daily-exhausted."""
    for key in keys:
        if not _is_daily_exhausted(key):
            return key
    return None


def _enforce_rate_limit():
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
    Call an OpenRouter free model.

    429 handling strategy:
    - First 429 → wait 65s (per-minute window reset), retry SAME key
    - Second 429 on same key → mark key daily-exhausted, switch to next key
    - All keys daily-exhausted → wait for soonest reset
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

    keys = _load_keys()

    # Track consecutive 429s per key to distinguish per-minute vs daily
    consecutive_429: dict = {_key_id(k): 0 for k in keys}

    for attempt in range(retries):
        _enforce_rate_limit()

        key = _get_available_key(keys)
        if key is None:
            # All keys daily-exhausted
            now      = time.time()
            min_wait = min(
                max(0, _daily_exhausted_until.get(_key_id(k), 0) - now)
                for k in keys
            )
            print(f"  ⏳ All {len(keys)} keys daily-exhausted. Waiting {int(min_wait//60)}m {int(min_wait%60)}s...")
            time.sleep(min_wait + 5)
            key = _get_available_key(keys)
            if key is None:
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
                kid = _key_id(key)
                consecutive_429[kid] = consecutive_429.get(kid, 0) + 1

                if consecutive_429[kid] >= 2:
                    # Two 429s in a row on same key = daily limit hit
                    _mark_daily_exhausted(key)
                    consecutive_429[kid] = 0
                    # Try next key immediately (don't sleep)
                    continue
                else:
                    # First 429 = per-minute limit — wait for window reset
                    remaining = response.headers.get("X-RateLimit-Remaining", "?")
                    print(f"  ⚠️  Per-minute limit (remaining={remaining}) — waiting {PER_MINUTE_WAIT}s for reset...")
                    time.sleep(PER_MINUTE_WAIT)
                    # Retry the SAME key after waiting
                    continue

            # Successful response — reset consecutive 429 counter
            consecutive_429[_key_id(key)] = 0

            if response.status_code == 503:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"  ⚠️  503 unavailable — retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data    = response.json()
            choices = data.get("choices") or []

            if not choices:
                print(f"  ⚠️  Empty choices on '{role}'. Raw: {str(data)[:200]}")
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue

            message = choices[0].get("message") or {}
            content = message.get("content")

            if content is None:
                finish_reason = choices[0].get("finish_reason", "unknown")
                print(f"  ⚠️  Null content on '{role}' (finish_reason={finish_reason}) — retrying...")
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue

            key_num = (keys.index(key) + 1) if key in keys else "?"
            print(f"  ✓  [{role}] key#{key_num} → {model.split('/')[1][:25]}")
            return content.strip()

        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  Timeout (attempt {attempt+1}) — retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ❌ Attempt {attempt+1}: {e} — retrying in {wait}s...")
            time.sleep(wait)

    print(f"  ❌ All {retries} attempts failed for role '{role}'.")
    return None


def call_llm_json(role: str, messages: list, **kwargs) -> Optional[dict]:
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
    raw = raw.strip()
    if not raw.startswith("["):
        return None
    salvaged, depth, start = [], 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    salvaged.append(json.loads(raw[start: i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return salvaged if salvaged else None