"""
openrouter_client.py
Unified LLM client for all pipeline stages via OpenRouter free models.

Key design: a GLOBAL rate limiter enforces minimum 4s between every call,
keeping total throughput under 15 req/min — safely below the free tier limit
of ~20 req/min regardless of which model or role is used.
"""

import os
import time
import json
import requests
from typing import Optional

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Model assignments ──────────────────────────────────────────────────────────
MODELS = {
    "agent":        "meta-llama/llama-3.3-70b-instruct:free",   # agent executor
    "agent_backup": "openai/gpt-oss-120b:free",                  # fallback executor
    "labeler":      "nvidia/nemotron-3-super-120b-a12b:free",    # primary labeler
    "secondary":    "qwen/qwen3-next-80b-a3b-instruct:free",     # secondary labeler
    "generator":    "meta-llama/llama-3.3-70b-instruct:free",    # task generator (same as agent — reliable text model)
    "quality_gate": "nvidia/nemotron-nano-9b-v2:free",           # quality checker
}

# ── Rate limit config ──────────────────────────────────────────────────────────
# OpenRouter free tier = ~20 requests/minute TOTAL across ALL models.
# 4s gap between every call = max 15 req/min, safely under the limit.
MIN_SECONDS_BETWEEN_CALLS = 4.0
RATE_LIMIT_WAIT           = 65
MAX_RETRIES               = 4
RETRY_BACKOFF             = [2, 5, 15, 30]

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
    max_tokens: int = 2048,
    json_mode: bool = False,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Call an OpenRouter free model.
    Returns response text string, or None if all retries fail.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    model = MODELS.get(role)
    if not model:
        raise ValueError(f"Unknown role '{role}'. Valid roles: {list(MODELS.keys())}")

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
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/your-username/agent-behavior-dataset",
        "X-Title":       "Agent Behavior Dataset Pipeline",
    }

    for attempt in range(retries):
        # Enforce global rate limit before every single request
        _enforce_rate_limit()

        try:
            response = requests.post(
                BASE_URL, headers=headers, json=payload, timeout=120
            )

            # ── 429: rate limit hit despite our throttle — wait and retry ─────
            if response.status_code == 429:
                print(f"  ⚠️  429 on '{role}' — waiting {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
                continue

            # ── 503: model temporarily unavailable ────────────────────────────
            if response.status_code == 503:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"  ⚠️  503 model unavailable — retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            # ── Safely extract content ─────────────────────────────────────────
            # Some models return None content or omit the field entirely.
            # Never call .strip() directly on data["choices"][0]["message"]["content"]
            # as it crashes when content is None.
            choices = data.get("choices") or []
            if not choices:
                print(f"  ⚠️  No choices in response for '{role}'. Raw: {str(data)[:200]}")
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue

            message = choices[0].get("message") or {}
            content = message.get("content")

            if content is None:
                # Model returned null content — happens when it hits its own
                # content filter or returns a tool_call block instead of text.
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
    Wrapper around call_llm with json_mode=True that auto-parses the result.
    If normal parsing fails due to truncation, attempts JSON repair before giving up.
    Returns a parsed dict or list, or None on failure.
    """
    raw = call_llm(role, messages, json_mode=True, **kwargs)
    if raw is None:
        return None
    try:
        clean = raw.strip()
        # Strip accidental markdown fences
        if clean.startswith("```"):
            parts = clean.split("```")
            clean = parts[1] if len(parts) > 1 else clean
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        # Normal parse failed — try to salvage truncated arrays
        salvaged = repair_truncated_json(raw)
        if salvaged:
            print(f"  ⚠️  JSON truncated — salvaged {len(salvaged)} item(s) from partial response.")
            return salvaged
        print(f"  ❌ JSON parse error: {e} | Raw preview: {raw[:300]}")
        return None


def repair_truncated_json(raw: str) -> Optional[list]:
    """
    Attempt to salvage a truncated JSON array.
    When a model hits its token limit mid-response, the JSON is cut off.
    This function extracts whatever complete objects exist before the cutoff.
    Returns a list of valid objects, or None if nothing salvageable.
    """
    raw = raw.strip()

    # Only attempt repair on arrays
    if not raw.startswith("["):
        return None

    salvaged = []
    depth    = 0
    start    = None

    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i   # start of a new top-level object
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                # We have a complete object — try to parse it
                candidate = raw[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    salvaged.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    return salvaged if salvaged else None