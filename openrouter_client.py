"""
openrouter_client.py
Unified LLM client for all pipeline stages via OpenRouter free models.
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
    "agent":        "meta-llama/llama-3.3-70b-instruct:free",       # agent executor
    "agent_backup": "openai/gpt-oss-120b:free",                      # fallback executor
    "labeler":      "nvidia/nemotron-3-super-120b-a12b:free",        # primary labeler
    "secondary":    "qwen/qwen3-next-80b-a3b-instruct:free",         # secondary labeler
    "generator":    "nvidia/nemotron-3-nano-30b-a3b:free",           # task generator
    "quality_gate": "minimax/minimax-m2.5:free",                    # quality checker
}

# ── Rate limit config ──────────────────────────────────────────────────────────
RATE_LIMIT_WAIT   = 65   # seconds to wait on 429
MAX_RETRIES       = 4
RETRY_BACKOFF     = [2, 5, 15, 30]   # seconds between retries


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

    Args:
        role:        Key from MODELS dict (agent / labeler / generator / etc.)
        messages:    Standard OpenAI-style message list
        temperature: Sampling temperature
        max_tokens:  Max output tokens
        json_mode:   If True, adds instruction to return pure JSON
        retries:     Number of retry attempts

    Returns:
        Response text string or None on failure
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    model = MODELS.get(role)
    if not model:
        raise ValueError(f"Unknown role '{role}'. Choose from: {list(MODELS.keys())}")

    # Inject JSON instruction if needed
    if json_mode:
        messages = messages.copy()
        messages[-1]["content"] += "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no explanation, no backticks."

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
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=120)

            if response.status_code == 429:
                print(f"  ⚠️  Rate limit hit on '{role}'. Waiting {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
                continue

            if response.status_code == 503:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"  ⚠️  Model unavailable (503). Retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return content.strip()

        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  Timeout on attempt {attempt+1}. Retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ❌ Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  ❌ All {retries} attempts failed for role '{role}'.")
    return None


def call_llm_json(role: str, messages: list, **kwargs) -> Optional[dict]:
    """
    Wrapper that calls call_llm with json_mode=True and auto-parses the result.
    Returns parsed dict or None.
    """
    raw = call_llm(role, messages, json_mode=True, **kwargs)
    if raw is None:
        return None
    try:
        # Strip any accidental markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}\n  Raw: {raw[:200]}")
        return None
