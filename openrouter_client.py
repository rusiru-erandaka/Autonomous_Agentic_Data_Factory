"""
openrouter_client.py

Stage-based API key assignment — each pipeline stage uses a dedicated key:

  Stage 1 — Task Generation  → OPENROUTER_API_KEY_1
    roles: generator
    ~8 calls/day (signal conversion + mutations)

  Stage 2 — Agent Execution  → OPENROUTER_API_KEY_2  (heaviest stage)
    roles: agent, agent_backup
    ~48 calls/day (12 tasks × 4 steps)

  Stage 3 — Labeling         → OPENROUTER_API_KEY_3
    roles: labeler, secondary
    ~24 calls/day (12 tasks × 2 labelers)

  Stages 4 & 5 (Quality Gate, HF Upload) make zero LLM calls.

Why this works better than round-robin:
  - Each key stays well under 50 req/day
  - No cross-stage interference
  - Per-minute limit (20 req/min) per key is respected by MIN_SECONDS_BETWEEN_CALLS
  - On 429: wait for per-minute reset on same key (don't switch — switching
    doesn't help for per-minute limits since all keys share account-level quota)
"""

import os
import time
import json
import requests
from typing import Optional

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Stage → Role → Model mapping ──────────────────────────────────────────────
MODELS = {
    # Stage 1: Task generation
    "generator":    "nvidia/nemotron-3-super-120b-a12b:free",

    # Stage 2: Agent execution (strongest free model for best task solving)
    "agent":        "nvidia/nemotron-3-super-120b-a12b:free",
    "agent_backup": "nvidia/nemotron-3-super-120b-a12b:free",

    # Stage 3: Labeling (two different models for genuine dual-label)
    "labeler":      "nvidia/nemotron-3-super-120b-a12b:free",
    "secondary":    "nvidia/nemotron-3-super-120b-a12b:free",

    # Utility (uses KEY_1 by default)
    "quality_gate": "nvidia/nemotron-3-super-120b-a12b:free",
}

# ── Role → which API key to use ───────────────────────────────────────────────
# Roles are assigned to keys based on which pipeline stage they belong to.
ROLE_TO_KEY_SLOT = {
    "generator":    1,   # Stage 1 — Task Generation   → KEY_1
    "quality_gate": 1,   # Stage 1 utility             → KEY_1
    "secondary":    1,   # Stage 3 secondary labeler   → KEY_1 (shares with generator)
                         # Generator uses ~8 req/day, leaving ~42 for secondary (~7 traces)
    "agent":        2,   # Stage 2 — Agent Execution   → KEY_2
    "agent_backup": 2,   # Stage 2 fallback            → KEY_2
    "labeler":      3,   # Stage 3 — Primary labeler   → KEY_3
}

# ── Rate limit config ──────────────────────────────────────────────────────────
MIN_SECONDS_BETWEEN_CALLS = 6.0   # 10 req/min — under the 20 req/min limit
PER_MINUTE_WAIT           = 70    # wait after per-minute 429
MAX_RETRIES               = 5
RETRY_BACKOFF             = [5, 10, 20]

_last_call_time: float = 0.0


def _load_key(slot: int) -> str:
    """
    Load a specific API key by slot number (1, 2, or 3) from environment.
    Called at request time — never at import time.
    """
    key = os.environ.get(f"OPENROUTER_API_KEY_{slot}", "").strip()
    if not key or key.startswith("sk-or-v1-replace"):
        # Fallback: if the specific slot isn't set, try others in order
        for fallback in [1, 2, 3]:
            k = os.environ.get(f"OPENROUTER_API_KEY_{fallback}", "").strip()
            if k and not k.startswith("sk-or-v1-replace"):
                print(f"  ⚠️  KEY_{slot} not set — falling back to KEY_{fallback}")
                return k
        raise EnvironmentError(
            f"\n❌ OPENROUTER_API_KEY_{slot} not found in environment.\n"
            f"   Make sure your .env file has:\n"
            f"   OPENROUTER_API_KEY_1=sk-or-v1-...\n"
            f"   OPENROUTER_API_KEY_2=sk-or-v1-...\n"
            f"   OPENROUTER_API_KEY_3=sk-or-v1-...\n"
        )
    return key


def _key_display(key: str) -> str:
    """Short display string — never logs the full key."""
    return key[:16] + "..."


def _enforce_rate_limit():
    """Enforce minimum gap between ALL calls to stay under 20 req/min."""
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
    Call an OpenRouter free model using stage-based key assignment.

    Each role maps to a fixed API key slot:
      generator/quality_gate → KEY_1  (Stage 1)
      agent/agent_backup     → KEY_2  (Stage 2)
      labeler/secondary      → KEY_3  (Stage 3)

    On 429 (per-minute limit): wait PER_MINUTE_WAIT seconds then retry
    same key. Switching keys does NOT help for per-minute limits since
    all keys on the same account share the 20 req/min bucket.
    """
    model = MODELS.get(role)
    if not model:
        raise ValueError(f"Unknown role '{role}'. Valid: {list(MODELS.keys())}")

    key_slot = ROLE_TO_KEY_SLOT.get(role, 1)
    key      = _load_key(key_slot)

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
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/your-username/agent-behavior-dataset",
        "X-Title":       "Agent Behavior Dataset Pipeline",
    }

    for attempt in range(retries):
        _enforce_rate_limit()

        try:
            response = requests.post(
                BASE_URL, headers=headers, json=payload, timeout=120
            )

            if response.status_code == 429:
                remaining   = response.headers.get("X-RateLimit-Remaining", "?")
                retry_after = response.headers.get("Retry-After", "")

                # Determine wait time — use server's Retry-After if available
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after) + 5
                else:
                    wait = PER_MINUTE_WAIT

                print(f"  ⚠️  429 on KEY_{key_slot} [{role}] "
                      f"(remaining={remaining}) — waiting {wait}s...")
                time.sleep(wait)
                continue   # retry same key after waiting

            if response.status_code == 503:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"  ⚠️  503 model unavailable — retrying in {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            data    = response.json()
            choices = data.get("choices") or []

            if not choices:
                raw_preview = str(data)[:200]
                print(f"  ⚠️  Empty choices on '{role}'. Raw: {raw_preview}")
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue

            message = choices[0].get("message") or {}
            content = message.get("content")

            if content is None:
                finish_reason = choices[0].get("finish_reason", "unknown")
                print(f"  ⚠️  Null content on '{role}' "
                      f"(finish_reason={finish_reason}) — retrying...")
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
                continue

            print(f"  ✓  [{role}] KEY_{key_slot} → {model.split('/')[1][:25]}")
            return content.strip()

        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  Timeout (attempt {attempt+1}) — retrying in {wait}s...")
            time.sleep(wait)

        except Exception as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ❌ Attempt {attempt+1}: {e} — retrying in {wait}s...")
            time.sleep(wait)

    print(f"  ❌ All {retries} attempts failed for role '{role}' on KEY_{key_slot}.")
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