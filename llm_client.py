"""
llm_client.py
Multi-provider LLM client — Groq primary, OpenRouter fallback.

Google AI Studio removed — too many 429 issues on free tier.

Provider assignment:
  All roles → Groq (primary)
  All roles → OpenRouter (fallback, 3 keys × 50 req/day = 150 req/day)

Groq free limits:
  llama-3.3-70b-versatile    : 1K  RPD, 30 RPM
  llama-3.1-8b-instant       : 14.4K RPD, 30 RPM
  qwen/qwen3-32b             : 1K  RPD, 60 RPM
  meta-llama/llama-4-scout   : 1K  RPD, 30 RPM
  openai/gpt-oss-120b        : 1K  RPD, 30 RPM

Role pools — tried top to bottom on failure:
  generator    → 8b-instant (14.4K RPD, fast)
  agent        → gpt-oss-120b (strongest coding baseline)
  agent_backup → 70b-versatile (execution fallback)
  labeler      → 70b-versatile (strong evaluation)
  secondary    → qwen3-32b (60 RPM, different model for diversity)
  quality_gate → qwen3-32b (60 RPM syntax/bug-checking role)
"""

import os
import time
import json
import requests
from typing import Optional

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OR_URL   = "https://openrouter.ai/api/v1/chat/completions"

# ── Role → model pool (Groq model IDs) ────────────────────────────────────────
ROLE_CONFIG = {
    "generator": {
        "provider": "groq",
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
        ],
        "rpm": 30,
    },
    "agent": {
        "provider": "groq",
        "models": [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
        "rpm": 30,
    },
    "agent_backup": {
        "provider": "groq",
        "models": [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
        "rpm": 30,
    },
    "labeler": {
        "provider": "groq",
        "models": [
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
        ],
        "rpm": 30,
    },
    "secondary": {
        "provider": "groq",
        "models": [
            "qwen/qwen3-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.1-8b-instant",
        ],
        "rpm": 60,
    },
    "quality_gate": {
        "provider": "groq",
        "models": [
            "qwen/qwen3-32b",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
        ],
        "rpm": 60,
    },
}

# ── OpenRouter fallback pool ───────────────────────────────────────────────────
OR_FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
]

# ── Rate limit config ──────────────────────────────────────────────────────────
MIN_SECONDS_BETWEEN_CALLS = 4.0   # 15 req/min — under all RPM limits
RETRY_BACKOFF             = [5, 10, 20]
MAX_RETRIES               = 10

# ── State ──────────────────────────────────────────────────────────────────────
_last_call_time:   float = 0.0
_active_model_idx: dict  = {}
_null_count:       dict  = {}


def _groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("❌ GROQ_API_KEY not set in .env")
    return key

def _openrouter_key() -> str:
    """Return first available OpenRouter key."""
    for i in range(1, 4):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "").strip()
        if k and not k.startswith("sk-or-v1-replace"):
            return k
    raise EnvironmentError("❌ No OpenRouter fallback key found")

def _get_active_model(role: str) -> tuple:
    pool = ROLE_CONFIG[role]["models"]
    idx  = min(_active_model_idx.get(role, 0), len(pool) - 1)
    return pool[idx], idx

def _advance_model(role: str) -> bool:
    pool    = ROLE_CONFIG[role]["models"]
    current = _active_model_idx.get(role, 0)
    if current + 1 < len(pool):
        _active_model_idx[role] = current + 1
        print(f"  ➡️  [{role}] pool[{current+1}/{len(pool)-1}]: {pool[current+1]}")
        _null_count[role] = 0
        return True
    print(f"  ⚠️  [{role}] all Groq models tried — switching to OpenRouter fallback")
    return False

def _is_permanent_error(status: int, body: str) -> bool:
    if status in (401, 402, 403, 404):
        return True
    if status in (502, 504):
        return True
    if status == 400 and "model" in body.lower():
        return True
    return False

def _enforce_rate_limit():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    gap     = MIN_SECONDS_BETWEEN_CALLS - elapsed
    if gap > 0:
        time.sleep(gap)
    _last_call_time = time.time()

def _post(url: str, headers: dict, payload: dict) -> tuple:
    """Make one HTTP request. Returns (content_or_None, status_code, body_preview)."""
    _enforce_rate_limit()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        body = resp.text[:300]
        if resp.status_code == 200:
            data    = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return None, 200, body
            msg     = choices[0].get("message") or {}
            content = msg.get("content")
            if not content or str(content).strip() == "":
                return None, 200, f"null_content|finish={choices[0].get('finish_reason','?')}"
            return str(content).strip(), 200, ""
        return None, resp.status_code, body
    except requests.exceptions.Timeout:
        return None, 408, "timeout"
    except Exception as e:
        return None, 0, str(e)[:200]

def call_llm(
    role:        str,
    messages:    list,
    temperature: float = 0.3,
    max_tokens:  int   = 1024,
    json_mode:   bool  = False,
    retries:     int   = MAX_RETRIES,
) -> Optional[str]:
    """
    Call LLM with Groq primary + OpenRouter fallback.

    Error handling:
      429            → wait for per-minute reset, retry SAME model
      401/402/502    → permanent error, advance to next pool model
      null content×2 → model unsuitable, advance to next pool model
      timeout        → retry same model with backoff
      Groq pool exhausted → try OpenRouter fallback models
    """
    if role not in ROLE_CONFIG:
        raise ValueError(f"Unknown role '{role}'. Valid: {list(ROLE_CONFIG.keys())}")

    if json_mode:
        messages = messages.copy()
        messages[-1]["content"] += (
            "\n\nIMPORTANT: Return ONLY valid JSON. "
            "No markdown fences, no preamble, no explanation."
        )

    groq_key     = _groq_key()
    pool_size    = len(ROLE_CONFIG[role]["models"])
    using_fallback = False

    for attempt in range(retries):
        # Determine which provider/model to use
        pool_exhausted = _active_model_idx.get(role, 0) >= pool_size

        if pool_exhausted:
            # Use OpenRouter fallback
            using_fallback = True
            or_idx    = (attempt - pool_size) % len(OR_FALLBACK_MODELS)
            model     = OR_FALLBACK_MODELS[or_idx]
            try:
                key = _openrouter_key()
            except EnvironmentError:
                print(f"  ❌ No fallback available for '{role}'")
                return None
            url     = OR_URL
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://github.com/rusiru-erandaka/Autonomous_Agentic_Data_Factory",
                "X-Title":       "Agent Behavior Dataset Pipeline",
            }
        else:
            # Use Groq primary
            using_fallback = False
            model, pool_idx = _get_active_model(role)
            key     = groq_key
            url     = GROQ_URL
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
            }

        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }

        content, status, body = _post(url, headers, payload)
        short_model = model.split("/")[-1][:28]
        provider    = "fallback" if using_fallback else "groq"

        # ── Handle response ────────────────────────────────────────────────────
        if status == 200 and content:
            _null_count[role] = 0
            pidx = _active_model_idx.get(role, 0)
            print(f"  ✓  [{role}] {provider} pool[{pidx}/{pool_size-1}] → {short_model}")
            return content

        if status == 429:
            wait = 65
            print(f"  ⚠️  429 [{role}] {short_model} — waiting {wait}s...")
            time.sleep(wait)
            continue   # retry SAME model

        if _is_permanent_error(status, body):
            print(f"  ⚠️  {status} [{role}] '{short_model}' — permanent error")
            if not using_fallback:
                ok = _advance_model(role)
                if not ok:
                    _active_model_idx[role] = pool_size  # trigger fallback
            continue

        if status == 503:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  503 — retrying in {wait}s...")
            time.sleep(wait)
            continue

        if status == 200 and content is None:
            _null_count[role] = _null_count.get(role, 0) + 1
            print(f"  ⚠️  Null [{role}] '{short_model}' (count={_null_count[role]}) {body}")
            if _null_count.get(role, 0) >= 2:
                if not using_fallback:
                    ok = _advance_model(role)
                    if not ok:
                        _active_model_idx[role] = pool_size
                _null_count[role] = 0
            else:
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
            continue

        if status == 408:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  Timeout — retrying in {wait}s...")
            time.sleep(wait)
            continue

        wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
        print(f"  ❌ [{role}] status={status}: {body[:80]} — retrying in {wait}s...")
        time.sleep(wait)

    print(f"  ❌ All {retries} attempts failed for '{role}'.")
    return None


def call_llm_json(role: str, messages: list, **kwargs) -> Optional[dict]:
    """call_llm with json_mode=True and auto-parsing + repair."""
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
            print(f"  ⚠️  JSON truncated — salvaged {len(salvaged)} items.")
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


def get_active_models_summary() -> dict:
    return {
        role: {
            "provider": cfg["provider"],
            "model":    cfg["models"][min(_active_model_idx.get(role, 0), len(cfg["models"]) - 1)],
            "pool":     cfg["models"],
        }
        for role, cfg in ROLE_CONFIG.items()
    }
