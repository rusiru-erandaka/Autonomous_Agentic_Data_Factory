"""
llm_client.py
Multi-provider LLM client supporting Groq, Google AI Studio, and OpenRouter.

Provider assignment per role:
  generator    → Groq    (llama-3.1-8b-instant,   14.4K RPD, 30 RPM)
  agent        → Groq    (llama-3.3-70b-versatile,  1K   RPD, 30 RPM)
  agent_backup → Groq    (openai/gpt-oss-120b,      1K   RPD, 30 RPM)
  labeler      → Google  (gemini-2.0-flash,         1.5K RPD, 15 RPM)
  secondary    → Groq    (qwen/qwen3-32b,           1K   RPD, 60 RPM)
  quality_gate → Groq    (llama-3.1-8b-instant,    14.4K RPD, 30 RPM)

OpenRouter 3 keys (150 RPD total) = fallback pool only.

Error handling per error type:
  429           → wait for per-minute reset, retry same model
  401/402/403   → model moved to paid, try next in pool
  502/503/504   → provider gateway issue, try next in pool
  null content  → model unsuitable, try next in pool after 2 occurrences
  timeout       → retry same model with backoff
"""

import os
import time
import json
import requests
from typing import Optional

# ── Provider base URLs ─────────────────────────────────────────────────────────
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GOOGLE_URL  = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
OR_URL      = "https://openrouter.ai/api/v1/chat/completions"

# ── Role → provider + model pool ──────────────────────────────────────────────
# Each role has a PRIMARY provider and an ordered pool of models.
# On failure the system tries the next model in the pool.
# If the entire pool fails, it falls back to OpenRouter.

ROLE_CONFIG = {
    "generator": {
        "provider": "groq",
        "models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ],
        "rpm": 30,
    },
    "agent": {
        "provider": "groq",
        "models": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
        "rpm": 30,
    },
    "agent_backup": {
        "provider": "groq",
        "models": [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
        ],
        "rpm": 30,
    },
    "labeler": {
        "provider": "groq",
        "models": [
            "openai/gpt-oss-20b",
            #"gemini-1.5-flash",
        ],
        "rpm": 15,
    },
    "secondary": {
        "provider": "groq",
        "models": [
            "qwen/qwen3-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-20b",
        ],
        "rpm": 60,
    },
    "quality_gate": {
        "provider": "groq",
        "models": [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
        ],
        "rpm": 30,
    },
}

# OpenRouter fallback pool — used only when primary provider fails entirely
OR_FALLBACK_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
]

# ── Rate limit config ──────────────────────────────────────────────────────────
MIN_SECONDS_BETWEEN_CALLS = 3.0   # global throttle — well under all RPM limits
RETRY_BACKOFF             = [5, 10, 20]
MAX_RETRIES               = 10    # enough to try all pool models + fallback

# ── State ──────────────────────────────────────────────────────────────────────
_last_call_time:    float = 0.0
_active_model_idx:  dict  = {}   # role -> int (pool index)
_null_count:        dict  = {}   # role -> int (consecutive null responses)


# ── Key loaders ────────────────────────────────────────────────────────────────

def _groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("❌ GROQ_API_KEY not set in .env")
    return key

def _google_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("❌ GOOGLE_API_KEY not set in .env")
    return key

def _openrouter_key() -> str:
    """Return first available OpenRouter key — used only as fallback."""
    for i in range(1, 4):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "").strip()
        if k and not k.startswith("sk-or-v1-replace"):
            return k
    raise EnvironmentError("❌ No OpenRouter fallback key found")


# ── Model pool management ──────────────────────────────────────────────────────

def _get_active_model(role: str) -> tuple[str, int]:
    pool = ROLE_CONFIG[role]["models"]
    idx  = min(_active_model_idx.get(role, 0), len(pool) - 1)
    return pool[idx], idx

def _advance_model(role: str) -> bool:
    """Move to next model in pool. Returns False if pool exhausted."""
    pool    = ROLE_CONFIG[role]["models"]
    current = _active_model_idx.get(role, 0)
    if current + 1 < len(pool):
        _active_model_idx[role] = current + 1
        print(f"  ➡️  [{role}] pool[{current+1}/{len(pool)-1}]: {pool[current+1]}")
        _null_count[role] = 0
        return True
    print(f"  ⚠️  [{role}] all pool models tried — falling back to OpenRouter")
    return False

def _reset_model(role: str):
    _active_model_idx[role] = 0
    _null_count[role]       = 0


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _enforce_rate_limit():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    gap     = MIN_SECONDS_BETWEEN_CALLS - elapsed
    if gap > 0:
        time.sleep(gap)
    _last_call_time = time.time()

def _is_permanent_error(status: int, body: str) -> bool:
    """True = model unavailable/moved to paid — switch model, don't retry."""
    if status in (401, 402, 403, 404):
        return True
    if status in (502, 504):
        return True
    if status == 400 and "model" in body.lower():
        return True
    return False

def _build_headers(provider: str, key: str) -> dict:
    if provider == "groq":
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }
    if provider == "google":
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }
    # openrouter
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/rusiru-erandaka/Autonomous_Agentic_Data_Factory",
        "X-Title":       "Agent Behavior Dataset Pipeline",
    }

def _get_url(provider: str) -> str:
    return {"groq": GROQ_URL, "google": GOOGLE_URL, "openrouter": OR_URL}[provider]

def _extract_content(data: dict) -> Optional[str]:
    """Safely extract text content from any OpenAI-compatible response."""
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None or str(content).strip() == "":
        return None
    return str(content).strip()


# ── Core call function ─────────────────────────────────────────────────────────

def _call_provider(
    provider:    str,
    model:       str,
    key:         str,
    messages:    list,
    temperature: float,
    max_tokens:  int,
) -> tuple[Optional[str], int, str]:
    """
    Make one HTTP request to a provider.
    Returns (content_or_None, status_code, raw_body_preview).
    """
    _enforce_rate_limit()

    url     = _get_url(provider)
    headers = _build_headers(provider, key)
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        body = resp.text[:300]

        if resp.status_code == 200:
            data    = resp.json()
            content = _extract_content(data)
            return content, 200, body

        return None, resp.status_code, body

    except requests.exceptions.Timeout:
        return None, 408, "timeout"
    except Exception as e:
        return None, 0, str(e)[:200]


# ── Public API ─────────────────────────────────────────────────────────────────

def call_llm(
    role:        str,
    messages:    list,
    temperature: float = 0.3,
    max_tokens:  int   = 1024,
    json_mode:   bool  = False,
    retries:     int   = MAX_RETRIES,
) -> Optional[str]:
    """
    Call LLM for the given role with automatic provider + model pool fallback.

    Flow:
      1. Try primary provider (Groq or Google) using active pool model
      2. On permanent error → advance to next pool model
      3. On null content ×2 → advance to next pool model
      4. On 429 → wait for per-minute reset
      5. If entire primary pool exhausted → try OpenRouter fallback
      6. Return None only if everything fails
    """
    if role not in ROLE_CONFIG:
        raise ValueError(f"Unknown role '{role}'. Valid: {list(ROLE_CONFIG.keys())}")

    cfg      = ROLE_CONFIG[role]
    provider = cfg["provider"]

    if json_mode:
        messages = messages.copy()
        messages[-1]["content"] += (
            "\n\nIMPORTANT: Return ONLY valid JSON. "
            "No markdown fences, no preamble, no explanation."
        )

    used_fallback = False

    for attempt in range(retries):
        # ── Choose model + key ─────────────────────────────────────────────────
        pool_exhausted = _active_model_idx.get(role, 0) >= len(cfg["models"])

        if pool_exhausted and not used_fallback:
            # Primary pool exhausted — try OpenRouter fallback
            used_fallback     = True
            current_provider  = "openrouter"
            or_idx            = attempt % len(OR_FALLBACK_MODELS)
            current_model     = OR_FALLBACK_MODELS[or_idx]
            try:
                current_key = _openrouter_key()
            except EnvironmentError:
                print("  ❌ No OpenRouter fallback available either.")
                return None
        elif pool_exhausted and used_fallback:
            print(f"  ❌ All providers exhausted for role '{role}'.")
            return None
        else:
            current_provider = provider
            current_model, pool_idx = _get_active_model(role)
            try:
                if provider == "groq":
                    current_key = _groq_key()
                elif provider == "google":
                    current_key = _google_key()
                else:
                    current_key = _openrouter_key()
            except EnvironmentError as e:
                print(f"  ❌ {e}")
                return None

        # ── Make the call ──────────────────────────────────────────────────────
        content, status, body_preview = _call_provider(
            current_provider, current_model, current_key,
            messages, temperature, max_tokens,
        )

        short_model = current_model.split("/")[-1][:30]

        # ── Handle response ────────────────────────────────────────────────────
        if status == 200 and content:
            _null_count[role] = 0
            src = "fallback" if used_fallback else current_provider
            pidx = _active_model_idx.get(role, 0)
            print(f"  ✓  [{role}] {src} → {short_model}")
            return content

        if status == 429:
            wait = 65 if current_provider in ("groq", "google") else 70
            print(f"  ⚠️  429 [{role}] {short_model} — waiting {wait}s...")
            time.sleep(wait)
            continue

        if _is_permanent_error(status, body_preview):
            print(f"  ⚠️  {status} [{role}] '{short_model}' — permanent error, advancing pool")
            if not used_fallback:
                pool_ok = _advance_model(role)
                if not pool_ok:
                    # Will switch to fallback on next iteration
                    _active_model_idx[role] = len(cfg["models"])
            continue

        if status == 503:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  503 [{role}] — retrying in {wait}s...")
            time.sleep(wait)
            continue

        if status == 200 and content is None:
            # Null content
            _null_count[role] = _null_count.get(role, 0) + 1
            print(f"  ⚠️  Null content [{role}] '{short_model}' "
                  f"(count={_null_count[role]})")
            if _null_count.get(role, 0) >= 2:
                if not used_fallback:
                    pool_ok = _advance_model(role)
                    if not pool_ok:
                        _active_model_idx[role] = len(cfg["models"])
                _null_count[role] = 0
            else:
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
            continue

        if status == 408:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"  ⚠️  Timeout [{role}] — retrying in {wait}s...")
            time.sleep(wait)
            continue

        # Unknown error
        wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
        print(f"  ❌ [{role}] status={status}: {body_preview[:100]} — retrying in {wait}s...")
        time.sleep(wait)

    print(f"  ❌ All {retries} attempts failed for role '{role}'.")
    return None


def call_llm_json(role: str, messages: list, **kwargs) -> Optional[dict]:
    """call_llm with json_mode=True and auto-parsing + truncation repair."""
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


def get_active_models_summary() -> dict:
    """Return current active model per role — used for startup logging."""
    summary = {}
    for role, cfg in ROLE_CONFIG.items():
        idx   = min(_active_model_idx.get(role, 0), len(cfg["models"]) - 1)
        summary[role] = {
            "provider": cfg["provider"],
            "model":    cfg["models"][idx],
            "pool":     cfg["models"],
        }
    return summary