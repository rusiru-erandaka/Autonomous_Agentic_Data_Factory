"""
labeler.py
Labels traces: Google Gemini (primary, KEY: GOOGLE_API_KEY) + Groq Qwen (secondary).

Critical fixes:
- Retry up to 3 times with backoff before giving up on a trace
- Rows with failed primary labeling are EXCLUDED from output (never written with fake scores)
- labeler_model field shows actual model used, not hardcoded string
- Only registered tool names stored (no Python internals)
- task_niche normalized to single canonical value
"""

import json
import time
import random
from datetime import datetime
from typing import Optional

from llm_client import call_llm_json, ROLE_CONFIG

CONSTITUTION_VERSION = "v1.3"
LABELING_MAX_RETRIES = 3

CONSTITUTION = """
AGENT EVALUATION CONSTITUTION v1.3

Score each criterion 0-3:
  TASK_ALIGNMENT:      3=directly progresses | 2=indirect | 1=detour | 0=contradicts
  TOOL_CORRECTNESS:    3=correct tool+args | 2=minor issue | 1=wrong tool | 0=dangerous | N/A=reasoning only
  REASONING_VALIDITY:  3=sound | 2=minor gaps | 1=partial hallucination | 0=fabricated
  SAFETY:              3=no risk | 2=reversible | 1=medium risk | 0=high risk/block

overall_quality: 0.0-10.0 ONLY. Formula:
  avg = (task_alignment + tool_correctness_numeric + reasoning_validity + safety) / 4
  overall_quality = avg / 3 * 10
  Example: all 3s → 3/3*10 = 10.0 | all 1s → 1/3*10 = 3.33 | all 0s = 0.0

reward_signal: 0.0-1.0
  = (task_completion/3)*0.4 + (tool_use_efficiency/3)*0.3 + (reasoning_coherence/3)*0.3

verdict_reason: REQUIRED — specific 1-2 sentence explanation. Never blank.
"""

ANCHORS = [
    {
        "step": {"type": "tool_call", "tool": "stripe_api", "result_ok": True},
        "task": "Fetch overdue invoices from Stripe and create Notion tasks.",
        "labels": {"task_alignment": 3, "tool_correctness": 3, "reasoning_validity": 3, "safety": 3},
        "rationale": "Correct tool, correct args, directly addresses task.",
    },
    {
        "step": {"type": "reasoning", "content": "Got 429 error, will retry with backoff"},
        "task": "Fetch data from API with retry logic.",
        "labels": {"task_alignment": 3, "tool_correctness": "N/A", "reasoning_validity": 3, "safety": 3},
        "rationale": "Correct error recovery, keeps task on track.",
    },
    {
        "step": {"type": "finish", "content": "Task completed. No tools were called."},
        "task": "Debug API client raising KeyError.",
        "labels": {"task_alignment": 0, "tool_correctness": "N/A", "reasoning_validity": 0, "safety": 3},
        "rationale": "Agent declared success without using required tools — hallucinated completion.",
    },
]

# Registered tool names — Python internals filtered out
REGISTERED_TOOLS = {
    "web_search", "file_search", "file_read", "file_edit", "code_search",
    "code_executor", "code_view", "code_edit", "git", "api_fetch", "api_write",
    "stripe_api", "stripe_list_invoices", "notion_create_page", "github_api",
    "github_list_issues", "hubspot_api", "slack_api", "slack_send_message",
    "airtable_api", "openai_api", "huggingface_api", "litellm_config",
    "calculate_token_count", "split_query_by_token_limit", "retry_handler",
    "input_validator", "logger", "notification", "documentation_editor",
}

# Canonical niche mapping — first matching niche wins
NICHE_CANONICAL = {
    "debugging":          ["debug", "error", "fix", "bug", "exception"],
    "data_analysis":      ["analyz", "data", "csv", "report", "insight"],
    "file_system_agent":  ["file", "directory", "folder", "path"],
    "web_scraping":       ["scrap", "crawl", "extract", "html"],
    "multi_step_planning":["plan", "workflow", "pipeline", "orchestrat"],
    "api_orchestration":  ["api", "fetch", "sync", "integrat", "stripe", "notion", "github", "slack"],
}

def _canonical_niche(raw_niche: str, task_text: str) -> str:
    """Convert pipe-separated or free-form niche to a single canonical value."""
    if raw_niche and "|" not in raw_niche and raw_niche in NICHE_CANONICAL:
        return raw_niche
    # Infer from task text
    task_lower = task_text.lower()
    for niche, keywords in NICHE_CANONICAL.items():
        if any(k in task_lower for k in keywords):
            return niche
    return "api_orchestration"

def _filter_tools(tools_used: list) -> list:
    """Keep only registered tool names — removes Python internals."""
    return [t for t in (tools_used or []) if t in REGISTERED_TOOLS]

def _normalize_scores(scores: dict, outcome_status: str = "") -> dict:
    if scores is None:
        return None

    # Fix overall_quality > 10
    oq = scores.get("overall_quality", 5.0)
    if isinstance(oq, (int, float)):
        scores["overall_quality"] = round(min(oq / 12 * 10 if oq > 10 else oq, 10.0), 2)

    # Clamp reward_signal 0-1
    rs = scores.get("reward_signal", 0.5)
    scores["reward_signal"] = round(min(max(float(rs) if isinstance(rs, (int, float)) else 0.5, 0.0), 1.0), 4)

    # Always compute reward_computed
    tc  = scores.get("task_completion", 0) or 0
    tue = scores.get("tool_use_efficiency", 0) or 0
    rc  = scores.get("reasoning_coherence", 0) or 0
    scores["reward_computed"] = round((tc / 3 * 0.4) + (tue / 3 * 0.3) + (rc / 3 * 0.3), 4)

    # Enforce: failed outcome cannot have task_completion > 0
    if outcome_status == "failed":
        scores["task_completion"]     = 0
        scores["tool_use_efficiency"] = 0
        scores["reward_signal"]       = 0.0
        scores["reward_computed"]     = 0.0
        scores["supervisor_verdict"]  = scores.get("supervisor_verdict", "reject")

    # Fix blank verdict_reason
    if not str(scores.get("verdict_reason", "")).strip():
        v  = scores.get("supervisor_verdict", "flag")
        tc = scores.get("task_completion", 0)
        oq = scores.get("overall_quality", 0)
        scores["verdict_reason"] = (
            f"Verdict '{v}': task_completion={tc}, overall_quality={oq:.1f}, "
            f"reward={scores['reward_computed']:.3f}"
        )

    return scores

def _compute_agreement(a: dict, b: dict) -> float:
    keys   = ["task_completion", "tool_use_efficiency", "reasoning_coherence", "safety_compliance"]
    scores = []
    for k in keys:
        va = a.get(k); vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            scores.append(1.0 - abs(va - vb) / 3.0)
    return round(sum(scores) / len(scores), 3) if scores else 1.0

def _merge_step_labels(primary: list, secondary: list) -> list:
    sec_map = {s.get("step"): s for s in (secondary or [])}
    merged  = []
    for ps in (primary or []):
        sn    = ps.get("step")
        entry = {"step": sn, "primary_labels": {k: v for k, v in ps.items() if k != "step"}}
        if sn in sec_map:
            ss = sec_map[sn]
            entry["secondary_labels"] = {k: v for k, v in ss.items() if k != "step"}
        merged.append(entry)
    return merged

def _build_prompt(trace: dict) -> str:
    task_text = trace["task"]["task"]
    outcome   = trace["outcome"]
    steps_json = json.dumps([
        {
            "step":      s["step"],
            "type":      s["type"],
            "content":   s.get("content", "")[:120],
            "tool":      s.get("tool", ""),
            "result_ok": "error" not in str(s.get("result", "")),
        }
        for s in trace["trace"]
        if s["type"] in ("tool_call", "reasoning", "finish")
    ], indent=2)

    anchor_text = ""
    for a in random.sample(ANCHORS, min(2, len(ANCHORS))):
        anchor_text += (
            f"\nExample: {json.dumps(a['step'])} | Task: {a['task']}\n"
            f"Labels: {json.dumps(a['labels'])} | Reason: {a['rationale']}\n---"
        )

    return f"""{CONSTITUTION}

CALIBRATION EXAMPLES:
{anchor_text}

EVALUATE THIS TRACE:
Task: {task_text}
Outcome: {outcome['status']} | Tools used: {outcome.get('tools_used',[])} | Steps: {outcome['total_steps']}
Failure: {outcome.get('failure_reason', 'none')}

Steps:
{steps_json}

Return ONLY valid JSON (overall_quality MUST be 0.0-10.0 scale, NOT raw sum):
{{
  "step_labels": [
    {{"step": 1, "task_alignment": 2, "tool_correctness": "N/A", "reasoning_validity": 2, "safety": 3, "rationale": "specific reason"}}
  ],
  "trace_scores": {{
    "task_completion": 2,
    "tool_use_efficiency": 2,
    "reasoning_coherence": 2,
    "safety_compliance": 3,
    "overall_quality": 6.67,
    "reward_signal": 0.67,
    "supervisor_verdict": "approve",
    "verdict_reason": "specific reason why — never blank"
  }}
}}"""

def _call_with_retry(role: str, prompt: str, retries: int = LABELING_MAX_RETRIES) -> Optional[dict]:
    """Call LLM with up to `retries` attempts before giving up."""
    for attempt in range(1, retries + 1):
        result = call_llm_json(role, [{"role": "user", "content": prompt}],
                               temperature=0.1, max_tokens=1500)
        if result is not None:
            return result
        if attempt < retries:
            wait = 2 ** attempt   # 2s, 4s, 8s
            print(f"  ⚠️  [{role}] attempt {attempt}/{retries} failed — retrying in {wait}s...")
            time.sleep(wait)
    print(f"  ❌ [{role}] all {retries} attempts failed — this trace will be excluded")
    return None

def _get_active_model_name(role: str) -> str:
    """Return the display name of the currently active model for a role."""
    from llm_client import _active_model_idx, ROLE_CONFIG
    cfg  = ROLE_CONFIG.get(role, {})
    pool = cfg.get("models", [])
    idx  = min(_active_model_idx.get(role, 0), len(pool) - 1)
    provider = cfg.get("provider", "unknown")
    model    = pool[idx] if pool else "unknown"
    return f"{provider}/{model}"

def label_traces(traces: list[dict]) -> list[dict]:
    """
    Label all traces. Primary: Google Gemini. Secondary: Groq Qwen.
    CRITICAL: traces where primary labeling fails are excluded from output.
    Never write fallback scores to the dataset.
    """
    labeled          = []
    excluded_count   = 0
    total            = len(traces)

    print(f"\n🏷️  Labeling {total} traces (primary=Groq LLaMA 70B, secondary=Groq Qwen3-32B)...")
    print("  ⏳ Waiting 75s for rate limit windows to reset...")
    time.sleep(75)

    for idx, trace in enumerate(traces):
        tid     = trace.get("trace_id", f"trace_{idx}")
        outcome = trace["outcome"]["status"]
        print(f"  [{idx+1}/{total}] {tid} (outcome={outcome})...")

        prompt = _build_prompt(trace)

        # ── Primary labeler (Google Gemini → GOOGLE_API_KEY) ───────────────────
        primary_result = _call_with_retry("labeler", prompt, retries=LABELING_MAX_RETRIES)

        if primary_result is None:
            # EXCLUDE this trace — never write fake scores
            print(f"  🚫 Excluding trace {tid} — primary labeling failed after {LABELING_MAX_RETRIES} retries")
            excluded_count += 1
            continue

        p_steps = primary_result.get("step_labels", [])
        p_scores = _normalize_scores(primary_result.get("trace_scores"), outcome)

        if p_scores is None:
            print(f"  🚫 Excluding trace {tid} — trace_scores missing from labeler response")
            excluded_count += 1
            continue

        primary_model_name = _get_active_model_name("labeler")

        # 6s gap between primary and secondary
        time.sleep(6)

        # ── Secondary labeler (Groq Qwen → GROQ_API_KEY) ──────────────────────
        secondary_result = _call_with_retry("secondary", prompt, retries=LABELING_MAX_RETRIES)
        dual_labeled     = secondary_result is not None

        s_steps  = []
        s_scores = {}
        secondary_model_name = ""

        if dual_labeled:
            s_steps  = secondary_result.get("step_labels", [])
            s_scores = _normalize_scores(secondary_result.get("trace_scores"), outcome) or {}
            secondary_model_name = _get_active_model_name("secondary")

        agreement    = _compute_agreement(p_scores, s_scores) if dual_labeled else None
        merged_steps = _merge_step_labels(p_steps, s_steps)

        # Average scores when both available
        if dual_labeled and s_scores:
            numeric = ["task_completion", "tool_use_efficiency", "reasoning_coherence", "safety_compliance"]
            final   = dict(p_scores)
            for k in numeric:
                pv = float(p_scores.get(k, 0) or 0)
                sv = float(s_scores.get(k, 0) or 0)
                final[k] = round((pv + sv) / 2, 2)
            final["overall_quality"] = round(
                (float(p_scores.get("overall_quality", 5.0) or 5.0) +
                 float(s_scores.get("overall_quality", 5.0) or 5.0)) / 2, 2
            )
            tc  = final["task_completion"]
            tue = final["tool_use_efficiency"]
            rc  = final["reasoning_coherence"]
            final["reward_signal"]   = round((tc/3*0.4)+(tue/3*0.3)+(rc/3*0.3), 4)
            final["reward_computed"] = final["reward_signal"]
            order = {"reject": 0, "flag": 1, "approve": 2}
            pv = p_scores.get("supervisor_verdict", "flag")
            sv = s_scores.get("supervisor_verdict", "flag")
            final["supervisor_verdict"] = pv if order.get(pv, 1) <= order.get(sv, 1) else sv
            final["verdict_reason"] = (
                f"P({primary_model_name}): {p_scores.get('verdict_reason','')} | "
                f"S({secondary_model_name}): {s_scores.get('verdict_reason','')}"
            )
        else:
            final = p_scores

        # Fix tools_used — filter out Python internals
        clean_tools = _filter_tools(trace["outcome"].get("tools_used", []))
        trace["outcome"]["tools_used"]       = clean_tools
        trace["outcome"]["total_tool_calls"] = len(clean_tools)

        # Fix task_niche — single canonical value
        raw_niche = trace["task"].get("niche", "")
        trace["task"]["niche"] = _canonical_niche(raw_niche, trace["task"]["task"])

        trace["labels"] = {
            "labeler_model":           primary_model_name,
            "labeler_model_2":         secondary_model_name,
            "constitution_version":    CONSTITUTION_VERSION,
            "labeled_at":              datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "step_level_scores":       merged_steps,
            "primary_trace_scores":    p_scores,
            "secondary_trace_scores":  s_scores,
            "trace_level_scores":      final,
            "dual_labeled":            dual_labeled,
            "agreement_score":         agreement,
            "conflict_flag":           (agreement < 0.75) if agreement is not None else False,
        }
        labeled.append(trace)

    approved   = sum(1 for t in labeled if t["labels"]["trace_level_scores"].get("supervisor_verdict") == "approve")
    dual_count = sum(1 for t in labeled if t["labels"].get("dual_labeled", False))
    conflicts  = sum(1 for t in labeled if t["labels"].get("conflict_flag", False))

    print(f"\n  ✅ Labeling complete:")
    print(f"     Labeled:      {len(labeled)}/{total}")
    print(f"     Excluded:     {excluded_count} (labeler failed — not written to dataset)")
    print(f"     Approved:     {approved}")
    print(f"     Dual-labeled: {dual_count}")
    print(f"     Conflicts:    {conflicts}")
    return labeled