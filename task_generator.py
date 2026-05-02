"""
task_generator.py
Converts real-world signals into clean, executable agent tasks.
Priority: signal-based (real data) > template-based > mutation-based.

Key fixes:
- Simplified task synthesis prompt — produces tasks completable in 6 steps
- source_url stored per task for verification
- Niche diversity from extended template set
- is_duplicate checks text hash only (not source URL)
"""

import json
import sqlite3
import hashlib
import os
import time
import random
from datetime import datetime

from llm_client import call_llm_json
from task_sources import collect_all_signals, mark_signal_used

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry", "tasks.db")

# ── Daily budget ───────────────────────────────────────────────────────────────
DAILY_BUDGET = {
    "llm_generative": 10,   # highest priority — real signals
    "template_based":  4,   # fallback
    "mutation_based":  4,   # fallback
}

# ── Task templates — 6 niches ──────────────────────────────────────────────────
TASK_TEMPLATES = [
    # API orchestration
    {
        "niche": "api_orchestration",
        "pattern": "Fetch {data_type} from {source_api} and sync it to {target_api} with error handling.",
        "slots": {
            "data_type":   ["overdue invoices", "customer records", "open issues", "webhook events"],
            "source_api":  ["Stripe", "GitHub", "Shopify", "HubSpot"],
            "target_api":  ["Notion", "Slack", "Airtable", "Google Sheets"],
        },
        "difficulty": "medium",
        "expected_tools": ["api_fetch", "api_write"],
        "likely_failure_points": ["auth token missing", "rate limit", "field mismatch"],
    },
    # Code debugging
    {
        "niche": "debugging",
        "pattern": "Debug why {component} raises {error} when {condition} and write a working fix.",
        "slots": {
            "component":  ["the API client", "the webhook handler", "the retry logic", "the data parser"],
            "error":      ["a KeyError", "a 401 Unauthorized", "a timeout", "a JSONDecodeError"],
            "condition":  ["given malformed input", "under concurrent requests", "after token refresh", "with missing fields"],
        },
        "difficulty": "medium",
        "expected_tools": ["code_search", "code_executor", "web_search"],
        "likely_failure_points": ["root cause misidentified", "fix breaks other paths"],
    },
    # Data analysis
    {
        "niche": "data_analysis",
        "pattern": "Analyze {data_source} to find {insight} and produce a {output} report.",
        "slots": {
            "data_source": ["sales CSV", "API response logs", "user activity JSON", "database query results"],
            "insight":     ["revenue trends", "top error codes", "usage patterns", "anomalies"],
            "output":      ["markdown summary", "structured JSON", "ranked list"],
        },
        "difficulty": "medium",
        "expected_tools": ["code_executor", "file_read", "api_fetch"],
        "likely_failure_points": ["malformed data", "missing columns", "wrong dtypes"],
    },
    # File system
    {
        "niche": "file_system_agent",
        "pattern": "Search {location} for {file_type} files matching {criteria} and {action}.",
        "slots": {
            "location":  ["the project directory", "the logs folder", "uploaded documents"],
            "file_type": ["Python", "JSON config", "CSV", "log"],
            "criteria":  ["errors in last 24h", "size over 1MB", "keyword TODO"],
            "action":    ["summarize findings", "extract key fields", "move to archive"],
        },
        "difficulty": "medium",
        "expected_tools": ["file_search", "file_read", "code_executor"],
        "likely_failure_points": ["file not found", "permission denied", "encoding error"],
    },
    # Multi-step planning
    {
        "niche": "multi_step_planning",
        "pattern": "Design and execute a {goal} workflow with {steps} steps that handles {constraint}.",
        "slots": {
            "goal":       ["data migration", "automated reporting", "incident response", "onboarding"],
            "steps":      ["3", "4", "5"],
            "constraint": ["rollback on failure", "idempotency", "rate limit compliance", "audit logging"],
        },
        "difficulty": "complex",
        "expected_tools": ["api_fetch", "api_write", "code_executor", "logger"],
        "likely_failure_points": ["partial completion", "step ordering", "rollback failure"],
    },
    # Web scraping
    {
        "niche": "web_scraping",
        "pattern": "Scrape {target} to extract {data}, handle {edge_case}, and store results in {destination}.",
        "slots": {
            "target":       ["a GitHub repo issue list", "a product pricing page", "an API docs page"],
            "data":         ["open issue titles and labels", "price and availability", "endpoint descriptions"],
            "edge_case":    ["pagination", "missing fields", "rate limiting"],
            "destination":  ["a JSON file", "an Airtable base", "a Slack message"],
        },
        "difficulty": "complex",
        "expected_tools": ["web_search", "api_fetch", "file_edit", "airtable_api"],
        "likely_failure_points": ["schema change", "timeout", "anti-bot protection"],
    },
]

# ── Registry helpers ───────────────────────────────────────────────────────────
def init_registry():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id            TEXT PRIMARY KEY,
            task               TEXT NOT NULL,
            niche              TEXT,
            difficulty         TEXT,
            expected_tools     TEXT,
            failure_points     TEXT,
            generation_strategy TEXT,
            freshness_source   TEXT,
            source_url         TEXT,
            created_at         TEXT,
            executed_count     INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def task_fingerprint(task_text: str) -> str:
    return hashlib.md5(task_text.strip().lower().encode()).hexdigest()

def is_duplicate(task_text: str) -> bool:
    """Text-hash check only — source URL check removed (was blocking valid tasks)."""
    fp = task_fingerprint(task_text)
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute("SELECT 1 FROM tasks WHERE task_id=?", (fp,)).fetchone()
    conn.close()
    return row is not None

def save_task(task: dict) -> bool:
    fp = task_fingerprint(task["task"])
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO tasks
            (task_id, task, niche, difficulty, expected_tools, failure_points,
             generation_strategy, freshness_source, source_url, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            fp, task["task"],
            task.get("niche", "api_orchestration"),
            task.get("difficulty", "medium"),
            json.dumps(task.get("expected_tools", [])),
            json.dumps(task.get("likely_failure_points", [])),
            task.get("generation_strategy", "unknown"),
            task.get("freshness_source", ""),
            task.get("source_url", ""),
            datetime.now().strftime("%Y-%m-%d"),
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"  ❌ DB save error: {e}")
        return False

def mark_executed(task_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE tasks SET executed_count=executed_count+1 WHERE task_id=?", (task_id,))
    conn.commit()
    conn.close()

def get_registry_sample(n: int = 10) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT task_id, task, niche, difficulty, expected_tools FROM tasks ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [{"task_id": r[0], "task": r[1], "niche": r[2],
             "difficulty": r[3], "expected_tools": json.loads(r[4])} for r in rows]

def load_approved_tasks(limit: int = 100) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT task_id, task, niche, difficulty, expected_tools, failure_points, freshness_source, source_url
        FROM tasks ORDER BY executed_count ASC, created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [{
        "task_id": r[0], "task": r[1], "niche": r[2], "difficulty": r[3],
        "expected_tools": json.loads(r[4]), "likely_failure_points": json.loads(r[5]),
        "freshness_source": r[6], "source_url": r[7],
    } for r in rows]

# ── Quality check ──────────────────────────────────────────────────────────────
NICHE_KEYWORDS = [
    "api", "stripe", "notion", "github", "slack", "airtable", "openai",
    "code", "script", "debug", "error", "fix", "fetch", "sync", "integrate",
    "langchain", "crewai", "agent", "tool", "function", "analyze", "extract",
]

def passes_quality_check(task: dict) -> bool:
    if not task or not task.get("task"):
        return False
    text = task["task"].lower()
    if len(text) < 30:
        return False
    if is_duplicate(task["task"]):
        return False
    return any(k in text for k in NICHE_KEYWORDS)

# ── Signal → Task conversion ───────────────────────────────────────────────────
def convert_signal_to_task(signal: dict) -> dict | None:
    """
    Use Groq LLaMA to convert a GitHub issue / SO question into a clean agent task.
    Prompt explicitly asks for tasks completable in 6 agent steps.
    """
    prompt = f"""You are a dataset designer for AI agent behavior research.

Convert this real developer problem into ONE clean, executable AI agent task.

Source: {signal['source']}
Problem:
{signal['raw_text'][:800]}

Requirements:
- Task must be solvable by an AI agent using tools in 4-6 steps
- Must involve at least 2 distinct tool calls (api_fetch, code_executor, web_search, file_read, etc.)
- Must be specific enough that success/failure is objectively measurable
- Keep it to 2 sentences maximum

Return ONLY this JSON:
{{
  "task": "specific 1-2 sentence executable task",
  "niche": "api_orchestration|debugging|data_analysis|file_system_agent|multi_step_planning|web_scraping",
  "difficulty": "simple|medium|complex",
  "expected_tools": ["tool1", "tool2"],
  "likely_failure_points": ["point1", "point2"],
  "generation_strategy": "llm_generative",
  "freshness_source": "{signal['source']}",
  "source_url": "{signal.get('source_url', '')}"
}}"""

    result = call_llm_json("generator", [{"role": "user", "content": prompt}], max_tokens=500)
    return result

def convert_signal_rule_based(signal: dict) -> dict:
    """Deterministic fallback — no LLM call needed."""
    title = signal.get("title", signal["raw_text"].split("\n")[0])[:100]
    text  = signal["raw_text"].lower()

    if any(k in text for k in ["stripe", "payment", "invoice"]):
        niche, tools = "api_orchestration", ["stripe_api", "api_fetch"]
    elif any(k in text for k in ["debug", "error", "exception", "traceback"]):
        niche, tools = "debugging", ["code_search", "code_executor"]
    elif any(k in text for k in ["langchain", "crewai", "agent", "llm"]):
        niche, tools = "api_orchestration", ["code_executor", "api_fetch"]
    else:
        niche, tools = "api_orchestration", ["api_fetch", "api_write"]

    return {
        "task": f"Investigate and resolve: {title}",
        "niche": niche,
        "difficulty": "medium",
        "expected_tools": tools,
        "likely_failure_points": ["tool error", "unexpected response"],
        "generation_strategy": "llm_generative",
        "freshness_source": signal["source"],
        "source_url": signal.get("source_url", ""),
    }

# ── Template generation ────────────────────────────────────────────────────────
def generate_template_tasks(count: int = 10) -> list[dict]:
    tasks = []
    for _ in range(count):
        template = random.choice(TASK_TEMPLATES)
        filled   = template["pattern"]
        for slot, options in template["slots"].items():
            filled = filled.replace(f"{{{slot}}}", random.choice(options))
        tasks.append({
            "task":                  filled,
            "niche":                 template["niche"],
            "difficulty":            template["difficulty"],
            "expected_tools":        template["expected_tools"],
            "likely_failure_points": template["likely_failure_points"],
            "generation_strategy":   "template_based",
            "freshness_source":      "template_library",
            "source_url":            "",
        })
    return tasks

# ── Mutation ───────────────────────────────────────────────────────────────────
MUTATIONS = [
    " Handle the case where the API returns 429 — implement exponential backoff.",
    " Validate all inputs first and return structured error messages for invalid data.",
    " Log every step and send a Slack notification on completion or failure.",
    " Ensure idempotency — running twice must not create duplicate records.",
    " Add retry logic with 3 attempts before marking the task as failed.",
]

def mutate_task_rule_based(seed: dict) -> dict:
    suffix = random.choice(MUTATIONS)
    return {
        "task":                  seed["task"] + suffix,
        "niche":                 seed.get("niche", "api_orchestration"),
        "difficulty":            "complex",
        "expected_tools":        seed.get("expected_tools", ["api_fetch", "api_write"]),
        "likely_failure_points": (seed.get("likely_failure_points", []) + ["retry exhausted"])[:3],
        "generation_strategy":   "mutation_based",
        "freshness_source":      "mutation_of_existing",
        "source_url":            "",
    }

# ── Main entry point ───────────────────────────────────────────────────────────
def generate_tasks(total: int = 18) -> list[dict]:
    """
    Priority: Signal-based → Template → Mutation
    Returns list of approved tasks ready for agent execution.
    """
    init_registry()
    approved = []

    # ── 1. Signal-based (highest priority) ────────────────────────────────────
    print("\n🌐 Strategy 1 (Priority): Signal-based from real-world sources...")
    signals   = collect_all_signals()
    llm_count = 0
    for signal in signals:
        if llm_count >= DAILY_BUDGET["llm_generative"]:
            break
        task = convert_signal_to_task(signal)
        if task is None:
            task = convert_signal_rule_based(signal)
        if task and passes_quality_check(task):
            task["task_id"] = task_fingerprint(task["task"])
            save_task(task)
            approved.append(task)
            mark_signal_used(signal)
            llm_count += 1
    print(f"  ✅ Signal-based approved: {llm_count}")

    # ── 2. Template-based (fallback) ──────────────────────────────────────────
    print("\n📋 Strategy 2 (Fallback): Template-based...")
    for t in generate_template_tasks(count=DAILY_BUDGET["template_based"] + 5):
        if len([x for x in approved if x.get("generation_strategy") == "template_based"]) >= DAILY_BUDGET["template_based"]:
            break
        if passes_quality_check(t):
            t["task_id"] = task_fingerprint(t["task"])
            save_task(t)
            approved.append(t)
    tmpl_count = len([x for x in approved if x.get("generation_strategy") == "template_based"])
    print(f"  ✅ Template approved: {tmpl_count}")

    # ── 3. Mutation (fill remaining quota) ────────────────────────────────────
    remaining = total - len(approved)
    if remaining > 0:
        print(f"\n🔀 Strategy 3 (Fallback): Mutation-based ({remaining} needed)...")
        seeds = get_registry_sample(n=DAILY_BUDGET["mutation_based"] + 5)
        mut_count = 0
        for seed in seeds:
            if mut_count >= min(DAILY_BUDGET["mutation_based"], remaining):
                break
            mutated = mutate_task_rule_based(seed)
            if passes_quality_check(mutated):
                mutated["task_id"] = task_fingerprint(mutated["task"])
                save_task(mutated)
                approved.append(mutated)
                mut_count += 1
        print(f"  ✅ Mutation approved: {mut_count}")

    print(f"\n✅ Total tasks approved: {len(approved)}")
    return approved[:total]