"""
task_generator.py
Converts raw real-world signals into clean, structured agent tasks.
Uses 3 strategies: template-based, LLM-generative, mutation-based.
Runs all tasks through a quality gate before storing in SQLite registry.
"""

import json
import sqlite3
import hashlib
import os
import time
from datetime import datetime
from typing import Optional

from openrouter_client import call_llm_json
from task_sources import collect_all_signals

DB_PATH = os.path.join(os.path.dirname(__file__), "registry", "tasks.db")

# ── YAML-style templates stored as Python dicts (no extra file needed) ─────────
TASK_TEMPLATES = [
    {
        "id": "api_chain_001",
        "pattern": "Fetch {data_type} from {source_api} and sync it to {target_api} with proper error handling.",
        "slots": {
            "data_type":   ["overdue invoices", "customer records", "product inventory", "support tickets", "subscription data"],
            "source_api":  ["Stripe", "Shopify", "Salesforce", "HubSpot", "Twilio"],
            "target_api":  ["Notion", "Airtable", "Google Sheets", "Slack", "Linear"],
        },
        "difficulty": "medium",
        "expected_tools": ["api_fetch", "api_write"],
        "likely_failure_points": ["pagination not handled", "rate limit hit", "missing auth token"],
    },
    {
        "id": "code_debug_001",
        "pattern": "Debug why {component} throws {error_type} when {condition} and write a fix.",
        "slots": {
            "component":  ["the API client", "the webhook handler", "the data parser", "the retry logic"],
            "error_type": ["a 429 error", "a null pointer exception", "a timeout", "a schema mismatch"],
            "condition":  ["processing large payloads", "handling concurrent requests", "running in production", "parsing nested JSON"],
        },
        "difficulty": "medium",
        "expected_tools": ["code_executor", "web_search"],
        "likely_failure_points": ["root cause misidentified", "fix introduces new bug"],
    },
    {
        "id": "multi_step_001",
        "pattern": "Write a Python script that calls {api_a} to get {resource}, transforms the data, then posts results to {api_b}.",
        "slots": {
            "api_a":     ["GitHub API", "Stripe API", "OpenAI API", "HuggingFace API"],
            "resource":  ["open issues", "failed payments", "model usage stats", "dataset metadata"],
            "api_b":     ["a Slack webhook", "a Notion database", "a Google Sheet", "an Airtable base"],
        },
        "difficulty": "complex",
        "expected_tools": ["code_executor", "api_fetch", "api_write"],
        "likely_failure_points": ["auth handling", "data shape mismatch", "API response pagination"],
    },
    {
        "id": "agent_orchestration_001",
        "pattern": "Build an agent that monitors {source} for {event} and automatically triggers {action}.",
        "slots": {
            "source": ["a GitHub repo", "a Stripe webhook", "an RSS feed", "a Slack channel"],
            "event":  ["new pull requests", "failed payments", "new posts", "keyword mentions"],
            "action": ["creates a Notion task", "sends a Slack alert", "updates a spreadsheet", "opens a Linear issue"],
        },
        "difficulty": "complex",
        "expected_tools": ["webhook_listener", "api_write", "notification"],
        "likely_failure_points": ["event deduplication", "async handling", "auth scope missing"],
    },
]

import random

def generate_template_tasks(count: int = 40) -> list[dict]:
    """Fill template slots randomly to generate structured tasks."""
    tasks = []
    for _ in range(count):
        template = random.choice(TASK_TEMPLATES)
        filled = template["pattern"]
        for slot, options in template["slots"].items():
            filled = filled.replace(f"{{{slot}}}", random.choice(options))

        tasks.append({
            "task":                 filled,
            "difficulty":          template["difficulty"],
            "expected_tools":      template["expected_tools"],
            "likely_failure_points": template["likely_failure_points"],
            "generation_strategy": "template_based",
            "freshness_source":    "template_library",
        })
    return tasks


BATCH_SIZE = 3   # used by mutation batching only

# ── Rule-based signal → task conversion (zero LLM calls) ─────────────────────
# Map source keywords to tools and difficulty
SOURCE_TOOL_MAP = {
    "stripe":     (["stripe_api", "api_fetch"],         "medium"),
    "notion":     (["notion_api", "api_write"],          "medium"),
    "github":     (["github_api", "code_executor"],      "medium"),
    "slack":      (["slack_api", "api_write"],           "simple"),
    "shopify":    (["shopify_api", "api_fetch"],         "medium"),
    "langchain":  (["code_executor", "llm_api"],         "complex"),
    "crewai":     (["code_executor", "llm_api"],         "complex"),
    "autogen":    (["code_executor", "llm_api"],         "complex"),
    "openai":     (["openai_api", "code_executor"],      "medium"),
    "huggingface":["huggingface_api", "code_executor"],
    "airtable":   (["airtable_api", "api_write"],        "simple"),
    "webhook":    (["webhook_listener", "api_write"],    "medium"),
    "api":        (["api_fetch", "api_write"],           "medium"),
    "code":       (["code_executor", "web_search"],      "medium"),
}

FAILURE_MAP = {
    "stripe":    ["rate limit hit", "missing webhook signature"],
    "notion":    ["database ID not found", "property type mismatch"],
    "github":    ["auth token expired", "repo not found"],
    "langchain": ["agent loop not terminating", "tool not found"],
    "crewai":    ["task delegation failure", "agent timeout"],
    "api":       ["401 unauthorized", "timeout", "malformed response"],
    "code":      ["syntax error", "import not found", "wrong output type"],
}

TASK_PATTERNS = [
    "Fetch {resource} from {service} and process the results with error handling.",
    "Debug and fix the issue described: {title}. Write a working solution.",
    "Build an agent that automates: {title}. Handle edge cases gracefully.",
    "Write a script to integrate {service} with another service based on: {title}",
    "Investigate and resolve: {title}. Document the root cause and fix.",
]

def convert_signal_rule_based(signal: dict) -> dict:
    """
    Convert a raw signal to a task using pure string matching — zero LLM calls.
    Extracts service name, picks matching tools/difficulty/failures, fills a template.
    """
    import re
    text   = (signal["raw_text"] or "").lower()
    source = (signal["source"] or "").lower()
    title  = signal["raw_text"].split("\n")[0].strip()[:120]

    # Detect which service/niche this signal belongs to
    detected_service = "api"
    tools            = ["api_fetch", "api_write"]
    difficulty       = "medium"
    failures         = ["timeout", "auth error", "malformed response"]

    for keyword, mapping in SOURCE_TOOL_MAP.items():
        if keyword in text or keyword in source:
            detected_service = keyword
            if isinstance(mapping, tuple):
                tools, difficulty = mapping
            else:
                tools = mapping
                difficulty = "medium"
            failures = FAILURE_MAP.get(keyword, failures)
            break

    # Fill a task pattern with the signal context
    pattern = random.choice(TASK_PATTERNS)
    task_text = pattern.format(
        resource=f"data related to {detected_service}",
        service=detected_service,
        title=title,
    )

    return {
        "task":                  task_text,
        "difficulty":            difficulty,
        "expected_tools":        tools,
        "likely_failure_points": failures[:2],
        "generation_strategy":   "llm_generative",
        "freshness_source":      signal["source"],
    }


def mutate_task_rule_based(seed: dict) -> dict:
    """
    Mutate a seed task using pure string manipulation — zero LLM calls.
    Applies one of 5 mutation types deterministically.
    """
    mutations = [
        # (suffix to append, difficulty bump, extra failure)
        (
            " Handle the case where the initial API call fails with a 429 and implement exponential backoff.",
            "complex",
            "rate limit not handled correctly",
        ),
        (
            " Additionally validate all inputs before processing and return structured error messages for invalid data.",
            "complex",
            "input validation missing",
        ),
        (
            " The integration must also log every step to a file and send a Slack notification on completion or failure.",
            "complex",
            "notification delivery failure",
        ),
        (
            " Ensure idempotency — if the task is run twice, the second run should detect duplicates and skip them.",
            "complex",
            "duplicate records created",
        ),
        (
            " Break this into two agents: one that fetches and validates the data, one that writes and confirms the output.",
            "complex",
            "agent coordination failure",
        ),
    ]

    suffix, new_difficulty, extra_failure = random.choice(mutations)

    base_task    = seed.get("task", "")
    base_tools   = seed.get("expected_tools",        ["api_fetch", "api_write"])
    base_failures= seed.get("likely_failure_points", ["timeout", "auth error"])

    return {
        "task":                  base_task + suffix,
        "difficulty":            new_difficulty,
        "expected_tools":        base_tools,
        "likely_failure_points": (base_failures + [extra_failure])[:3],
        "generation_strategy":   "mutation_based",
        "freshness_source":      "mutation_of_existing",
        "mutation_type":         "rule_based",
    }


# ── Task Registry (SQLite) ─────────────────────────────────────────────────────

def init_registry():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id            TEXT PRIMARY KEY,
            task               TEXT NOT NULL,
            difficulty         TEXT,
            expected_tools     TEXT,
            failure_points     TEXT,
            generation_strategy TEXT,
            freshness_source   TEXT,
            created_at         TEXT,
            executed_count     INTEGER DEFAULT 0,
            niche_score        REAL DEFAULT 0.0,
            status             TEXT DEFAULT 'approved'
        )
    """)
    conn.commit()
    conn.close()


def task_fingerprint(task_text: str) -> str:
    """Create a short hash to detect near-duplicates."""
    return hashlib.md5(task_text.strip().lower().encode()).hexdigest()


def is_duplicate(task_text: str, threshold: float = 0.85) -> bool:
    """Simple duplicate check — exact hash match for now."""
    fp = task_fingerprint(task_text)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT 1 FROM tasks WHERE task_id = ?", (fp,)
    ).fetchone()
    conn.close()
    return row is not None


def save_task(task: dict) -> bool:
    """Save a validated task to the registry. Returns True if saved."""
    fp = task_fingerprint(task["task"])
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO tasks
            (task_id, task, difficulty, expected_tools, failure_points,
             generation_strategy, freshness_source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fp,
            task["task"],
            task.get("difficulty", "medium"),
            json.dumps(task.get("expected_tools", [])),
            json.dumps(task.get("likely_failure_points", [])),
            task.get("generation_strategy", "unknown"),
            task.get("freshness_source", "unknown"),
            datetime.now().strftime("%Y-%m-%d"),
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"  ❌ DB save error: {e}")
        return False


def load_approved_tasks(limit: int = 100) -> list[dict]:
    """Load approved tasks from registry, prioritising least-executed ones."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT task_id, task, difficulty, expected_tools, failure_points,
               freshness_source, created_at
        FROM tasks
        WHERE status = 'approved'
        ORDER BY executed_count ASC, created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    tasks = []
    for row in rows:
        tasks.append({
            "task_id":               row[0],
            "task":                  row[1],
            "difficulty":            row[2],
            "expected_tools":        json.loads(row[3]),
            "likely_failure_points": json.loads(row[4]),
            "freshness_source":      row[5],
            "created_at":            row[6],
        })
    return tasks


def mark_executed(task_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE tasks SET executed_count = executed_count + 1 WHERE task_id = ?",
        (task_id,)
    )
    conn.commit()
    conn.close()


def get_registry_sample(n: int = 5) -> list[dict]:
    """Return a small sample of existing tasks for mutation."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT task, difficulty FROM tasks ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [{"task": r[0], "difficulty": r[1]} for r in rows]


# ── Quality Gate ───────────────────────────────────────────────────────────────

# Niche keywords for fast rule-based check (no LLM call needed)
NICHE_KEYWORDS = [
    "api", "stripe", "notion", "github", "slack", "airtable", "shopify",
    "salesforce", "hubspot", "webhook", "endpoint", "request", "response",
    "fetch", "sync", "integrate", "code", "script", "debug", "function",
    "python", "deploy", "pipeline", "automate", "parse", "extract", "transform",
]

def passes_rule_based_check(task: dict) -> bool:
    """
    Fast rule-based quality check — no LLM call, no rate limit risk.
    Used for template tasks which are already well-structured.
    """
    if not task or not task.get("task"):
        return False
    text = task["task"].lower()
    if len(text) < 20:
        return False
    if is_duplicate(task["task"]):
        return False
    # Must contain at least one niche keyword
    return any(kw in text for kw in NICHE_KEYWORDS)



# ── Main Generate Function ─────────────────────────────────────────────────────

# Daily budget breakdown
DAILY_BUDGET = {
    "template_based":  2,
    "llm_generative":  2,
    "mutation_based":  1,
}  # 5 tasks/day × 8 calls = 40 req/day — fits free 50 req/day hard limit

def generate_tasks(total: int = 100) -> list[dict]:
    """
    Full task generation run.
    Returns list of validated tasks ready for agent execution.

    Rate-limit strategy:
      template_based  → rule-based check only, zero LLM calls
      llm_generative  → rule-based keyword matching, ZERO LLM calls
      mutation_based  → batched LLM calls, 5s delay between batches
    """
    init_registry()
    approved = []

    # ── Strategy 1: Template-based (zero LLM calls) ───────────────────────────
    print("\n📋 Strategy 1: Template-based generation...")
    template_tasks = generate_template_tasks(count=DAILY_BUDGET["template_based"] + 10)
    for t in template_tasks:
        if len([x for x in approved if x.get("generation_strategy") == "template_based"]) >= DAILY_BUDGET["template_based"]:
            break
        if passes_rule_based_check(t):
            save_task(t)
            approved.append(t)
    print(f"  ✅ Template tasks approved: {len([x for x in approved if x.get('generation_strategy') == 'template_based'])}")

    # ── Strategy 2: Rule-based signal conversion (ZERO LLM calls) ──────────────
    # Signals are converted using keyword matching + templates — no API calls.
    # This reserves all API quota for the labeling stage where quality matters.
    print("\n🌐 Strategy 2: LLM-generative from real-world signals...")
    signals   = collect_all_signals()
    llm_count = 0
    for signal in signals:
        if llm_count >= DAILY_BUDGET["llm_generative"]:
            break
        task = convert_signal_rule_based(signal)
        if passes_rule_based_check(task):
            save_task(task)
            approved.append(task)
            llm_count += 1
    print(f"  ✅ Signal-based tasks approved: {llm_count}")

    # ── Strategy 3: Mutation-based (rule-based, zero LLM calls) ─────────────────
    print("\n🔀 Strategy 3: Mutation-based generation...")
    seed_tasks     = get_registry_sample(n=DAILY_BUDGET["mutation_based"] + 5)
    mutation_count = 0
    for seed in seed_tasks:
        if mutation_count >= DAILY_BUDGET["mutation_based"]:
            break
        task = mutate_task_rule_based(seed)
        if passes_rule_based_check(task):
            save_task(task)
            approved.append(task)
            mutation_count += 1
    print(f"  ✅ Mutation-based tasks approved: {mutation_count}")

    print(f"\n✅ Total tasks approved for today: {len(approved)}")
    return approved[:total]