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


def convert_signal_to_task(signal: dict) -> Optional[dict]:
    """Use LLM to convert a raw real-world signal into a clean agent task."""
    prompt = f"""
You are a task designer for an AI agent behavior dataset focused on API Orchestration and Code Agents.

Below is a real-world signal (GitHub issue, Stack Overflow question, or API changelog).
Your job is to extract or rewrite it as a clean, executable agent task.

RULES:
1. Task must require at least 2 tool calls to complete
2. Task must have at least one realistic failure point
3. Task must fit the "API Orchestration" or "Code Agent" niche
4. Task must be solvable by an AI agent with standard tool access
5. If the signal is irrelevant or too vague, return null

Signal source: {signal['source']}
Signal text:
{signal['raw_text'][:800]}

Return ONLY this JSON (or the word null if not relevant):
{{
  "task": "one or two sentence executable agent task",
  "difficulty": "simple|medium|complex",
  "expected_tools": ["tool1", "tool2"],
  "likely_failure_points": ["point1", "point2"],
  "generation_strategy": "llm_generative",
  "freshness_source": "{signal['source']}"
}}
"""
    result = call_llm_json("generator", [{"role": "user", "content": prompt}])
    return result


def mutate_task(base_task: dict) -> Optional[dict]:
    """Apply a random mutation to a high-quality existing task to create a harder variant."""
    mutations = {
        "escalate_difficulty":    "Add an ambiguous constraint that requires the agent to infer intent from incomplete information.",
        "add_failure_injection":  "Add a mid-task condition that causes the agent's first approach to fail and requires recovery.",
        "change_domain_context":  "Swap the API/tool context to a different but structurally similar service.",
        "add_adversarial_input":  "Inject a malformed or unexpected input the agent must detect and handle gracefully.",
        "multi_agent_expansion":  "Expand this single-agent task into a 2-agent coordination task where one agent delegates to another.",
    }
    mutation_type, mutation_desc = random.choice(list(mutations.items()))

    prompt = f"""
You are mutating an existing AI agent task to create a harder, more diverse variant.

Original task: {base_task['task']}
Mutation to apply: {mutation_desc}

Return ONLY this JSON:
{{
  "task": "the mutated task description",
  "difficulty": "medium|complex",
  "expected_tools": ["tool1", "tool2"],
  "likely_failure_points": ["point1", "point2"],
  "generation_strategy": "mutation_based",
  "freshness_source": "mutation_of_existing",
  "mutation_type": "{mutation_type}"
}}
"""
    result = call_llm_json("generator", [{"role": "user", "content": prompt}])
    return result


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

def passes_quality_gate(task: dict) -> bool:
    """Quick LLM-based feasibility + niche check."""
    if not task or not task.get("task"):
        return False
    if len(task["task"]) < 20:
        return False
    if is_duplicate(task["task"]):
        return False

    prompt = f"""
Rate this proposed AI agent task on two criteria.

Task: {task['task']}

Return ONLY this JSON:
{{
  "niche_relevant": true|false,   // Is it API Orchestration or Code Agent niche?
  "feasible": true|false,         // Can a standard AI agent realistically complete it?
  "reason": "one sentence"
}}
"""
    result = call_llm_json("quality_gate", [{"role": "user", "content": prompt}])
    if result is None:
        return True   # allow through if checker fails (don't drop data)
    return result.get("niche_relevant", False) and result.get("feasible", False)


# ── Main Generate Function ─────────────────────────────────────────────────────

# Daily budget breakdown
DAILY_BUDGET = {
    "template_based":  40,
    "llm_generative":  35,
    "mutation_based":  25,
}

def generate_tasks(total: int = 100) -> list[dict]:
    """
    Full task generation run.
    Returns list of validated tasks ready for agent execution.
    """
    init_registry()
    approved = []

    # ── Strategy 1: Template-based ─────────────────────────────────────────────
    print("\n📋 Strategy 1: Template-based generation...")
    template_tasks = generate_template_tasks(count=DAILY_BUDGET["template_based"] + 10)
    for t in template_tasks:
        if len([x for x in approved if x.get("generation_strategy") == "template_based"]) >= DAILY_BUDGET["template_based"]:
            break
        if passes_quality_gate(t):
            save_task(t)
            approved.append(t)
    print(f"  ✅ Template tasks approved: {len([x for x in approved if x.get('generation_strategy') == 'template_based'])}")

    # ── Strategy 2: LLM-generative from real signals ───────────────────────────
    print("\n🌐 Strategy 2: LLM-generative from real-world signals...")
    signals = collect_all_signals()
    llm_count = 0
    for signal in signals:
        if llm_count >= DAILY_BUDGET["llm_generative"]:
            break
        task = convert_signal_to_task(signal)
        if task and passes_quality_gate(task):
            save_task(task)
            approved.append(task)
            llm_count += 1
    print(f"  ✅ LLM-generative tasks approved: {llm_count}")

    # ── Strategy 3: Mutation-based ─────────────────────────────────────────────
    print("\n🔀 Strategy 3: Mutation-based generation...")
    seed_tasks = get_registry_sample(n=DAILY_BUDGET["mutation_based"] + 5)
    mutation_count = 0
    for seed in seed_tasks:
        if mutation_count >= DAILY_BUDGET["mutation_based"]:
            break
        mutated = mutate_task(seed)
        if mutated and passes_quality_gate(mutated):
            save_task(mutated)
            approved.append(mutated)
            mutation_count += 1
    print(f"  ✅ Mutation-based tasks approved: {mutation_count}")

    print(f"\n✅ Total tasks approved for today: {len(approved)}")
    return approved[:total]
