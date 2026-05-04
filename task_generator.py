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
from task_sources import collect_all_signals, mark_signal_used
#from tool_registry import TOOL_NAMES, TOOL_LIST_FOR_PROMPT

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

# ── LLM-based signal → task conversion ───────────────────────────────────────
# Uses the generator model (gemma) to convert real signals into clean tasks.
# Batches 3 signals per call to stay within rate limits.
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

def convert_signal_to_task(signal: dict) -> Optional[dict]:
    """
    Convert a raw real-world signal into a clean agent task using LLM.
    Single signal per call to keep output tokens low and avoid truncation.
    """
    prompt = f"""Convert this real-world developer signal into an AI agent task.

Signal ({signal['source']}):
{signal['raw_text'][:200]}

Rules: requires 2+ tool calls, fits API Orchestration or Code Agent niche, solvable by AI agent.
If irrelevant return null.

Return ONLY JSON (or null):
{{
  "task": "one sentence executable agent task",
  "difficulty": "simple|medium|complex",
  "expected_tools": ["tool1", "tool2"],
  "likely_failure_points": ["point1"],
  "generation_strategy": "llm_generative",
  "freshness_source": "{signal['source']}"
}}"""
    return call_llm_json("generator", [{"role": "user", "content": prompt}], max_tokens=300)


def convert_signals_llm(signals: list[dict]) -> list[dict]:
    """
    Convert a batch of real-world signals into clean agent tasks using LLM.
    Batches BATCH_SIZE signals per call to respect rate limits.
    """
    from openrouter_client import call_llm_json

    numbered = ""
    for i, s in enumerate(signals):
        numbered += f"\n--- Signal {i+1} (source: {s['source']}) ---\n{s['raw_text'][:200]}\n"

    prompt = f"""You are a task designer for an AI agent behavior dataset.
Convert each signal below into a clean, executable agent task.
Focus on API Orchestration and Code Agent tasks.

{numbered}

CRITICAL RULES:
1. expected_tools MUST be chosen ONLY from this exact list:
{TOOL_NAMES}
2. Do NOT invent tool names — only use tools from the list above
3. Each task needs 2-3 realistic tool calls
4. Must have at least one realistic failure point
5. Return null for irrelevant signals

Return ONLY a JSON array with exactly {len(signals)} items:
[
  {{
    "task": "one or two sentence agent task",
    "difficulty": "simple|medium|complex",
    "expected_tools": ["web_search", "api_fetch"],
    "likely_failure_points": ["401 unauthorized", "timeout"],
    "generation_strategy": "llm_generative",
    "freshness_source": "source_here"
  }}
]"""

    result = call_llm_json("generator", [{"role": "user", "content": prompt}], max_tokens=1200)
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return []
    return [_validate_task_tools(r) for r in result if r and isinstance(r, dict) and r.get("task")]


def _validate_task_tools(task: dict) -> dict:
    """Ensure expected_tools only contains registered tool names. Fix invalid ones."""
    tools = task.get("expected_tools", [])
    valid = [t for t in tools if t in TOOL_NAMES]
    if not valid:
        # Assign default tools based on task content
        text = task.get("task", "").lower()
        if any(k in text for k in ["stripe", "invoice", "payment"]):
            valid = ["stripe_list_invoices", "api_fetch", "slack_send_message"]
        elif any(k in text for k in ["github", "issue", "repo"]):
            valid = ["github_list_issues", "api_fetch", "file_write"]
        elif any(k in text for k in ["notion", "database"]):
            valid = ["notion_query_database", "notion_create_page"]
        elif any(k in text for k in ["code", "script", "debug", "fix"]):
            valid = ["code_executor", "web_search", "file_write"]
        else:
            valid = ["api_fetch", "api_write", "web_search"]
    task["expected_tools"] = valid
    return task


def convert_signal_rule_based(signal: dict) -> dict:
    """Fallback rule-based converter used if LLM call fails."""
def convert_signal_rule_based(signal: dict) -> dict:
    """Fallback: rule-based conversion if LLM call fails."""
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


def is_duplicate(task_text: str, freshness_source: str = "") -> bool:
    """
    Duplicate check on both task text hash AND freshness_source.
    Prevents same GitHub issue appearing twice with slightly different wording.
    """
    fp = task_fingerprint(task_text)
    conn = sqlite3.connect(DB_PATH)
    # Check task hash
    row = conn.execute("SELECT 1 FROM tasks WHERE task_id = ?", (fp,)).fetchone()
    if row:
        conn.close()
        return True
    # Check source URL — same source = likely same task
    if freshness_source and freshness_source not in ("template_library", "mutation_of_existing", ""):
        row = conn.execute(
            "SELECT 1 FROM tasks WHERE freshness_source = ?", (freshness_source,)
        ).fetchone()
        if row:
            conn.close()
            return True
    conn.close()
    return False


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
    if is_duplicate(task["task"], task.get("freshness_source", "")):
        return False
    # Must contain at least one niche keyword
    return any(kw in text for kw in NICHE_KEYWORDS)



def mutate_tasks_llm(seed_tasks: list[dict]) -> list[dict]:
    """Mutate a batch of tasks using LLM to create harder, more diverse variants."""
    from openrouter_client import call_llm_json

    mutations = [
        "Add a mid-task failure that requires recovery (e.g. 429 rate limit, auth error).",
        "Add input validation requirement — agent must handle malformed or missing fields.",
        "Expand into 2-agent coordination: one fetches data, one writes output.",
        "Add idempotency requirement — running twice must not create duplicates.",
        "Add a logging + notification requirement on completion or failure.",
    ]

    numbered = ""
    chosen   = []
    for i, seed in enumerate(seed_tasks):
        m = random.choice(mutations)
        chosen.append(m)
        numbered += f"\n--- Task {i+1} ---\nOriginal: {seed['task']}\nMutation: {m}\n"

    prompt = f"""Mutate each agent task below to create a harder, more diverse variant.
Apply the specified mutation to each one.

{numbered}

Return ONLY a JSON array with exactly {len(seed_tasks)} items:
[
  {{
    "task": "mutated task description",
    "difficulty": "complex",
    "expected_tools": ["tool1", "tool2"],
    "likely_failure_points": ["point1"],
    "generation_strategy": "mutation_based",
    "freshness_source": "mutation_of_existing"
  }}
]"""

    result = call_llm_json("generator", [{"role": "user", "content": prompt}], max_tokens=1200)
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        return []
    return [r for r in result if r and isinstance(r, dict) and r.get("task")]


# ── Main Generate Function ─────────────────────────────────────────────────────

# Daily budget breakdown
DAILY_BUDGET = {
    "template_based":  4,    # 0 LLM calls (pure template)
    "llm_generative":  4,    # 4 LLM calls (generator model)
    "mutation_based":  4,    # 4 LLM calls (generator model)
}
# Total call budget breakdown (3 keys = 150 req/day):
#   task generation:  8  calls  (4+4 LLM, templates free)
#   agent execution:  48 calls  (12 tasks × 4 steps)
#   labeling:         24 calls  (12 tasks × 2: primary+secondary)
#   buffer:           70 calls  (retries, quality checks)
#   TOTAL:           ~150 calls — fits exactly

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

    # ── Strategy 2: LLM-generative from real signals (rule-based fallback) ──────
    print("\n🌐 Strategy 2: LLM-generative from real-world signals...")
    signals   = collect_all_signals()
    llm_count = 0
    for signal in signals:
        if llm_count >= DAILY_BUDGET["llm_generative"]:
            break
        # Try LLM first — fall back to rule-based if it fails
        task = convert_signal_to_task(signal)
        if task is None:
            task = convert_signal_rule_based(signal)
        if task and passes_rule_based_check(task):
            save_task(task)
            approved.append(task)
            llm_count += 1
    print(f"  ✅ Signal-based tasks approved: {llm_count}")

    # ── Strategy 3: LLM mutation (rule-based fallback) ───────────────────────────
    print("\n🔀 Strategy 3: Mutation-based generation...")
    seed_tasks     = get_registry_sample(n=DAILY_BUDGET["mutation_based"] + 5)
    mutation_count = 0
    mutations = [
        "Add a mid-task failure that requires exponential backoff recovery.",
        "Add input validation and structured error messages for all edge cases.",
        "The agent must also send a Slack notification on completion or failure.",
        "Ensure idempotency — detect and skip duplicates on repeat runs.",
        "Expand to 2-agent coordination: one fetches/validates, one writes/confirms.",
    ]
    for seed in seed_tasks:
        if mutation_count >= DAILY_BUDGET["mutation_based"]:
            break
        mutation = random.choice(mutations)
        prompt = f"""Mutate this agent task to be harder and more realistic:

Original: {seed['task']}
Mutation to apply: {mutation}

Return ONLY JSON:
{{
  "task": "mutated task (1-2 sentences)",
  "difficulty": "complex",
  "expected_tools": ["tool1", "tool2"],
  "likely_failure_points": ["point1"],
  "generation_strategy": "mutation_based",
  "freshness_source": "mutation_of_existing"
}}"""
        task = call_llm_json("generator", [{"role": "user", "content": prompt}], max_tokens=250)
        if task is None:
            task = mutate_task_rule_based(seed)
        if task and passes_rule_based_check(task):
            save_task(task)
            approved.append(task)
            mutation_count += 1
    print(f"  ✅ Mutation-based tasks approved: {mutation_count}")

    print(f"\n✅ Total tasks approved for today: {len(approved)}")
    return approved[:total]