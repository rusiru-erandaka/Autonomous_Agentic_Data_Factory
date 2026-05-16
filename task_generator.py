"""
task_generator.py
Converts source signals into executable tasks and stores them in a registry.

The pipeline is transitioning from synthetic prompts toward repo-grounded coding
tasks. Real GitHub issues now carry repo provenance and execution metadata all
the way into the task registry.
"""

import hashlib
import json
import os
import random
import sqlite3
from datetime import datetime
from typing import Optional

from llm_client import call_llm_json
from task_sources import collect_all_signals, mark_signal_used

DB_PATH = os.path.join(os.path.dirname(__file__), "registry", "tasks.db")

REAL_CODING_TOOL_NAMES = [
    "file_search", "file_read", "file_edit", "code_search", "code_view",
    "code_edit", "code_executor", "git", "web_search", "api_fetch",
]

TASK_TEMPLATES = [
    {
        "id": "api_chain_001",
        "pattern": "Fetch {data_type} from {source_api} and sync it to {target_api} with proper error handling.",
        "slots": {
            "data_type": ["overdue invoices", "customer records", "product inventory", "support tickets", "subscription data"],
            "source_api": ["Stripe", "Shopify", "Salesforce", "HubSpot", "Twilio"],
            "target_api": ["Notion", "Airtable", "Google Sheets", "Slack", "Linear"],
        },
        "difficulty": "medium",
        "expected_tools": ["api_fetch", "file_edit"],
        "likely_failure_points": ["pagination not handled", "rate limit hit", "missing auth token"],
    },
    {
        "id": "code_debug_001",
        "pattern": "Debug why {component} throws {error_type} when {condition} and write a fix.",
        "slots": {
            "component": ["the API client", "the webhook handler", "the data parser", "the retry logic"],
            "error_type": ["a 429 error", "a null pointer exception", "a timeout", "a schema mismatch"],
            "condition": ["processing large payloads", "handling concurrent requests", "running in production", "parsing nested JSON"],
        },
        "difficulty": "medium",
        "expected_tools": ["code_search", "code_edit", "code_executor"],
        "likely_failure_points": ["root cause misidentified", "fix introduces new bug"],
    },
]

SOURCE_TOOL_MAP = {
    "github": (["file_search", "code_search", "code_edit", "code_executor", "git"], "complex"),
    "openai": (["file_search", "code_search", "code_edit", "code_executor"], "complex"),
    "langchain": (["file_search", "code_search", "code_edit", "code_executor"], "complex"),
    "autogen": (["file_search", "code_search", "code_edit", "code_executor"], "complex"),
    "crewai": (["file_search", "code_search", "code_edit", "code_executor"], "complex"),
    "api": (["api_fetch", "file_edit", "code_executor"], "medium"),
    "code": (["code_search", "code_edit", "code_executor"], "medium"),
}

FAILURE_MAP = {
    "github": ["issue not reproducible locally", "failing tests not isolated", "wrong target file edited"],
    "openai": ["client API change misread", "response schema mismatch", "test fixture update missing"],
    "langchain": ["integration path misidentified", "snapshot expectations stale", "tool wiring incomplete"],
    "autogen": ["agent flow regression", "async behavior mismatch", "fixture coverage missing"],
    "crewai": ["task orchestration regression", "flow contract mismatch", "test setup incomplete"],
    "api": ["401 unauthorized", "timeout", "malformed response"],
    "code": ["syntax error", "import not found", "wrong output type"],
}

TASK_PATTERNS = [
    "Reproduce and fix the GitHub issue: {title}. Work inside the repository, change the relevant code, and validate the fix with the most relevant tests you can run.",
    "Investigate the repository issue '{title}', identify the root cause, implement a code fix, and verify the result with targeted execution evidence.",
    "Patch the codebase to resolve: {title}. Use repository search, inspect the affected files, edit the fix, and run the best available validation command.",
]


def _issue_task_metadata(signal: dict) -> dict:
    return {
        "source_url": signal.get("source_url", ""),
        "repo_url": signal.get("repo_url", ""),
        "repo_clone_url": signal.get("repo_clone_url", ""),
        "repo_full_name": signal.get("repo_full_name", ""),
        "repo_default_branch": signal.get("repo_default_branch", ""),
        "repo_language": signal.get("repo_language", ""),
        "issue_number": signal.get("issue_number"),
        "issue_title": signal.get("title", ""),
        "issue_labels": signal.get("issue_labels", []),
        "path_hints": signal.get("path_hints", []),
        "execution_target": signal.get("execution_target", "synthetic"),
        "task_type": "repo_issue_fix",
    }


def _merge_task_metadata(task: dict, signal: Optional[dict] = None) -> dict:
    merged = dict(task or {})
    if signal:
        merged.update(_issue_task_metadata(signal))
    merged.setdefault("source_url", "")
    merged.setdefault("repo_url", "")
    merged.setdefault("repo_clone_url", "")
    merged.setdefault("repo_full_name", "")
    merged.setdefault("repo_default_branch", "")
    merged.setdefault("repo_language", "")
    merged.setdefault("issue_number", None)
    merged.setdefault("issue_title", merged.get("task", ""))
    merged.setdefault("issue_labels", [])
    merged.setdefault("path_hints", [])
    merged.setdefault("execution_target", "synthetic")
    merged.setdefault("task_type", "generic_task")
    return merged


def generate_template_tasks(count: int = 40) -> list[dict]:
    tasks = []
    for _ in range(count):
        template = random.choice(TASK_TEMPLATES)
        filled = template["pattern"]
        for slot, options in template["slots"].items():
            filled = filled.replace(f"{{{slot}}}", random.choice(options))
        tasks.append(_merge_task_metadata({
            "task": filled,
            "difficulty": template["difficulty"],
            "expected_tools": template["expected_tools"],
            "likely_failure_points": template["likely_failure_points"],
            "generation_strategy": "template_based",
            "freshness_source": "template_library",
            "execution_target": "synthetic",
            "task_type": "synthetic_template",
        }))
    return tasks


def _validate_task_tools(task: dict) -> dict:
    tools = task.get("expected_tools", [])
    valid = [tool for tool in tools if tool in REAL_CODING_TOOL_NAMES]
    if not valid:
        text = task.get("task", "").lower()
        if any(k in text for k in ["test", "pytest", "failing", "bug", "fix"]):
            valid = ["file_search", "code_search", "code_edit", "code_executor"]
        elif any(k in text for k in ["api", "client", "response", "request"]):
            valid = ["file_search", "code_search", "code_edit", "api_fetch", "code_executor"]
        else:
            valid = ["file_search", "code_search", "code_edit", "code_executor"]
    task["expected_tools"] = valid
    return task


def convert_signal_to_task(signal: dict) -> Optional[dict]:
    prompt = f"""You are creating a repo-grounded coding task for a real coding agent.

Repository: {signal.get('repo_full_name', signal.get('source', 'unknown'))}
Issue URL: {signal.get('source_url', '')}
Issue title: {signal.get('title', '')}
Issue labels: {signal.get('issue_labels', [])}
Path hints: {signal.get('path_hints', [])}

Issue body:
{signal.get('body', signal.get('raw_text', ''))[:1200]}

Requirements:
- The task must require repository inspection and code changes.
- The task must ask the agent to produce execution evidence, ideally targeted tests or another concrete validation command.
- expected_tools must be chosen only from: {REAL_CODING_TOOL_NAMES}
- Prefer 3-5 tools, not 1.
- If the issue does not look locally executable, return null.

Return ONLY JSON:
{{
  "task": "two-sentence repo-grounded coding task",
  "difficulty": "medium|complex",
  "expected_tools": ["file_search", "code_search", "code_edit", "code_executor"],
  "likely_failure_points": ["point1", "point2"],
  "generation_strategy": "llm_generative",
  "freshness_source": "{signal.get('source', '')}",
  "execution_target": "real_repo_issue",
  "task_type": "repo_issue_fix"
}}"""
    task = call_llm_json("generator", [{"role": "user", "content": prompt}], max_tokens=400)
    if not task or not isinstance(task, dict):
        return None
    task = _validate_task_tools(task)
    return _merge_task_metadata(task, signal)


def convert_signal_rule_based(signal: dict) -> dict:
    text = (signal.get("raw_text") or "").lower()
    source = (signal.get("source") or "").lower()
    title = signal.get("title") or signal.get("raw_text", "").split("\n")[0].strip()[:160]

    detected_service = "github"
    tools = ["file_search", "code_search", "code_edit", "code_executor"]
    difficulty = "complex"
    failures = ["issue not reproduced locally", "wrong file changed", "validation missing"]

    for keyword, mapping in SOURCE_TOOL_MAP.items():
        if keyword in text or keyword in source:
            detected_service = keyword
            tools, difficulty = mapping
            failures = FAILURE_MAP.get(keyword, failures)
            break

    task_text = random.choice(TASK_PATTERNS).format(title=title)
    task = {
        "task": task_text,
        "difficulty": difficulty,
        "expected_tools": tools,
        "likely_failure_points": failures[:3],
        "generation_strategy": "llm_generative",
        "freshness_source": signal.get("source", "unknown"),
        "execution_target": signal.get("execution_target", "real_repo_issue"),
        "task_type": "repo_issue_fix",
    }
    return _merge_task_metadata(_validate_task_tools(task), signal)


def mutate_task_rule_based(seed: dict) -> dict:
    mutations = [
        (" Also capture a failing test or exact validation command before editing, then rerun it after the patch.", "complex", "no before/after validation evidence"),
        (" Minimize the patch surface and explain which repository files are safe to ignore while debugging.", "complex", "edited unrelated files"),
        (" Ensure the fix works for the reported issue and a nearby edge case found during code inspection.", "complex", "edge case regression missed"),
        (" Record the most relevant command outputs so a supervisor can verify the fix is grounded in execution evidence.", "complex", "insufficient execution evidence"),
    ]
    suffix, new_difficulty, extra_failure = random.choice(mutations)
    base_task = seed.get("task", "")
    base_tools = seed.get("expected_tools", ["file_search", "code_search", "code_edit", "code_executor"])
    base_failures = seed.get("likely_failure_points", ["validation missing", "wrong file changed"])
    task = {
        "task": base_task + suffix,
        "difficulty": new_difficulty,
        "expected_tools": base_tools,
        "likely_failure_points": (base_failures + [extra_failure])[:4],
        "generation_strategy": "mutation_based",
        "freshness_source": "mutation_of_existing",
        "execution_target": seed.get("execution_target", "real_repo_issue"),
        "task_type": seed.get("task_type", "repo_issue_fix"),
        "source_url": seed.get("source_url", ""),
        "repo_url": seed.get("repo_url", ""),
        "repo_clone_url": seed.get("repo_clone_url", ""),
        "repo_full_name": seed.get("repo_full_name", ""),
        "repo_default_branch": seed.get("repo_default_branch", ""),
        "repo_language": seed.get("repo_language", ""),
        "issue_number": seed.get("issue_number"),
        "issue_title": seed.get("issue_title", ""),
        "issue_labels": seed.get("issue_labels", []),
        "path_hints": seed.get("path_hints", []),
    }
    return _validate_task_tools(task)


def init_registry():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            difficulty TEXT,
            expected_tools TEXT,
            failure_points TEXT,
            generation_strategy TEXT,
            freshness_source TEXT,
            source_url TEXT,
            repo_url TEXT,
            repo_clone_url TEXT,
            repo_full_name TEXT,
            repo_default_branch TEXT,
            repo_language TEXT,
            issue_number INTEGER,
            issue_title TEXT,
            issue_labels TEXT,
            path_hints TEXT,
            execution_target TEXT,
            task_type TEXT,
            created_at TEXT,
            executed_count INTEGER DEFAULT 0,
            niche_score REAL DEFAULT 0.0,
            status TEXT DEFAULT 'approved'
        )
    """)
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
    }
    extra_columns = {
        "source_url": "TEXT",
        "repo_url": "TEXT",
        "repo_clone_url": "TEXT",
        "repo_full_name": "TEXT",
        "repo_default_branch": "TEXT",
        "repo_language": "TEXT",
        "issue_number": "INTEGER",
        "issue_title": "TEXT",
        "issue_labels": "TEXT",
        "path_hints": "TEXT",
        "execution_target": "TEXT",
        "task_type": "TEXT",
    }
    for col, col_type in extra_columns.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


def task_fingerprint(task_text: str) -> str:
    return hashlib.md5(task_text.strip().lower().encode()).hexdigest()


def is_duplicate(task_text: str, freshness_source: str = "", source_url: str = "") -> bool:
    fp = task_fingerprint(task_text)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT 1 FROM tasks WHERE task_id = ?", (fp,)).fetchone()
    if row:
        conn.close()
        return True
    if source_url:
        row = conn.execute("SELECT 1 FROM tasks WHERE source_url = ?", (source_url,)).fetchone()
        if row:
            conn.close()
            return True
    if freshness_source and freshness_source not in ("template_library", "mutation_of_existing", ""):
        row = conn.execute("SELECT 1 FROM tasks WHERE freshness_source = ?", (freshness_source,)).fetchone()
        if row:
            conn.close()
            return True
    conn.close()
    return False


def save_task(task: dict) -> bool:
    fp = task_fingerprint(task["task"])
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR IGNORE INTO tasks (
                task_id, task, difficulty, expected_tools, failure_points,
                generation_strategy, freshness_source, source_url, repo_url,
                repo_clone_url, repo_full_name, repo_default_branch, repo_language,
                issue_number, issue_title, issue_labels, path_hints,
                execution_target, task_type, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fp,
            task["task"],
            task.get("difficulty", "medium"),
            json.dumps(task.get("expected_tools", [])),
            json.dumps(task.get("likely_failure_points", [])),
            task.get("generation_strategy", "unknown"),
            task.get("freshness_source", "unknown"),
            task.get("source_url", ""),
            task.get("repo_url", ""),
            task.get("repo_clone_url", ""),
            task.get("repo_full_name", ""),
            task.get("repo_default_branch", ""),
            task.get("repo_language", ""),
            task.get("issue_number"),
            task.get("issue_title", ""),
            json.dumps(task.get("issue_labels", [])),
            json.dumps(task.get("path_hints", [])),
            task.get("execution_target", "synthetic"),
            task.get("task_type", "generic_task"),
            datetime.now().strftime("%Y-%m-%d"),
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"  ❌ DB save error: {e}")
        return False


def load_approved_tasks(limit: int = 100) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT
            task_id, task, difficulty, expected_tools, failure_points,
            freshness_source, created_at, source_url, repo_url, repo_clone_url,
            repo_full_name, repo_default_branch, repo_language, issue_number,
            issue_title, issue_labels, path_hints, execution_target, task_type
        FROM tasks
        WHERE status = 'approved'
        ORDER BY executed_count ASC, created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    tasks = []
    for row in rows:
        tasks.append({
            "task_id": row[0],
            "task": row[1],
            "difficulty": row[2],
            "expected_tools": json.loads(row[3]),
            "likely_failure_points": json.loads(row[4]),
            "freshness_source": row[5],
            "created_at": row[6],
            "source_url": row[7] or "",
            "repo_url": row[8] or "",
            "repo_clone_url": row[9] or "",
            "repo_full_name": row[10] or "",
            "repo_default_branch": row[11] or "",
            "repo_language": row[12] or "",
            "issue_number": row[13],
            "issue_title": row[14] or "",
            "issue_labels": json.loads(row[15] or "[]"),
            "path_hints": json.loads(row[16] or "[]"),
            "execution_target": row[17] or "synthetic",
            "task_type": row[18] or "generic_task",
        })
    return tasks


def mark_executed(task_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE tasks SET executed_count = executed_count + 1 WHERE task_id = ?",
        (task_id,),
    )
    conn.commit()
    conn.close()


def get_registry_sample(n: int = 5) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT
            task, difficulty, expected_tools, failure_points, source_url, repo_url,
            repo_clone_url, repo_full_name, repo_default_branch, repo_language,
            issue_number, issue_title, issue_labels, path_hints,
            execution_target, task_type
        FROM tasks
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()
    return [{
        "task": row[0],
        "difficulty": row[1],
        "expected_tools": json.loads(row[2] or "[]"),
        "likely_failure_points": json.loads(row[3] or "[]"),
        "source_url": row[4] or "",
        "repo_url": row[5] or "",
        "repo_clone_url": row[6] or "",
        "repo_full_name": row[7] or "",
        "repo_default_branch": row[8] or "",
        "repo_language": row[9] or "",
        "issue_number": row[10],
        "issue_title": row[11] or "",
        "issue_labels": json.loads(row[12] or "[]"),
        "path_hints": json.loads(row[13] or "[]"),
        "execution_target": row[14] or "synthetic",
        "task_type": row[15] or "generic_task",
    } for row in rows]


NICHE_KEYWORDS = [
    "api", "stripe", "notion", "github", "slack", "airtable", "shopify",
    "salesforce", "hubspot", "webhook", "endpoint", "request", "response",
    "fetch", "sync", "integrate", "code", "script", "debug", "function",
    "python", "deploy", "pipeline", "automate", "parse", "extract", "transform",
    "repo", "repository", "test", "pytest", "issue", "patch",
]


def passes_rule_based_check(task: dict) -> bool:
    if not task or not task.get("task"):
        return False
    text = task["task"].lower()
    if len(text) < 40:
        return False
    if is_duplicate(task["task"], task.get("freshness_source", ""), task.get("source_url", "")):
        return False
    if task.get("execution_target") == "real_repo_issue":
        if not task.get("repo_full_name") or not task.get("source_url"):
            return False
        if len(task.get("expected_tools", [])) < 3:
            return False
    return any(keyword in text for keyword in NICHE_KEYWORDS)


DAILY_BUDGET = {
    "template_based": 2,
    "llm_generative": 6,
    "mutation_based": 4,
}


def generate_tasks(total: int = 100) -> list[dict]:
    init_registry()
    approved = []

    print("\n📋 Strategy 1: Template-based generation...")
    template_tasks = generate_template_tasks(count=DAILY_BUDGET["template_based"] + 6)
    for task in template_tasks:
        if len([x for x in approved if x.get("generation_strategy") == "template_based"]) >= DAILY_BUDGET["template_based"]:
            break
        if passes_rule_based_check(task):
            save_task(task)
            approved.append(task)
    print(f"  ✅ Template tasks approved: {len([x for x in approved if x.get('generation_strategy') == 'template_based'])}")

    print("\n🌐 Strategy 2: Repo-grounded GitHub issue tasks...")
    signals = collect_all_signals()
    llm_count = 0
    for signal in signals:
        if llm_count >= DAILY_BUDGET["llm_generative"]:
            break
        task = convert_signal_to_task(signal)
        if task is None:
            task = convert_signal_rule_based(signal)
        if task and passes_rule_based_check(task):
            save_task(task)
            mark_signal_used(signal)
            approved.append(task)
            llm_count += 1
    print(f"  ✅ Repo-grounded tasks approved: {llm_count}")

    print("\n🔀 Strategy 3: Mutation-based generation...")
    seed_tasks = [seed for seed in get_registry_sample(n=DAILY_BUDGET["mutation_based"] + 6) if seed.get("execution_target") == "real_repo_issue"]
    mutation_count = 0
    for seed in seed_tasks:
        if mutation_count >= DAILY_BUDGET["mutation_based"]:
            break
        task = mutate_task_rule_based(seed)
        if task and passes_rule_based_check(task):
            save_task(task)
            approved.append(task)
            mutation_count += 1
    print(f"  ✅ Mutation-based tasks approved: {mutation_count}")

    print(f"\n✅ Total tasks approved for today: {len(approved)}")
    return approved[:total]
