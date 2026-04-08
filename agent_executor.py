"""
agent_executor.py
Runs a ReAct-style agent on each task using OpenRouter free models.
Captures the full step-by-step trace: reasoning, tool calls, tool results, outcome.
Simulates realistic tool environments for API Orchestration + Code Agent tasks.
"""

import json
import uuid
import time
from datetime import datetime
from typing import Optional

from openrouter_client import call_llm, call_llm_json

# ── Tool Simulator ─────────────────────────────────────────────────────────────
# In a real deployment you'd call actual APIs.
# Here we simulate realistic tool responses so the agent can reason over them.

SIMULATED_TOOLS = {
    "stripe_list_invoices": {
        "description": "List invoices from Stripe. Args: status (open|paid|void), limit (int), due_date_lt (YYYY-MM-DD)",
        "mock_response": {
            "invoices": [
                {"id": "in_001", "customer": "Acme Corp",   "amount_due": 4500, "due_date": "2026-03-15", "currency": "usd"},
                {"id": "in_002", "customer": "Globex Ltd",  "amount_due": 1200, "due_date": "2026-03-28", "currency": "usd"},
                {"id": "in_003", "customer": "Initech Inc", "amount_due": 8900, "due_date": "2026-03-10", "currency": "usd"},
            ],
            "has_more": False,
        },
    },
    "notion_create_page": {
        "description": "Create a page in a Notion database. Args: database_id (str), properties (dict)",
        "mock_response": {"page_id": "pg_auto_generated", "status": "success", "url": "https://notion.so/page/xxx"},
    },
    "github_list_issues": {
        "description": "List open issues from a GitHub repo. Args: repo (owner/repo), labels (str), state (open|closed)",
        "mock_response": {
            "issues": [
                {"number": 101, "title": "Agent loop not terminating on tool error", "labels": ["bug"], "created_at": "2026-04-07"},
                {"number": 102, "title": "Add retry logic to API client", "labels": ["enhancement"], "created_at": "2026-04-06"},
            ]
        },
    },
    "code_executor": {
        "description": "Execute Python code in a sandbox. Args: code (str)",
        "mock_response": {"stdout": "Execution successful\nResult: 42", "stderr": "", "exit_code": 0},
    },
    "web_search": {
        "description": "Search the web for information. Args: query (str)",
        "mock_response": {
            "results": [
                {"title": "How to handle Stripe 429 errors", "snippet": "Use exponential backoff...", "url": "https://stripe.com/docs"},
                {"title": "Notion API rate limits", "snippet": "Notion allows 3 requests per second...", "url": "https://developers.notion.com"},
            ]
        },
    },
    "slack_send_message": {
        "description": "Send a message to a Slack channel. Args: channel (str), message (str)",
        "mock_response": {"ok": True, "ts": "1712345678.000100"},
    },
    "airtable_create_record": {
        "description": "Create a record in an Airtable base. Args: base_id (str), table_name (str), fields (dict)",
        "mock_response": {"id": "rec_auto", "createdTime": "2026-04-08T02:00:00.000Z", "fields": {}},
    },
    "api_fetch": {
        "description": "Make a GET request to any REST API. Args: url (str), headers (dict), params (dict)",
        "mock_response": {"status_code": 200, "data": {"items": [], "total": 0}},
    },
    "api_write": {
        "description": "Make a POST/PUT request to any REST API. Args: url (str), headers (dict), body (dict), method (str)",
        "mock_response": {"status_code": 201, "data": {"id": "new_resource_id", "status": "created"}},
    },
}

TOOL_LIST_STR = "\n".join([
    f"- {name}: {info['description']}"
    for name, info in SIMULATED_TOOLS.items()
])

# ── ReAct Agent System Prompt ──────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = f"""You are an expert AI agent specializing in API Orchestration and Code tasks.
You solve tasks step by step using a Thought → Action → Observation loop (ReAct pattern).

Available tools:
{TOOL_LIST_STR}

For each step, respond ONLY with this JSON format:
{{
  "thought": "Your reasoning about what to do next",
  "action": "tool_name OR 'finish'",
  "action_input": {{"arg1": "val1"}},
  "final_answer": "Only set this when action is 'finish'"
}}

Rules:
- Always think before acting
- Check tool results before proceeding to next step
- If a tool fails, reason about why and try an alternative approach
- When the task is fully complete, use action: "finish" with a final_answer
- Maximum 10 steps per task
"""


# ── Agent Runner ───────────────────────────────────────────────────────────────

def simulate_tool_call(tool_name: str, tool_input: dict) -> dict:
    """Return a simulated tool response for the given tool and input."""
    if tool_name not in SIMULATED_TOOLS:
        return {"error": f"Tool '{tool_name}' not found. Available tools: {list(SIMULATED_TOOLS.keys())}"}

    # Occasionally inject a realistic failure to make dataset diverse
    import random
    failure_chance = 0.15   # 15% chance of simulated failure per tool call
    if random.random() < failure_chance:
        failures = [
            {"error": "429 Too Many Requests", "retry_after": 60},
            {"error": "401 Unauthorized", "message": "Invalid API key"},
            {"error": "timeout", "message": "Request timed out after 30s"},
            {"error": "422 Unprocessable Entity", "message": "Missing required field"},
        ]
        return random.choice(failures)

    response = dict(SIMULATED_TOOLS[tool_name]["mock_response"])
    response["_tool_called"] = tool_name
    response["_args_received"] = tool_input
    return response


def run_agent_on_task(task: dict, max_steps: int = 10) -> dict:
    """
    Run the ReAct agent on a single task.
    Returns a complete trace dict ready for labeling.
    """
    trace_id = f"trace_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    messages  = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    steps     = []
    start_time = time.time()

    # Inject the task
    messages.append({
        "role":    "user",
        "content": f"Complete this task:\n\n{task['task']}"
    })

    outcome_status   = "failed"
    failure_reason   = "max_steps_reached"
    final_answer     = None
    tool_calls_made  = []

    for step_num in range(1, max_steps + 1):
        raw = call_llm(
            "agent",
            messages,
            temperature=0.4,
            max_tokens=1024,
        )

        if raw is None:
            # Try backup model
            raw = call_llm("agent_backup", messages, temperature=0.4, max_tokens=1024)

        if raw is None:
            failure_reason = "model_unavailable"
            break

        # Parse agent response
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            agent_response = json.loads(clean.strip())
        except json.JSONDecodeError:
            # Agent didn't follow JSON format — record as reasoning step
            agent_response = {
                "thought": raw,
                "action":  "reasoning_only",
                "action_input": {},
            }

        thought      = agent_response.get("thought", "")
        action       = agent_response.get("action", "")
        action_input = agent_response.get("action_input", {})

        step_record = {
            "step":         step_num,
            "type":         "reasoning",
            "content":      thought,
        }

        if action == "finish":
            final_answer   = agent_response.get("final_answer", "Task complete.")
            outcome_status = "success"
            failure_reason = None
            steps.append(step_record)
            steps.append({
                "step":         step_num,
                "type":         "finish",
                "content":      final_answer,
            })
            break

        elif action and action != "reasoning_only":
            # Execute the tool
            tool_start = time.time()
            tool_result = simulate_tool_call(action, action_input)
            latency_ms  = int((time.time() - tool_start) * 1000) + 150   # simulate network

            tool_calls_made.append(action)

            tool_step = {
                "step":        step_num,
                "type":        "tool_call",
                "tool":        action,
                "arguments":   action_input,
                "result":      tool_result,
                "latency_ms":  latency_ms,
            }
            steps.append(step_record)
            steps.append(tool_step)

            # Check if tool returned an error
            if "error" in tool_result:
                failure_reason = f"tool_error:{action}"
                # Don't break — let agent try to recover

            # Feed tool result back to agent
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role":    "user",
                "content": f"Tool result for {action}:\n{json.dumps(tool_result, indent=2)}\n\nContinue."
            })
        else:
            steps.append(step_record)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Continue with the next step."})

    duration_s = round(time.time() - start_time, 2)

    # Determine final outcome
    if outcome_status != "success":
        if any("error" in str(s.get("result", "")) for s in steps if s.get("type") == "tool_call"):
            outcome_status  = "partial"
            failure_reason  = "tool_error_unrecovered"
        else:
            outcome_status  = "failed"

    return {
        "trace_id":   trace_id,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "task":       task,
        "trace":      steps,
        "outcome": {
            "status":          outcome_status,
            "total_steps":     len(steps),
            "total_tool_calls": len(tool_calls_made),
            "tools_used":      list(set(tool_calls_made)),
            "failure_occurred": outcome_status != "success",
            "failure_reason":   failure_reason,
            "final_answer":    final_answer,
            "duration_seconds": duration_s,
        },
        "metadata": {
            "agent_framework":    "react",
            "agent_model":       "llama-3.3-70b-instruct",
            "world_context_date": datetime.now().strftime("%Y-%m-%d"),
            "schema_version":    "v1.0",
        }
    }


# ── Batch Executor ─────────────────────────────────────────────────────────────

def execute_tasks(tasks: list[dict]) -> list[dict]:
    """Run the agent on all tasks. Returns list of trace dicts."""
    from task_generator import mark_executed

    traces = []
    total  = len(tasks)

    print(f"\n🤖 Executing {total} tasks with ReAct agent...")
    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{total}] {task['task'][:70]}...")
        try:
            trace = run_agent_on_task(task)
            traces.append(trace)
            # Mark task as executed in registry
            if task.get("task_id"):
                mark_executed(task["task_id"])
        except Exception as e:
            print(f"  ❌ Task execution failed: {e}")

        # Small delay between tasks to respect rate limits
        time.sleep(2)

    success_count = sum(1 for t in traces if t["outcome"]["status"] == "success")
    partial_count = sum(1 for t in traces if t["outcome"]["status"] == "partial")
    failed_count  = sum(1 for t in traces if t["outcome"]["status"] == "failed")

    print(f"\n✅ Execution complete:")
    print(f"   Success:  {success_count}")
    print(f"   Partial:  {partial_count}")
    print(f"   Failed:   {failed_count}")

    return traces
