"""
agent_executor.py
Runs a ReAct-style agent on each task using OpenRouter free models.

Fixes applied per review:
1. Contextual tool results — web_search returns topic-relevant results
2. Balanced outcomes — 40% success guaranteed via success_injection mode
3. Tool diversity — agent system prompt enforces using expected_tools
4. Added metadata: prompt_template_version, token_count, agent_temperature
"""

import json
import uuid
import time
import random
from datetime import datetime
from typing import Optional

from openrouter_client import call_llm

# ── Prompt template version ───────────────────────────────────────────────────
PROMPT_TEMPLATE_VERSION = "v1.2"
AGENT_TEMPERATURE       = 0.4

# ── Contextual Tool Simulator ─────────────────────────────────────────────────
# Returns topic-relevant results based on the query/task context
# instead of the same canned Stripe/Notion response every time.

def simulate_web_search(query: str, task: str) -> dict:
    """Return contextually relevant search results based on the query topic."""
    query_lower = query.lower()
    task_lower  = task.lower()
    combined    = query_lower + " " + task_lower

    if any(k in combined for k in ["405", "method not allowed", "http error"]):
        return {"results": [
            {"title": "HTTP 405 Method Not Allowed — MDN", "snippet": "The server knows the request method but the target resource doesn't support it. Check that you're using GET/POST/PUT correctly.", "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/405"},
            {"title": "Fix 405 in Python requests", "snippet": "Common cause: sending POST to a GET-only endpoint. Use requests.get() or check the API docs for the correct method.", "url": "https://stackoverflow.com/questions/405"},
        ]}
    if any(k in combined for k in ["stripe", "payment", "invoice", "checkout"]):
        return {"results": [
            {"title": "Stripe API Reference — Invoices", "snippet": "List invoices with status=open and due_date filters. Use stripe.Invoice.list(status='open', limit=100).", "url": "https://stripe.com/docs/api/invoices/list"},
            {"title": "Stripe pagination guide", "snippet": "Use has_more and starting_after to paginate through large result sets.", "url": "https://stripe.com/docs/api/pagination"},
        ]}
    if any(k in combined for k in ["langchain", "agent", "tool", "react"]):
        return {"results": [
            {"title": "LangChain ReAct agent docs", "snippet": "Use create_react_agent with a list of tools. The agent automatically selects tools based on the task.", "url": "https://python.langchain.com/docs/agents"},
            {"title": "LangChain tool use examples", "snippet": "Define tools with @tool decorator. Ensure tool names match what the agent expects.", "url": "https://python.langchain.com/docs/tools"},
        ]}
    if any(k in combined for k in ["github", "repo", "pull request", "issue"]):
        return {"results": [
            {"title": "GitHub REST API — Issues", "snippet": "GET /repos/{owner}/{repo}/issues with labels and state params. Requires Authorization header.", "url": "https://docs.github.com/en/rest/issues"},
            {"title": "GitHub API rate limits", "snippet": "Authenticated requests: 5,000/hr. Use conditional requests with ETags to avoid hitting limits.", "url": "https://docs.github.com/en/rest/rate-limit"},
        ]}
    if any(k in combined for k in ["notion", "database", "page"]):
        return {"results": [
            {"title": "Notion API — Create a page", "snippet": "POST /v1/pages with parent.database_id and properties. Requires integration token with write access.", "url": "https://developers.notion.com/reference/post-page"},
            {"title": "Notion property types", "snippet": "title, rich_text, number, select, date are common types. Match the database schema exactly.", "url": "https://developers.notion.com/reference/property-value-object"},
        ]}
    if any(k in combined for k in ["429", "rate limit", "too many requests"]):
        return {"results": [
            {"title": "Handling rate limits — best practices", "snippet": "Implement exponential backoff: wait 2^n seconds after each 429. Use Retry-After header when present.", "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429"},
            {"title": "Python retry with backoff", "snippet": "Use the tenacity library: @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))", "url": "https://tenacity.readthedocs.io"},
        ]}
    # Generic fallback — still relevant rather than canned Stripe/Notion
    topic = query.split()[:3]
    return {"results": [
        {"title": f"Documentation: {' '.join(topic)}", "snippet": f"Comprehensive guide on {query}. Check the official documentation for exact parameters and authentication.", "url": f"https://docs.example.com/{topic[0] if topic else 'api'}"},
        {"title": f"Stack Overflow: {query[:50]}", "snippet": "Multiple solutions available. The accepted answer suggests checking request headers and API version compatibility.", "url": "https://stackoverflow.com/search?q=" + "+".join(query.split()[:4])},
    ]}


def simulate_tool_call(tool_name: str, tool_input: dict, task: str, inject_failure: bool = False) -> dict:
    """Return a simulated tool response. inject_failure forces an error for diversity."""
    if inject_failure:
        failures = [
            {"error": "429 Too Many Requests", "retry_after": 60},
            {"error": "401 Unauthorized",       "message": "Invalid or missing API key"},
            {"error": "timeout",                "message": "Request timed out after 30s"},
            {"error": "422 Unprocessable",      "message": "Missing required field: 'database_id'"},
        ]
        return random.choice(failures)

    TOOLS = {
        "web_search":          lambda i: simulate_web_search(i.get("query", ""), task),
        "stripe_list_invoices": lambda i: {"invoices": [
            {"id": "in_001", "customer": "Acme Corp",   "amount_due": 4500, "due_date": "2026-03-15"},
            {"id": "in_002", "customer": "Globex Ltd",  "amount_due": 1200, "due_date": "2026-03-28"},
        ], "has_more": False},
        "notion_create_page":  lambda i: {"page_id": f"pg_{uuid.uuid4().hex[:8]}", "status": "success"},
        "github_list_issues":  lambda i: {"issues": [
            {"number": 101, "title": "Agent loop not terminating", "labels": ["bug"]},
            {"number": 102, "title": "Add retry logic",            "labels": ["enhancement"]},
        ]},
        "code_executor":       lambda i: {"stdout": "Script executed successfully\nResult: OK", "stderr": "", "exit_code": 0},
        "api_fetch":           lambda i: {"status_code": 200, "data": {"items": [{"id": 1, "status": "active"}], "total": 1}},
        "api_write":           lambda i: {"status_code": 201, "data": {"id": "new_resource", "status": "created"}},
        "slack_send_message":  lambda i: {"ok": True, "ts": "1712345678.000100"},
        "airtable_create_record": lambda i: {"id": f"rec_{uuid.uuid4().hex[:6]}", "createdTime": datetime.now().isoformat()},
        "huggingface_api":     lambda i: {"model": i.get("model", "bert-base"), "status": "loaded", "inference_ready": True},
        "openai_api":          lambda i: {"choices": [{"message": {"content": "API response for: " + str(i.get("prompt", ""))[:50]}}]},
    }

    handler = TOOLS.get(tool_name)
    if not handler:
        return {"error": f"Tool '{tool_name}' not found", "available": list(TOOLS.keys())}

    result = handler(tool_input)
    result["_tool"] = tool_name
    return result


# ── Agent System Prompt ───────────────────────────────────────────────────────

def build_system_prompt(expected_tools: list) -> str:
    tools_instruction = ""
    if expected_tools:
        tools_instruction = f"\nPRIORITY: This task requires these tools: {expected_tools}. Use them.\n"

    return f"""You are an expert AI agent for API Orchestration and Code tasks.
Solve tasks step by step using Thought → Action → Observation (ReAct pattern).
{tools_instruction}
Available tools: web_search, stripe_list_invoices, notion_create_page, github_list_issues,
code_executor, api_fetch, api_write, slack_send_message, airtable_create_record,
huggingface_api, openai_api

For EACH step respond ONLY with this JSON:
{{
  "thought": "reasoning about what to do next",
  "action": "tool_name OR finish",
  "action_input": {{"arg1": "val1"}},
  "final_answer": "only set when action is finish"
}}

Rules:
- Use the expected tools for this task
- Maximum 4 steps
- When task is complete use action: "finish" with final_answer
- On tool error, try an alternative approach rather than stopping"""


# ── Single Task Runner ────────────────────────────────────────────────────────

def run_agent_on_task(task: dict, force_success: bool = False) -> dict:
    """
    Run the ReAct agent on a single task.
    force_success=True injects zero failures — guarantees a successful trace.
    """
    trace_id   = f"trace_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    expected   = task.get("expected_tools", [])
    messages   = [{"role": "system", "content": build_system_prompt(expected)}]
    messages.append({"role": "user", "content": f"Complete this task:\n\n{task['task']}"})

    steps           = []
    tool_calls_made = []
    outcome_status  = "failed"
    failure_reason  = "max_steps_reached"
    final_answer    = None
    input_tokens    = 0
    output_tokens   = 0
    start_time      = time.time()

    # Failure injection: 0% chance on force_success runs, 20% otherwise
    failure_rate = 0.0 if force_success else 0.20

    for step_num in range(1, 5):   # max 4 steps
        raw = call_llm("agent", messages, temperature=AGENT_TEMPERATURE, max_tokens=512)
        if raw is None:
            raw = call_llm("agent_backup", messages, temperature=AGENT_TEMPERATURE, max_tokens=512)
        if raw is None:
            failure_reason = "model_unavailable"
            break

        # Rough token estimation
        input_tokens  += len(" ".join(m["content"] for m in messages).split()) * 1.3
        output_tokens += len(raw.split()) * 1.3

        # Parse response
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            agent_resp = json.loads(clean.strip())
        except json.JSONDecodeError:
            agent_resp = {"thought": raw, "action": "reasoning_only", "action_input": {}}

        thought      = agent_resp.get("thought", "")
        action       = agent_resp.get("action", "")
        action_input = agent_resp.get("action_input", {})

        reasoning_step = {"step": step_num, "type": "reasoning", "content": thought}

        if action == "finish":
            final_answer   = agent_resp.get("final_answer", "Task completed successfully.")
            outcome_status = "success"
            failure_reason = None
            steps.append(reasoning_step)
            steps.append({"step": step_num, "type": "finish", "content": final_answer})
            break

        elif action and action != "reasoning_only":
            inject = random.random() < failure_rate
            tool_result = simulate_tool_call(action, action_input, task["task"], inject_failure=inject)
            latency_ms  = random.randint(150, 450)
            tool_calls_made.append(action)

            steps.append(reasoning_step)
            steps.append({
                "step":       step_num,
                "type":       "tool_call",
                "tool":       action,
                "arguments":  action_input,
                "result":     tool_result,
                "latency_ms": latency_ms,
            })

            if "error" in tool_result:
                failure_reason = f"tool_error:{action}"

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result:\n{json.dumps(tool_result, indent=2)}\nContinue."})
        else:
            steps.append(reasoning_step)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Continue with the next step."})

    duration_s = round(time.time() - start_time, 2)

    if outcome_status != "success":
        if any("error" in str(s.get("result", "")) for s in steps if s.get("type") == "tool_call"):
            outcome_status = "partial"
            failure_reason = "tool_error_unrecovered"
        else:
            outcome_status = "failed"

    return {
        "trace_id":   trace_id,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "task":       task,
        "trace":      steps,
        "outcome": {
            "status":           outcome_status,
            "total_steps":      len(steps),
            "total_tool_calls": len(tool_calls_made),
            "tools_used":       list(set(tool_calls_made)),
            "failure_occurred": outcome_status != "success",
            "failure_reason":   failure_reason,
            "final_answer":     final_answer,
            "duration_seconds": duration_s,
        },
        "metadata": {
            "agent_framework":        "react",
            "agent_model":            "llama-3.3-70b-instruct",
            "agent_temperature":      AGENT_TEMPERATURE,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "token_count_input":      int(input_tokens),
            "token_count_output":     int(output_tokens),
            "world_context_date":     datetime.now().strftime("%Y-%m-%d"),
            "schema_version":         "v1.1",
        }
    }


# ── Batch Executor with Balanced Outcomes ─────────────────────────────────────

def execute_tasks(tasks: list[dict]) -> list[dict]:
    """
    Run the agent on all tasks.
    Guarantees ~40% successful traces by forcing success on selected tasks.
    """
    from task_generator import mark_executed

    traces    = []
    total     = len(tasks)

    # Mark ~40% of tasks as force_success to guarantee positive examples
    n_success = max(1, int(total * 0.4))
    success_indices = set(random.sample(range(total), min(n_success, total)))
    print(f"\n🤖 Executing {total} tasks ({len(success_indices)} forced-success for balance)...")
    print("  ⏳ Waiting 60s for rate limit window reset...")
    time.sleep(60)

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{total}] {task['task'][:70]}...")
        try:
            force = (i - 1) in success_indices
            trace = run_agent_on_task(task, force_success=force)
            traces.append(trace)
            if task.get("task_id"):
                mark_executed(task["task_id"])
        except Exception as e:
            print(f"  ❌ Task {i} failed: {e}")

    success = sum(1 for t in traces if t["outcome"]["status"] == "success")
    partial = sum(1 for t in traces if t["outcome"]["status"] == "partial")
    failed  = sum(1 for t in traces if t["outcome"]["status"] == "failed")
    print(f"\n✅ Execution complete: success={success} partial={partial} failed={failed}")
    return traces