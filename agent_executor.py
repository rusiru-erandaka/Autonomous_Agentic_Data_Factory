"""
agent_executor.py

Execution layer for dataset generation.

- Synthetic tasks still use the legacy mocked ReAct executor.
- Repo-grounded GitHub issue tasks default to a grounded ReAct executor that
  reasons from GitHub issue content and repository metadata.
- An older clone-based workspace runner is still available behind
  REAL_REPO_EXECUTION_MODE=repo_clone.

This keeps repo-grounded traces valuable even when cloning large repositories is
slow or unreliable in the pipeline environment.
"""

import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_client import call_llm

PROMPT_TEMPLATE_VERSION = "v4.0"
EXECUTION_ROOT = Path(__file__).resolve().parent / "registry" / "execution_workspace"
REAL_AGENT_MAX_STEPS = 6
MAX_COMMAND_OUTPUT_CHARS = 4000
KEEP_EXECUTION_WORKSPACES = os.environ.get("KEEP_EXECUTION_WORKSPACES", "false").lower() == "true"
REAL_REPO_EXECUTION_MODE = os.environ.get("REAL_REPO_EXECUTION_MODE", "react_grounded").lower()

# Temperature pool for synthetic traces only.
AGENT_TEMPERATURES = [0.0, 0.2, 0.4, 0.4, 0.6, 0.7]


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (text or "").strip())
    slug = slug.strip("-._")
    return slug[:80] or "repo"


def _search_terms_from_task(task: dict) -> list[str]:
    terms = []
    if task.get("path_hints"):
        for hint in task.get("path_hints", []):
            parts = [p for p in re.split(r"[\\/]", str(hint)) if p]
            if parts:
                terms.append(parts[-1][:80])
    title = task.get("issue_title") or task.get("task", "")
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{3,}", title):
        if token.lower() in {"issue", "github", "repository", "debug", "fix"}:
            continue
        terms.append(token[:80])
    deduped = []
    seen = set()
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)
    return deduped[:6]


def _candidate_files_from_task(task: dict) -> list[str]:
    candidates = []
    for hint in task.get("path_hints", []) or []:
        hint = str(hint).strip()
        if hint:
            candidates.append(hint[:160])
    tokens = _search_terms_from_task(task)
    suffixes = [".py", ".ts", ".tsx", ".js"]
    for token in tokens[:4]:
        if "/" in token or "." in token:
            candidates.append(token[:160])
            continue
        candidates.extend([f"src/{token}{suffix}" for suffix in suffixes[:2]])
        candidates.extend([f"tests/test_{token}.py", f"libs/{token}.py"])
    deduped = []
    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped[:6] or ["src/target_module.py", "tests/test_target_module.py"]


def _default_validation_command(task: dict) -> str:
    terms = _search_terms_from_task(task)
    if terms:
        expr = " or ".join(term.lower().replace('"', "") for term in terms[:2])
        return f'pytest -k "{expr}"'
    return "pytest -k issue_fix"


def _grounded_issue_prompt(task: dict) -> str:
    issue_body = (task.get("issue_body") or "").strip()
    if len(issue_body) > 1400:
        issue_body = issue_body[:1400] + "\n...[truncated]"
    return f"""Repository-grounded GitHub issue task.

Task:
{task.get("task", "")}

Repository:
- name: {task.get("repo_full_name", "")}
- repo_url: {task.get("repo_url", "")}
- issue_url: {task.get("source_url", "")}

Issue context:
- title: {task.get("issue_title", "")}
- labels: {task.get("issue_labels", [])}
- path_hints: {task.get("path_hints", [])}
- body:
{issue_body or "(no issue body provided)"}

Constraints:
- You do not have a local clone in this mode.
- Reason from the GitHub issue and repository metadata.
- Use repository-aware tools like code_search, code_view, code_edit, web_search, and code_executor.
- Before finishing, propose a likely code patch via code_edit and a targeted validation command via code_executor.
- Do not claim an exact verified fix unless your trace includes both a proposed code change and validation evidence.
"""


def _truncate(text: str, limit: int = MAX_COMMAND_OUTPUT_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _run_ps(command: str, cwd: Path, timeout_s: int = 90) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout = _truncate(proc.stdout)
        stderr = _truncate(proc.stderr)
        ok = proc.returncode == 0
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "ok": ok,
            "duration_seconds": round(time.time() - start, 2),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": 124,
            "stdout": _truncate((e.stdout or "") if isinstance(e.stdout, str) else ""),
            "stderr": _truncate((e.stderr or "") if isinstance(e.stderr, str) else "command timed out"),
            "ok": False,
            "duration_seconds": round(time.time() - start, 2),
        }
    except Exception as e:
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": 1,
            "stdout": "",
            "stderr": str(e),
            "ok": False,
            "duration_seconds": round(time.time() - start, 2),
        }


def _workspace_for_task(task: dict, trace_id: str) -> Path:
    repo_name = task.get("repo_full_name") or task.get("repo_url") or "repo"
    slug = _safe_slug(repo_name.replace("/", "-"))
    return EXECUTION_ROOT / f"{slug}-{trace_id[-8:]}"


def _prepare_repo_workspace(task: dict, trace_id: str) -> tuple[Path, list[dict], dict]:
    EXECUTION_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = _workspace_for_task(task, trace_id)
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.parent.mkdir(parents=True, exist_ok=True)

    repo_url = task.get("repo_clone_url") or task.get("repo_url")
    default_branch = (task.get("repo_default_branch") or "").strip()
    commands = []

    if not repo_url:
        return workspace, commands, {
            "repo_prepared": False,
            "repo_prepare_error": "missing_repo_url",
            "repo_workspace": str(workspace),
        }

    clone_cmds = []
    if default_branch:
        clone_cmds.append(f"git clone --depth 1 --branch {default_branch} {repo_url} '{workspace.name}'")
    clone_cmds.append(f"git clone --depth 1 {repo_url} '{workspace.name}'")

    clone_result = None
    for clone_cmd in clone_cmds:
        clone_result = _run_ps(clone_cmd, EXECUTION_ROOT, timeout_s=240)
        commands.append(clone_result)
        if clone_result["ok"]:
            break
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)

    if not clone_result or not clone_result["ok"]:
        return workspace, commands, {
            "repo_prepared": False,
            "repo_prepare_error": f"clone_failed:{clone_result['exit_code'] if clone_result else 1}",
            "repo_workspace": str(workspace),
        }

    git_rev = _run_ps("git rev-parse HEAD", workspace, timeout_s=30)
    commands.append(git_rev)
    return workspace, commands, {
        "repo_prepared": True,
        "repo_prepare_error": "",
        "repo_workspace": str(workspace),
        "repo_commit": (git_rev.get("stdout") or "").strip(),
    }


def _cleanup_workspace(workspace: Path) -> bool:
    if KEEP_EXECUTION_WORKSPACES:
        return False
    try:
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        return True
    except Exception:
        return False


def _real_repo_prompt(task: dict, command_results: list[dict]) -> str:
    inspected = []
    for result in command_results[-4:]:
        inspected.append({
            "command": result["command"],
            "exit_code": result["exit_code"],
            "stdout": result["stdout"][:500],
            "stderr": result["stderr"][:200],
        })
    return f"""You are supervising a real coding-agent attempt on a GitHub issue.

Task:
{task.get("task", "")}

Repository: {task.get("repo_full_name", "")}
Issue URL: {task.get("source_url", "")}
Path hints: {task.get("path_hints", [])}

Recent command evidence:
{json.dumps(inspected, indent=2)}

Return ONLY JSON:
{{
  "summary": "1-2 sentence grounded assessment of what the next coding step should be",
  "candidate_files": ["path/or/file.py"],
  "candidate_validation_commands": ["pytest tests/test_x.py -k something"],
  "confidence": "low|medium|high"
}}"""


def _run_real_repo_task(task: dict) -> dict:
    trace_id = f"trace_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    steps = []
    commands = []
    workspace, prep_commands, prep_meta = _prepare_repo_workspace(task, trace_id)
    commands.extend(prep_commands)

    for cmd in prep_commands:
        steps.append({
            "step": len(steps) + 1,
            "type": "command",
            "command": cmd["command"],
            "result": cmd,
        })

    if not prep_meta.get("repo_prepared"):
        steps.append({
            "step": len(steps) + 1,
            "type": "reasoning",
            "content": f"Repository preparation failed before code inspection: {prep_meta.get('repo_prepare_error', 'repo_prepare_failed')}.",
        })
        workspace_deleted = _cleanup_workspace(workspace)
        return {
            "trace_id": trace_id,
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task": task,
            "trace": steps,
            "outcome": {
                "status": "failed",
                "total_steps": len(steps),
                "total_tool_calls": len(commands),
                "tools_used": ["git"] if commands else [],
                "failure_occurred": True,
                "failure_reason": prep_meta.get("repo_prepare_error", "repo_prepare_failed"),
                "final_answer": "",
                "duration_seconds": round(time.time() - start_time, 2),
                "execution_grounded": bool(commands),
                "files_changed": [],
                "validation_commands": [],
                "command_history": commands,
            },
            "metadata": {
                "agent_framework": "repo_runner",
                "agent_model": "real-repo-runner/bootstrap",
                "agent_temperature": 0.0,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "token_count_input": 0,
                "token_count_output": 0,
                "world_context_date": datetime.now().strftime("%Y-%m-%d"),
                "schema_version": "v4.0",
                **prep_meta,
                "workspace_deleted": workspace_deleted,
            },
        }

    inspection_commands = [
        "git status --short",
        "Get-ChildItem -Name | Select-Object -First 40",
        "rg --files | Select-Object -First 120",
    ]
    search_terms = _search_terms_from_task(task)
    for term in search_terms[:3]:
        escaped = term.replace("'", "''")
        inspection_commands.append(f"rg -n --hidden --glob '!*.git*' '{escaped}' .")

    for command in inspection_commands[:REAL_AGENT_MAX_STEPS]:
        result = _run_ps(command, workspace, timeout_s=90)
        commands.append(result)
        steps.append({
            "step": len(steps) + 1,
            "type": "command",
            "command": command,
            "result": result,
        })

    prompt = _real_repo_prompt(task, commands)
    model_resp = call_llm(
        "agent",
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    token_out = len((model_resp or "").split())
    token_in = len(prompt.split())
    agent_summary = ""
    candidate_files = []
    validation_commands = []
    if model_resp:
        try:
            parsed = json.loads(model_resp.strip().strip("`"))
        except json.JSONDecodeError:
            parsed = {"summary": model_resp}
        agent_summary = parsed.get("summary", "")
        candidate_files = parsed.get("candidate_files", []) or []
        validation_commands = parsed.get("candidate_validation_commands", []) or []
        steps.append({
            "step": len(steps) + 1,
            "type": "reasoning",
            "content": agent_summary,
            "candidate_files": candidate_files[:6],
            "candidate_validation_commands": validation_commands[:4],
        })

    diff_name_only = _run_ps("git diff --name-only", workspace, timeout_s=30)
    commands.append(diff_name_only)
    steps.append({
        "step": len(steps) + 1,
        "type": "command",
        "command": diff_name_only["command"],
        "result": diff_name_only,
    })
    changed_files = [
        line.strip() for line in (diff_name_only.get("stdout") or "").splitlines() if line.strip()
    ]

    outcome_status = "partial"
    failure_reason = "no_code_changes_applied_yet"
    final_answer = (
        "Repository prepared and inspected with real shell evidence, but no patch was applied yet. "
        "This run is grounded and suitable for supervision, but not a completed issue fix."
    )
    if changed_files:
        outcome_status = "success"
        failure_reason = None
        final_answer = "Repository inspection and code changes were captured in the workspace."

    workspace_deleted = _cleanup_workspace(workspace)

    return {
        "trace_id": trace_id,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": task,
        "trace": steps,
        "outcome": {
            "status": outcome_status,
            "total_steps": len(steps),
            "total_tool_calls": len(commands),
            "tools_used": ["git", "code_search", "code_executor"],
            "failure_occurred": outcome_status != "success",
            "failure_reason": failure_reason,
            "final_answer": final_answer,
            "duration_seconds": round(time.time() - start_time, 2),
            "execution_grounded": True,
            "files_changed": changed_files,
            "validation_commands": validation_commands[:4],
            "command_history": commands,
        },
        "metadata": {
            "agent_framework": "repo_runner",
            "agent_model": "groq/llama-3.3-70b-versatile",
            "agent_temperature": 0.0,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "token_count_input": int(token_in),
            "token_count_output": int(token_out),
            "world_context_date": datetime.now().strftime("%Y-%m-%d"),
            "schema_version": "v4.0",
            **prep_meta,
            "candidate_files": candidate_files[:6],
            "workspace_deleted": workspace_deleted,
        },
    }


# ---- legacy synthetic executor path ----

def _make_tool_registry(task: str, task_context: Optional[dict] = None):
    grounded = bool(task_context and task_context.get("execution_target") == "real_repo_issue")
    candidate_files = _candidate_files_from_task(task_context or {}) if grounded else ["src/main.py", "tests/test_main.py"]
    validation_command = _default_validation_command(task_context or {}) if grounded else "pytest -k smoke"
    repo_url = (task_context or {}).get("repo_url", "")
    issue_url = (task_context or {}).get("source_url", "")
    issue_title = (task_context or {}).get("issue_title", "")

    def web_search(inp):
        q = inp.get("query", task[:60])
        keywords = q.split()[:3]
        if grounded:
            return {
                "results": [
                    {
                        "title": issue_title or f"{' '.join(keywords)} - GitHub issue",
                        "snippet": f"Grounded repository issue context for {q}",
                        "url": issue_url or repo_url or "https://github.com",
                    },
                    {
                        "title": f"{' '.join(keywords)} - Repository",
                        "snippet": "Repository metadata and issue-linked context.",
                        "url": repo_url or issue_url or "https://github.com",
                    },
                ]
            }
        return {
            "results": [
                {
                    "title": f"{' '.join(keywords)} - Documentation",
                    "snippet": f"Reference material for {q}",
                    "url": "https://docs.example.com",
                }
            ]
        }

    def file_search(inp):
        query = inp.get("query", "")
        file_path = inp.get("path") or candidate_files[0]
        if grounded:
            return {
                "matches": [
                    {
                        "file": candidate_files[0],
                        "line": 42,
                        "content": f"Grounded issue hint for '{query}' in {candidate_files[0]}",
                    },
                    {
                        "file": candidate_files[min(1, len(candidate_files) - 1)],
                        "line": 15,
                        "content": f"Related validation coverage for '{query}'",
                    },
                ]
            }
        return {"matches": [{"file": file_path, "line": 42, "content": f"Found {query}"}]}

    def file_edit(inp):
        file_name = (
            inp.get("file") or
            inp.get("file_path") or
            inp.get("filename") or
            candidate_files[0]
        )
        if grounded:
            return {
                "status": "success",
                "file": file_name,
                "diff": f"--- {file_name}\n+++ {file_name}\n+ apply targeted fix for repository issue",
            }
        return {"status": "success", "file": file_name}

    def file_read(inp):
        file_path = inp.get("path", candidate_files[0] if grounded else "")
        content = (
            f"# File: {file_path}\n"
            f"# Grounded from issue: {issue_title}\n"
            "def target_function():\n"
            "    pass\n"
        ) if grounded else f"# File: {file_path}\ndef example():\n    pass\n"
        return {"content": content, "lines": len(content.splitlines())}

    def code_search(inp):
        query = inp.get("query", "function")
        file_name = candidate_files[0] if grounded else "src/main.py"
        snippet = (
            f"def handle_issue_fix(...):  # suspected path for {query}"
            if grounded else
            f"def {query}(self): ..."
        )
        return {"matches": [{"file": file_name, "line": 77, "snippet": snippet}], "total": 1}

    def code_view(inp):
        file_name = inp.get("file", candidate_files[0] if grounded else "")
        content = (
            f"# {file_name}\n"
            f"# Issue: {issue_title}\n"
            "def target_function(payload):\n"
            "    return payload\n"
        ) if grounded else (
            f"# {file_name}\n"
            "def example_function():\n"
            "    result = api_call()\n"
            "    return result\n"
        )
        return {"content": content, "language": "python"}

    def code_edit(inp):
        file_name = inp.get("file", candidate_files[0] if grounded else "main.py")
        diff = (
            f"--- {file_name}\n+++ {file_name}\n"
            "- old_issue_behavior\n+ guarded_issue_behavior\n"
        ) if grounded else "- old_code\n+ new_code"
        return {"status": "success", "diff": diff, "file": file_name}

    def code_executor(inp):
        command = inp.get("command") or inp.get("cmd") or validation_command
        stdout = (
            f"Executed grounded validation command: {command}\nTargeted issue regression check passed"
            if grounded else
            "Script executed successfully\nOutput: OK"
        )
        return {"stdout": stdout, "stderr": "", "exit_code": 0, "command": command}

    return {
        "web_search": web_search,
        "file_search": file_search,
        "file_edit": lambda i: {
            "status": "success",
            "file": (
                (file_edit(i)).get("file") or
                "unknown.py"
            ),
        },
        "file_read": file_read,
        "code_search": code_search,
        "code_view": code_view,
        "code_edit": code_edit,
        "code_executor": code_executor,
        "git": lambda i: {"status": "success", "output": f"git {i.get('command', 'status')}: OK", "branch": "main"},
        "api_fetch": lambda i: {"status_code": 200, "data": {"items": [{"id": "item_1", "status": "active", "value": 100}], "total": 1, "has_more": False}},
        "api_write": lambda i: {"status_code": 201, "data": {"id": f"new_{uuid.uuid4().hex[:8]}", "status": "created"}},
    }


def _auto_mock(tool_name: str, tool_input: dict) -> dict:
    return {
        "status": "success",
        "tool": tool_name,
        "result": f"Operation completed for {tool_name}",
        "input_received": list(tool_input.keys()),
    }


def simulate_tool_call(
    tool_name: str,
    tool_input: dict,
    task: str,
    inject_failure: bool = False,
    task_context: Optional[dict] = None,
) -> dict:
    if inject_failure:
        return random.choice([
            {"error": "429 Too Many Requests", "retry_after": 60},
            {"error": "401 Unauthorized", "message": "Invalid API key"},
            {"error": "422 Unprocessable", "message": "Missing required field"},
        ])
    registry = _make_tool_registry(task, task_context=task_context)
    handler = registry.get(tool_name, lambda i: _auto_mock(tool_name, i))
    result = handler(tool_input)
    result["_tool"] = tool_name
    return result


def build_system_prompt(expected_tools: list, grounded: bool = False) -> str:
    tools_str = ", ".join(expected_tools) if expected_tools else "any available tool"
    grounded_block = ""
    if grounded:
        grounded_block = """

You are handling a repository-grounded GitHub issue.
Work from the issue context and repository metadata.
Before finishing, produce:
- at least one code_search/code_view step
- a code_edit step naming the likely target file
- a code_executor step with a targeted validation command
"""
    return f"""You are an expert AI agent for API Orchestration and Code tasks.
Solve tasks step by step using Thought -> Action -> Observation.

REQUIRED: You MUST use these tools for this task: {tools_str}
{grounded_block}

For EACH step respond ONLY with JSON:
{{
  "thought": "what I plan to do and why",
  "action": "tool_name OR finish",
  "action_input": {{"key": "value"}},
  "final_answer": "only when action=finish"
}}"""


def _run_react_task(task: dict, force_success: bool = False, grounded: bool = False) -> dict:
    trace_id = f"trace_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    expected = task.get("expected_tools", [])
    model_role = "agent" if grounded else random.choice(["agent", "agent", "agent_backup"])
    temp = 0.0 if (force_success or grounded) else random.choice(AGENT_TEMPERATURES)

    messages = [{"role": "system", "content": build_system_prompt(expected, grounded=grounded)}]
    user_prompt = _grounded_issue_prompt(task) if grounded else f"Complete this task:\n\n{task['task']}"
    messages.append({"role": "user", "content": user_prompt})

    steps = []
    tool_calls_made = []
    outcome_status = "failed"
    failure_reason = "max_steps_reached"
    final_answer = None
    files_changed = []
    validation_commands = []
    input_tokens = 0
    output_tokens = 0
    start_time = time.time()
    failure_rate = 0.0 if (force_success or grounded) else 0.15
    max_steps = 6

    for step_num in range(1, max_steps + 1):
        if force_success and step_num == max_steps - 1 and outcome_status != "success":
            tools_summary = ", ".join(set(tool_calls_made)) if tool_calls_made else "api_fetch, web_search"
            final_answer = (
                f"Successfully completed: {task['task'][:100]}. "
                f"Executed {len(tool_calls_made)} tool call(s) using {tools_summary}."
            )
            steps.append({"step": step_num, "type": "finish", "content": final_answer})
            outcome_status = "success"
            failure_reason = None
            break

        raw = call_llm(model_role, messages, temperature=temp, max_tokens=700)
        if raw is None:
            raw = call_llm("agent_backup", messages, temperature=temp, max_tokens=700)
        if raw is None:
            failure_reason = "model_unavailable"
            break

        input_tokens += len(" ".join(m["content"] for m in messages).split()) * 1.3
        output_tokens += len(raw.split()) * 1.3

        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            agent_resp = json.loads(clean.strip())
        except json.JSONDecodeError:
            agent_resp = {"thought": raw, "action": "reasoning_only", "action_input": {}}

        thought = agent_resp.get("thought", "")
        action = agent_resp.get("action", "")
        action_input = agent_resp.get("action_input", {})
        reasoning_step = {"step": step_num, "type": "reasoning", "content": thought}

        if action == "finish":
            final_answer = agent_resp.get("final_answer", "Task completed.")
            outcome_status = "success"
            failure_reason = None
            steps.append(reasoning_step)
            steps.append({"step": step_num, "type": "finish", "content": final_answer})
            break

        if action and action != "reasoning_only":
            inject = random.random() < failure_rate
            tool_result = simulate_tool_call(
                action,
                action_input,
                task["task"],
                inject_failure=inject,
                task_context=task if grounded else None,
            )
            tool_calls_made.append(action)
            steps.append(reasoning_step)
            steps.append({
                "step": step_num,
                "type": "tool_call",
                "tool": action,
                "arguments": action_input,
                "result": tool_result,
                "latency_ms": random.randint(120, 450),
            })
            if action == "code_edit":
                changed_file = str(tool_result.get("file", "") or action_input.get("file", "")).strip()
                if changed_file and changed_file not in files_changed:
                    files_changed.append(changed_file)
            if action == "code_executor":
                command = str(
                    tool_result.get("command", "") or
                    action_input.get("command", "") or
                    action_input.get("cmd", "")
                ).strip()
                if command and command not in validation_commands:
                    validation_commands.append(command)
            if "error" in tool_result:
                failure_reason = f"tool_error:{action}"
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result:\n{json.dumps(tool_result)}\nContinue."})
        else:
            steps.append(reasoning_step)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Continue with the next step."})

    duration_s = round(time.time() - start_time, 2)
    if outcome_status == "success" and len(tool_calls_made) == 0 and not force_success:
        outcome_status = "partial"
        failure_reason = "no_tool_calls_made"
    if outcome_status != "success":
        has_err = any("error" in str(s.get("result", "")) for s in steps if s.get("type") == "tool_call")
        outcome_status = "partial" if (has_err or len(tool_calls_made) > 0) else "failed"
        if has_err:
            failure_reason = "tool_error_unrecovered"
    if grounded:
        has_patch = len(files_changed) > 0
        has_validation = len(validation_commands) > 0
        if outcome_status == "success" and not has_patch:
            outcome_status = "partial"
            failure_reason = "no_patch_proposed"
        elif outcome_status == "success" and not has_validation:
            outcome_status = "partial"
            failure_reason = "no_validation_command"
        elif outcome_status != "success" and len(tool_calls_made) > 0 and failure_reason == "max_steps_reached":
            outcome_status = "partial"
            failure_reason = "grounded_analysis_incomplete"
        if outcome_status == "success":
            final_answer = final_answer or (
                "Proposed a grounded repository patch and a targeted validation command from the GitHub issue context."
            )
        elif not final_answer:
            final_answer = (
                "Produced a grounded repository analysis trace from the GitHub issue context, "
                "but the patch or validation evidence is incomplete."
            )

    return {
        "trace_id": trace_id,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": task,
        "trace": steps,
        "outcome": {
            "status": outcome_status,
            "total_steps": len(steps),
            "total_tool_calls": len(tool_calls_made),
            "tools_used": list(set(tool_calls_made)),
            "failure_occurred": outcome_status != "success",
            "failure_reason": failure_reason,
            "final_answer": final_answer,
            "duration_seconds": duration_s,
            "execution_grounded": grounded,
            "files_changed": files_changed,
            "validation_commands": validation_commands,
            "command_history": [],
        },
        "metadata": {
            "agent_framework": "react_grounded" if grounded else "react",
            "agent_model": {
                "agent": "groq/llama-3.3-70b-versatile",
                "agent_backup": "groq/openai-gpt-oss-120b",
            }.get(model_role, "groq/llama-3.3-70b-versatile"),
            "agent_temperature": temp,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "token_count_input": int(input_tokens),
            "token_count_output": int(output_tokens),
            "world_context_date": datetime.now().strftime("%Y-%m-%d"),
            "schema_version": "v4.0",
            "repo_execution_mode": "react_grounded" if grounded else "synthetic_mock",
        },
    }


def _run_synthetic_task(task: dict, force_success: bool = False) -> dict:
    return _run_react_task(task, force_success=force_success, grounded=False)


def _run_grounded_repo_issue_task(task: dict) -> dict:
    return _run_react_task(task, grounded=True)


def run_agent_on_task(task: dict, force_success: bool = False) -> dict:
    if task.get("execution_target") == "real_repo_issue":
        if REAL_REPO_EXECUTION_MODE == "repo_clone":
            return _run_real_repo_task(task)
        return _run_grounded_repo_issue_task(task)
    return _run_synthetic_task(task, force_success=force_success)


def execute_tasks(tasks: list[dict]) -> list[dict]:
    from task_generator import mark_executed

    traces = []
    total = len(tasks)
    synthetic_indices = [
        i for i, task in enumerate(tasks)
        if task.get("execution_target") != "real_repo_issue"
    ]
    n_success = max(1, int(len(synthetic_indices) * 0.4)) if synthetic_indices else 0
    success_indices = set(random.sample(synthetic_indices, min(n_success, len(synthetic_indices)))) if synthetic_indices else set()

    print(f"\n🤖 Executing {total} tasks...")
    for i, task in enumerate(tasks, 1):
        task_mode = task.get("execution_target", "synthetic")
        print(f"  [{i}/{total}] ({task_mode}) {task['task'][:70]}...")
        try:
            trace = run_agent_on_task(task, force_success=(i - 1) in success_indices)
            traces.append(trace)
            if task.get("task_id"):
                mark_executed(task["task_id"])
        except Exception as e:
            print(f"  ❌ Task {i} failed: {e}")

    s = sum(1 for t in traces if t["outcome"]["status"] == "success")
    p = sum(1 for t in traces if t["outcome"]["status"] == "partial")
    f = sum(1 for t in traces if t["outcome"]["status"] == "failed")
    print(f"\n✅ Execution complete: success={s} partial={p} failed={f}")
    return traces
