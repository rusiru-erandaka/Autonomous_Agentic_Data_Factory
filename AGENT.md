# AGENT.md

Guide for coding agents starting a fresh session in this repository.

## Project Purpose

This repository is an offline data pipeline that generates training data for
coding-agent supervisor models.

The intended output is a dataset of rows containing:
- a coding task
- an execution trace
- trace/outcome metadata
- supervisor-style labels and scalar rewards
- provenance for real repository issues when available

The project is in transition:
- old path: mostly synthetic tasks with mocked execution
- current direction: repo-grounded GitHub issue tasks with execution evidence

Do not treat this as a web app, CLI product, SDK, or library-first codebase.
It is a batch pipeline.

## High-Level Flow

The orchestrator is `main.py`. It runs five stages:

1. Task generation
2. Agent execution
3. Trace labeling
4. Quality gate
5. Hugging Face export

Core modules:
- `main.py`: top-level orchestrator and scheduler
- `task_sources.py`: signal collection, mainly GitHub issues
- `task_generator.py`: task construction, registry writes, mutation
- `agent_executor.py`: synthetic executor and grounded execution paths
- `labeler.py`: dual labeling, score merge, conflict detection
- `quality_gate.py`: validation and drift checks
- `hf_uploader.py`: flattening and dataset upload
- `llm_client.py`: active LLM routing
- `openrouter_client.py`: legacy/older fallback path, not the main runtime

## Current Architecture Reality

This repo mixes three task sources:
- template-based synthetic tasks
- repo-grounded GitHub issue tasks
- mutation-based tasks derived from earlier grounded tasks

This is intentional. The pipeline still depends partly on synthetic rows to keep
output volume stable.

Important: the project currently produces supervision traces more reliably than
it produces actual repository fixes.

## Execution Modes

### 1. Synthetic tasks

Synthetic tasks use a mocked ReAct-style executor.

Characteristics:
- no real repository
- mocked tool responses
- easier to label
- some success rate is intentionally manufactured for dataset balance

### 2. Repo-grounded tasks

Repo-grounded tasks are generated from GitHub issues and carry provenance such
as:
- `repo_url`
- `repo_clone_url`
- `repo_full_name`
- `issue_number`
- `issue_title`
- `issue_labels`
- `path_hints`
- `execution_target=real_repo_issue`

There are two grounded execution modes in `agent_executor.py`:

#### `react_grounded` (default)

This is the default runtime path.

It does not clone the repository. Instead it:
- reasons from GitHub issue text and repo metadata
- simulates repository-aware tools like `code_search`, `code_edit`, and
  `code_executor`
- records a grounded-looking trace with issue context

This path is useful for supervision data, but it is not a real repo fix.

#### `repo_clone`

This path attempts a real repository run:
- clone target repo into `registry/execution_workspace`
- inspect files
- run shell commands
- collect command history
- summarize likely files and validation commands

Current limitation:
- it does not yet implement a full patch-authoring and validation loop
- many runs end as honest `partial` or `failed`

Platform caveat:
- the clone path currently uses `powershell` commands
- the provided CI workflow targets `ubuntu-latest`
- this mismatch matters if you are trying to make `repo_clone` production-ready

## Signal Collection

Primary signal source: GitHub issues.

The intake logic in `task_sources.py` scores issues for local executability
using heuristics such as:
- bug/fix/regression labels
- repro steps
- stack traces or explicit failures
- test references
- file path hints
- discussion context

Signals are deduplicated in `registry/seen_signals.db`.

By default the repo is GitHub-only:
- `GITHUB_ONLY_SIGNALS=true`

Optional secondary sources exist but are off by default:
- Stack Overflow
- Hugging Face papers
- API changelogs

## Task Registry

Generated tasks are stored in `registry/tasks.db`.

The task registry includes both classic fields and grounded metadata:
- task text
- difficulty
- expected tools
- failure points
- source URL
- repo metadata
- issue metadata
- execution target
- task type
- executed count

Mutation-based generation reuses grounded tasks from the registry to create
variant prompts that demand better evidence or narrower patches.

## Labeling Model

`labeler.py` is one of the most important files in the project.

It does more than assign scalar scores:
- labels step-level behavior
- computes trace-level scores
- merges primary and secondary labeler outputs
- records disagreement with `agreement_score`
- records conflicts in `conflict_dimensions`
- applies hard supervisor-policy rules for grounded traces

Important policy behavior:
- failed traces cannot keep completion credit
- real repo issue traces without grounded progress must be flagged
- real repo issue successes without file changes are not approvable

The labeler is therefore part evaluator, part policy enforcement layer.

## Quality Gate

`quality_gate.py` drops rows that should not enter the dataset.

It checks:
- schema completeness
- reward consistency
- failed-labeling signatures
- suspicious fallback score patterns
- grounded-trace requirements for real repo issues
- drift against a saved baseline in `registry/baseline_scores.json`

This means a trace can be generated and labeled successfully but still be
excluded from export.

## Export Shape

`hf_uploader.py` flattens records into Hugging Face-ready rows.

Key exported groups:
- task fields
- trace JSON
- outcome fields
- grounded execution metadata
- label merge/audit fields
- scalar rewards and quality
- model/runtime metadata

The uploader also:
- writes a local JSONL backup into `registry/`
- writes the same flattened Hugging Face rows into timestamped `data/` batches
- appends to an existing HF dataset if configured
- creates train/validation/test splits when enough rows exist

## LLM Routing

The live client is `llm_client.py`.

Current routing:
- Groq is primary
- OpenRouter is fallback

Role-based pools are configured for:
- `generator`
- `agent`
- `agent_backup`
- `labeler`
- `secondary`
- `quality_gate`

Do not assume older comments about Google-first behavior are accurate. Some
module docstrings and banners are stale relative to the active routing.

## Important Repository State

`registry/` is not sample data; it is active local state.

It contains:
- `tasks.db`
- `seen_signals.db`
- `baseline_scores.json`
- run logs such as `run_YYYY-MM-DD.json`
- batch backups such as `batch_YYYY-MM-DD.jsonl`
- grounded execution workspaces under `registry/execution_workspace/`

Be careful with changes that delete or invalidate this state.

`registry/` is ignored by git.

## Scheduler and CI Notes

The active workflow is `.github/workflows/daily-pipeline.yml`.

It runs the pipeline on a daily schedule or manual dispatch, caches the task and
signal registries, uploads run artifacts, and commits generated flattened JSONL
batches under `data/`. The root `daily_pipeline.yml` is a legacy reference and
is not loaded by GitHub Actions.

## Local Run Expectations

Typical local run:

```bash
RUN_ONCE=true DAILY_TASK_TARGET=4 python main.py
```

Default scheduled mode:

```bash
python main.py
```

This starts the scheduler and also runs immediately once on startup.

Useful env vars:
- `GROQ_API_KEY`
- `HF_TOKEN`
- `HF_DATASET_REPO`
- `GITHUB_TOKEN`
- `RUN_ONCE`
- `DAILY_TASK_TARGET`
- `GITHUB_ONLY_SIGNALS`
- `MIN_GITHUB_ISSUE_SCORE`
- `KEEP_EXECUTION_WORKSPACES`
- `SCHEDULE_TIME`
- `REAL_REPO_EXECUTION_MODE`

## Things a Coding Agent Should Not Misassume

- Do not assume all successful rows are grounded. Many are still synthetic.
- Do not assume grounded mode means a real clone happened. Default grounded mode
  is simulated from issue metadata.
- Do not assume the project already writes patches into cloned repos. That is
  still incomplete.
- Do not assume the root `daily_pipeline.yml` is an active GitHub Actions
  workflow.
- Do not assume comments/docstrings about provider assignments are all current.
  Check `llm_client.py`.
- Do not assume `registry/` can be discarded safely.

## Best First Reads

If you are new to the repo, read in this order:

1. `README.md`
2. `main.py`
3. `task_sources.py`
4. `task_generator.py`
5. `agent_executor.py`
6. `labeler.py`
7. `quality_gate.py`
8. `hf_uploader.py`
9. `llm_client.py`

## Good Next-Step Areas

If you are asked to improve the project, the highest-value areas are usually:
- real patch authoring in `repo_clone` mode
- Linux-safe grounded execution commands
- stronger validation command selection
- reducing synthetic dependence
- improving observability around why grounded traces pass or fail

## Working Rule

When making changes, preserve the core intent:

This repository is optimizing for high-quality supervision data over coding
agent behavior, especially honest grounded traces, not just inflated success
rates.
