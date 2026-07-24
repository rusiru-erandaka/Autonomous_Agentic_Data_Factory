# AI Coding Supervisor Dataset Pipeline

An automated pipeline for generating training data for **coding-agent supervisor models**.

The current direction of the project is:
- harvest real GitHub issues
- turn them into repo-grounded coding tasks
- run agents against cloned repositories
- capture execution evidence and traces
- label the traces with supervisor-style judgments and scalar rewards
- publish filtered rows to Hugging Face

This repository is still in transition from a mostly synthetic pipeline to a grounded repo-execution pipeline. The code reflects that mixed state.

## Current Status

What is already implemented:
- GitHub issue harvesting with executable-issue scoring
- repo/issue provenance carried through task generation and export
- real repo workspace preparation for grounded tasks
- command-history capture for repo-grounded attempts
- dual labeling with merged supervisor scores
- deterministic exported `reward_signal`, `reward_computed`, and `overall_quality`
- conflict detection with `conflict_dimensions`

What is still in progress:
- autonomous patch authoring inside cloned repos
- stronger grounded success rates for repo-based tasks
- broader safety-score variation
- reducing dependence on synthetic template tasks

## Pipeline Flow

```text
Real-World Signals
  GitHub issues (primary)
  Optional other sources
        |
        v
Task Generation
  template tasks
  repo-grounded GitHub issue tasks
  mutation tasks
        |
        v
Execution
  synthetic tasks -> mocked executor
  real_repo_issue tasks -> clone repo, inspect files, capture commands
        |
        v
Labeling
  primary + secondary LLM review
  merged scalar scores
  step-level labels
  conflict detection
        |
        v
Quality Gate
  schema validation
  verdict / reward consistency checks
  grounded-trace checks
  drift detection
        |
        v
Hugging Face Export
  flattened rows
  registry backup
  tracked data/ JSONL batch
  Hugging Face dataset push
```

## Project Structure

```text
main.py               orchestrator
task_sources.py       signal collection, mainly GitHub issue harvesting
task_generator.py     task conversion, registry storage, mutation
agent_executor.py     synthetic executor + grounded repo runner
labeler.py            dual labeling, merge policy, conflict detection
quality_gate.py       validation and drift checks
hf_uploader.py        row flattening and Hugging Face upload
llm_client.py         Groq-first LLM client with fallback handling
openrouter_client.py  older fallback client, not the primary path now
requirements.txt
.github/workflows/
  daily-pipeline.yml  active scheduled GitHub Actions workflow
data/                 tracked flattened JSONL dataset batches
registry/             generated DBs, logs, backups, workspaces
SECURITY.md           security policy
```

## Execution Modes

### Synthetic Tasks

Template-style tasks still exist to keep the pipeline producing stable baseline rows.

Characteristics:
- faster
- easier to label
- useful as clean anchors
- not repository-grounded

### Repo-Grounded Tasks

GitHub issue tasks now carry:
- `repo_url`
- `repo_clone_url`
- `repo_full_name`
- `issue_number`
- `issue_title`
- `issue_labels`
- `path_hints`
- `execution_target=real_repo_issue`

Current grounded execution does:
- clone the target repository into `registry/execution_workspace`
- inspect files and search likely paths
- record command history and repo metadata
- emit honest `failed` / `partial` traces when no patch is applied

It does **not** yet fully implement autonomous patch writing and validation loops. That is the next major implementation step.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the repo root.

Minimum useful settings:

```bash
GROQ_API_KEY=...
HF_TOKEN=...
HF_DATASET_REPO=your-username/your-dataset
GITHUB_TOKEN=ghp_...
RUN_ONCE=true
DAILY_TASK_TARGET=4
```

Useful optional settings:

```bash
GITHUB_ONLY_SIGNALS=true
GITHUB_TIMEOUT_SECONDS=25
MIN_GITHUB_ISSUE_SCORE=5
KEEP_EXECUTION_WORKSPACES=false
SCHEDULE_TIME=02:00
```

## Running

### Run once locally

```bash
RUN_ONCE=true DAILY_TASK_TARGET=4 python main.py
```

### Scheduled mode

```bash
python main.py
```

This starts the scheduler and also runs immediately on startup.

## Model Routing

The live pipeline is **Groq-first**, with fallback support handled in `llm_client.py`.

Current role intent:
- `generator`: `llama-3.1-8b-instant`
- `agent`: `openai/gpt-oss-120b`
- `agent_backup`: `llama-3.3-70b-versatile`
- `labeler`: `llama-3.3-70b-versatile`
- `secondary`: `qwen/qwen3-32b`
- `quality_gate`: `qwen/qwen3-32b` (reserved; the current gate is pure Python)

Do not rely on the old README-era assumption that the project is OpenRouter-first or Nemotron-only. That is no longer accurate.

## Dataset Row Shape

Exported rows now include both classic synthetic fields and newer grounded-execution fields.

Core fields:
- `task`
- `task_difficulty`
- `trace_json`
- `outcome_status`
- `reward_signal`
- `reward_computed`
- `overall_quality`
- `supervisor_verdict`
- `verdict_reason`
- `step_level_scores`

Grounded execution fields:
- `source_url`
- `repo_url`
- `repo_clone_url`
- `repo_full_name`
- `repo_default_branch`
- `repo_language`
- `issue_number`
- `issue_title`
- `issue_labels`
- `path_hints`
- `execution_target`
- `execution_grounded`
- `files_changed`
- `validation_commands`
- `command_history`

Label merge / audit fields:
- `labeler_model`
- `labeler_model_2`
- `agreement_score`
- `conflict_flag`
- `conflict_dimensions`
- `merge_strategy`
- `reward_adjustment_reason`
- `quality_formula`
- `reward_formula`
- `rubric_hash`

## Known Limitations

- Many successful rows are still synthetic.
- Grounded repo runs currently produce more honest failures than successful fixes.
- Safety-score diversity is still weaker than it should be.
- The secondary labeler path should continue to be strengthened.
- `openrouter_client.py` remains in the repo, but it is not the main execution path.

## GitHub Actions

`.github/workflows/daily-pipeline.yml` runs daily at 02:00 UTC and can also be
started manually. After a quality-gated run, it commits the new flattened JSONL
batch under `data/` and pushes the same rows to the configured Hugging Face
dataset. The workflow preserves registry databases through the Actions cache.

Recommended secrets / variables:
- `GROQ_API_KEY`
- `HF_TOKEN`
- `GH_PAT`
- `OPENROUTER_API_KEY_1`
- `OPENROUTER_API_KEY_2`
- `OPENROUTER_API_KEY_3`
- `HF_DATASET_REPO`

The workflow requires `contents: write` permission so the GitHub Actions bot can
commit generated batches. Repository or organization policies must also permit
GitHub Actions to create commits.

## Security

See [SECURITY.md](./SECURITY.md).

## License

Use your intended repository and dataset license here if you have not finalized it yet. The previous README’s licensing statement should not be treated as authoritative unless you explicitly confirm it.
