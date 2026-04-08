# 🤖 AI Agent Behavior Dataset — Automated Pipeline

A fully automated, continuously updating dataset for training **AI agent supervisors, evaluators, and reward models**.

**Niche:** API Orchestration + Code Agent tasks  
**Updates:** Daily via GitHub Actions  
**Labeler:** Nvidia Nemotron-3-Super-120B with constitutional rubric  
**Dataset:** Published on HuggingFace — anyone can use it

---

## Pipeline Architecture

```
Real-World Signals (GitHub / StackOverflow / HF Papers / API Changelogs)
        │
        ▼
Task Generator (3 strategies: template / LLM-generative / mutation)
        │
        ▼
Agent Executor (Llama-3.3-70B ReAct agent, full trace capture)
        │
        ▼
Nemotron Labeler (constitutional rubric, dual-check on 10%)
        │
        ▼
Quality Gate (schema + label + drift validation)
        │
        ▼
HuggingFace Dataset (daily append + version history)
```

---

## Project Structure

```
pipeline/
├── main.py                    # orchestrator — run this
├── openrouter_client.py       # unified LLM client (OpenRouter free models)
├── task_sources.py            # GitHub / SO / HF / changelog scrapers
├── task_generator.py          # 3-strategy task engine + SQLite registry
├── agent_executor.py          # ReAct agent runner + trace capture
├── labeler.py                 # Nemotron constitutional labeling
├── quality_gate.py            # validation + drift detection
├── hf_uploader.py             # HuggingFace push
├── requirements.txt
└── registry/                  # auto-created, stores SQLite DB + run logs

.github/
└── workflows/
    └── daily_pipeline.yml     # GitHub Actions — runs every day at 2am UTC
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/agent-behavior-dataset-pipeline
cd agent-behavior-dataset-pipeline
pip install -r pipeline/requirements.txt
```

### 2. Set environment variables

```bash
export OPENROUTER_API_KEY="sk-or-..."
export HF_TOKEN="hf_..."
export GITHUB_TOKEN="ghp_..."          # optional but recommended
export HF_DATASET_REPO="your-username/agent-behavior-dataset"
```

### 3. Run once locally to test

```bash
cd pipeline
RUN_ONCE=true DAILY_TASK_TARGET=5 python main.py
```

### 4. Set up GitHub Actions (for daily automation)

In your GitHub repo:
- Go to **Settings → Secrets → Actions**
- Add: `OPENROUTER_API_KEY`, `HF_TOKEN`, `GH_PAT`
- Go to **Settings → Variables → Actions**
- Add: `HF_DATASET_REPO` = `your-username/agent-behavior-dataset`

That's it — it runs every night at 2am UTC automatically.

---

## Models Used (all free via OpenRouter)

| Role | Model |
|---|---|
| Agent Executor | `meta-llama/llama-3.3-70b-instruct:free` |
| Agent Backup | `openai/gpt-oss-120b:free` |
| Primary Labeler | `nvidia/nemotron-3-super-120b-a12b:free` |
| Secondary Labeler | `qwen/qwen3-next-80b-a3b-instruct:free` |
| Task Generator | `nvidia/nemotron-3-nano-30b-a3b:free` |
| Quality Gate | `google/gemma-4-31b-it:free` |

---

## Dataset Schema

Each record contains:

| Field | Description |
|---|---|
| `task` | The agent task instruction |
| `task_difficulty` | simple / medium / complex |
| `trace_json` | Full step-by-step agent execution trace |
| `outcome_status` | success / partial / failed |
| `reward_signal` | 0.0–1.0 reward score from Nemotron |
| `supervisor_verdict` | approve / flag / reject |
| `step_level_scores` | Per-step labels with rationale |
| `world_context_date` | Date task was grounded in real-world context |
| `constitution_version` | Labeling rubric version for consistency |

---

## License

Dataset: Apache 2.0 — free to use for research and commercial training.
