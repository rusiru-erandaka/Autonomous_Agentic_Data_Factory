"""
hf_uploader.py
Pushes validated records to HuggingFace dataset repo.
Fixes:
- Reads HF_TOKEN at call time (not import time) so .env loading order doesn't matter
- Graceful sklearn fallback if not installed
- Verbose error reporting
"""

import os
import json
import traceback
import pandas as pd
from datetime import datetime


def _get_hf_token() -> str:
    """Read token at call time — not at import time."""
    return os.environ.get("HF_TOKEN", "").strip()

def _get_dataset_repo() -> str:
    return os.environ.get("HF_DATASET_REPO", "").strip()


def flatten_record(record: dict) -> dict:
    """Flatten nested trace record into a single-row dict."""
    trace_scores = record["labels"]["trace_level_scores"]
    meta         = record.get("metadata", {})

    tc  = trace_scores.get("task_completion", 0) or 0
    tue = trace_scores.get("tool_use_efficiency", 0) or 0
    rc  = trace_scores.get("reasoning_coherence", 0) or 0
    reward_computed = round((tc / 3 * 0.4) + (tue / 3 * 0.3) + (rc / 3 * 0.3), 4)

    return {
        "trace_id":                  record.get("trace_id", ""),
        "created_at":                record.get("created_at", ""),
        "task":                      record["task"]["task"],
        "task_difficulty":           record["task"].get("difficulty", ""),
        "task_niche":                record["task"].get("niche", "api_orchestration+code_agent"),
        "expected_tools":            json.dumps(record["task"].get("expected_tools", [])),
        "likely_failure_points":     json.dumps(record["task"].get("likely_failure_points", [])),
        "freshness_source":          record["task"].get("freshness_source", ""),
        "generation_strategy":       record["task"].get("generation_strategy", ""),
        "trace_json":                json.dumps(record.get("trace", [])),
        "outcome_status":            record["outcome"]["status"],
        "total_steps":               record["outcome"]["total_steps"],
        "total_tool_calls":          record["outcome"]["total_tool_calls"],
        "tools_used":                json.dumps(record["outcome"].get("tools_used", [])),
        "failure_occurred":          bool(record["outcome"]["failure_occurred"]),
        "failure_reason":            record["outcome"].get("failure_reason") or "",
        "final_answer":              record["outcome"].get("final_answer") or "",
        "duration_seconds":          record["outcome"].get("duration_seconds", 0),
        "labeler_model":             record["labels"]["labeler_model"],
        "constitution_version":      record["labels"]["constitution_version"],
        "labeled_at":                record["labels"]["labeled_at"],
        "step_level_scores":         json.dumps(record["labels"].get("step_level_scores", [])),
        "task_completion":           int(tc),
        "tool_use_efficiency":       int(tue),
        "reasoning_coherence":       int(rc),
        "safety_compliance":         int(trace_scores.get("safety_compliance", 0) or 0),
        "overall_quality":           float(trace_scores.get("overall_quality", 0) or 0),
        "reward_signal":             float(trace_scores.get("reward_signal", 0) or 0),
        "reward_computed":           reward_computed,
        "reward_formula":            "task_completion*0.4 + tool_use_efficiency*0.3 + reasoning_coherence*0.3 (each /3)",
        "supervisor_verdict":        trace_scores.get("supervisor_verdict", ""),
        "verdict_reason":            trace_scores.get("verdict_reason", ""),
        "dual_labeled":              bool(record["labels"].get("dual_labeled", False)),
        "agent_framework":           meta.get("agent_framework", ""),
        "agent_model":               meta.get("agent_model", ""),
        "agent_temperature":         float(meta.get("agent_temperature", 0.4)),
        "prompt_template_version":   meta.get("prompt_template_version", ""),
        "token_count_input":         int(meta.get("token_count_input", 0)),
        "token_count_output":        int(meta.get("token_count_output", 0)),
        "world_context_date":        meta.get("world_context_date", ""),
        "schema_version":            meta.get("schema_version", "v1.1"),
    }


def make_splits(df: pd.DataFrame) -> dict:
    """Stratified 70/15/15 train/val/test split. Falls back to train-only if too small."""
    if len(df) < 30:
        print(f"  ℹ️  {len(df)} records — using train-only split (need 30+ for stratified splits)")
        return {"train": df}
    try:
        from sklearn.model_selection import train_test_split
        strat = df["outcome_status"].fillna("unknown") + "_" + df["task_difficulty"].fillna("medium")
        train, temp   = train_test_split(df,   test_size=0.30, stratify=strat,            random_state=42)
        strat_temp    = strat.loc[temp.index]
        val,   test   = train_test_split(temp, test_size=0.50, stratify=strat_temp,       random_state=42)
        return {"train": train, "validation": val, "test": test}
    except ImportError:
        print("  ⚠️  scikit-learn not installed — using train-only split. Run: pip install scikit-learn")
        return {"train": df}
    except Exception as e:
        print(f"  ⚠️  Split failed ({e}) — using train-only split")
        return {"train": df}


def push_to_hf(records: list[dict]):
    """Flatten, append, split, and push to HuggingFace."""
    if not records:
        print("  ⚠️  No records to push.")
        return

    # Read tokens at call time — not at import time
    HF_TOKEN     = _get_hf_token()
    DATASET_REPO = _get_dataset_repo()

    today = datetime.now().strftime("%Y-%m-%d")

    # ── Flatten records ────────────────────────────────────────────────────────
    rows = []
    for i, r in enumerate(records):
        try:
            rows.append(flatten_record(r))
        except Exception as e:
            print(f"  ⚠️  Could not flatten record {i}: {e}")
    df_new = pd.DataFrame(rows)
    print(f"  📦 Prepared {len(df_new)} rows for upload.")

    # ── Local backup — always written regardless of HF result ─────────────────
    os.makedirs("registry", exist_ok=True)
    backup_path = f"registry/batch_{today}.jsonl"
    with open(backup_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    print(f"  💾 Local backup saved: {backup_path}")

    # ── Validate credentials ───────────────────────────────────────────────────
    if not HF_TOKEN:
        print("  ❌ HF_TOKEN is not set.")
        print("     Add this to your .env: export HF_TOKEN='hf_...'")
        print("     Then re-run: source .env  (or restart terminal on Windows)")
        return

    if not DATASET_REPO:
        print("  ❌ HF_DATASET_REPO is not set.")
        print("     Add: export HF_DATASET_REPO='your-username/your-dataset-name'")
        return

    print(f"  🔑 HF_TOKEN: {HF_TOKEN[:8]}... (set)")
    print(f"  📁 Target:   {DATASET_REPO}")

    # ── Import HF libraries ────────────────────────────────────────────────────
    try:
        from datasets import Dataset, DatasetDict, load_dataset
        from huggingface_hub import HfApi
    except ImportError as e:
        print(f"  ❌ Missing HF library: {e}")
        print("     Run: pip install datasets huggingface_hub")
        return

    # ── Ensure repo exists ─────────────────────────────────────────────────────
    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
        print(f"  ✅ Repo found: {DATASET_REPO}")
    except Exception:
        print(f"  📝 Repo not found — creating it...")
        try:
            api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False)
            print(f"  ✅ Repo created: https://huggingface.co/datasets/{DATASET_REPO}")
        except Exception as e:
            print(f"  ❌ Could not create repo: {e}")
            traceback.print_exc()
            return

    # ── Load existing data and append ─────────────────────────────────────────
    try:
        existing    = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
        df_existing = existing.to_pandas()
        print(f"  📂 Existing dataset: {len(df_existing)} rows")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except Exception as e:
        print(f"  📂 No existing data ({type(e).__name__}) — starting fresh")
        df_combined = df_new

    print(f"  📊 Total after append: {len(df_combined)} rows")

    # ── Split and push ─────────────────────────────────────────────────────────
    splits     = make_splits(df_combined)
    split_info = {k: len(v) for k, v in splits.items()}
    print(f"  📊 Splits: {split_info}")

    try:
        dataset_dict = DatasetDict({
            name: Dataset.from_pandas(df.reset_index(drop=True))
            for name, df in splits.items()
        })
        dataset_dict.push_to_hub(
            DATASET_REPO,
            token=HF_TOKEN,
            commit_message=f"Daily update {today} — +{len(rows)} new records (total: {len(df_combined)})",
        )
        print(f"  ✅ Pushed! View at: https://huggingface.co/datasets/{DATASET_REPO}")
        print(f"  📊 Dataset now has {len(df_combined)} total records.")

    except Exception as e:
        print(f"  ❌ Push failed: {e}")
        traceback.print_exc()
        print(f"  💾 Data is safe locally at: {backup_path}")