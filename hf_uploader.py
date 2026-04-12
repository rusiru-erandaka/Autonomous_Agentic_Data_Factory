"""
hf_uploader.py
Pushes validated records to a HuggingFace dataset repo.
Appends to existing data, maintains version history.
Produces stratified train/val/test splits once dataset is large enough.
"""

import os
import json
import pandas as pd
from datetime import datetime

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
DATASET_REPO = os.environ.get("HF_DATASET_REPO", "your-username/agent-behavior-dataset")


def flatten_record(record: dict) -> dict:
    """Flatten nested trace record into a single-row dict."""
    trace_scores = record["labels"]["trace_level_scores"]
    meta         = record.get("metadata", {})

    tc  = trace_scores.get("task_completion", 0)
    tue = trace_scores.get("tool_use_efficiency", 0)
    rc  = trace_scores.get("reasoning_coherence", 0)
    reward_computed = round((tc/3*0.4) + (tue/3*0.3) + (rc/3*0.3), 4)

    return {
        # Identity
        "trace_id":              record.get("trace_id", ""),
        "created_at":            record.get("created_at", ""),

        # Task
        "task":                  record["task"]["task"],
        "task_difficulty":       record["task"].get("difficulty", ""),
        "task_niche":            record["task"].get("niche", "api_orchestration+code_agent"),
        "expected_tools":        json.dumps(record["task"].get("expected_tools", [])),
        "likely_failure_points": json.dumps(record["task"].get("likely_failure_points", [])),
        "freshness_source":      record["task"].get("freshness_source", ""),
        "generation_strategy":   record["task"].get("generation_strategy", ""),

        # Trace (full, no truncation)
        "trace_json":            json.dumps(record.get("trace", [])),

        # Outcome
        "outcome_status":        record["outcome"]["status"],
        "total_steps":           record["outcome"]["total_steps"],
        "total_tool_calls":      record["outcome"]["total_tool_calls"],
        "tools_used":            json.dumps(record["outcome"].get("tools_used", [])),
        "failure_occurred":      record["outcome"]["failure_occurred"],
        "failure_reason":        record["outcome"].get("failure_reason") or "",
        "final_answer":          record["outcome"].get("final_answer") or "",
        "duration_seconds":      record["outcome"].get("duration_seconds", 0),

        # Labels
        "labeler_model":         record["labels"]["labeler_model"],
        "constitution_version":  record["labels"]["constitution_version"],
        "labeled_at":            record["labels"]["labeled_at"],
        "step_level_scores":     json.dumps(record["labels"].get("step_level_scores", [])),
        "task_completion":       tc,
        "tool_use_efficiency":   tue,
        "reasoning_coherence":   rc,
        "safety_compliance":     trace_scores.get("safety_compliance", 0),
        "overall_quality":       trace_scores.get("overall_quality", 0),
        "reward_signal":         trace_scores.get("reward_signal", 0),
        "reward_computed":       reward_computed,
        "reward_formula":        "task_completion*0.4 + tool_use_efficiency*0.3 + reasoning_coherence*0.3 (each /3)",
        "supervisor_verdict":    trace_scores.get("supervisor_verdict", ""),
        "verdict_reason":        trace_scores.get("verdict_reason", ""),
        "dual_labeled":          record["labels"].get("dual_labeled", False),
        "agreement_score":       record["labels"].get("agreement_score", None),

        # Metadata
        "agent_framework":           meta.get("agent_framework", ""),
        "agent_model":               meta.get("agent_model", ""),
        "agent_temperature":         meta.get("agent_temperature", 0.4),
        "prompt_template_version":   meta.get("prompt_template_version", ""),
        "token_count_input":         meta.get("token_count_input", 0),
        "token_count_output":        meta.get("token_count_output", 0),
        "world_context_date":        meta.get("world_context_date", ""),
        "schema_version":            meta.get("schema_version", "v1.1"),
    }


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Stratified train/val/test split (70/15/15).
    Stratifies on outcome_status × task_difficulty so all splits
    have balanced difficulty and success/fail ratios.
    Only applies when dataset has 30+ records; otherwise all rows go to train.
    """
    if len(df) < 30:
        return {"train": df}

    from sklearn.model_selection import train_test_split
    strat_col = df["outcome_status"].fillna("unknown") + "_" + df["task_difficulty"].fillna("medium")

    train, temp = train_test_split(df, test_size=0.30, stratify=strat_col, random_state=42)
    strat_temp  = strat_col.loc[temp.index]
    val, test   = train_test_split(temp, test_size=0.50, stratify=strat_temp, random_state=42)
    return {"train": train, "validation": val, "test": test}


def push_to_hf(records: list[dict]):
    """Flatten, append, split, and push to HuggingFace."""
    if not records:
        print("  ⚠️  No records to push.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    rows  = [flatten_record(r) for r in records]
    df_new = pd.DataFrame(rows)
    print(f"  📦 Prepared {len(df_new)} rows.")

    # Local backup first — always
    os.makedirs("registry", exist_ok=True)
    backup_path = f"registry/batch_{today}.jsonl"
    with open(backup_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  💾 Local backup: {backup_path}")

    if not HF_TOKEN:
        print("  ⚠️  HF_TOKEN not set — saved locally only.")
        return

    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi

        # Load existing and append
        try:
            from datasets import load_dataset
            existing = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
            df_existing = existing.to_pandas()
            print(f"  📂 Existing: {len(df_existing)} rows")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            print("  📂 No existing dataset — creating new.")
            df_combined = df_new

        # Drop sklearn import fallback
        splits = make_splits(df_combined)
        split_info = {k: len(v) for k, v in splits.items()}
        print(f"  📊 Splits: {split_info}")

        dataset_dict = DatasetDict({
            split: Dataset.from_pandas(df.reset_index(drop=True))
            for split, df in splits.items()
        })
        dataset_dict.push_to_hub(
            DATASET_REPO,
            token=HF_TOKEN,
            commit_message=f"Daily update {today} — +{len(rows)} records (total: {len(df_combined)})",
        )
        print(f"  ✅ Pushed to {DATASET_REPO}")

    except ImportError as e:
        print(f"  ❌ Missing library: {e}. Run: pip install datasets huggingface_hub scikit-learn")
    except Exception as e:
        print(f"  ❌ HF push failed: {e} | Records safe in {backup_path}")