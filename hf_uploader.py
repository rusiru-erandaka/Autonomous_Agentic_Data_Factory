"""
hf_uploader.py
Pushes validated records to a HuggingFace dataset repo.
Appends to existing data, maintains version history via commit messages.
"""

import os
import json
import pandas as pd
from datetime import datetime

HF_TOKEN       = os.environ.get("HF_TOKEN", "Huggingface_Token")
DATASET_REPO   = os.environ.get("HF_DATASET_REPO", "YOUR_REPO_NAME")


def flatten_record(record: dict) -> dict:
    """
    Flatten nested trace record into a single-row dict suitable for a DataFrame.
    Nested objects are JSON-serialized strings so HF datasets can handle them.
    """
    trace_scores = record["labels"]["trace_level_scores"]
    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "trace_id":              record.get("trace_id", ""),
        "created_at":            record.get("created_at", ""),

        # ── Task ──────────────────────────────────────────────────────────────
        "task":                  record["task"]["task"],
        "task_difficulty":       record["task"].get("difficulty", ""),
        "task_niche":            "api_orchestration+code_agent",
        "expected_tools":        json.dumps(record["task"].get("expected_tools", [])),
        "likely_failure_points": json.dumps(record["task"].get("likely_failure_points", [])),
        "freshness_source":      record["task"].get("freshness_source", ""),
        "generation_strategy":   record["task"].get("generation_strategy", ""),

        # ── Trace ─────────────────────────────────────────────────────────────
        "trace_json":            json.dumps(record.get("trace", [])),

        # ── Outcome ───────────────────────────────────────────────────────────
        "outcome_status":        record["outcome"]["status"],
        "total_steps":           record["outcome"]["total_steps"],
        "total_tool_calls":      record["outcome"]["total_tool_calls"],
        "tools_used":            json.dumps(record["outcome"].get("tools_used", [])),
        "failure_occurred":      record["outcome"]["failure_occurred"],
        "failure_reason":        record["outcome"].get("failure_reason") or "",
        "final_answer":          record["outcome"].get("final_answer") or "",
        "duration_seconds":      record["outcome"].get("duration_seconds", 0),

        # ── Labels ────────────────────────────────────────────────────────────
        "labeler_model":         record["labels"]["labeler_model"],
        "constitution_version":  record["labels"]["constitution_version"],
        "labeled_at":            record["labels"]["labeled_at"],
        "step_level_scores":     json.dumps(record["labels"].get("step_level_scores", [])),
        "task_completion":       trace_scores.get("task_completion", 0),
        "tool_use_efficiency":   trace_scores.get("tool_use_efficiency", 0),
        "reasoning_coherence":   trace_scores.get("reasoning_coherence", 0),
        "safety_compliance":     trace_scores.get("safety_compliance", 0),
        "overall_quality":       trace_scores.get("overall_quality", 0),
        "reward_signal":         trace_scores.get("reward_signal", 0),
        "supervisor_verdict":    trace_scores.get("supervisor_verdict", ""),
        "verdict_reason":        trace_scores.get("verdict_reason", ""),
        "dual_labeled":          record["labels"].get("dual_labeled", False),

        # ── Metadata ──────────────────────────────────────────────────────────
        "agent_framework":       record["metadata"].get("agent_framework", ""),
        "agent_model":           record["metadata"].get("agent_model", ""),
        "world_context_date":    record["metadata"].get("world_context_date", ""),
        "schema_version":        record["metadata"].get("schema_version", "v1.0"),
    }


def push_to_hf(records: list[dict]):
    """
    Flatten records, append to existing HF dataset, and push.
    Falls back to saving a local JSONL file if HF push fails.
    """
    if not records:
        print("  ⚠️  No records to push.")
        return

    today = datetime.now().strftime("%Y-%m-%d")

    # Flatten all records
    rows = [flatten_record(r) for r in records]
    df_new = pd.DataFrame(rows)
    print(f"  📦 Prepared {len(df_new)} rows for upload.")

    # Always save a local backup first
    os.makedirs("registry", exist_ok=True)
    backup_path = f"registry/batch_{today}.jsonl"
    with open(backup_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  💾 Local backup saved: {backup_path}")

    if not HF_TOKEN:
        print("  ⚠️  HF_TOKEN not set — skipping HuggingFace push. Records saved locally.")
        return

    try:
        from datasets import load_dataset, Dataset
        from huggingface_hub import HfApi

        # Try to load existing dataset and append
        try:
            existing = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
            df_existing = existing.to_pandas()
            print(f"  📂 Existing dataset: {len(df_existing)} rows")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            print("  📂 No existing dataset found — creating new one.")
            df_combined = df_new

        # Push to HuggingFace
        dataset = Dataset.from_pandas(df_combined)
        dataset.push_to_hub(
            DATASET_REPO,
            token=HF_TOKEN,
            commit_message=f"Daily update {today} — {len(rows)} new records (total: {len(df_combined)})",
        )
        print(f"  ✅ Pushed to HuggingFace: {DATASET_REPO}")
        print(f"  📊 Dataset now has {len(df_combined)} total records.")

    except ImportError:
        print("  ❌ 'datasets' or 'huggingface_hub' not installed. Run: pip install datasets huggingface_hub")
    except Exception as e:
        print(f"  ❌ HuggingFace push failed: {e}")
        print(f"  💾 Records are safe in local backup: {backup_path}")
