"""
hf_uploader.py
Pushes validated records to HuggingFace dataset repo.
- Reads tokens at call time (never at import time)
- reward_computed always populated — never NULL
- Normalized overall_quality (0-10)
- Real model names in agent_model field
- source_url stored per record
- Stratified train/val/test splits at 30+ records
"""

import os
import json
import traceback
import pandas as pd
from datetime import datetime


def _get_hf_token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()

def _get_dataset_repo() -> str:
    return os.environ.get("HF_DATASET_REPO", "").strip()


def flatten_record(record: dict) -> dict:
    """Flatten nested record into a single HuggingFace-ready row."""
    trace_scores = record["labels"]["trace_level_scores"]
    meta         = record.get("metadata", {})

    tc  = int(trace_scores.get("task_completion",    0) or 0)
    tue = int(trace_scores.get("tool_use_efficiency", 0) or 0)
    rc  = int(trace_scores.get("reasoning_coherence", 0) or 0)

    # reward_computed — always calculated, never NULL
    reward_computed = round((tc / 3 * 0.4) + (tue / 3 * 0.3) + (rc / 3 * 0.3), 4)

    # Use labeler reward_signal if available, else use computed
    rs_raw        = trace_scores.get("reward_signal")
    reward_signal = round(float(rs_raw), 4) if isinstance(rs_raw, (int, float)) else reward_computed

    # Normalize overall_quality to strict 0-10 range
    oq_raw        = trace_scores.get("overall_quality", 5.0)
    oq            = float(oq_raw) if isinstance(oq_raw, (int, float)) else 5.0
    overall_quality = round(min(oq / 12 * 10, 10.0) if oq > 10 else min(oq, 10.0), 2)

    # Niche — check multiple possible field locations
    task_niche = (
        record["task"].get("niche") or
        record["task"].get("task_niche") or
        "api_orchestration"
    )

    # Agreement score
    agreement = record["labels"].get("agreement_score")

    # Verdict reason — ensure never blank
    verdict_reason = trace_scores.get("verdict_reason", "") or ""
    if not verdict_reason.strip():
        verdict_reason = (
            f"Auto: verdict={trace_scores.get('supervisor_verdict','flag')}, "
            f"task_completion={tc}, reward={reward_computed}"
        )

    return {
        # ── Identity ──────────────────────────────────────────────────────────
        "trace_id":                    record.get("trace_id", ""),
        "created_at":                  record.get("created_at", ""),
        "schema_version":              meta.get("schema_version", "v3.0"),

        # ── Task ──────────────────────────────────────────────────────────────
        "task":                        record["task"]["task"],
        "task_difficulty":             record["task"].get("difficulty", "medium"),
        "task_niche":                  task_niche,
        "expected_tools":              json.dumps(record["task"].get("expected_tools", [])),
        "likely_failure_points":       json.dumps(record["task"].get("likely_failure_points", [])),
        "generation_strategy":         record["task"].get("generation_strategy", ""),
        "freshness_source":            record["task"].get("freshness_source", ""),
        "source_url":                  record["task"].get("source_url", ""),

        # ── Trace ─────────────────────────────────────────────────────────────
        "trace_json":                  json.dumps(record.get("trace", [])),

        # ── Outcome ───────────────────────────────────────────────────────────
        "outcome_status":              record["outcome"]["status"],
        "total_steps":                 int(record["outcome"]["total_steps"]),
        "total_tool_calls":            int(record["outcome"]["total_tool_calls"]),
        "tools_used":                  json.dumps(record["outcome"].get("tools_used", [])),
        "failure_occurred":            bool(record["outcome"]["failure_occurred"]),
        "failure_reason":              record["outcome"].get("failure_reason") or "",
        "final_answer":                record["outcome"].get("final_answer") or "",
        "duration_seconds":            float(record["outcome"].get("duration_seconds", 0)),

        # ── Labels ────────────────────────────────────────────────────────────
        "labeler_model":               record["labels"]["labeler_model"],
        "labeler_model_2":             record["labels"].get("labeler_model_2", ""),
        "constitution_version":        record["labels"]["constitution_version"],
        "labeled_at":                  record["labels"]["labeled_at"],
        "step_level_scores":           json.dumps(record["labels"].get("step_level_scores", [])),
        "primary_trace_scores":        json.dumps(record["labels"].get("primary_trace_scores", {})),
        "secondary_trace_scores":      json.dumps(record["labels"].get("secondary_trace_scores", {})),
        "dual_labeled":                bool(record["labels"].get("dual_labeled", False)),
        "agreement_score":             float(agreement) if isinstance(agreement, (int, float)) else None,
        "conflict_flag":               bool(record["labels"].get("conflict_flag", False)),

        # ── Scores ────────────────────────────────────────────────────────────
        "task_completion":             tc,
        "tool_use_efficiency":         tue,
        "reasoning_coherence":         rc,
        "safety_compliance":           int(trace_scores.get("safety_compliance", 3) or 3),
        "overall_quality":             overall_quality,
        "reward_signal":               reward_signal,
        "reward_computed":             reward_computed,
        "reward_formula":              "task_completion*0.4 + tool_use_efficiency*0.3 + reasoning_coherence*0.3 (each /3)",
        "supervisor_verdict":          trace_scores.get("supervisor_verdict", "flag"),
        "verdict_reason":              verdict_reason,

        # ── Metadata ──────────────────────────────────────────────────────────
        "agent_framework":             meta.get("agent_framework", "react"),
        "agent_model":                 meta.get("agent_model", "groq/openai/gpt-oss-120b"),
        "agent_temperature":           float(meta.get("agent_temperature", 0.4)),
        "prompt_template_version":     meta.get("prompt_template_version", "v3.0"),
        "token_count_input":           int(meta.get("token_count_input", 0)),
        "token_count_output":          int(meta.get("token_count_output", 0)),
        "world_context_date":          meta.get("world_context_date", ""),
    }


def make_splits(df: pd.DataFrame) -> dict:
    """Stratified 70/15/15 split. Falls back to train-only if < 30 records."""
    if len(df) < 30:
        print(f"  ℹ️  {len(df)} records — train-only (need 30+ for splits)")
        return {"train": df}
    try:
        from sklearn.model_selection import train_test_split
        strat        = df["outcome_status"].fillna("unknown") + "_" + df["task_difficulty"].fillna("medium")
        train, temp  = train_test_split(df,   test_size=0.30, stratify=strat,            random_state=42)
        strat_temp   = strat.loc[temp.index]
        val,   test  = train_test_split(temp, test_size=0.50, stratify=strat_temp,       random_state=42)
        return {"train": train, "validation": val, "test": test}
    except ImportError:
        print("  ⚠️  scikit-learn not installed — train-only split")
        return {"train": df}
    except Exception as e:
        print(f"  ⚠️  Split failed ({e}) — train-only split")
        return {"train": df}


def push_to_hf(records: list[dict]):
    """Flatten, append to existing dataset, split, and push to HuggingFace."""
    if not records:
        print("  ⚠️  No records to push.")
        return

    HF_TOKEN     = _get_hf_token()
    DATASET_REPO = _get_dataset_repo()
    today        = datetime.now().strftime("%Y-%m-%d")

    # Flatten records
    rows = []
    for i, r in enumerate(records):
        try:
            rows.append(flatten_record(r))
        except Exception as e:
            print(f"  ⚠️  Could not flatten record {i}: {e}")
    df_new = pd.DataFrame(rows)
    print(f"  📦 Prepared {len(df_new)} rows.")

    # Always save local backup first
    os.makedirs("registry", exist_ok=True)
    backup_path = f"registry/batch_{today}.jsonl"
    with open(backup_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    print(f"  💾 Local backup: {backup_path}")

    if not HF_TOKEN:
        print("  ❌ HF_TOKEN not set — saved locally only.")
        return
    if not DATASET_REPO:
        print("  ❌ HF_DATASET_REPO not set — saved locally only.")
        return

    print(f"  🔑 HF_TOKEN: {HF_TOKEN[:8]}... | Repo: {DATASET_REPO}")

    try:
        from datasets import Dataset, DatasetDict, load_dataset
        from huggingface_hub import HfApi
    except ImportError as e:
        print(f"  ❌ Missing library: {e}. Run: pip install datasets huggingface_hub")
        return

    # Ensure repo exists
    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
        print(f"  ✅ Repo exists: {DATASET_REPO}")
    except Exception:
        try:
            api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False)
            print(f"  ✅ Repo created: {DATASET_REPO}")
        except Exception as e:
            print(f"  ❌ Cannot create repo: {e}")
            return

    # Load existing data and append
    try:
        existing    = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN)
        df_existing = existing.to_pandas()
        # Align columns — new schema may have extra columns
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = None
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"  📂 Existing: {len(df_existing)} rows → Combined: {len(df_combined)} rows")
    except Exception:
        df_combined = df_new
        print(f"  📂 No existing data — starting fresh with {len(df_combined)} rows")

    # Split and push
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
            commit_message=f"Daily update {today} — +{len(rows)} records (total: {len(df_combined)})",
        )
        print(f"  ✅ Pushed! https://huggingface.co/datasets/{DATASET_REPO}")
        print(f"  📊 Dataset now has {len(df_combined)} total records.")
    except Exception as e:
        print(f"  ❌ Push failed: {e}")
        traceback.print_exc()
        print(f"  💾 Data safe at: {backup_path}")