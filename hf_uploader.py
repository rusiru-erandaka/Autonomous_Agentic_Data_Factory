"""
hf_uploader.py
Pushes validated records to HuggingFace dataset repo.
"""

import json
import os
import traceback
from datetime import datetime

import pandas as pd


def _get_hf_token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def _get_dataset_repo() -> str:
    return os.environ.get("HF_DATASET_REPO", "").strip()


def _normalize_date(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return value
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        return value
    for fmt in ("%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


def _fallback_source_url(task: dict) -> str:
    existing = task.get("source_url", "")
    if existing:
        return existing
    freshness = task.get("freshness_source", "")
    if freshness.startswith("github:") and "#" in freshness:
        repo, issue_no = freshness[len("github:"):].split("#", 1)
        if repo and issue_no.isdigit():
            return f"https://github.com/{repo}/issues/{issue_no}"
    if task.get("generation_strategy") in ("template_based", "mutation_based"):
        return f"template:{task.get('freshness_source', 'unknown')}"
    return ""


def flatten_record(record: dict) -> dict:
    """Flatten nested record into a single HuggingFace-ready row."""
    trace_scores = record["labels"]["trace_level_scores"]
    meta = record.get("metadata", {})

    tc = float(trace_scores.get("task_completion", 0) or 0)
    tue = float(trace_scores.get("tool_use_efficiency", 0) or 0)
    rc = float(trace_scores.get("reasoning_coherence", 0) or 0)
    sc = float(trace_scores.get("safety_compliance", 3) or 3)

    reward_computed = round((tc / 3 * 0.4) + (tue / 3 * 0.3) + (rc / 3 * 0.3), 4)
    reward_signal = reward_computed
    overall_quality = round(((tc + tue + rc + sc) / 12.0) * 10.0, 2)

    task_niche = (
        record["task"].get("niche") or
        record["task"].get("task_niche") or
        "api_orchestration"
    )
    agreement = record["labels"].get("agreement_score")

    verdict_reason = trace_scores.get("verdict_reason", "") or ""
    if not verdict_reason.strip():
        verdict_reason = (
            f"Auto: verdict={trace_scores.get('supervisor_verdict', 'flag')}, "
            f"task_completion={tc}, reward={reward_computed}"
        )

    return {
        "trace_id": record.get("trace_id", ""),
        "created_at": record.get("created_at", "") or datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": meta.get("schema_version", "v4.0"),

        "task": record["task"]["task"],
        "task_difficulty": record["task"].get("difficulty", "medium"),
        "task_niche": task_niche,
        "expected_tools": json.dumps(record["task"].get("expected_tools", [])),
        "likely_failure_points": json.dumps(record["task"].get("likely_failure_points", [])),
        "generation_strategy": record["task"].get("generation_strategy", ""),
        "freshness_source": record["task"].get("freshness_source", ""),
        "source_url": _fallback_source_url(record["task"]),
        "repo_url": record["task"].get("repo_url", ""),
        "repo_clone_url": record["task"].get("repo_clone_url", ""),
        "repo_full_name": record["task"].get("repo_full_name", ""),
        "repo_default_branch": record["task"].get("repo_default_branch", ""),
        "repo_language": record["task"].get("repo_language", ""),
        "issue_number": record["task"].get("issue_number"),
        "issue_title": record["task"].get("issue_title", ""),
        "issue_labels": json.dumps(record["task"].get("issue_labels", [])),
        "path_hints": json.dumps(record["task"].get("path_hints", [])),
        "execution_target": record["task"].get("execution_target", "synthetic"),
        "task_type": record["task"].get("task_type", "generic_task"),

        "trace_json": json.dumps(record.get("trace", [])),

        "outcome_status": record["outcome"]["status"],
        "total_steps": int(record["outcome"]["total_steps"]),
        "total_tool_calls": int(record["outcome"]["total_tool_calls"]),
        "tools_used": json.dumps(record["outcome"].get("tools_used", [])),
        "failure_occurred": bool(record["outcome"]["failure_occurred"]),
        "failure_reason": record["outcome"].get("failure_reason") or "",
        "final_answer": record["outcome"].get("final_answer") or "",
        "duration_seconds": float(record["outcome"].get("duration_seconds", 0)),
        "execution_grounded": bool(record["outcome"].get("execution_grounded", False)),
        "files_changed": json.dumps(record["outcome"].get("files_changed", [])),
        "validation_commands": json.dumps(record["outcome"].get("validation_commands", [])),
        "command_history": json.dumps(record["outcome"].get("command_history", [])),

        "labeler_model": record["labels"]["labeler_model"],
        "labeler_model_2": record["labels"].get("labeler_model_2", ""),
        "constitution_version": record["labels"]["constitution_version"],
        "labeled_at": record["labels"].get("labeled_at", "") or "",
        "step_level_scores": json.dumps(record["labels"].get("step_level_scores", [])),
        "primary_trace_scores": json.dumps(record["labels"].get("primary_trace_scores", {})),
        "secondary_trace_scores": json.dumps(record["labels"].get("secondary_trace_scores", {})),
        "dual_labeled": bool(record["labels"].get("dual_labeled", False)),
        "agreement_score": float(agreement) if isinstance(agreement, (int, float)) else None,
        "conflict_flag": bool(record["labels"].get("conflict_flag", False)),
        "conflict_dimensions": json.dumps(record["labels"].get("conflict_dimensions", [])),
        "merge_strategy": record["labels"].get("merge_strategy", ""),
        "reward_adjustment_reason": record["labels"].get("reward_adjustment_reason", ""),
        "quality_formula": record["labels"].get("quality_formula", "((task_completion + tool_use_efficiency + reasoning_coherence + safety_compliance) / 12) * 10"),
        "rubric_hash": record["labels"].get("rubric_hash", ""),

        "task_completion": tc,
        "tool_use_efficiency": tue,
        "reasoning_coherence": rc,
        "safety_compliance": sc,
        "overall_quality": overall_quality,
        "reward_signal": reward_signal,
        "reward_computed": reward_computed,
        "reward_formula": record["labels"].get("reward_formula", "task_completion*0.4 + tool_use_efficiency*0.3 + reasoning_coherence*0.3 (each /3)"),
        "supervisor_verdict": trace_scores.get("supervisor_verdict", "flag"),
        "verdict_reason": verdict_reason,

        "agent_framework": meta.get("agent_framework", "react"),
        "agent_model": meta.get("agent_model", "groq/llama-3.3-70b-versatile"),
        "agent_temperature": float(meta.get("agent_temperature", 0.4)),
        "prompt_template_version": meta.get("prompt_template_version", "v4.0"),
        "token_count_input": int(meta.get("token_count_input", 0)),
        "token_count_output": int(meta.get("token_count_output", 0)),
        "world_context_date": _normalize_date(meta.get("world_context_date", "")),
    }


def make_splits(df: pd.DataFrame) -> dict:
    """Stratified 70/15/15 split. Falls back to train-only if < 30 records."""
    if len(df) < 30:
        print(f"  ℹ️  {len(df)} records - train-only (need 30+ for splits)")
        return {"train": df}
    try:
        from sklearn.model_selection import train_test_split

        strat = df["outcome_status"].fillna("unknown") + "_" + df["task_difficulty"].fillna("medium")
        train, temp = train_test_split(df, test_size=0.30, stratify=strat, random_state=42)
        strat_temp = strat.loc[temp.index]
        val, test = train_test_split(temp, test_size=0.50, stratify=strat_temp, random_state=42)
        return {"train": train, "validation": val, "test": test}
    except ImportError:
        print("  ⚠️  scikit-learn not installed - train-only split")
        return {"train": df}
    except Exception as e:
        print(f"  ⚠️  Split failed ({e}) - train-only split")
        return {"train": df}


def push_to_hf(records: list[dict]):
    """Flatten, append to existing dataset, split, and push to HuggingFace."""
    if not records:
        print("  ⚠️  No records to push.")
        return

    hf_token = _get_hf_token()
    dataset_repo = _get_dataset_repo()
    today = datetime.now().strftime("%Y-%m-%d")

    rows = []
    for i, record in enumerate(records):
        try:
            rows.append(flatten_record(record))
        except Exception as e:
            print(f"  ⚠️  Could not flatten record {i}: {e}")
    df_new = pd.DataFrame(rows)
    print(f"  Prepared {len(df_new)} rows.")

    os.makedirs("registry", exist_ok=True)
    backup_path = f"registry/batch_{today}.jsonl"
    with open(backup_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    print(f"  Local backup: {backup_path}")

    if not hf_token:
        print("  HF_TOKEN not set - saved locally only.")
        return
    if not dataset_repo:
        print("  HF_DATASET_REPO not set - saved locally only.")
        return

    try:
        from datasets import Dataset, DatasetDict, load_dataset
        from huggingface_hub import HfApi
    except ImportError as e:
        print(f"  Missing library: {e}. Run: pip install datasets huggingface_hub")
        return

    api = HfApi(token=hf_token)
    try:
        api.repo_info(repo_id=dataset_repo, repo_type="dataset")
    except Exception:
        try:
            api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=False)
        except Exception as e:
            print(f"  Cannot create repo: {e}")
            return

    try:
        existing = load_dataset(dataset_repo, split="train", token=hf_token)
        df_existing = existing.to_pandas()
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = None
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"  Existing: {len(df_existing)} rows -> Combined: {len(df_combined)} rows")
    except Exception:
        df_combined = df_new
        print(f"  No existing data - starting fresh with {len(df_combined)} rows")

    splits = make_splits(df_combined)
    split_info = {k: len(v) for k, v in splits.items()}
    print(f"  Splits: {split_info}")

    try:
        dataset_dict = DatasetDict({
            name: Dataset.from_pandas(df.reset_index(drop=True))
            for name, df in splits.items()
        })
        dataset_dict.push_to_hub(
            dataset_repo,
            token=hf_token,
            commit_message=f"Daily update {today} - +{len(rows)} records (total: {len(df_combined)})",
        )
        print(f"  Pushed! https://huggingface.co/datasets/{dataset_repo}")
    except Exception as e:
        print(f"  Push failed: {e}")
        traceback.print_exc()
        print(f"  Data safe at: {backup_path}")
