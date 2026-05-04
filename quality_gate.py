"""
quality_gate.py
Validates records before HuggingFace push.
- Schema completeness check
- Reward signal threshold
- Label drift detection
- No LLM calls — pure Python logic
"""

import json
import os
import statistics
from datetime import datetime

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry", "baseline_scores.json")

MIN_REWARD_SIGNAL     = 0.20   # lowered from 0.30 — allows partially labeled traces
MIN_STEPS             = 2
DRIFT_ALERT_THRESHOLD = 0.30


def validate_schema(record: dict) -> tuple[bool, str]:
    required = [
        ("task", "task"),
        ("trace",),
        ("outcome", "status"),
        ("labels", "labeler_model"),
        ("labels", "trace_level_scores"),
        ("metadata", "schema_version"),
    ]
    for path in required:
        obj = record
        for key in path:
            if not isinstance(obj, dict) or key not in obj:
                return False, f"Missing: {' > '.join(path)}"
            obj = obj[key]
        if obj is None or obj == "":
            return False, f"Empty: {' > '.join(path)}"

    if len(record.get("trace", [])) < MIN_STEPS:
        return False, f"Trace too short: {len(record['trace'])} steps"

    return True, "ok"


def validate_labels(record: dict) -> tuple[bool, str]:
    scores = record["labels"]["trace_level_scores"]

    # reward_computed must exist and not be None
    rc = scores.get("reward_computed")
    if rc is None:
        return False, "reward_computed is None"

    # Catch labeling failure verdict_reason strings — these rows should have been excluded
    verdict_reason = str(scores.get("verdict_reason", ""))
    if "primary_call_failed" in verdict_reason or "labeling_unavailable" in verdict_reason:
        return False, "Row has failed-labeling verdict_reason — exclude from dataset"

    rs = scores.get("reward_signal", 0)
    if isinstance(rs, (int, float)) and rs < MIN_REWARD_SIGNAL:
        return False, f"reward_signal too low: {rs}"

    # Catch identical default scores (labeler silently failing)
    tc  = scores.get("task_completion", -1)
    tue = scores.get("tool_use_efficiency", -1)
    rc2 = scores.get("reasoning_coherence", -1)
    oq  = scores.get("overall_quality", -1)
    if tc == 1 and tue == 1 and rc2 == 1 and round(float(oq), 2) == 3.33:
        return False, "Identical fallback scores — labeler likely failed silently"

    # Validate: failed outcome must not have task_completion > 0
    outcome = record.get("outcome", {}).get("status", "")
    if outcome == "failed" and tc > 0:
        scores["task_completion"]     = 0
        scores["reward_signal"]       = 0.0
        scores["reward_computed"]     = 0.0
        record["labels"]["trace_level_scores"] = scores

    return True, "ok"


def filter_valid(records: list[dict]) -> list[dict]:
    passed  = []
    dropped = {"schema": 0, "labels": 0}

    for record in records:
        ok, reason = validate_schema(record)
        if not ok:
            print(f"  ⚠️  Schema drop: {reason}")
            dropped["schema"] += 1
            continue

        ok, reason = validate_labels(record)
        if not ok:
            print(f"  ⚠️  Label drop: {reason}")
            dropped["labels"] += 1
            continue

        passed.append(record)

    print(f"\n🔍 Quality gate:")
    print(f"   Passed:         {len(passed)}")
    print(f"   Dropped schema: {dropped['schema']}")
    print(f"   Dropped labels: {dropped['labels']}")
    return passed


def update_baseline(records: list[dict]):
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    rewards = [
        r["labels"]["trace_level_scores"].get("reward_signal", 0)
        for r in records
        if isinstance(r["labels"]["trace_level_scores"].get("reward_signal"), (int, float))
    ]
    if not rewards:
        return
    baseline = {
        "mean":        round(statistics.mean(rewards), 4),
        "stdev":       round(statistics.stdev(rewards) if len(rewards) > 1 else 0, 4),
        "n":           len(rewards),
        "recorded_at": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  📊 Baseline saved: mean={baseline['mean']}, n={baseline['n']}")


def check_label_drift(records: list[dict]) -> bool:
    if not os.path.exists(BASELINE_PATH):
        update_baseline(records)
        return False

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    current = [
        r["labels"]["trace_level_scores"].get("reward_signal", 0)
        for r in records
        if isinstance(r["labels"]["trace_level_scores"].get("reward_signal"), (int, float))
    ]
    if not current:
        return False

    mean  = statistics.mean(current)
    drift = abs(mean - baseline["mean"])
    print(f"  📊 Drift: baseline={baseline['mean']:.3f}, current={mean:.3f}, drift={drift:.3f}")

    if drift > DRIFT_ALERT_THRESHOLD:
        print(f"  ⚠️  DRIFT DETECTED ({drift:.3f} > {DRIFT_ALERT_THRESHOLD}) — audit labeling pipeline")
        return True

    update_baseline(records)
    return False