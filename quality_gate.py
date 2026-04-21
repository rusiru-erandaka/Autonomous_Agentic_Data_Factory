"""
quality_gate.py
Final validation pass before records enter the HuggingFace dataset.
Checks schema completeness, label consistency, and reward signal threshold.
Also runs label drift detection against a static validation batch.
"""

import json
import os
import statistics
from datetime import datetime
from typing import Optional

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_REWARD_SIGNAL      = 0.30   # drop records with reward signal below this
MIN_STEPS              = 2      # traces must have at least 2 steps
MAX_CONFLICT_RATE      = 0.50   # reject if >50% of dual-labeled steps conflict
DRIFT_ALERT_THRESHOLD  = 0.30   # alert if mean reward shifts by this much

# Path to store the validation batch baseline
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "registry", "baseline_scores.json")


def validate_schema(record: dict) -> tuple[bool, str]:
    """Check that all required fields are present and non-empty."""
    required_paths = [
        ("task", "task"),
        ("trace",),
        ("outcome", "status"),
        ("labels", "labeler_model"),
        ("labels", "trace_level_scores"),
        ("metadata", "schema_version"),
    ]
    for path in required_paths:
        obj = record
        for key in path:
            if not isinstance(obj, dict) or key not in obj:
                return False, f"Missing field: {' > '.join(path)}"
            obj = obj[key]
        if obj is None or obj == "":
            return False, f"Empty field: {' > '.join(path)}"

    if len(record.get("trace", [])) < MIN_STEPS:
        return False, f"Trace too short: {len(record['trace'])} steps (min {MIN_STEPS})"

    return True, "ok"


def validate_labels(record: dict) -> tuple[bool, str]:
    """Check label quality and conflict rates."""
    trace_scores = record["labels"]["trace_level_scores"]
    reward       = trace_scores.get("reward_signal", 0)

    if isinstance(reward, (int, float)) and reward < MIN_REWARD_SIGNAL:
        return False, f"Reward signal too low: {reward} (min {MIN_REWARD_SIGNAL})"

    # Check dual-label conflict rate
    step_scores    = record["labels"].get("step_level_scores", [])
    dual_steps     = [s for s in step_scores if "agreement_score" in s]
    if dual_steps:
        conflict_rate = sum(1 for s in dual_steps if s.get("conflict_flag", False)) / len(dual_steps)
        if conflict_rate > MAX_CONFLICT_RATE:
            return False, f"High label conflict rate: {conflict_rate:.0%}"

    return True, "ok"


def filter_valid(records: list[dict]) -> list[dict]:
    """Run all validation checks. Returns only passing records."""
    passed  = []
    dropped = {"schema": 0, "labels": 0}

    for record in records:
        ok, reason = validate_schema(record)
        if not ok:
            dropped["schema"] += 1
            continue

        ok, reason = validate_labels(record)
        if not ok:
            dropped["labels"] += 1
            continue

        passed.append(record)

    print(f"\n🔍 Quality gate results:")
    print(f"   Passed:         {len(passed)}")
    print(f"   Dropped schema: {dropped['schema']}")
    print(f"   Dropped labels: {dropped['labels']}")
    return passed


# ── Drift Detection ────────────────────────────────────────────────────────────

def update_baseline(records: list[dict]):
    """Save reward signal stats from current batch as new baseline."""
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    rewards = [
        r["labels"]["trace_level_scores"].get("reward_signal", 0)
        for r in records
        if isinstance(r["labels"]["trace_level_scores"].get("reward_signal"), (int, float))
    ]
    if not rewards:
        return
    baseline = {
        "mean":     round(statistics.mean(rewards), 4),
        "stdev":    round(statistics.stdev(rewards) if len(rewards) > 1 else 0, 4),
        "n":        len(rewards),
        "recorded_at": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  📊 Baseline updated: mean={baseline['mean']}, n={baseline['n']}")


def check_label_drift(records: list[dict]) -> bool:
    """
    Compare current batch reward distribution against stored baseline.
    Returns True if drift is detected (alerts but does NOT block pipeline).
    """
    if not os.path.exists(BASELINE_PATH):
        update_baseline(records)
        return False

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    current_rewards = [
        r["labels"]["trace_level_scores"].get("reward_signal", 0)
        for r in records
        if isinstance(r["labels"]["trace_level_scores"].get("reward_signal"), (int, float))
    ]
    if not current_rewards:
        return False

    current_mean = statistics.mean(current_rewards)
    drift        = abs(current_mean - baseline["mean"])

    print(f"  📊 Drift check: baseline={baseline['mean']:.3f}, current={current_mean:.3f}, drift={drift:.3f}")

    if drift > DRIFT_ALERT_THRESHOLD:
        print(f"  ⚠️  LABEL DRIFT DETECTED (drift={drift:.3f} > threshold={DRIFT_ALERT_THRESHOLD})")
        print(f"  ⚠️  Consider auditing the labeling pipeline.")
        return True

    update_baseline(records)
    return False