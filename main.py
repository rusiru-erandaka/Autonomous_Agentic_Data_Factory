"""
main.py
Daily pipeline orchestrator.
Run manually:  python main.py
Scheduled:     GitHub Actions runs this every day at 2am UTC
"""

import os
import sys

# ── Load .env file FIRST before any other imports read env vars ───────────────
def _load_env():
    """Load .env file — works on Windows and Linux, with or without export prefix."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        print("\u26a0\ufe0f  No .env file found — create one from .env.example")
        return
    loaded = 0
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
                val = val[1:-1]
            if not key or val in ("", "...", "replace_with_your_key"):
                continue
            os.environ[key] = val
            loaded += 1
    print(f"\u2705 .env loaded ({loaded} keys set)")
_load_env()

import time
import json
import schedule
from datetime import datetime

from task_generator import generate_tasks
from agent_executor import execute_tasks
from labeler import label_traces
from quality_gate import filter_valid, check_label_drift
from hf_uploader import push_to_hf

# ── Config ─────────────────────────────────────────────────────────────────────
DAILY_TASK_TARGET = int(os.environ.get("DAILY_TASK_TARGET", "15"))   # 15 tasks = ~80 API calls, fits 3-key 150 req/day budget
RUN_ONCE          = os.environ.get("RUN_ONCE", "false").lower() == "true"
SCHEDULE_TIME     = os.environ.get("SCHEDULE_TIME", "02:00")         # UTC


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AI Agent Behavior Dataset — Automated Pipeline          ║
║     Niche: API Orchestration + Code Agent                   ║
╚══════════════════════════════════════════════════════════════╝
""")


def _print_key_assignment():
    """Show which API key handles each stage — helps verify keys are loaded."""
    print("\n📋 API Key Assignment:")
    llm_stages = [
        ("Stage 1 — Task Gen + Secondary Label", "OPENROUTER_API_KEY_1", "~8 gen + ~12 secondary = ~20 req/day"),
        ("Stage 2 — Agent Execution",            "OPENROUTER_API_KEY_2", "~48 req/day  ← heaviest"),
        ("Stage 3 — Primary Labeling",           "OPENROUTER_API_KEY_3", "~12 req/day"),
    ]
    for stage, key_name, budget in llm_stages:
        val    = os.environ.get(key_name, "")
        status = f"{val[:16]}..." if val and not val.startswith("sk-or-v1-replace") else "❌ NOT SET"
        print(f"   {stage:<32} {key_name} = {status}  ({budget})")
    print(f"   {'Stage 4 — Quality Gate':<32} pure Python — no LLM calls  (0 req/day)")
    print(f"   {'Stage 5 — HF Upload':<32} pure Python — no LLM calls  (0 req/day)")
    print()


def run_pipeline():
    start   = time.time()
    today   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"🚀 Pipeline run started: {today}")
    print(f"{'='*60}")
    _print_key_assignment()

    stats = {
        "date":            datetime.now().strftime("%Y-%m-%d"),
        "tasks_generated": 0,
        "traces_captured": 0,
        "traces_labeled":  0,
        "records_passed":  0,
        "drift_detected":  False,
        "errors":          [],
    }

    try:
        # ── Step 1: Generate Tasks ─────────────────────────────────────────────
        print("\n[1/5] 📋 Generating tasks...")
        tasks = generate_tasks(total=DAILY_TASK_TARGET)
        stats["tasks_generated"] = len(tasks)
        if not tasks:
            print("  ❌ No tasks generated. Aborting pipeline.")
            return

        # ── Step 2: Execute Agent ──────────────────────────────────────────────
        print("\n[2/5] 🤖 Running agent on tasks...")
        traces = execute_tasks(tasks)
        stats["traces_captured"] = len(traces)
        if not traces:
            print("  ❌ No traces captured. Aborting pipeline.")
            return

        # ── Step 3: Label with Nemotron ────────────────────────────────────────
        print("\n[3/5] 🏷️  Labeling traces...")
        labeled = label_traces(traces)
        stats["traces_labeled"] = len(labeled)

        # ── Step 4: Quality Gate + Drift Detection ─────────────────────────────
        print("\n[4/5] 🔍 Quality gate + drift detection...")
        clean = filter_valid(labeled)
        stats["records_passed"] = len(clean)

        drift = check_label_drift(clean)
        stats["drift_detected"] = drift

        if not clean:
            print("  ❌ All records failed quality gate. Nothing to push.")
            return

        # ── Step 5: Push to HuggingFace ────────────────────────────────────────
        print("\n[5/5] 📤 Pushing to HuggingFace...")
        push_to_hf(clean)

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        error_msg = f"Pipeline error: {e}"
        print(f"\n❌ {error_msg}")
        stats["errors"].append(error_msg)
        import traceback
        traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────────────
    duration = round(time.time() - start, 1)
    print(f"""
{'='*60}
✅ Pipeline run complete in {duration}s
   Tasks generated:  {stats['tasks_generated']}
   Traces captured:  {stats['traces_captured']}
   Traces labeled:   {stats['traces_labeled']}
   Records passed:   {stats['records_passed']}
   Drift detected:   {stats['drift_detected']}
   Errors:           {len(stats['errors'])}
{'='*60}
""")

    # Save run log
    os.makedirs("registry", exist_ok=True)
    log_path = f"registry/run_{stats['date']}.json"
    with open(log_path, "w") as f:
        json.dump({**stats, "duration_seconds": duration}, f, indent=2)
    print(f"  📝 Run log saved: {log_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner()

    # GitHub Actions sets RUN_ONCE=true — run immediately and exit
    if RUN_ONCE:
        print("▶️  Single run mode (GitHub Actions)")
        run_pipeline()
        sys.exit(0)

    # Local dev mode — run on a schedule
    print(f"⏰ Scheduled mode: pipeline will run daily at {SCHEDULE_TIME} UTC")
    print("   Press Ctrl+C to stop.\n")
    print("   Tip: To run immediately, press R + Enter or set RUN_ONCE=true\n")

    schedule.every().day.at(SCHEDULE_TIME).do(run_pipeline)

    # Also run immediately on first start
    run_pipeline()

    while True:
        schedule.run_pending()
        time.sleep(60)