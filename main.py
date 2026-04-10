"""
main.py
Daily pipeline orchestrator.
Run manually:  python main.py
Scheduled:     GitHub Actions runs this every day at 2am UTC
"""

import os
import sys
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
DAILY_TASK_TARGET = int(os.environ.get("DAILY_TASK_TARGET", "10"))   # start small
RUN_ONCE          = os.environ.get("RUN_ONCE", "false").lower() == "true"
SCHEDULE_TIME     = os.environ.get("SCHEDULE_TIME", "02:00")         # UTC


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     AI Agent Behavior Dataset — Automated Pipeline          ║
║     Niche: API Orchestration + Code Agent                   ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_pipeline():
    start   = time.time()
    today   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"🚀 Pipeline run started: {today}")
    print(f"{'='*60}")

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