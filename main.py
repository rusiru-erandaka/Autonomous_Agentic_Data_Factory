"""
main.py
Pipeline orchestrator — loads .env, prints startup info, runs all 5 stages.
Run manually:  python main.py
Scheduled:     GitHub Actions cron
"""

import os
import sys

# ── Load .env FIRST — before any other import reads env vars ──────────────────
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        print("⚠️  No .env file found — create one from .env.example")
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
    print(f"✅ .env loaded ({loaded} keys set)")

_load_env()

import time
import json
import schedule
from datetime import datetime

from task_generator import generate_tasks
from agent_executor import execute_tasks
from labeler        import label_traces
from quality_gate   import filter_valid, check_label_drift
from hf_uploader    import push_to_hf

DAILY_TASK_TARGET = int(os.environ.get("DAILY_TASK_TARGET", "18"))
RUN_ONCE          = os.environ.get("RUN_ONCE", "false").lower() == "true"
SCHEDULE_TIME     = os.environ.get("SCHEDULE_TIME", "02:00")


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Autonomous Agentic Data Factory — Pipeline v3.0           ║
║   Niche: 6 domains | Providers: Groq + Google + OpenRouter  ║
╚══════════════════════════════════════════════════════════════╝""")


def _print_startup_info():
    """Show provider assignment and active model pools."""
    from llm_client import ROLE_CONFIG, get_active_models_summary

    print("\n📋 Provider & Key Assignment:")
    rows = [
        ("Stage 1 — Task Generation",   "Groq",        "GROQ_API_KEY",  "generator / quality_gate", "~20 req/day"),
        ("Stage 2 — Agent Execution",   "Groq",        "GROQ_API_KEY",  "agent / agent_backup",     "~72 req/day"),
        ("Stage 3 — Primary Labeling",  "Groq",        "GROQ_API_KEY",  "labeler",                  "~18 req/day"),
        ("Stage 3 — Secondary Label",   "Groq",        "GROQ_API_KEY",  "secondary",                "~18 req/day"),
        ("Stage 4 — Quality Gate",      "—",           "—",             "pure Python",              "0 req/day"),
        ("Stage 5 — HF Upload",         "—",           "HF_TOKEN",      "pure Python",              "0 req/day"),
    ]
    for stage, provider, key_name, roles, budget in rows:
        val    = os.environ.get(key_name, "") if key_name != "—" else "N/A"
        status = f"{val[:14]}..." if val and val not in ("", "N/A") and not val.startswith("replace") else ("✅ N/A" if key_name == "—" else "❌ NOT SET")
        print(f"   {stage:<32} {provider:<8} {status:<22} ({budget})")

    print("\n🤖 Active Model Pools:")
    summary = get_active_models_summary()
    for role, info in summary.items():
        pool_str = " → ".join(m.split("/")[-1][:22] for m in info["pool"])
        print(f"   [{role:<14}] {info['provider']:<8}: {pool_str}")

    # OpenRouter fallback status
    or_keys = sum(1 for i in range(1, 4) if os.environ.get(f"OPENROUTER_API_KEY_{i}", "").strip()
                  and not os.environ.get(f"OPENROUTER_API_KEY_{i}", "").startswith("sk-or-v1-replace"))
    groq_set = "✅" if os.environ.get("GROQ_API_KEY", "") else "❌ NOT SET"
    print(f"\n   Groq API key: {groq_set}")
    print(f"   OpenRouter fallback: {or_keys}/3 keys set")
    print()


def run_pipeline():
    start = time.time()
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"🚀 Pipeline started: {today}")
    print(f"{'='*60}")
    _print_startup_info()

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
        # ── Stage 1: Generate tasks ────────────────────────────────────────────
        print("\n[1/5] 📋 Generating tasks...")
        tasks = generate_tasks(total=DAILY_TASK_TARGET)
        stats["tasks_generated"] = len(tasks)
        if not tasks:
            print("  ❌ No tasks generated. Aborting.")
            return

        # ── Stage 2: Execute agent ─────────────────────────────────────────────
        print("\n[2/5] 🤖 Running agent on tasks...")
        traces = execute_tasks(tasks)
        stats["traces_captured"] = len(traces)
        if not traces:
            print("  ❌ No traces captured. Aborting.")
            return

        # ── Stage 3: Label traces ──────────────────────────────────────────────
        print("\n[3/5] 🏷️  Labeling traces...")
        labeled = label_traces(traces)
        stats["traces_labeled"] = len(labeled)

        # ── Stage 4: Quality gate ──────────────────────────────────────────────
        print("\n[4/5] 🔍 Quality gate...")
        clean = filter_valid(labeled)
        stats["records_passed"] = len(clean)
        stats["drift_detected"] = check_label_drift(clean) if clean else False

        if not clean:
            print("  ❌ All records failed quality gate.")
            return

        # ── Stage 5: Push to HuggingFace ──────────────────────────────────────
        print("\n[5/5] 📤 Pushing to HuggingFace...")
        push_to_hf(clean)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        msg = f"Pipeline error: {e}"
        print(f"\n❌ {msg}")
        stats["errors"].append(msg)
        import traceback
        traceback.print_exc()

    duration = round(time.time() - start, 1)
    print(f"""
{'='*60}
✅ Pipeline complete in {duration}s
   Tasks generated:  {stats['tasks_generated']}
   Traces captured:  {stats['traces_captured']}
   Traces labeled:   {stats['traces_labeled']}
   Records passed:   {stats['records_passed']}
   Drift detected:   {stats['drift_detected']}
   Errors:           {len(stats['errors'])}
{'='*60}
""")

    os.makedirs("registry", exist_ok=True)
    log_path = f"registry/run_{stats['date']}.json"
    with open(log_path, "w") as f:
        json.dump({**stats, "duration_seconds": duration}, f, indent=2)
    print(f"  📝 Run log: {log_path}")


if __name__ == "__main__":
    print_banner()

    if RUN_ONCE:
        print("▶️  Single run mode")
        run_pipeline()
        sys.exit(0)

    print(f"⏰ Scheduled mode: daily at {SCHEDULE_TIME} UTC")
    print("   Press Ctrl+C to stop.\n")
    schedule.every().day.at(SCHEDULE_TIME).do(run_pipeline)
    run_pipeline()   # run immediately on start
    while True:
        schedule.run_pending()
        time.sleep(60)