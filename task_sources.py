"""
task_sources.py
Real-world signal collection with quality filtering.

Fixes:
- GitHub: scrapes full issue body (1000 chars), filters for bug/help-wanted labels
- Dedup: signals marked seen only AFTER successful task creation
- Relevance: stronger keyword filter before returning signals
"""

import os
import time
import hashlib
import sqlite3
import requests
import feedparser
from datetime import datetime, timezone
from bs4 import BeautifulSoup

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

_SEEN_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry", "seen_signals.db")

def _init_seen_db():
    os.makedirs(os.path.dirname(_SEEN_DB), exist_ok=True)
    conn = sqlite3.connect(_SEEN_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS seen (sig_id TEXT PRIMARY KEY, seen_at TEXT)")
    conn.commit()
    conn.close()

def _is_seen(sig_id: str) -> bool:
    try:
        conn = sqlite3.connect(_SEEN_DB)
        row  = conn.execute("SELECT 1 FROM seen WHERE sig_id=?", (sig_id,)).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False

def _mark_seen(sig_id: str):
    try:
        conn = sqlite3.connect(_SEEN_DB)
        conn.execute("INSERT OR IGNORE INTO seen VALUES (?,?)",
                     (sig_id, datetime.now().strftime("%Y-%m-%d")))
        conn.commit()
        conn.close()
    except Exception:
        pass

def _sig_id(source: str, text: str) -> str:
    return hashlib.md5(f"{source}::{text[:120]}".encode()).hexdigest()

def mark_signal_used(signal: dict):
    """Call AFTER signal is successfully converted to an approved task."""
    sig_id = signal.get("_sig_id")
    if sig_id:
        _mark_seen(sig_id)

RELEVANT_KEYWORDS = [
    "api", "agent", "tool", "webhook", "endpoint", "request", "response",
    "code", "script", "debug", "error", "fix", "implement", "integrate",
    "stripe", "notion", "github", "slack", "airtable", "openai", "langchain",
    "crewai", "autogen", "llm", "function", "async", "timeout", "retry",
    "bug", "exception", "traceback", "import", "module", "client", "auth",
]

def _is_relevant(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in RELEVANT_KEYWORDS)

# ── GitHub Issues ──────────────────────────────────────────────────────────────
# Focus on bug/help-wanted issues — these have concrete problems to solve

GITHUB_REPOS = [
    ("langchain-ai/langchain",  ["bug", "help wanted"]),
    ("crewAIInc/crewAI",        ["bug", "help wanted"]),
    ("microsoft/autogen",       ["bug", "help wanted"]),
    ("openai/openai-python",    ["bug", "question"]),
    ("run-llama/llama_index",   ["bug", "help wanted"]),
    ("BerriAI/litellm",         ["bug", "help wanted"]),
]

def fetch_github_issues(max_per_repo: int = 8) -> list[dict]:
    _init_seen_db()
    results = []
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

    for repo, preferred_labels in GITHUB_REPOS:
        fetched = []
        # Try preferred labels first, then open issues without label filter
        label_attempts = [",".join(preferred_labels), ""]
        for label_str in label_attempts:
            if len(fetched) >= max_per_repo:
                break
            try:
                params = {
                    "state":     "open",
                    "per_page":  max_per_repo,
                    "sort":      "created",
                    "direction": "desc",
                }
                if label_str:
                    params["labels"] = label_str

                resp = requests.get(
                    f"https://api.github.com/repos/{repo}/issues",
                    headers=headers, params=params, timeout=15,
                )
                if resp.status_code != 200:
                    continue

                for issue in resp.json():
                    if not isinstance(issue, dict):
                        continue
                    if issue.get("pull_request"):
                        continue  # skip PRs

                    source = f"github:{repo}#{issue['number']}"
                    title  = (issue.get("title") or "").strip()

                    # Use full body up to 1000 chars — contains error msgs + reproduction steps
                    body   = (issue.get("body") or "")[:1000].strip()
                    text   = f"{title}\n{body}"

                    if not _is_relevant(text):
                        continue
                    sig = _sig_id(source, text)
                    if _is_seen(sig):
                        continue

                    # Extract issue labels for metadata
                    issue_labels = [l["name"] for l in issue.get("labels", [])]

                    fetched.append({
                        "raw_text":     text,
                        "title":        title,
                        "source":       source,
                        "source_url":   issue.get("html_url", ""),
                        "issue_labels": issue_labels,
                        "date":         issue["created_at"][:10],
                        "_sig_id":      sig,
                    })
                time.sleep(0.5)
            except Exception as e:
                print(f"  ⚠️  GitHub {repo}: {e}")

        results.extend(fetched[:max_per_repo])

    print(f"  📦 GitHub: {len(results)} new signals")
    return results

# ── Stack Overflow ─────────────────────────────────────────────────────────────
SO_TAGS = [
    "langchain", "openai-api", "stripe-api", "notion-api",
    "github-api", "llm-agent", "autogen", "crewai", "litellm",
]

def fetch_stackoverflow_questions(max_per_tag: int = 4) -> list[dict]:
    _init_seen_db()
    results = []
    for tag in SO_TAGS:
        try:
            resp = requests.get(
                "https://api.stackexchange.com/2.3/questions",
                params={
                    "order": "desc", "sort": "creation", "tagged": tag,
                    "site": "stackoverflow", "pagesize": max_per_tag,
                    "filter": "withbody",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            for q in resp.json().get("items", []):
                source = f"stackoverflow:{q['question_id']}"
                title  = (q.get("title") or "").strip()
                body   = BeautifulSoup(q.get("body", ""), "html.parser").get_text()[:600]
                text   = f"{title}\n{body}"

                if not _is_relevant(text):
                    continue
                sig = _sig_id(source, text)
                if _is_seen(sig):
                    continue
                results.append({
                    "raw_text":   text,
                    "title":      title,
                    "source":     source,
                    "source_url": f"https://stackoverflow.com/q/{q['question_id']}",
                    "date":       datetime.fromtimestamp(
                        q["creation_date"], tz=timezone.utc).strftime("%Y-%m-%d"),
                    "_sig_id":    sig,
                })
            time.sleep(1)
        except Exception as e:
            print(f"  ⚠️  SO '{tag}': {e}")
    print(f"  📦 Stack Overflow: {len(results)} new signals")
    return results

# ── HuggingFace Papers ─────────────────────────────────────────────────────────
def fetch_hf_daily_papers(max_papers: int = 6) -> list[dict]:
    _init_seen_db()
    results = []
    try:
        resp = requests.get("https://huggingface.co/papers", timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        for article in soup.select("article")[:max_papers]:
            title_el   = article.select_one("h3")
            summary_el = article.select_one("p")
            if not title_el:
                continue
            title   = title_el.get_text(strip=True)
            summary = summary_el.get_text(strip=True) if summary_el else ""
            text    = f"{title}\n{summary[:400]}"
            if not _is_relevant(text):
                continue
            sig = _sig_id("hf_papers", text)
            if _is_seen(sig):
                continue
            results.append({
                "raw_text":   text,
                "title":      title,
                "source":     "huggingface_papers",
                "source_url": "https://huggingface.co/papers",
                "date":       datetime.now().strftime("%Y-%m-%d"),
                "_sig_id":    sig,
            })
    except Exception as e:
        print(f"  ⚠️  HF papers: {e}")
    print(f"  📦 HuggingFace Papers: {len(results)} new signals")
    return results

# ── API Changelogs ─────────────────────────────────────────────────────────────
CHANGELOG_FEEDS = {
    "stripe": "https://stripe.com/blog/changelog.rss",
    "github": "https://github.blog/changelog/feed/",
    "openai": "https://openai.com/blog/rss.xml",
}

def fetch_api_changelogs(max_per_feed: int = 3) -> list[dict]:
    _init_seen_db()
    results = []
    for api_name, feed_url in CHANGELOG_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                summary = BeautifulSoup(
                    entry.get("summary", ""), "html.parser"
                ).get_text()[:400]
                text   = f"{entry.title}\n{summary}"
                source = f"{api_name}_changelog"
                if not _is_relevant(text):
                    continue
                sig = _sig_id(source, text)
                if _is_seen(sig):
                    continue
                results.append({
                    "raw_text":   text,
                    "title":      entry.title,
                    "source":     source,
                    "source_url": entry.get("link", ""),
                    "date":       datetime.now().strftime("%Y-%m-%d"),
                    "_sig_id":    sig,
                })
        except Exception as e:
            print(f"  ⚠️  Changelog '{api_name}': {e}")
    print(f"  📦 API Changelogs: {len(results)} new signals")
    return results

# ── Aggregator ─────────────────────────────────────────────────────────────────
def collect_all_signals() -> list[dict]:
    print("🌐 Collecting real-world signals...")
    signals = []
    signals += fetch_github_issues()
    signals += fetch_stackoverflow_questions()
    signals += fetch_hf_daily_papers()
    signals += fetch_api_changelogs()
    print(f"  ✅ Total new signals: {len(signals)}")
    return signals