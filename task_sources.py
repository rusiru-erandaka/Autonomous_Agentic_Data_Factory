"""
task_sources.py
Collects repo-grounded real-world signals, with GitHub issues as the primary source.

The current supervisor goal is real coding-agent oversight, so GitHub issues are
scored for local executability and code-change potential before they are admitted
into the pipeline.
"""

import hashlib
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import feedparser
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_ONLY_SIGNALS = os.environ.get("GITHUB_ONLY_SIGNALS", "true").lower() == "true"
MIN_GITHUB_ISSUE_SCORE = int(os.environ.get("MIN_GITHUB_ISSUE_SCORE", "5"))
MAX_BODY_CHARS = 1800
GITHUB_TIMEOUT_SECONDS = int(os.environ.get("GITHUB_TIMEOUT_SECONDS", "25"))
GITHUB_MAX_RETRIES = int(os.environ.get("GITHUB_MAX_RETRIES", "2"))

_SEEN_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry", "seen_signals.db")

CODE_KEYWORDS = [
    "bug", "fix", "failing", "failure", "regression", "traceback", "exception",
    "error", "incorrect", "broken", "crash", "test", "pytest", "unit test",
    "integration test", "repro", "steps to reproduce", "expected behavior",
    "actual behavior", "stack trace", "module", "import", "function", "class",
    "file", "line", "schema", "parser", "client", "timeout", "retry",
]

ANTI_PATTERNS = [
    "feature request", "documentation request", "question", "usage question",
    "support request", "how do i", "how can i", "proposal", "discussion",
]

PATH_HINT_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?")


def _github_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=GITHUB_MAX_RETRIES,
        connect=GITHUB_MAX_RETRIES,
        read=GITHUB_MAX_RETRIES,
        status=GITHUB_MAX_RETRIES,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _init_seen_db():
    os.makedirs(os.path.dirname(_SEEN_DB), exist_ok=True)
    conn = sqlite3.connect(_SEEN_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS seen (sig_id TEXT PRIMARY KEY, seen_at TEXT)")
    conn.commit()
    conn.close()


def _is_seen(sig_id: str) -> bool:
    try:
        conn = sqlite3.connect(_SEEN_DB)
        row = conn.execute("SELECT 1 FROM seen WHERE sig_id=?", (sig_id,)).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _mark_seen(sig_id: str):
    try:
        conn = sqlite3.connect(_SEEN_DB)
        conn.execute(
            "INSERT OR IGNORE INTO seen VALUES (?,?)",
            (sig_id, datetime.now().strftime("%Y-%m-%d")),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _sig_id(source: str, text: str) -> str:
    return hashlib.md5(f"{source}::{text[:120]}".encode()).hexdigest()


def mark_signal_used(signal: dict):
    """Call after a signal is converted into a saved task."""
    sig_id = signal.get("_sig_id")
    if sig_id:
        _mark_seen(sig_id)


def _is_relevant(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CODE_KEYWORDS)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_path_hints(text: str) -> list[str]:
    hints = []
    for match in PATH_HINT_RE.findall(text or ""):
        if "/" not in match:
            continue
        hints.append(match[:160])
    seen = set()
    deduped = []
    for hint in hints:
        if hint not in seen:
            seen.add(hint)
            deduped.append(hint)
    return deduped[:8]


def _score_github_issue(title: str, body: str, labels: list[str], comments: int) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    title_l = (title or "").lower()
    body_l = (body or "").lower()
    labels_l = [str(x).lower() for x in labels]
    combined = f"{title_l}\n{body_l}"

    if any(lbl in labels_l for lbl in ("bug", "regression", "good first issue", "help wanted")):
        score += 2
        reasons.append("useful labels")
    if "question" in labels_l:
        score -= 2
        reasons.append("question label")
    if any(term in combined for term in ("steps to reproduce", "repro", "expected behavior", "actual behavior")):
        score += 2
        reasons.append("reproduction detail")
    if any(term in combined for term in ("traceback", "stack trace", "exception", "error:", "fails with")):
        score += 2
        reasons.append("failure evidence")
    if any(term in combined for term in ("test", "pytest", "unit test", "integration test")):
        score += 1
        reasons.append("test signal")
    if _extract_path_hints(body):
        score += 1
        reasons.append("file path hints")
    if comments >= 2:
        score += 1
        reasons.append("discussion context")
    if len(body or "") < 120:
        score -= 1
        reasons.append("thin description")
    if any(term in combined for term in ANTI_PATTERNS):
        score -= 3
        reasons.append("non-execution issue")
    return score, reasons


def _build_issue_signal(repo: str, issue: dict, repo_meta: dict) -> dict | None:
    title = _normalize_text(issue.get("title", ""))
    body = _normalize_text((issue.get("body") or "")[:MAX_BODY_CHARS])
    text = f"{title}\n{body}".strip()
    if not _is_relevant(text):
        return None

    labels = [label.get("name", "") for label in issue.get("labels", []) if isinstance(label, dict)]
    score, reasons = _score_github_issue(title, body, labels, int(issue.get("comments", 0) or 0))
    if score < MIN_GITHUB_ISSUE_SCORE:
        return None

    issue_number = int(issue["number"])
    source = f"github:{repo}#{issue_number}"
    sig = _sig_id(source, text)
    if _is_seen(sig):
        return None

    return {
        "signal_kind": "github_issue",
        "raw_text": text,
        "title": title,
        "body": body,
        "issue_body": body,
        "source": source,
        "source_url": issue.get("html_url", ""),
        "date": issue.get("created_at", "")[:10],
        "issue_number": issue_number,
        "issue_state": issue.get("state", "open"),
        "issue_comments": int(issue.get("comments", 0) or 0),
        "issue_labels": labels,
        "issue_author": (issue.get("user") or {}).get("login", ""),
        "repo_full_name": repo,
        "repo_url": repo_meta.get("repo_url", f"https://github.com/{repo}"),
        "repo_clone_url": repo_meta.get("repo_clone_url", f"https://github.com/{repo}.git"),
        "repo_default_branch": repo_meta.get("repo_default_branch", ""),
        "repo_language": repo_meta.get("repo_language", ""),
        "repo_stars": int(repo_meta.get("repo_stars", 0) or 0),
        "path_hints": _extract_path_hints(body),
        "execution_target": "real_repo_issue",
        "candidate_score": score,
        "candidate_score_reasons": reasons,
        "_sig_id": sig,
    }


GITHUB_REPOS = [
    ("langchain-ai/langchain", ["bug", "help wanted"]),
    ("crewAIInc/crewAI", ["bug", "help wanted"]),
    ("microsoft/autogen", ["bug", "help wanted"]),
    ("openai/openai-python", ["bug", "question"]),
    ("run-llama/llama_index", ["bug", "help wanted"]),
    ("BerriAI/litellm", ["bug", "help wanted"]),
]


def fetch_github_issues(max_per_repo: int = 8) -> list[dict]:
    _init_seen_db()
    results = []
    session = _github_session()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "agent-behavior-dataset-pipeline",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    for repo, preferred_labels in GITHUB_REPOS:
        fetched = []
        repo_meta = {
            "repo_url": f"https://github.com/{repo}",
            "repo_clone_url": f"https://github.com/{repo}.git",
            "repo_default_branch": "",
            "repo_language": "",
            "repo_stars": 0,
        }
        label_attempts = [",".join(preferred_labels), ""]

        for label_str in label_attempts:
            if len(fetched) >= max_per_repo:
                break
            try:
                params = {
                    "state": "open",
                    "per_page": max_per_repo,
                    "sort": "created",
                    "direction": "desc",
                }
                if label_str:
                    params["labels"] = label_str

                resp = session.get(
                    f"https://api.github.com/repos/{repo}/issues",
                    headers=headers,
                    params=params,
                    timeout=GITHUB_TIMEOUT_SECONDS,
                )
                if resp.status_code != 200:
                    continue

                for issue in resp.json():
                    if not isinstance(issue, dict) or issue.get("pull_request"):
                        continue
                    signal = _build_issue_signal(repo, issue, repo_meta)
                    if signal:
                        fetched.append(signal)
                time.sleep(0.5)
            except Exception as e:
                print(f"  ⚠️  GitHub {repo}: {e}")

        fetched.sort(key=lambda x: (x.get("candidate_score", 0), x.get("issue_comments", 0)), reverse=True)
        results.extend(fetched[:max_per_repo])

    print(f"  📦 GitHub: {len(results)} executable issue signals")
    return results


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
                    "order": "desc",
                    "sort": "creation",
                    "tagged": tag,
                    "site": "stackoverflow",
                    "pagesize": max_per_tag,
                    "filter": "withbody",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            for q in resp.json().get("items", []):
                source = f"stackoverflow:{q['question_id']}"
                title = _normalize_text(q.get("title") or "")
                body = BeautifulSoup(q.get("body", ""), "html.parser").get_text()[:600]
                text = f"{title}\n{_normalize_text(body)}"
                if not _is_relevant(text):
                    continue
                sig = _sig_id(source, text)
                if _is_seen(sig):
                    continue
                results.append({
                    "signal_kind": "stackoverflow_question",
                    "raw_text": text,
                    "title": title,
                    "source": source,
                    "source_url": f"https://stackoverflow.com/q/{q['question_id']}",
                    "date": datetime.fromtimestamp(
                        q["creation_date"], tz=timezone.utc,
                    ).strftime("%Y-%m-%d"),
                    "_sig_id": sig,
                })
            time.sleep(1)
        except Exception as e:
            print(f"  ⚠️  SO '{tag}': {e}")
    print(f"  📦 Stack Overflow: {len(results)} new signals")
    return results


def fetch_hf_daily_papers(max_papers: int = 6) -> list[dict]:
    _init_seen_db()
    results = []
    try:
        resp = requests.get("https://huggingface.co/papers", timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        for article in soup.select("article")[:max_papers]:
            title_el = article.select_one("h3")
            summary_el = article.select_one("p")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            summary = summary_el.get_text(strip=True) if summary_el else ""
            text = f"{title}\n{summary[:400]}"
            if not _is_relevant(text):
                continue
            sig = _sig_id("hf_papers", text)
            if _is_seen(sig):
                continue
            results.append({
                "signal_kind": "hf_paper",
                "raw_text": text,
                "title": title,
                "source": "huggingface_papers",
                "source_url": "https://huggingface.co/papers",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "_sig_id": sig,
            })
    except Exception as e:
        print(f"  ⚠️  HF papers: {e}")
    print(f"  📦 HuggingFace Papers: {len(results)} new signals")
    return results


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
                summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:400]
                text = f"{entry.title}\n{summary}"
                source = f"{api_name}_changelog"
                if not _is_relevant(text):
                    continue
                sig = _sig_id(source, text)
                if _is_seen(sig):
                    continue
                results.append({
                    "signal_kind": "api_changelog",
                    "raw_text": text,
                    "title": entry.title,
                    "source": source,
                    "source_url": entry.get("link", ""),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "_sig_id": sig,
                })
        except Exception as e:
            print(f"  ⚠️  Changelog '{api_name}': {e}")
    print(f"  📦 API Changelogs: {len(results)} new signals")
    return results


def collect_all_signals() -> list[dict]:
    print("🌐 Collecting real-world signals...")
    signals = fetch_github_issues()
    if not GITHUB_ONLY_SIGNALS:
        signals += fetch_stackoverflow_questions()
        signals += fetch_hf_daily_papers()
        signals += fetch_api_changelogs()
    print(f"  ✅ Total new signals: {len(signals)}")
    return signals
