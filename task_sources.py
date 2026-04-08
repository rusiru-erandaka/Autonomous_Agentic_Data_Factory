"""
task_sources.py
Pulls real-world signals daily from GitHub, Stack Overflow,
HuggingFace Papers, and API changelogs to ground task generation
in real developer problems.
"""

import os
import time
import requests
import feedparser
from datetime import datetime, timezone
from typing import Optional
from bs4 import BeautifulSoup

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


# ── GitHub Issues ──────────────────────────────────────────────────────────────

GITHUB_REPOS = [
    ("langchain-ai/langchain",      "agent"),
    ("crewAIInc/crewAI",            "bug"),
    ("microsoft/autogen",           "enhancement"),
    ("openai/openai-python",        "question"),
    ("run-llama/llama_index",       "agent"),
]

def fetch_github_issues(max_per_repo: int = 10) -> list[dict]:
    """Fetch recent open issues from agentic AI repos."""
    results = []
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    for repo, label in GITHUB_REPOS:
        try:
            url = f"https://api.github.com/repos/{repo}/issues"
            response = requests.get(url, headers=headers, params={
                "labels":    label,
                "state":     "open",
                "per_page":  max_per_repo,
                "sort":      "created",
                "direction": "desc",
            }, timeout=15)

            if response.status_code != 200:
                print(f"  ⚠️  GitHub {repo}: status {response.status_code}")
                continue

            for issue in response.json():
                body = issue.get("body") or ""
                results.append({
                    "raw_text":  f"{issue['title']}\n{body[:800]}",
                    "source":    f"github:{repo}#{issue['number']}",
                    "date":      issue["created_at"][:10],
                })

            time.sleep(1)   # be polite to GitHub API

        except Exception as e:
            print(f"  ❌ GitHub {repo} fetch failed: {e}")

    print(f"  📦 GitHub: {len(results)} raw signals")
    return results


# ── Stack Overflow ─────────────────────────────────────────────────────────────

SO_TAGS = [
    "langchain", "openai-api", "stripe-api",
    "notion-api", "github-api", "python-requests",
    "llm-agent", "autogen", "crewai",
]

def fetch_stackoverflow_questions(max_per_tag: int = 5) -> list[dict]:
    """Fetch recent Stack Overflow questions for agentic/API tags."""
    results = []

    for tag in SO_TAGS:
        try:
            response = requests.get(
                "https://api.stackexchange.com/2.3/questions",
                params={
                    "order":    "desc",
                    "sort":     "creation",
                    "tagged":   tag,
                    "site":     "stackoverflow",
                    "pagesize": max_per_tag,
                    "filter":   "withbody",
                },
                timeout=15,
            )

            if response.status_code != 200:
                continue

            for q in response.json().get("items", []):
                body = BeautifulSoup(q.get("body", ""), "html.parser").get_text()
                results.append({
                    "raw_text": f"{q['title']}\n{body[:600]}",
                    "source":   f"stackoverflow:{q['question_id']}",
                    "date":     datetime.fromtimestamp(
                        q["creation_date"], tz=timezone.utc
                    ).strftime("%Y-%m-%d"),
                })

            time.sleep(1)

        except Exception as e:
            print(f"  ❌ SO tag '{tag}' fetch failed: {e}")

    print(f"  📦 Stack Overflow: {len(results)} raw signals")
    return results


# ── HuggingFace Daily Papers ───────────────────────────────────────────────────

def fetch_hf_daily_papers(max_papers: int = 10) -> list[dict]:
    """Scrape HuggingFace daily papers page for AI agent / tool-use papers."""
    results = []
    try:
        response = requests.get("https://huggingface.co/papers", timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        for article in soup.select("article")[:max_papers]:
            title_el   = article.select_one("h3")
            summary_el = article.select_one("p")
            if not title_el:
                continue

            title   = title_el.get_text(strip=True)
            summary = summary_el.get_text(strip=True) if summary_el else ""

            # Only keep papers relevant to agents / tool use
            keywords = ["agent", "tool", "api", "orchestrat", "code", "workflow", "reward"]
            combined = (title + summary).lower()
            if not any(k in combined for k in keywords):
                continue

            results.append({
                "raw_text": f"{title}\n{summary[:500]}",
                "source":   "huggingface_papers",
                "date":     datetime.now().strftime("%Y-%m-%d"),
            })

    except Exception as e:
        print(f"  ❌ HF papers fetch failed: {e}")

    print(f"  📦 HuggingFace Papers: {len(results)} raw signals")
    return results


# ── API Changelog RSS Feeds ────────────────────────────────────────────────────

CHANGELOG_FEEDS = {
    "stripe":  "https://stripe.com/blog/changelog.rss",
    "github":  "https://github.blog/changelog/feed/",
    "openai":  "https://openai.com/blog/rss.xml",
    "notion":  "https://www.notion.so/releases/rss.xml",
}

def fetch_api_changelogs(max_per_feed: int = 3) -> list[dict]:
    """Parse RSS changelog feeds from major API providers."""
    results = []

    for api_name, feed_url in CHANGELOG_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_feed]:
                summary = BeautifulSoup(
                    entry.get("summary", ""), "html.parser"
                ).get_text()
                results.append({
                    "raw_text": f"{entry.title}\n{summary[:500]}",
                    "source":   f"{api_name}_changelog",
                    "date":     datetime.now().strftime("%Y-%m-%d"),
                })
        except Exception as e:
            print(f"  ❌ Changelog feed '{api_name}' failed: {e}")

    print(f"  📦 API Changelogs: {len(results)} raw signals")
    return results


# ── Aggregator ─────────────────────────────────────────────────────────────────

def collect_all_signals() -> list[dict]:
    """Pull from all 4 sources and return combined raw signal list."""
    print("🌐 Collecting real-world signals...")
    signals = []
    signals += fetch_github_issues()
    signals += fetch_stackoverflow_questions()
    signals += fetch_hf_daily_papers()
    signals += fetch_api_changelogs()
    print(f"  ✅ Total raw signals collected: {len(signals)}")
    return signals
