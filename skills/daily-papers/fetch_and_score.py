#!/usr/bin/env python3
"""
fetch_and_score.py — Phase 1+2: Fetch, score, merge, dedup, select top 30.

Replaces the two LLM Task Agents with pure Python. Zero token cost.

Usage:
    python3 fetch_and_score.py > /tmp/daily_papers_top30.json
    python3 fetch_and_score.py --date 2026-02-25 > /tmp/daily_papers_top30.json
    python3 fetch_and_score.py --days 7 > /tmp/daily_papers_top30.json

Stderr: progress logs.  Stdout: JSON array of top papers (30 * days).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_SHARED_DIR = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from user_config import daily_papers_config, daily_papers_dir

# ── Configuration ──────────────────────────────────────────────────────────

_CONFIG = daily_papers_config()

KEYWORDS = _CONFIG["keywords"]
NEGATIVE_KEYWORDS = _CONFIG["negative_keywords"]
DOMAIN_BOOST_KEYWORDS = _CONFIG["domain_boost_keywords"]
ARXIV_CATEGORIES = _CONFIG["arxiv_categories"]
MIN_SCORE = _CONFIG["min_score"]
TOP_N = _CONFIG["top_n"]
COMPANY_BLOGS = _CONFIG.get("company_blogs", []) or []
# Which fetchers to enable. Supported: "arxiv", "hf-daily", "hf-trending", "company-blogs"
SOURCES = set(_CONFIG.get("sources", ["arxiv", "hf-daily", "hf-trending", "company-blogs"]))

DAILYPAPERS_DIR = daily_papers_dir()
HISTORY_PATH = DAILYPAPERS_DIR / ".history.json"

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# ── Scoring ────────────────────────────────────────────────────────────────


def score_paper(paper: dict, is_trending: bool = False) -> int:
    text = (paper["title"] + " " + paper["abstract"]).lower()
    title_lower = paper["title"].lower()

    # 1. Negative keywords → instant reject
    for neg in NEGATIVE_KEYWORDS:
        if neg in text:
            return -999

    score = 0

    # 2. Positive keywords
    keyword_hits = 0
    for kw in KEYWORDS:
        if kw in title_lower:
            score += 3
            keyword_hits += 1
        elif kw in text:
            score += 1
            keyword_hits += 1

    # 3. Domain boost
    domain_hits = sum(1 for kw in DOMAIN_BOOST_KEYWORDS if kw in text)
    if domain_hits >= 2:
        score += 2
    elif domain_hits == 1:
        score += 1

    # 4. Trending boost (HF sources only)
    #    GATE: only apply if paper has at least 1 keyword or domain match,
    #    to prevent irrelevant but popular papers from flooding the list
    has_relevance = keyword_hits > 0 or domain_hits > 0
    if is_trending:
        upvotes = paper.get("hf_upvotes", 0) or 0
        if has_relevance:
            # Relevant + trending → full boost
            if upvotes >= 10:
                score += 3
            elif upvotes >= 5:
                score += 2
            elif upvotes >= 2:
                score += 1
        else:
            # No relevance → minimal boost (only very popular papers get a chance)
            if upvotes >= 20:
                score += 1

    # Company blogs are editorially curated (not a firehose), so +3 matches
    # the top trending tier — but gated on relevance to skip funding/partnership PR.
    if paper.get("source") == "company-blog" and has_relevance:
        score += 3

    return score


# ── Fetchers ───────────────────────────────────────────────────────────────


def fetch_url(url: str, timeout: int = 30) -> str:
    try:
        req = Request(url, headers={"User-Agent": "daily-papers-bot/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        print(f"  [WARN] fetch failed {url}: {e}", file=sys.stderr)
        return ""


def _parse_hf_item(item: dict, source: str):
    """Parse a single HF API item into (arxiv_id, paper_dict). Returns None on skip."""
    p = item.get("paper", {})
    arxiv_id = p.get("id", "")
    if not arxiv_id:
        return None

    upvotes = p.get("upvotes", 0)

    # Authors
    authors_raw = p.get("authors", [])
    if isinstance(authors_raw, list):
        names = []
        for a in authors_raw:
            if isinstance(a, dict):
                names.append(a.get("name", ""))
            elif isinstance(a, str):
                names.append(a)
        authors = ", ".join(n for n in names if n)
    else:
        authors = str(authors_raw)

    paper = {
        "title": p.get("title", ""),
        "authors": authors,
        "affiliations": "",
        "abstract": p.get("summary", ""),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
        "date": (p.get("publishedAt") or "")[:10],
        "score": 0,
        "category": "",
        "source": source,
        "hf_upvotes": upvotes,
    }

    is_trending = source == "hf-trending"
    paper["score"] = score_paper(paper, is_trending=is_trending)

    if paper["score"] < 0:
        return None

    return arxiv_id, paper


def _ingest_hf_endpoint(endpoint: str, source: str, label: str, papers: dict) -> None:
    """Fetch an HF daily_papers endpoint and merge parsed items into papers (keep higher score)."""
    print(f"  Fetching {label}...", file=sys.stderr)
    raw = fetch_url(endpoint)
    if not raw:
        return
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] bad JSON from {label}", file=sys.stderr)
        return
    for item in items:
        result = _parse_hf_item(item, source)
        if not result:
            continue
        arxiv_id, paper = result
        if arxiv_id not in papers or paper["score"] > papers[arxiv_id]["score"]:
            papers[arxiv_id] = paper


def fetch_hf_papers(start_date=None, end_date=None) -> list[dict]:
    papers: dict = {}  # arxiv_id → paper

    hf_daily_enabled = "hf-daily" in SOURCES
    hf_trending_enabled = "hf-trending" in SOURCES

    if not hf_daily_enabled and not hf_trending_enabled:
        print("  HF sources disabled via config (skipping hf-daily & hf-trending)", file=sys.stderr)
        return []

    if hf_daily_enabled:
        if start_date and end_date:
            d = start_date
            while d <= end_date:
                date_str = d.isoformat()
                _ingest_hf_endpoint(
                    f"https://huggingface.co/api/daily_papers?date={date_str}&limit=100",
                    "hf-daily", f"hf-daily {date_str}", papers,
                )
                d += timedelta(days=1)
        else:
            _ingest_hf_endpoint(
                "https://huggingface.co/api/daily_papers?limit=50",
                "hf-daily", "hf-daily", papers,
            )
    else:
        print("  hf-daily disabled via config", file=sys.stderr)

    if hf_trending_enabled:
        _ingest_hf_endpoint(
            "https://huggingface.co/api/daily_papers?sort=trending&limit=50",
            "hf-trending", "hf-trending", papers,
        )
    else:
        print("  hf-trending disabled via config", file=sys.stderr)

    result = list(papers.values())
    print(f"  HF: {len(result)} papers after scoring", file=sys.stderr)
    return result


def _newest_published_date(root):
    """Return the newest <published> date across entries, for staleness detection."""
    newest = None
    for entry in root.findall("atom:entry", ATOM_NS):
        pub_el = entry.find("atom:published", ATOM_NS)
        if pub_el is None or not pub_el.text:
            continue
        try:
            d = datetime.strptime(pub_el.text[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if newest is None or d > newest:
            newest = d
    return newest


def fetch_arxiv_papers(start_date=None, end_date=None, days: int = 1) -> list[dict]:
    import time as _time

    max_results = min(2000 * days, 3000)
    cats = "+OR+".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    base_url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query=({cats})"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )

    timeout = max(60, 30 * days)
    today_ref = end_date or datetime.now().date()
    # arXiv mirrors occasionally serve a stale snapshot where the newest
    # returned paper is >=2 days old — this misses the most recent announcement
    # batch entirely. Retry with a cache-buster when this happens.
    STALENESS_THRESHOLD_DAYS = 2
    MAX_ATTEMPTS = 3

    root = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        url = base_url if attempt == 1 else f"{base_url}&_cb={int(_time.time())}_{attempt}"
        print(
            f"  Fetching arXiv (attempt {attempt}/{MAX_ATTEMPTS}, "
            f"max_results={max_results}, timeout={timeout}s)...",
            file=sys.stderr,
        )
        xml_text = fetch_url(url, timeout=timeout)
        if not xml_text:
            if attempt < MAX_ATTEMPTS:
                _time.sleep(5)
                continue
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"  [WARN] arXiv XML parse error: {e}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                _time.sleep(5)
                continue
            return []

        newest = _newest_published_date(root)
        if newest is None:
            break  # nothing to validate; proceed with what we have
        lag_days = (today_ref - newest).days
        if lag_days >= STALENESS_THRESHOLD_DAYS and attempt < MAX_ATTEMPTS:
            print(
                f"  [WARN] arXiv result looks stale "
                f"(newest_published={newest}, target={today_ref}, lag={lag_days}d); "
                f"retrying with cache-buster",
                file=sys.stderr,
            )
            _time.sleep(5)
            continue
        break

    if root is None:
        return []

    papers = []
    filtered_by_date = 0
    for entry in root.findall("atom:entry", ATOM_NS):
        title_el = entry.find("atom:title", ATOM_NS)
        summary_el = entry.find("atom:summary", ATOM_NS)
        published_el = entry.find("atom:published", ATOM_NS)
        id_el = entry.find("atom:id", ATOM_NS)

        if title_el is None or summary_el is None:
            continue

        title = " ".join(title_el.text.split())
        abstract = " ".join(summary_el.text.split())
        entry_url = id_el.text.strip() if id_el is not None else ""
        date = published_el.text[:10] if published_el is not None else ""
        arxiv_id = entry_url.split("/abs/")[-1] if "/abs/" in entry_url else ""

        # Date filter: only apply in multi-day mode (days > 1)
        # In single-day mode, arXiv batches span 2-3 days, so filtering would be too strict
        if days > 1 and start_date and end_date and date:
            try:
                pub_date = datetime.strptime(date, "%Y-%m-%d").date()
                if pub_date < start_date or pub_date > end_date:
                    filtered_by_date += 1
                    continue
            except ValueError:
                pass  # keep papers with unparseable dates

        author_els = entry.findall("atom:author", ATOM_NS)
        names = []
        affiliations = set()
        for a in author_els:
            name_el = a.find("atom:name", ATOM_NS)
            if name_el is not None and name_el.text:
                names.append(name_el.text.strip())
            for aff_el in a.findall("arxiv:affiliation", ATOM_NS):
                if aff_el.text and aff_el.text.strip():
                    affiliations.add(aff_el.text.strip())

        cat_el = entry.find("arxiv:primary_category", ATOM_NS)
        category = cat_el.get("term", "") if cat_el is not None else ""

        papers.append({
            "title": title,
            "authors": ", ".join(names),
            "affiliations": ", ".join(sorted(affiliations)) if affiliations else "",
            "abstract": abstract,
            "url": entry_url,
            "pdf": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
            "date": date,
            "score": 0,
            "category": category,
            "source": "arxiv",
        })

    scored = []
    for p in papers:
        p["score"] = score_paper(p)
        if p["score"] >= 0:
            scored.append(p)

    print(
        f"  arXiv: {len(scored)} papers after scoring (from {len(papers)} parsed, {filtered_by_date} filtered by date)",
        file=sys.stderr,
    )
    return scored


# ── Merge & Dedup ──────────────────────────────────────────────────────────


def extract_arxiv_id(url: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})", url)
    return m.group(1) if m else ""


def paper_id(p: dict) -> str:
    """Stable dedup ID for any paper source.

    Prefers explicit ``id`` field (used by company-blog posts like ``blog:abc123``),
    otherwise falls back to arXiv ID extracted from the URL.
    Returns "" if nothing usable is found.
    """
    if p.get("id"):
        return p["id"]
    return extract_arxiv_id(p.get("url", ""))


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return []


def load_fallback_ids(days: int = 7) -> set[str]:
    ids: set[str] = set()
    today = datetime.now().date()
    for d in range(1, days + 1):
        fpath = DAILYPAPERS_DIR / f"{(today - timedelta(days=d)).isoformat()}-论文推荐.md"
        if fpath.exists():
            try:
                text = fpath.read_text()
                for m in re.finditer(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", text):
                    ids.add(m.group(1))
            except IOError:
                pass
    return ids


def merge_and_dedup(
    hf_papers: list[dict],
    arxiv_papers: list[dict],
    target_date,
    days: int = 1,
    top_n: int = TOP_N,
    blog_papers: list[dict] | None = None,
) -> list[dict]:
    is_weekend = target_date.weekday() >= 5
    blog_papers = blog_papers or []

    # ── merge by stable ID (arXiv ID or blog synthetic ID), keep higher score ──
    by_id: dict[str, dict] = {}
    for p in hf_papers + arxiv_papers + blog_papers:
        aid = paper_id(p)
        if not aid:
            continue
        if aid not in by_id or p["score"] > by_id[aid]["score"]:
            by_id[aid] = p

    print(f"  Merged: {len(by_id)} unique papers", file=sys.stderr)

    if days > 1:
        # ── multi-day mode: skip history dedup ──
        # User explicitly wants to see all N days, don't filter out previously recommended
        print(f"  Multi-day mode (days={days}): skipping history dedup", file=sys.stderr)
        candidates = [p for p in by_id.values() if p["score"] >= MIN_SCORE]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:top_n]
        print(f"  Final: {len(top)} papers (top_n={top_n})", file=sys.stderr)
        return top

    # ── single-day mode: history dedup as before ──
    history = load_history()
    history_ids: dict[str, str] = {}  # id → earliest date
    for h in history:
        hid, hdate = h.get("id", ""), h.get("date", "")
        if hid and hdate:
            if hid not in history_ids or hdate < history_ids[hid]:
                history_ids[hid] = hdate

    if len(history) < 10:
        for fid in load_fallback_ids():
            history_ids.setdefault(fid, "unknown")

    # ── cross-day dedup ──
    deduped: dict[str, dict] = {}
    removed = 0
    for aid, p in by_id.items():
        if aid in history_ids:
            # Weekend: keep trending with upvotes >= 5
            if is_weekend and p.get("source") == "hf-trending" and (p.get("hf_upvotes") or 0) >= 5:
                p["is_re_recommend"] = True
                p["last_recommend_date"] = history_ids[aid]
                deduped[aid] = p
            else:
                removed += 1
        else:
            deduped[aid] = p

    # Mark any remaining that appear in history
    for aid, p in deduped.items():
        if aid in history_ids and not p.get("is_re_recommend"):
            p["is_re_recommend"] = True
            p["last_recommend_date"] = history_ids[aid]

    print(f"  After history dedup: {len(deduped)} (removed {removed})", file=sys.stderr)

    # ── filter + sort ──
    candidates = [p for p in deduped.values() if p["score"] >= MIN_SCORE]
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Back-fill from history if pool is thin
    if len(candidates) < 20 and removed > 0:
        backfill = []
        for aid, p in by_id.items():
            if aid not in deduped and p["score"] >= MIN_SCORE:
                p["is_re_recommend"] = True
                p["last_recommend_date"] = history_ids.get(aid, "unknown")
                backfill.append(p)
        backfill.sort(key=lambda x: x["score"], reverse=True)
        needed = 20 - len(candidates)
        candidates.extend(backfill[:needed])
        if backfill[:needed]:
            print(f"  Back-filled {min(needed, len(backfill))} from history", file=sys.stderr)

    top = candidates[:top_n]
    print(f"  Final: {len(top)} papers", file=sys.stderr)
    return top


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch (default: 1)")
    args = parser.parse_args()

    target_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else datetime.now().date()
    )
    days = max(1, args.days)
    start_date = target_date - timedelta(days=days - 1)
    top_n = TOP_N * days

    is_weekend = target_date.weekday() >= 5
    print(
        f"[fetch_and_score] {target_date} ({'weekend' if is_weekend else 'weekday'})"
        + (f", days={days} [{start_date} ~ {target_date}], top_n={top_n}" if days > 1 else ""),
        file=sys.stderr,
    )

    print(f"  Enabled sources: {sorted(SOURCES)}", file=sys.stderr)

    hf_papers = fetch_hf_papers(start_date, target_date)

    if "arxiv" in SOURCES:
        arxiv_papers = fetch_arxiv_papers(start_date, target_date, days)
    else:
        print("  arxiv disabled via config", file=sys.stderr)
        arxiv_papers = []

    # Company blog posts (Physical Intelligence, NVIDIA, DeepMind, etc.)
    blog_papers: list[dict] = []
    if "company-blogs" in SOURCES and COMPANY_BLOGS:
        try:
            from fetch_company_blogs import fetch_company_blogs
            raw_blogs = fetch_company_blogs(COMPANY_BLOGS)
            for p in raw_blogs:
                p["score"] = score_paper(p)
                if p["score"] >= 0:
                    blog_papers.append(p)
            print(f"  Blogs: {len(blog_papers)} posts after scoring", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] blog fetch failed: {e}", file=sys.stderr)
    elif "company-blogs" not in SOURCES:
        print("  company-blogs disabled via config", file=sys.stderr)

    top = merge_and_dedup(
        hf_papers, arxiv_papers, target_date,
        days=days, top_n=top_n, blog_papers=blog_papers,
    )

    # Output to stdout (UTF-8 encoded for Windows compatibility)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    json.dump(top, sys.stdout, ensure_ascii=False, indent=2)
    print(file=sys.stdout)  # trailing newline


if __name__ == "__main__":
    main()
