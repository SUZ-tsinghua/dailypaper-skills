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
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
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

OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arx": "http://arxiv.org/OAI/arXiv/",
}

OAI_ENDPOINT = "https://oaipmh.arxiv.org/oai"

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

    # Trending boost is gated on keyword/domain relevance — otherwise
    # a popular but off-topic paper would flood the top of the list.
    has_relevance = keyword_hits > 0 or domain_hits > 0
    if is_trending:
        upvotes = paper.get("hf_upvotes", 0) or 0
        if has_relevance:
            if upvotes >= 10:
                score += 3
            elif upvotes >= 5:
                score += 2
            elif upvotes >= 2:
                score += 1
        elif upvotes >= 20:
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


def _oai_fetch_once(url: str, timeout: int = 120) -> str:
    """Fetch an OAI-PMH URL with one retry on 503/transient failure."""
    for attempt in range(2):
        try:
            req = Request(url, headers={"User-Agent": "daily-papers-bot/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            retry_after = 5
            if hasattr(e, "headers") and e.headers and e.headers.get("Retry-After"):
                try:
                    retry_after = int(e.headers.get("Retry-After"))
                except ValueError:
                    pass
            if attempt == 0:
                print(f"  [WARN] OAI fetch {url}: {e} — retrying in {retry_after}s", file=sys.stderr)
                time.sleep(retry_after)
            else:
                print(f"  [ERROR] OAI fetch {url}: {e}", file=sys.stderr)
                return ""
    return ""


def _parse_oai_record(record, start_date, end_date) -> dict | None:
    """Parse one OAI-PMH <record> into a paper dict, applying date+category filters.

    Returns None if deleted, malformed, outside the date range, or not in
    ARXIV_CATEGORIES. Both new submissions and replacements/revisions are kept
    — all records announced on the target date(s). New vs replacement is tagged
    via `is_replacement`, using the arXiv ID's YYMM prefix (original submission
    month) — OAI's `<created>` tracks the latest version, not the first.
    """
    header = record.find("oai:header", OAI_NS)
    if header is None or header.get("status") == "deleted":
        return None

    datestamp_el = header.find("oai:datestamp", OAI_NS)
    meta = record.find(".//arx:arXiv", OAI_NS)
    if meta is None or datestamp_el is None:
        return None

    arxiv_id_el = meta.find("arx:id", OAI_NS)
    created_el = meta.find("arx:created", OAI_NS)
    title_el = meta.find("arx:title", OAI_NS)
    abstract_el = meta.find("arx:abstract", OAI_NS)
    categories_el = meta.find("arx:categories", OAI_NS)
    if None in (arxiv_id_el, created_el, title_el, abstract_el, categories_el):
        return None

    datestamp = (datestamp_el.text or "").strip()
    created = (created_el.text or "").strip()
    try:
        d_date = datetime.strptime(datestamp, "%Y-%m-%d").date()
    except ValueError:
        return None
    if d_date < start_date or d_date > end_date:
        return None

    paper_cats = (categories_el.text or "").split()
    if not any(c in ARXIV_CATEGORIES for c in paper_cats):
        return None
    primary_cat = paper_cats[0] if paper_cats else ""

    arxiv_id = (arxiv_id_el.text or "").strip()
    is_replacement = False
    id_match = re.match(r"(\d{2})(\d{2})\.\d+", arxiv_id)
    if id_match:
        id_ym = (2000 + int(id_match.group(1))) * 12 + int(id_match.group(2))
        ann_ym = d_date.year * 12 + d_date.month
        is_replacement = (ann_ym - id_ym) >= 2

    authors_list = []
    affiliations = set()
    for a in meta.findall("arx:authors/arx:author", OAI_NS):
        keyname_el = a.find("arx:keyname", OAI_NS)
        forenames_el = a.find("arx:forenames", OAI_NS)
        parts = []
        if forenames_el is not None and forenames_el.text:
            parts.append(forenames_el.text.strip())
        if keyname_el is not None and keyname_el.text:
            parts.append(keyname_el.text.strip())
        if parts:
            authors_list.append(" ".join(parts))
        for aff_el in a.findall("arx:affiliation", OAI_NS):
            if aff_el.text and aff_el.text.strip():
                affiliations.add(aff_el.text.strip())

    title = " ".join((title_el.text or "").split())
    abstract = " ".join((abstract_el.text or "").split())

    return {
        "title": title,
        "authors": ", ".join(authors_list),
        "affiliations": ", ".join(sorted(affiliations)) if affiliations else "",
        "abstract": abstract,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf": f"https://arxiv.org/pdf/{arxiv_id}",
        "date": created,
        "announcement_date": datestamp,
        "is_replacement": is_replacement,
        "score": 0,
        "category": primary_cat,
        "source": "arxiv",
    }


def fetch_arxiv_papers(start_date, end_date) -> list[dict]:
    """Fetch papers *announced* between start_date and end_date via OAI-PMH.

    Uses the announcement datestamp (matching arxiv.org/list/cs.XX/recent),
    not submittedDate — so Sat/Sun submissions correctly surface on Monday.
    Returns both new submissions and replacements announced in the window;
    each record is tagged with `is_replacement` for downstream use.
    """
    from_str = start_date.isoformat()
    until_str = end_date.isoformat()
    print(
        f"  Fetching arXiv OAI-PMH (announcement {from_str} ~ {until_str}, set=cs)...",
        file=sys.stderr,
    )

    papers: list[dict] = []
    resumption: str | None = None
    total_records = 0
    for request_num in range(1, 21):  # safety cap: ~26k records
        if resumption:
            url = f"{OAI_ENDPOINT}?verb=ListRecords&resumptionToken={resumption}"
        else:
            url = (
                f"{OAI_ENDPOINT}?verb=ListRecords"
                f"&from={from_str}&until={until_str}"
                f"&set=cs&metadataPrefix=arXiv"
            )

        xml_text = _oai_fetch_once(url)
        if not xml_text:
            break

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"  [WARN] OAI XML parse error (req #{request_num}): {e}", file=sys.stderr)
            break

        err = root.find("oai:error", OAI_NS)
        if err is not None:
            code = err.get("code", "")
            if code == "noRecordsMatch":
                print(f"  arXiv OAI: no records in range", file=sys.stderr)
            else:
                print(f"  [WARN] OAI error: {code} — {err.text}", file=sys.stderr)
            break

        batch = root.findall(".//oai:record", OAI_NS)
        total_records += len(batch)
        for r in batch:
            p = _parse_oai_record(r, start_date, end_date)
            if p is not None:
                papers.append(p)

        rt_el = root.find(".//oai:resumptionToken", OAI_NS)
        if rt_el is not None and rt_el.text and rt_el.text.strip():
            resumption = rt_el.text.strip()
            time.sleep(3)  # arXiv OAI requests a 3s pause between calls
        else:
            break
    else:
        print(f"  [WARN] OAI pagination hit 20-request safety cap", file=sys.stderr)

    scored = []
    for p in papers:
        p["score"] = score_paper(p)
        if p["score"] >= 0:
            scored.append(p)

    print(
        f"  arXiv OAI: {len(scored)} scored / {len(papers)} in-category "
        f"/ {total_records} total records",
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
        arxiv_papers = fetch_arxiv_papers(start_date, target_date)
    else:
        print("  arxiv disabled via config", file=sys.stderr)
        arxiv_papers = []

    blog_papers: list[dict] = []
    if "company-blogs" not in SOURCES:
        print("  company-blogs disabled via config", file=sys.stderr)
    elif not COMPANY_BLOGS:
        print("  company-blogs enabled but no blog URLs configured", file=sys.stderr)
    else:
        try:
            from fetch_company_blogs import fetch_company_blogs
            raw_blogs = fetch_company_blogs(COMPANY_BLOGS)
            for p in raw_blogs:
                p["score"] = score_paper(p)
                if p["score"] >= 0:
                    blog_papers.append(p)
            print(f"  Blogs: {len(blog_papers)} posts after scoring", file=sys.stderr)
        except Exception as e:
            print(f"  [WARN] blog fetch failed: {e}", file=sys.stderr)

    top = merge_and_dedup(
        hf_papers, arxiv_papers, target_date,
        days=days, top_n=top_n, blog_papers=blog_papers,
    )

    # Output to stdout (UTF-8 encoded for Windows compatibility)
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    json.dump(top, sys.stdout, ensure_ascii=False, indent=2)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()
