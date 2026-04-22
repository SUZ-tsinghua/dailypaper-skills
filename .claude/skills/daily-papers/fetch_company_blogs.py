#!/usr/bin/env python3
"""
fetch_company_blogs.py — Fetch recent posts from curated robotics/AI company blogs.

Returns paper-like dicts so they can flow through the same scoring / merge /
enrichment pipeline as arXiv and HuggingFace papers. This covers releases
like Physical Intelligence's pi0.7 or DeepMind's Gemini Robotics that
are announced via blog posts rather than arXiv preprints.

Usage (as a module):
    from fetch_company_blogs import fetch_company_blogs
    posts = fetch_company_blogs(blog_configs)  # returns list[dict]

Usage (standalone, for debugging):
    python3 fetch_company_blogs.py

Paper dict shape (same as other sources, with blog-specific extras):
    {
        "id":          "blog:{md5-hash-of-url}",      # used for dedup
        "title":       str,
        "authors":     "",
        "affiliations": str,   # company name
        "abstract":    str,    # post description/summary (HTML-stripped)
        "url":         str,    # canonical post URL
        "pdf":         "",
        "date":        "YYYY-MM-DD",
        "score":       int,
        "category":    "blog",
        "source":      "company-blog",
        "company":     str,    # "Google DeepMind", "NVIDIA", etc.
        "hf_upvotes":  0,
    }
"""

from __future__ import annotations

import hashlib
import html as _html
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler

_SHARED_DIR = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from user_config import daily_papers_config  # noqa: E402

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


# ── HTTP helper with redirect support (incl. HTTP 308, not in older stdlib) ──


class _RedirectHandler308(HTTPRedirectHandler):
    """Extend urllib's default handler to follow HTTP 308 Permanent Redirect.

    Python's ``HTTPRedirectHandler`` covers 301/302/303/307 across all supported
    versions; 308 was only added in 3.11. Physical Intelligence's sitemap (hosted
    at ``pi.website``) points at the old ``physicalintelligence.company`` domain
    which 308-redirects back, so we need this for broader compat.
    """

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)


_OPENER = build_opener(_RedirectHandler308())


def _fetch(url: str, timeout: int = 15) -> str:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 daily-papers-bot/1.0"})
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] blog fetch failed {url}: {e}", file=sys.stderr)
        return ""


# ── Date parsing helpers ──────────────────────────────────────────────────


def _parse_rfc822(s: str):
    if not s:
        return None
    try:
        return parsedate_to_datetime(s).date()
    except (TypeError, ValueError):
        return None


def _parse_iso(s: str):
    if not s:
        return None
    s = s.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.split("+")[0].split("-00:00")[0], fmt).date()
        except ValueError:
            continue
    return None


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _make_id(url: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    return f"blog:{h}"


# ── Feed parsers ──────────────────────────────────────────────────────────


def _parse_rss(root, company: str, max_items: int) -> list[dict]:
    papers = []
    for item in root.findall(".//item")[:max_items]:
        title = _strip_html(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        desc = item.findtext("description") or item.findtext(
            "{http://purl.org/rss/1.0/modules/content/}encoded"
        ) or ""
        desc_clean = _strip_html(desc)[:800]
        pub_raw = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or ""
        pub_date = _parse_rfc822(pub_raw) or _parse_iso(pub_raw)

        if not title or not link:
            continue

        papers.append({
            "id": _make_id(link),
            "title": title,
            "authors": "",
            "affiliations": company,
            "abstract": desc_clean,
            "url": link,
            "pdf": "",
            "date": pub_date.isoformat() if pub_date else "",
            "_pub_date": pub_date,
            "score": 0,
            "category": "blog",
            "source": "company-blog",
            "company": company,
            "hf_upvotes": 0,
        })
    return papers


def _parse_atom(root, company: str, max_items: int) -> list[dict]:
    papers = []
    for entry in root.findall("a:entry", _ATOM_NS)[:max_items]:
        title = _strip_html(entry.findtext("a:title", namespaces=_ATOM_NS) or "")
        link = ""
        for link_el in entry.findall("a:link", _ATOM_NS):
            rel = link_el.get("rel", "alternate")
            if rel == "alternate" or not link:
                link = link_el.get("href", "")
        summary = (
            entry.findtext("a:summary", namespaces=_ATOM_NS)
            or entry.findtext("a:content", namespaces=_ATOM_NS)
            or ""
        )
        desc_clean = _strip_html(summary)[:800]
        pub_raw = (
            entry.findtext("a:published", namespaces=_ATOM_NS)
            or entry.findtext("a:updated", namespaces=_ATOM_NS)
            or ""
        )
        pub_date = _parse_iso(pub_raw)

        if not title or not link:
            continue

        papers.append({
            "id": _make_id(link),
            "title": title,
            "authors": "",
            "affiliations": company,
            "abstract": desc_clean,
            "url": link,
            "pdf": "",
            "date": pub_date.isoformat() if pub_date else "",
            "_pub_date": pub_date,
            "score": 0,
            "category": "blog",
            "source": "company-blog",
            "company": company,
            "hf_upvotes": 0,
        })
    return papers


_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def _extract_meta(html: str, prop: str) -> str:
    """Extract <meta property=prop> or <meta name=prop> content, attr-order agnostic."""
    if not html:
        return ""
    escaped = re.escape(prop)
    for pattern in (
        rf'<meta\s+(?:property|name)=["\']{escaped}["\']\s+content=["\']([^"\']*)["\']',
        rf'<meta\s+content=["\']([^"\']*)["\']\s+(?:property|name)=["\']{escaped}["\']',
    ):
        m = re.search(pattern, html, re.I)
        if m:
            return _strip_html(m.group(1))
    return ""


def _extract_title_tag(html: str) -> str:
    if not html:
        return ""
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
    return _strip_html(m.group(1)) if m else ""


def _slug_as_title(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return tail.replace("-", " ").replace("_", " ").strip().title() or url


def _parse_sitemap(
    xml_text: str,
    company: str,
    path_prefix: str,
    lookback_days: int,
    max_items: int,
) -> list[dict]:
    """Parse a sitemap.xml and return paper-dicts for URLs matching path_prefix.

    For each candidate URL within the lookback window, fetches the page HTML to
    extract title (og:title / <title>) and description (og:description / meta
    description). Falls back to a slug-derived title if the page cannot be fetched.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  [WARN] sitemap parse error for {company}: {e}", file=sys.stderr)
        return []

    cutoff = datetime.now().date() - timedelta(days=lookback_days)

    candidates: list[tuple[str, date]] = []
    for url_el in root.findall("sm:url", _SITEMAP_NS):
        loc_el = url_el.find("sm:loc", _SITEMAP_NS)
        if loc_el is None or not loc_el.text:
            continue
        url = loc_el.text.strip()

        # Require the prefix AND something after it — skips the index page itself.
        if path_prefix:
            if path_prefix not in url:
                continue
            if not url.split(path_prefix, 1)[1].strip("/"):
                continue

        lastmod_el = url_el.find("sm:lastmod", _SITEMAP_NS)
        lastmod_raw = lastmod_el.text if lastmod_el is not None else ""
        lastmod = _parse_iso(lastmod_raw) if lastmod_raw else None
        if lastmod is not None and lastmod < cutoff:
            continue

        candidates.append((url, lastmod))

    candidates.sort(key=lambda x: x[1] or date.min, reverse=True)
    candidates = candidates[:max_items]

    if not candidates:
        return []

    with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as pool:
        htmls = list(pool.map(lambda u: _fetch(u, timeout=12), (c[0] for c in candidates)))

    papers = []
    for (url, lastmod), html in zip(candidates, htmls):
        title = (
            _extract_meta(html, "og:title")
            or _extract_title_tag(html)
            or _slug_as_title(url)
        )
        desc = (
            _extract_meta(html, "og:description")
            or _extract_meta(html, "description")
            or ""
        )
        papers.append({
            "id": _make_id(url),
            "title": title,
            "authors": "",
            "affiliations": company,
            "abstract": desc[:800],
            "url": url,
            "pdf": "",
            "date": lastmod.isoformat() if lastmod else "",
            "_pub_date": lastmod,
            "score": 0,
            "category": "blog",
            "source": "company-blog",
            "company": company,
            "hf_upvotes": 0,
        })
    return papers


def _auto_parse(xml_text: str, company: str, max_items: int) -> list[dict]:
    """Auto-detect RSS vs Atom from root element."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  [WARN] feed parse error for {company}: {e}", file=sys.stderr)
        return []
    if root.tag.lower().endswith("feed"):
        return _parse_atom(root, company, max_items)
    return _parse_rss(root, company, max_items)


# ── Main entry ────────────────────────────────────────────────────────────


def fetch_company_blogs(blog_configs: list[dict] | None = None) -> list[dict]:
    """Fetch and return recent posts from all configured blogs.

    Each config entry:
        {
            "name": str,             # human label for logs
            "url": str,              # feed URL (RSS, Atom, or sitemap.xml)
            "company": str,          # shown in review output
            "max_items": int,        # per-feed cap (default 20)
            "lookback_days": int,    # drop posts older than this (default 21)
            "type": str,             # "rss"/"atom"/"auto" (default) or "sitemap"
            "path_prefix": str,      # sitemap-only: only URLs containing this substring
                                     # (e.g. "/blog/" or "/news/") are treated as posts.
        }

    Returns: list of paper-like dicts (see module docstring).
    """
    if blog_configs is None:
        blog_configs = daily_papers_config().get("company_blogs", []) or []

    configs = [c for c in blog_configs if c.get("url")]
    if not configs:
        return []

    today = datetime.now().date()

    def _fetch_one(cfg: dict) -> tuple[dict, str]:
        return cfg, _fetch(cfg["url"])

    with ThreadPoolExecutor(max_workers=min(8, len(configs))) as pool:
        fetched = list(pool.map(_fetch_one, configs))

    results: list[dict] = []
    for cfg, raw in fetched:
        company = cfg.get("company") or cfg.get("name") or "Unknown"
        max_items = int(cfg.get("max_items", 20))
        lookback = int(cfg.get("lookback_days", 21))
        feed_type = (cfg.get("type") or "auto").lower()
        path_prefix = cfg.get("path_prefix", "")

        if not raw:
            continue

        if feed_type == "sitemap":
            posts = _parse_sitemap(raw, company, path_prefix, lookback, max_items)
        else:
            posts = _auto_parse(raw, company, max_items)

        cutoff = today - timedelta(days=lookback)
        kept = []
        for p in posts:
            pd = p.pop("_pub_date", None)
            if pd is None or pd >= cutoff:
                kept.append(p)

        print(f"    {company}: {len(kept)} posts within {lookback}d (total parsed: {len(posts)})",
              file=sys.stderr)
        results.extend(kept)

    print(f"  Blogs: {len(results)} total posts", file=sys.stderr)
    return results


# ── Standalone debug ──────────────────────────────────────────────────────


if __name__ == "__main__":
    import json
    posts = fetch_company_blogs()
    print(f"\n=== {len(posts)} blog posts ===", file=sys.stderr)
    for p in posts[:5]:
        print(f"  [{p['company']}] {p['date']}: {p['title'][:80]}", file=sys.stderr)
    json.dump(posts, sys.stdout, ensure_ascii=False, indent=2)
