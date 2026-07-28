from __future__ import annotations

import html
import re
import threading
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from typing import Any


USER_AGENT = "LitTraceQA-PDF-VLM-Symbolic-Baseline/1.0"
_INDEX_CACHE: dict[tuple[str, int], dict[str, str]] = {}
_AUTHOR_INDEX_CACHE: dict[tuple[str, int], list[tuple[set[str], str]]] = {}
_INDEX_LOCK = threading.Lock()
_ICML_PMLR_VOLUMES = {2025: "v267"}
_STOPWORDS = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or", "the", "to", "via", "with"}
_GREEK_LATEX = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "varepsilon": "epsilon",
    "theta": "theta",
    "lambda": "lambda",
    "mu": "mu",
    "pi": "pi",
    "sigma": "sigma",
    "omega": "omega",
}
_GREEK_UNICODE = {"α": " alpha ", "β": " beta ", "γ": " gamma ", "δ": " delta ", "ε": " epsilon ", "θ": " theta ", "λ": " lambda ", "μ": " mu ", "π": " pi ", "σ": " sigma ", "ω": " omega "}


def normalize_title(title: str | None) -> str:
    text = html.unescape(str(title or "")).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\[\"'`^~=.]\{?([a-z])\}?", r"\1", text)
    text = re.sub(r"\\([a-zA-Z]+)", lambda match: f" {_GREEK_LATEX.get(match.group(1).lower(), match.group(1).lower())} ", text)
    for char, replacement in _GREEK_UNICODE.items():
        text = text.replace(char, replacement)
    text = unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(title_key: str) -> set[str]:
    return {token for token in title_key.split() if token not in _STOPWORDS and len(token) > 1}


def _author_tokens(authors: Any) -> set[str]:
    if isinstance(authors, list):
        text = " ".join(str(author) for author in authors)
    else:
        text = str(authors or "")
    return {token for token in normalize_title(text).split() if len(token) > 2 and token not in _STOPWORDS}


def _close_title_lookup(title_key: str, index: dict[str, str]) -> str | None:
    tokens = _content_tokens(title_key)
    if len(tokens) < 4:
        return None
    candidates: list[tuple[float, str, str]] = []
    for indexed_key, pdf_url in index.items():
        ratio = SequenceMatcher(None, title_key, indexed_key).ratio()
        indexed_tokens = _content_tokens(indexed_key)
        if not indexed_tokens:
            continue
        overlap = len(tokens & indexed_tokens)
        smaller = min(len(tokens), len(indexed_tokens))
        union = len(tokens | indexed_tokens)
        containment = overlap / smaller if smaller else 0
        jaccard = overlap / union if union else 0
        score = max(ratio, (0.65 * containment) + (0.35 * jaccard))
        if ratio >= 0.92 or (overlap >= 5 and containment >= 0.85 and jaccard >= 0.55):
            candidates.append((score, indexed_key, pdf_url))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.03:
        return None
    return candidates[0][2]


def _author_lookup(author_tokens: set[str], author_index: list[tuple[set[str], str]]) -> str | None:
    if len(author_tokens) < 3:
        return None
    candidates: list[tuple[float, str]] = []
    for indexed_tokens, pdf_url in author_index:
        if len(indexed_tokens) < 3:
            continue
        overlap = len(author_tokens & indexed_tokens)
        smaller = min(len(author_tokens), len(indexed_tokens))
        containment = overlap / smaller if smaller else 0
        if overlap >= 3 and containment >= 0.8:
            candidates.append((containment + (overlap / 100), pdf_url))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.02:
        return None
    return candidates[0][1]


def _fetch_text(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", html.unescape(text or ""))


def _parse_hash_proceedings_index(text: str, base_url: str) -> tuple[dict[str, str], list[tuple[set[str], str]]]:
    index: dict[str, str] = {}
    author_index: list[tuple[set[str], str]] = []
    for block_match in re.finditer(r"<li\b[^>]*>(?P<block>.*?)</li>", text, flags=re.IGNORECASE | re.DOTALL):
        block = block_match.group("block")
        title_match = re.search(
            r"<a\b[^>]*href=[\"'](?P<href>[^\"']*/paper_files/paper/\d+/hash/[^\"']+-Abstract-[^\"']+\.html)[\"'][^>]*>(?P<title>.*?)</a>",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not title_match:
            continue
        title_key = normalize_title(_strip_tags(title_match.group("title")))
        if not title_key:
            continue
        href = html.unescape(title_match.group("href"))
        pdf_href = href.replace("/hash/", "/file/").replace("-Abstract-", "-Paper-")
        pdf_href = re.sub(r"\.html(?:#.*)?$", ".pdf", pdf_href)
        pdf_url = urllib.parse.urljoin(base_url, pdf_href)
        index.setdefault(title_key, pdf_url)
        authors_match = re.search(r"<span class=\"paper-authors\">(?P<authors>.*?)</span>", block, flags=re.IGNORECASE | re.DOTALL)
        if authors_match:
            author_index.append((_author_tokens(_strip_tags(authors_match.group("authors"))), pdf_url))
    return index, author_index


def _parse_pmlr_index(text: str) -> tuple[dict[str, str], list[tuple[set[str], str]]]:
    index: dict[str, str] = {}
    author_index: list[tuple[set[str], str]] = []
    for block_match in re.finditer(r"<div class=\"paper\">(?P<block>.*?)</div>", text, flags=re.IGNORECASE | re.DOTALL):
        block = block_match.group("block")
        title_match = re.search(r"<p class=\"title\">(?P<title>.*?)</p>", block, flags=re.IGNORECASE | re.DOTALL)
        pdf_match = re.search(r"<a\b[^>]*href=[\"'](?P<pdf>[^\"']+\.pdf)[\"'][^>]*>\s*Download PDF\s*</a>", block, flags=re.IGNORECASE | re.DOTALL)
        if not title_match or not pdf_match:
            continue
        title_key = normalize_title(_strip_tags(title_match.group("title")))
        if title_key:
            pdf_url = html.unescape(pdf_match.group("pdf"))
            index.setdefault(title_key, pdf_url)
            authors_match = re.search(r"<span class=\"authors\">(?P<authors>.*?)</span>", block, flags=re.IGNORECASE | re.DOTALL)
            if authors_match:
                author_index.append((_author_tokens(_strip_tags(authors_match.group("authors"))), pdf_url))
    return index, author_index


def _load_index(venue: str, year: int, timeout: int = 30) -> dict[str, str]:
    venue_key = venue.lower()
    cache_key = (venue_key, year)
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if venue_key == "iclr":
            index, author_index = _parse_hash_proceedings_index(_fetch_text(f"https://proceedings.iclr.cc/paper_files/paper/{year}", timeout=timeout), "https://proceedings.iclr.cc")
        elif venue_key == "neurips":
            index, author_index = _parse_hash_proceedings_index(_fetch_text(f"https://papers.nips.cc/paper_files/paper/{year}", timeout=timeout), "https://papers.nips.cc")
        elif venue_key == "icml" and year in _ICML_PMLR_VOLUMES:
            index, author_index = _parse_pmlr_index(_fetch_text(f"https://proceedings.mlr.press/{_ICML_PMLR_VOLUMES[year]}/", timeout=timeout))
        else:
            index, author_index = {}, []
        _INDEX_CACHE[cache_key] = index
        _AUTHOR_INDEX_CACHE[cache_key] = author_index
        return index


def proceedings_pdf_source(paper: dict[str, Any], timeout: int = 30) -> dict[str, str] | None:
    title_key = normalize_title(paper.get("title"))
    if not title_key:
        return None
    venue = str(paper.get("venue") or "").strip()
    try:
        year = int(paper.get("year") or 0)
    except (TypeError, ValueError):
        return None
    if venue not in {"ICLR", "ICML", "NeurIPS"}:
        return None
    cache_key = (venue.lower(), year)
    index = _load_index(venue, year, timeout=timeout)
    pdf_url = index.get(title_key) or _close_title_lookup(title_key, index)
    if not pdf_url:
        pdf_url = _author_lookup(_author_tokens(paper.get("authors")), _AUTHOR_INDEX_CACHE.get(cache_key, []))
    if not pdf_url:
        return None
    method = {"ICLR": "proceedings.iclr", "ICML": "proceedings.icml_pmlr", "NeurIPS": "proceedings.neurips"}[venue]
    return {"source_type": method, "url": pdf_url, "reason": f"matched {venue} {year} official proceedings mirror"}
