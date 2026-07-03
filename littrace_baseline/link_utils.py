from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse


URL_RE = re.compile(r"https?://[^\s<>\"]+")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
ARXIV_RE = re.compile(r"\b(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?", re.IGNORECASE)
OPENREVIEW_URL_RE = re.compile(r"openreview\.net/(?:forum|pdf|attachment)?\??[^\"'<>\s]*\bid=([A-Za-z0-9_-]+)", re.IGNORECASE)
OPENREVIEW_ID_RE = re.compile(r"\b[A-Za-z0-9_-]{8,}\b")

TRAILING_PUNCTUATION = ".,;)]}>\"'"
PDF_PRIORITIES = {
    "direct_pdf": 1,
    "arxiv": 2,
    "openreview": 3,
    "doi": 4,
    "direct_url": 5,
    "unknown": 9,
}


def _walk_values(obj: Any) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            values.append((str(key), value))
            values.extend(_walk_values(value))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_walk_values(item))
    return values


def _clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_identifier(value: str) -> str:
    return value.strip().strip(TRAILING_PUNCTUATION)


def _clean_url(value: str) -> str:
    return _clean_identifier(value)


def _looks_like_direct_pdf_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return url.lower().endswith(".pdf") or "/pdf" in url.lower()
    path = parsed.path.lower()
    return path.endswith(".pdf") or "/pdf/" in path or path.endswith("/pdf") or "/pdf" in path


def _extract_openreview_id_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if "openreview.net" not in parsed.netloc.lower():
        return None
    query_ids = parse_qs(parsed.query).get("id")
    if query_ids and query_ids[0].strip():
        return _clean_identifier(query_ids[0])
    match = OPENREVIEW_URL_RE.search(url)
    if match:
        return _clean_identifier(match.group(1))
    return None


def _iter_string_fields(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for key, value in _walk_values(metadata):
        if isinstance(value, str) or value is not None:
            text = _clean_text_value(value)
            if text:
                fields.append((key, text))
    return fields


def extract_identifiers(metadata: dict[str, Any]) -> dict[str, list[str]]:
    identifiers: dict[str, list[str]] = {"urls": [], "pdf_urls": [], "dois": [], "arxiv_ids": [], "openreview_ids": []}

    def add(kind: str, value: str) -> None:
        cleaned = _clean_identifier(value)
        if cleaned and cleaned not in identifiers[kind]:
            identifiers[kind].append(cleaned)

    for key, text in _iter_string_fields(metadata):
        key_lower = key.lower()
        for url in URL_RE.findall(text):
            cleaned_url = _clean_url(url)
            add("urls", cleaned_url)
            if "pdf" in key_lower or _looks_like_direct_pdf_url(cleaned_url):
                add("pdf_urls", cleaned_url)
            for match in ARXIV_URL_RE.findall(cleaned_url):
                add("arxiv_ids", match)
            openreview_id = _extract_openreview_id_from_url(cleaned_url)
            if openreview_id:
                add("openreview_ids", openreview_id)
        for doi in DOI_RE.findall(text):
            add("dois", doi)
        if "doi" in key_lower and text and not text.startswith("http"):
            for doi in DOI_RE.findall(text):
                add("dois", doi)
        if "arxiv" in key_lower or "arxiv" in text.lower():
            for match in ARXIV_RE.findall(text):
                add("arxiv_ids", match)
        if "openreview" in key_lower and text and not text.startswith("http"):
            for match in OPENREVIEW_ID_RE.findall(text):
                add("openreview_ids", match)
    return identifiers


def extract_pdf_candidate_urls(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract reproducible, public PDF download candidates from paper metadata."""
    identifiers = extract_identifiers(metadata)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(source_type: str, url: str, reason: str) -> None:
        cleaned_url = _clean_url(url)
        if not cleaned_url or cleaned_url in seen:
            return
        seen.add(cleaned_url)
        candidates.append(
            {
                "source_type": source_type,
                "url": cleaned_url,
                "priority": PDF_PRIORITIES.get(source_type, PDF_PRIORITIES["unknown"]),
                "reason": reason,
            }
        )

    for url in identifiers["pdf_urls"]:
        add("direct_pdf", url, "metadata string contains a direct PDF-looking URL")

    for arxiv_id in identifiers["arxiv_ids"]:
        add("arxiv", f"https://arxiv.org/pdf/{arxiv_id}.pdf", "metadata contains arXiv identifier")

    for openreview_id in identifiers["openreview_ids"]:
        if openreview_id.startswith("http"):
            add("openreview", openreview_id, "metadata contains OpenReview URL")
        else:
            add("openreview", f"https://openreview.net/pdf?id={openreview_id}", "metadata contains OpenReview identifier")

    for doi in identifiers["dois"]:
        add("doi", f"https://doi.org/{doi}", "metadata contains DOI")

    for url in identifiers["urls"]:
        if url not in identifiers["pdf_urls"]:
            source_type = "direct_url" if url.startswith(("http://", "https://")) else "unknown"
            add(source_type, url, "metadata contains non-PDF URL")

    candidates.sort(key=lambda item: (item["priority"], item["url"]))
    return candidates


def extract_online_links(metadata: dict[str, Any]) -> list[str]:
    return [candidate["url"] for candidate in extract_pdf_candidate_urls(metadata)]
