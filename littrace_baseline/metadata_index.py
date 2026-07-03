from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[a-z0-9]+")
URL_RE = re.compile(r"https?://\S+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def sanitize_metadata_text(text: str) -> str:
    return URL_RE.sub("[URL_REMOVED]", text or "")


def build_metadata_records(metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in metadata_rows:
        title = sanitize_metadata_text(str(row.get("title") or ""))
        abstract = sanitize_metadata_text(str(row.get("abstract") or ""))
        record = dict(row)
        record["paper_id"] = str(row.get("paper_id") or "")
        record["title"] = title
        record["abstract"] = abstract
        record["venue"] = row.get("venue", "")
        record["year"] = row.get("year", "")
        record["online_links"] = []
        record["_title_tokens"] = tokenize(title)
        record["_abstract_tokens"] = tokenize(abstract)
        records.append(record)
    return records


def retrieve_candidates(question: str, metadata_records: list[dict[str, Any]], top_k: int = 12) -> list[dict[str, Any]]:
    query_tokens = tokenize(question)
    if not query_tokens:
        return []
    corpus = [record["_title_tokens"] + record["_abstract_tokens"] for record in metadata_records]
    index = BM25Okapi(corpus)
    scores = index.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    candidates: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        record = metadata_records[idx]
        candidates.append(
            {
                "rank": rank,
                "score": round(score, 6),
                "bm25_score": float(score),
                "paper_id": record.get("paper_id", ""),
                "title": record.get("title", ""),
                "abstract": record.get("abstract", ""),
                "venue": record.get("venue", ""),
                "year": record.get("year", ""),
                "online_links": [],
            }
        )
    if not candidates and metadata_records:
        first = metadata_records[0]
        candidates.append(
            {
                "rank": 1,
                "score": 0.0,
                "paper_id": first.get("paper_id", ""),
                "title": first.get("title", ""),
                "abstract": first.get("abstract", ""),
                "online_links": [],
            }
        )
    return candidates
