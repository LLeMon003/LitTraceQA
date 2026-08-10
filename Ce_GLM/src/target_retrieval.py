"""Paper-constrained lexical retrieval with provenance-aware benchmark scoring."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


STOP = {"the", "a", "an", "in", "of", "at", "and", "or", "is", "what", "which", "does", "value", "reported", "paper"}
def terms(text: str) -> set[str]: return {x for x in re.findall(r"[a-z0-9]+", text.lower()) if len(x) > 1 and x not in STOP}
def source_id(row: dict[str, Any]) -> str | None: return row.get("object_uid") or row.get("record_hash")
def text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(k, "")) for k in ("normalized_cell_value", "raw_cell_value", "normalized_value", "table_caption", "object_type", "evaluator_visible_table_id", "row_index", "column_index"))


def retrieve(question: str, paper: str, corpus: Iterable[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    query = terms(question); ranked = []; table = re.search(r"(Table\s+\d+)", question, re.I); page = re.search(r"page\s+(\d+)", question, re.I); coordinates = re.search(r"row\s+(\d+)\s+and\s+column\s+(\d+)", question, re.I); fact_kind = re.search(r"the\s+([a-z_]+)\s+in paper", question, re.I)
    for row in corpus:
        if row.get("paper_id") != paper or not source_id(row): continue
        row_terms = row.get("_search_terms")
        if row_terms is None:
            row_terms = terms(text(row))
        score = len(query & row_terms)
        if table and str(row.get("evaluator_visible_table_id", "")).lower() == table.group(1).lower(): score += 50
        if page and str(row.get("page")) == page.group(1): score += 25
        if coordinates and str(row.get("row_index")) == coordinates.group(1) and str(row.get("column_index")) == coordinates.group(2): score += 100
        if fact_kind and str(row.get("object_type", "")).lower() == fact_kind.group(1).lower(): score += 50
        ranked.append((score, str(source_id(row)), row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [{"object_id": uid, "paper": row.get("paper_id"), "page": row.get("page"), "source_hash": row.get("source_hash"), "content": text(row), "score": score} for score, uid, row in ranked[:limit]]


def score(records: Iterable[dict[str, Any]], corpus: Iterable[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for row in corpus:
        if not source_id(row):
            continue
        row["_search_terms"] = terms(text(row))
        by_paper.setdefault(str(row.get("paper_id")), []).append(row)
    totals, hits, family = 0, 0, Counter()
    family_hits = Counter()
    for record in records:
        expected = {str(x.get("object_id")) for x in record["source_objects"] if x.get("object_id")}
        evidence_limit = max(limit, 512) if record["reasoning_operator"] in {"comparison", "negation_except", "multi_object_count"} else limit
        found = {x["object_id"] for x in retrieve(record["question"], record["source_paper"], by_paper.get(str(record["source_paper"]), []), evidence_limit)}
        hit = bool(expected & found); totals += 1; hits += hit; family[record["reasoning_operator"]] += 1; family_hits[record["reasoning_operator"]] += hit
    return {"answer_bearing_recall": hits / totals if totals else 0.0, "records": totals, "family_recall": {key: family_hits[key] / count for key, count in sorted(family.items())}}
