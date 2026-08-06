"""Hybrid content retriever that constructs the candidate reranking pool.

Measured design (retriever-before-rerank experiment, 2026-08-06): special
objects (figure/table/equation) are ALWAYS preserved; text and citation units
are scored by hybrid BM25 (BM25 + upper-alias + explicit object-label +
section-title boosts) and the top-k enter the pool.  The pool bounds the
full-Qwen reranking input.  Conditional evidence recall (reachable gold):
0.945 @ budget 160, 0.964 @ 240; retention of the frozen b160-selected gold
evidence: 0.927 @ 160, 0.964 @ 240; rerank unit savings ~58% @160 (~27% token
savings).  Budget 0 disables the stage (full-Qwen path, the iron-rule default).
"""
from __future__ import annotations

import re
from typing import Any

from .metadata_index import BM25Okapi, extract_upper_aliases, tokenize


SPECIAL_UNIT_TYPES = {"object_figure", "object_table", "object_equation_algorithm"}


def explicit_object_labels(question: str) -> set[str]:
    """Labels like 'Table 4', 'Figure 2', 'Equation 6' mentioned in the question."""
    labels: set[str] = set()
    for kind, value in re.findall(
        r"\b(table|figure|fig\.?|equation|eq\.?|algorithm)\s*\(?\s*(\d{1,4})\s*\)?",
        question,
        re.IGNORECASE,
    ):
        normalized = (
            "figure" if kind.lower().startswith("fig")
            else "equation" if kind.lower().startswith("eq")
            else kind.lower()
        )
        labels.add(f"{normalized} {value}")
    return labels


def hybrid_retriever_scores(
    question: str,
    units: list[dict[str, Any]],
    sections: list[dict[str, Any]] | None = None,
) -> list[float]:
    """Hybrid scores: BM25 + alias, explicit object-label, and section boosts."""
    texts = [tokenize(str(unit.get("text") or "")) for unit in units]
    scores = [float(value) for value in BM25Okapi(texts).get_scores(tokenize(question))] if texts else []
    aliases = {alias.lower() for alias in extract_upper_aliases(question)}
    question_terms = set(tokenize(question))
    labels = explicit_object_labels(question)
    section_title_by_id = {
        str(section.get("section_id") or ""): str(section.get("section_title") or "")
        for section in (sections or [])
    }
    boosted = list(scores)
    for index, unit in enumerate(units):
        text = str(unit.get("text") or "").lower()
        if aliases and any(alias in text for alias in aliases):
            boosted[index] += 3.0
        section_title = section_title_by_id.get(str(unit.get("section_id") or ""), "")
        if section_title and question_terms.intersection(set(tokenize(section_title))):
            boosted[index] += 2.0
        if labels and any(label in text for label in labels):
            boosted[index] += 4.0
        if str(unit.get("unit_type") or "") in SPECIAL_UNIT_TYPES:
            first_line_terms = set(tokenize(str(unit.get("text") or "").split("\n")[0]))
            if question_terms.intersection(first_line_terms):
                boosted[index] += 2.0
    return boosted


def build_retriever_pool(
    units: list[dict[str, Any]],
    scores: list[float],
    budget: int,
) -> set[int]:
    """Special objects always preserved; top-budget non-special by score."""
    special = {index for index, unit in enumerate(units) if str(unit.get("unit_type") or "") in SPECIAL_UNIT_TYPES}
    nonspecial = sorted(
        (index for index in range(len(units)) if index not in special),
        key=lambda index: scores[index] if index < len(scores) else float("-inf"),
        reverse=True,
    )[: max(0, budget)]
    return special | set(nonspecial)
