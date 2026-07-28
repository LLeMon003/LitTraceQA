from __future__ import annotations

import re
from typing import Any

from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES


TABLE_ID_RE = re.compile(r"\b(?:Table|Tab\.)\s+([A-Za-z]*\d+[A-Za-z0-9.\-]*)\b", re.IGNORECASE)
FIGURE_ID_RE = re.compile(r"\b(?:Figure|Fig\.)\s+([A-Za-z]*\d+[A-Za-z0-9.\-]*)\b", re.IGNORECASE)
EQUATION_ID_RE = re.compile(r"\b(?:Equation|Eq\.)\s*\(?([A-Za-z0-9.\-]+)\)?", re.IGNORECASE)
ALGORITHM_ID_RE = re.compile(r"\bAlgorithm\s+([A-Za-z0-9.\-]+)", re.IGNORECASE)
CITATION_ID_RE = re.compile(r"(?:^|\n|\s)\[(\d{1,3})\]\s+")
MARKDOWN_TABLE_RE = re.compile(r"(^|\n)\s*\|[^\n]+\|\s*(\n|$)")
EQUATION_SIGNAL_RE = re.compile(
    r"\b(?:loss|objective|minimi[sz]e|argmax|argmin|gradient|regularization|constraint)\b|[=∑∏∫≤≥]",
    re.IGNORECASE,
)


def _label(prefix: str, value: str | None) -> str:
    text = str(value or "").strip().rstrip(".,;:")
    if not text:
        return ""
    if text.lower().startswith(prefix.lower()):
        return text
    return f"{prefix} {text}"


def infer_source_type_hints(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer compact alternate source-type hints for a symbolic record.

    This does not duplicate records and does not change the primary source_type.
    It only marks text spans that visibly contain table, figure, equation, or
    citation evidence signals so VLM-2 and deterministic evidence normalization
    can treat the same paper/page/text as a typed evidence candidate.
    """

    primary_source = str(record.get("source_type") or "")
    text = str(record.get("text") or "")
    if not text.strip() or primary_source != "text_span":
        return []

    page = record.get("page")
    hints: list[dict[str, Any]] = []

    def add(source_type: str, *, label: str = "", locator_extra: dict[str, Any] | None = None, reason: str) -> None:
        if source_type == primary_source or source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            return
        locator = {"page": page}
        if locator_extra:
            locator.update({key: value for key, value in locator_extra.items() if value not in {None, ""}})
        hint = {
            "source_type": source_type,
            "label": label or None,
            "locator": locator,
        }
        key = (hint["source_type"], hint["label"], tuple(sorted((locator or {}).items())))
        for existing in hints:
            existing_key = (
                existing.get("source_type"),
                existing.get("label"),
                tuple(sorted(((existing.get("locator") or {}).items()))),
            )
            if existing_key == key:
                return
        hints.append(hint)

    table_match = TABLE_ID_RE.search(text)
    if table_match:
        label = _label("Table", table_match.group(1))
        add("table", label=label, locator_extra={"table_id": label}, reason="text contains a visible table id")
    elif MARKDOWN_TABLE_RE.search(text):
        add("table", label=str(record.get("label") or ""), reason="text contains markdown-like table rows")

    figure_match = FIGURE_ID_RE.search(text)
    if figure_match:
        label = _label("Figure", figure_match.group(1))
        add("figure", label=label, locator_extra={"figure_id": label}, reason="text contains a visible figure id")

    algorithm_match = ALGORITHM_ID_RE.search(text)
    if algorithm_match:
        label = _label("Algorithm", algorithm_match.group(1))
        add("equation_algorithm", label=label, locator_extra={"algorithm_id": label}, reason="text contains a visible algorithm id")
    else:
        equation_match = EQUATION_ID_RE.search(text)
        if equation_match:
            label = _label("Equation", equation_match.group(1))
            add("equation_algorithm", label=label, locator_extra={"equation_id": label}, reason="text contains a visible equation id")
        elif EQUATION_SIGNAL_RE.search(text) and len(re.findall(r"[=∑∏∫≤≥]", text)) >= 2:
            add("equation_algorithm", label=str(record.get("label") or ""), reason="text contains equation-like symbols or objective terms")

    citation_match = CITATION_ID_RE.search(text)
    if citation_match:
        citation_id = int(citation_match.group(1))
        add("citation_context", locator_extra={"citation_id": citation_id, "reference_id": citation_id}, reason="text contains a reference-list citation marker")
    elif re.search(r"\bReferences\b", text, re.IGNORECASE):
        add("citation_context", reason="text contains a references heading")

    return hints
