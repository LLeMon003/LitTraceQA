from __future__ import annotations

import re
from typing import Any


OFFICIAL_SOURCE_TYPES = {"text_span", "table", "figure", "equation_algorithm", "citation_context"}
SOURCE_ALIASES = {
    "paragraph": "text_span",
    "section_header": "text_span",
    "text": "text_span",
    "table_caption": "table",
    "figure_caption": "figure",
    "equation": "equation_algorithm",
    "algorithm": "equation_algorithm",
    "reference": "citation_context",
    "reference_entry": "citation_context",
}


def normalize_source_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return SOURCE_ALIASES.get(text, text if text in OFFICIAL_SOURCE_TYPES else text)


def normalize_label(value: Any, source_type: str | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    source = normalize_source_type(source_type)
    if source == "table":
        match = re.search(r"\b(?:table|tab\.)\s*([A-Za-z0-9.\-]+)", text, re.IGNORECASE)
        return f"Table {match.group(1).rstrip('.,;:')}" if match else text
    if source == "figure":
        match = re.search(r"\b(?:figure|fig\.)\s*([A-Za-z0-9.\-]+)", text, re.IGNORECASE)
        return f"Figure {match.group(1).rstrip('.,;:')}" if match else text
    if source == "equation_algorithm":
        alg = re.search(r"\balgorithm\s*([A-Za-z0-9.\-]+)", text, re.IGNORECASE)
        if alg:
            return f"Algorithm {alg.group(1).rstrip('.,;:')}"
        eq = re.search(r"\b(?:equation|eq\.)\s*\(?([A-Za-z0-9.\-]+)\)?", text, re.IGNORECASE) or re.search(r"^\(?([0-9]+[A-Za-z]?)\)?$", text)
        return f"Equation {eq.group(1).rstrip('.,;:')}" if eq else text
    if source == "citation_context":
        match = (
            re.search(r"\breference\s*([0-9]{1,4})\b", text, re.IGNORECASE)
            or re.search(r"\[([0-9]{1,4})\]", text)
            or re.search(r"\b([0-9]{1,4})(?:st|nd|rd|th)\s+reference\b", text, re.IGNORECASE)
            or re.search(r"^\s*([0-9]{1,4})\s*$", text)
        )
        return f"Reference {int(match.group(1))}" if match else text
    return text


def gold_page(gold: dict[str, Any]) -> int | None:
    locator = gold.get("locator") if isinstance(gold.get("locator"), dict) else {}
    value = gold.get("page") or locator.get("page")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def gold_source_type(gold: dict[str, Any]) -> str:
    return normalize_source_type(gold.get("source_type") or gold.get("record_type") or gold.get("primary_evidence_type"))


def gold_locator_label(gold: dict[str, Any]) -> str:
    locator = gold.get("locator") if isinstance(gold.get("locator"), dict) else {}
    source = gold_source_type(gold)
    if source == "table":
        return normalize_label(locator.get("table_id") or gold.get("label"), source)
    if source == "figure":
        return normalize_label(locator.get("figure_id") or gold.get("label"), source)
    if source == "equation_algorithm":
        return normalize_label(locator.get("algorithm_id") or locator.get("equation_id") or gold.get("label"), source)
    if source == "citation_context":
        return normalize_label(locator.get("citation_id") or gold.get("label") or gold.get("evidence_text_or_value"), source)
    return normalize_label(gold.get("label"), source)


def record_locator_label(record: dict[str, Any], source_type: str | None = None) -> str:
    source = normalize_source_type(source_type or record.get("source_type") or record.get("record_type"))
    locator = record.get("locator") if isinstance(record.get("locator"), dict) else {}
    if source == "table":
        return normalize_label(locator.get("table_id") or record.get("label"), source)
    if source == "figure":
        return normalize_label(locator.get("figure_id") or record.get("label"), source)
    if source == "equation_algorithm":
        return normalize_label(locator.get("algorithm_id") or locator.get("equation_id") or record.get("label"), source)
    if source == "citation_context":
        return normalize_label(locator.get("citation_id") or record.get("label") or record.get("text"), source)
    return normalize_label(record.get("label"), source)


def lexical_overlap(a: Any, b: Any) -> float:
    left = {t for t in re.findall(r"[a-z0-9]+", str(a or "").lower()) if len(t) > 2}
    right = {t for t in re.findall(r"[a-z0-9]+", str(b or "").lower()) if len(t) > 2}
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _base_match(gold: dict[str, Any], record: dict[str, Any]) -> tuple[bool, bool, bool]:
    paper_ok = str(record.get("paper_id") or "") == str(gold.get("paper_id") or "")
    page = gold_page(gold)
    try:
        record_page = int(record.get("page") or 0)
    except (TypeError, ValueError):
        record_page = 0
    page_ok = page is not None and record_page == page
    source_ok = normalize_source_type(record.get("source_type") or record.get("record_type")) == gold_source_type(gold)
    return paper_ok, page_ok, source_ok


def match_gold_to_record(gold: dict[str, Any], records: list[dict[str, Any]], match_mode: str = "strict") -> dict[str, Any]:
    mode = "relaxed" if str(match_mode).lower() == "relaxed" else "strict"
    page_candidates: list[dict[str, Any]] = []
    source_candidates: list[dict[str, Any]] = []
    locator_candidates: list[dict[str, Any]] = []
    text_candidates: list[tuple[float, dict[str, Any]]] = []
    gold_label = gold_locator_label(gold)
    gold_text = gold.get("evidence_text_or_value") or gold.get("text") or ""
    source = gold_source_type(gold)

    for record in records:
        paper_ok, page_ok, source_ok = _base_match(gold, record)
        if not (paper_ok and page_ok):
            continue
        page_candidates.append(record)
        if not source_ok:
            continue
        source_candidates.append(record)
        record_label = record_locator_label(record, source)
        if gold_label and record_label and gold_label.lower() == record_label.lower():
            locator_candidates.append(record)
        elif source == "text_span" and gold_text:
            overlap = lexical_overlap(gold_text, record.get("text"))
            if overlap >= 0.35:
                text_candidates.append((overlap, record))

    selected: dict[str, Any] | None = None
    level = ""
    reason = ""
    if mode == "relaxed" and source_candidates:
        selected = source_candidates[0]
        level = "source_type"
        reason = "paper_id + page + source_type matched"
    elif locator_candidates:
        selected = locator_candidates[0]
        level = "locator"
        reason = "paper_id + page + source_type + normalized locator/label matched"
    elif source == "text_span" and source_candidates:
        if text_candidates:
            text_candidates.sort(key=lambda item: item[0], reverse=True)
            selected = text_candidates[0][1]
            level = "text_overlap"
            reason = f"text overlap matched at {text_candidates[0][0]:.2f}"
        else:
            selected = source_candidates[0]
            level = "source_type"
            reason = "text_span uses page + source_type match when no stable locator is available"

    if selected:
        return {
            "matched": True,
            "match_level": level,
            "match_mode": mode,
            "reason": reason,
            "matched_record_id": selected.get("record_id"),
            "matched_global_record_id": selected.get("global_record_id"),
            "matched_label": selected.get("label"),
        }
    if source_candidates:
        level = "source_type"
        reason = "source_type matched but strict locator/label did not"
    elif page_candidates:
        level = "page"
        reason = "page matched but source_type did not"
    else:
        level = "none"
        reason = "no record on gold page"
    return {
        "matched": False,
        "match_level": level,
        "match_mode": mode,
        "reason": reason,
        "matched_record_id": None,
        "matched_global_record_id": None,
    }
