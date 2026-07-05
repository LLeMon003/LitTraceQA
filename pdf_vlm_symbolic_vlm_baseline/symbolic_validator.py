from __future__ import annotations

import re
from typing import Any

from .config import DEFAULT_ARTIFACT_VERSION, DEFAULT_PARSER_EXTRACTION_MODE


KIND_TO_RECORD_AND_SOURCE = {
    "text_span": ("paragraph", "text_span"),
    "table": ("table", "table"),
    "figure": ("figure", "figure"),
    "equation_algorithm": ("equation", "equation_algorithm"),
    "citation_context": ("citation_context", "citation_context"),
    "header_footer": ("header_footer", "text_span"),
    "unknown": ("unknown", "text_span"),
}

OLD_RECORD_TYPE_TO_KIND = {
    "title": "text_span",
    "section_header": "text_span",
    "paragraph": "text_span",
    "table": "table",
    "table_caption": "table",
    "figure": "figure",
    "figure_caption": "figure",
    "equation": "equation_algorithm",
    "algorithm": "equation_algorithm",
    "citation_context": "citation_context",
    "reference": "citation_context",
    "footer": "header_footer",
    "header": "header_footer",
    "header_footer": "header_footer",
    "unknown": "unknown",
}


def validate_page_structure(
    raw: dict[str, Any],
    expected_paper_id: str,
    expected_page: int,
    expected_parser_mode: str = DEFAULT_PARSER_EXTRACTION_MODE,
) -> dict[str, Any]:
    repaired = dict(raw) if isinstance(raw, dict) else {}
    warnings: list[str] = list(repaired.get("warnings", [])) if isinstance(repaired.get("warnings"), list) else []
    if not isinstance(repaired.get("records"), list):
        warnings.append("records repaired to empty list because it was not a list")
        repaired["records"] = []
    coverage = repaired.get("coverage")
    if not isinstance(coverage, dict):
        old_coverage = repaired.get("coverage_report")
        coverage = old_coverage if isinstance(old_coverage, dict) else {}
        if not coverage:
            warnings.append("coverage repaired because it was missing or invalid")
    if not isinstance(coverage.get("needs_continuation"), bool):
        coverage["needs_continuation"] = False
        warnings.append("coverage.needs_continuation repaired to false")
    if "next_start_hint" not in coverage:
        coverage["next_start_hint"] = ""
    if not isinstance(coverage.get("known_omissions"), list):
        coverage["known_omissions"] = []
    if not repaired["records"] and not coverage.get("needs_continuation") and not warnings:
        warnings.append("records empty and continuation false; page may be unreadable")
    repaired["paper_id"] = expected_paper_id
    repaired["page"] = expected_page
    repaired["parser_mode"] = expected_parser_mode
    repaired["coverage"] = coverage
    repaired["warnings"] = warnings
    return repaired


def _repair_bbox(value: Any) -> tuple[list[int] | None, list[str], bool]:
    errors: list[str] = []
    if value is None:
        return None, [], False
    if not isinstance(value, list) or len(value) != 4:
        return None, ["bbox_1000 is not a 4-item list"], False
    try:
        nums = [float(v) for v in value]
    except Exception:
        return None, ["bbox_1000 contains non-numeric values"], False
    x1, y1, x2, y2 = nums
    repaired = False
    if x1 > x2:
        x1, x2 = x2, x1
        repaired = True
        errors.append("bbox x coordinates swapped")
    if y1 > y2:
        y1, y2 = y2, y1
        repaired = True
        errors.append("bbox y coordinates swapped")
    clipped = [max(0.0, min(1000.0, v)) for v in [x1, y1, x2, y2]]
    if clipped != [x1, y1, x2, y2]:
        repaired = True
        errors.append("bbox clipped to 0-1000")
    x1, y1, x2, y2 = clipped
    if not (x1 < x2 and y1 < y2):
        return None, errors + ["bbox has zero or negative area"], repaired
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], errors, repaired


def _kind_from_raw(item: dict[str, Any]) -> tuple[str, list[str]]:
    errors: list[str] = []
    raw_kind = str(item.get("kind") or "").strip()
    if raw_kind in KIND_TO_RECORD_AND_SOURCE:
        return raw_kind, errors
    old_record_type = str(item.get("record_type") or "").strip()
    if old_record_type in OLD_RECORD_TYPE_TO_KIND:
        kind = OLD_RECORD_TYPE_TO_KIND[old_record_type]
        errors.append(f"legacy record_type mapped to kind={kind}")
        return kind, errors
    errors.append(f"kind repaired from {raw_kind or old_record_type or '<missing>'} to unknown")
    return "unknown", errors


def _looks_like_placeholder(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    placeholders = {
        "...",
        "n/a",
        "none",
        "null",
        "text",
        "string",
        "<text>",
        "visible page content",
        "record text",
    }
    if lowered in placeholders:
        return True
    instruction_markers = ["output json", "do not output", "use a minimal schema", "for each record"]
    return any(marker in lowered for marker in instruction_markers)


TABLE_ID_RE = re.compile(r"\b(?:Table|Tab\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)", re.IGNORECASE)
FIGURE_ID_RE = re.compile(r"\b(?:Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)", re.IGNORECASE)
EQUATION_ID_RE = re.compile(r"\b(?:Equation|Eq\.)\s*\(?\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\s*\)?|\(\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\s*\)", re.IGNORECASE)
ALGORITHM_ID_RE = re.compile(r"\bAlgorithm\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)", re.IGNORECASE)
CITATION_ID_RE = re.compile(r"\[\s*(\d{1,4})\s*\]|\bReference\s+(\d{1,4})\b", re.IGNORECASE)


def _visible_id(match: re.Match[str], prefix: str) -> str:
    value = next((group for group in match.groups() if group), "")
    return f"{prefix} {value}".strip()


def _extract_locator_ids(source_type: str, label: Any, text: str) -> dict[str, Any]:
    haystack = " ".join(part for part in [str(label or ""), text] if part).strip()
    locator: dict[str, Any] = {}
    if source_type == "table":
        match = TABLE_ID_RE.search(haystack)
        if match:
            locator["table_id"] = _visible_id(match, "Table")
    elif source_type == "figure":
        match = FIGURE_ID_RE.search(haystack)
        if match:
            locator["figure_id"] = _visible_id(match, "Figure")
    elif source_type == "equation_algorithm":
        algorithm_match = ALGORITHM_ID_RE.search(haystack)
        equation_match = EQUATION_ID_RE.search(haystack)
        if algorithm_match:
            locator["algorithm_id"] = _visible_id(algorithm_match, "Algorithm")
        if equation_match:
            locator["equation_id"] = _visible_id(equation_match, "Equation")
    elif source_type == "citation_context":
        match = CITATION_ID_RE.search(haystack)
        if match:
            locator["citation_id"] = int(next(group for group in match.groups() if group))
    return locator


def normalize_page_records(
    raw: dict[str, Any],
    parser_model: str,
    page_width: int,
    page_height: int,
    *,
    artifact_version: str = DEFAULT_ARTIFACT_VERSION,
    parser_mode: str = DEFAULT_PARSER_EXTRACTION_MODE,
    page_status: str = "partial",
    start_index: int = 1,
    pass_index: int = 1,
) -> list[dict[str, Any]]:
    paper_id = str(raw.get("paper_id", ""))
    page = int(raw.get("page", 0) or 0)
    records: list[dict[str, Any]] = []
    indexed_items = [
        (order, item)
        for order, item in enumerate(raw.get("records", []))
        if isinstance(item, dict)
    ]
    indexed_items.sort(key=lambda pair: int(pair[1].get("reading_order") or 10**9) if pair[1].get("reading_order") is not None else pair[0])
    for offset, (_, item) in enumerate(indexed_items, start=0):
        idx = start_index + offset
        errors: list[str] = []
        status = "valid"
        kind, kind_errors = _kind_from_raw(item)
        errors.extend(kind_errors)
        if kind_errors:
            status = "repaired"
        record_type, source_type = KIND_TO_RECORD_AND_SOURCE[kind]
        if item.get("bbox_1000") is not None:
            errors.append("bbox_1000 ignored by evaluator-grounded minimal schema")
        text_value = item.get("text")
        if not isinstance(text_value, str):
            text = "" if text_value is None else str(text_value)
            errors.append("text coerced to string")
            status = "repaired"
        else:
            text = text_value
        if _looks_like_placeholder(text):
            errors.append("text appears to be schema echo, instruction echo, or placeholder")
            status = "rejected"
        if not text.strip() and kind not in {"figure", "table", "equation_algorithm"}:
            errors.append("empty text for non-visual/equation record")
            status = "rejected"
        record_id = f"p{page:03d}_r{idx:04d}"
        label = item.get("label")
        if label in {"", "null", "None"}:
            label = None
        label = label if label is None else str(label)
        locator = {"page": page, **_extract_locator_ids(source_type, label, text)}
        if item.get("confidence") is not None:
            errors.append("confidence ignored by evaluator-grounded minimal schema")
        records.append(
            {
                "paper_id": paper_id,
                "page": page,
                "record_id": record_id,
                "global_record_id": f"{paper_id}::{record_id}",
                "record_type": record_type,
                "source_type": source_type,
                "text": text,
                "label": label,
                "locator": locator,
                "reading_order": idx,
                "parser_model": parser_model,
                "artifact_version": artifact_version,
                "validation_status": status,
                "validation_errors": errors,
                "page_status": page_status,
                "pass_index": pass_index,
                "raw_kind": kind,
                "raw_record": item,
            }
        )
    return records


def to_runtime_record(record: dict[str, Any]) -> dict[str, Any]:
    runtime = {
        "paper_id": record.get("paper_id"),
        "page": record.get("page"),
        "record_id": record.get("record_id"),
        "global_record_id": record.get("global_record_id"),
        "record_type": record.get("record_type"),
        "source_type": record.get("source_type"),
        "label": record.get("label"),
        "locator": record.get("locator") or {"page": record.get("page")},
        "text": record.get("text"),
        "reading_order": record.get("reading_order"),
    }
    return runtime


def migrate_legacy_record_to_runtime(record: dict[str, Any]) -> dict[str, Any]:
    kind, _ = _kind_from_raw(record)
    record_type, source_type = KIND_TO_RECORD_AND_SOURCE[kind]
    runtime = {
        "paper_id": record.get("paper_id"),
        "page": record.get("page"),
        "record_id": record.get("record_id"),
        "global_record_id": record.get("global_record_id") or f"{record.get('paper_id')}::{record.get('record_id')}",
        "record_type": record.get("record_type") if record.get("record_type") in {v[0] for v in KIND_TO_RECORD_AND_SOURCE.values()} else record_type,
        "source_type": record.get("source_type") or source_type,
        "label": record.get("label"),
        "locator": record.get("locator") or {"page": record.get("page")},
        "text": record.get("text"),
        "reading_order": record.get("reading_order"),
    }
    return runtime
