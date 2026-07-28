"""Explicit prose references for labeled scientific-paper objects."""
from __future__ import annotations

import re
from typing import Any

from .symbolic_schema import to_official_source_type


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or record.get("record_id") or record.get("id") or "")


def _record_order(record: dict[str, Any]) -> tuple[int, int, str]:
    try:
        page = int(record.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    try:
        reading = int(record.get("document_order") or record.get("reading_order") or 0)
    except (TypeError, ValueError):
        reading = 0
    return page, reading, _record_id(record)


def object_label_pattern(label: Any) -> re.Pattern[str] | None:
    """Build a conservative in-text reference matcher for a labeled object."""
    text = str(label or "").strip()
    match = re.match(
        r"^(table|figure|fig\.?|equation|eq\.?|algorithm)\s*\(?\s*((?:[A-Za-z]\.)?[0-9]+[A-Za-z]?(?:\.[0-9A-Za-z]+)*)\s*\)?$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    kind = match.group(1).lower().rstrip(".")
    identifier = re.escape(match.group(2))
    aliases = {
        "table": r"(?:table|tab\.?)",
        "figure": r"(?:figure|fig\.?)",
        "fig": r"(?:figure|fig\.?)",
        "equation": r"(?:equation|eq\.?)",
        "eq": r"(?:equation|eq\.?)",
        "algorithm": r"(?:algorithm|alg\.?)",
    }[kind]
    return re.compile(rf"(?<!\w){aliases}\s*\(?\s*{identifier}\s*\)?(?!\w)", re.IGNORECASE)


def same_object_label(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_label = re.sub(r"\s+", " ", str(left.get("label") or "").strip().lower())
    right_label = re.sub(r"\s+", " ", str(right.get("label") or "").strip().lower())
    return bool(left_label and left_label == right_label)


def object_reference_paragraphs(anchor: dict[str, Any], paper_records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Return prose that explicitly narrates a labeled table, figure, or equation."""
    pattern = object_label_pattern(anchor.get("label"))
    if pattern is None or limit <= 0:
        return []
    anchor_order = _record_order(anchor)
    candidates = [
        record
        for record in paper_records
        if _record_id(record) != _record_id(anchor)
        and to_official_source_type(record.get("record_type"), record.get("source_type")) == "text_span"
        and pattern.search(str(record.get("text") or ""))
    ]
    return sorted(
        candidates,
        key=lambda record: (
            int(record.get("section_id") != anchor.get("section_id")),
            abs(_record_order(record)[0] - anchor_order[0]),
            abs(_record_order(record)[1] - anchor_order[1]),
            _record_order(record),
        ),
    )[:limit]


__all__ = ["object_label_pattern", "object_reference_paragraphs", "same_object_label"]
