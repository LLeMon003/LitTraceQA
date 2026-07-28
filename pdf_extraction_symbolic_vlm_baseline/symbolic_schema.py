from __future__ import annotations


ALLOWED_RECORD_TYPES = {
    "title",
    "section_header",
    "paragraph",
    "table",
    "table_caption",
    "figure",
    "figure_caption",
    "equation",
    "algorithm",
    "citation_context",
    "reference",
    "footer",
    "header",
    "header_footer",
    "unknown",
}

RECORD_TYPE_TO_SOURCE_TYPE = {
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
    "footer": "text_span",
    "header": "text_span",
    "header_footer": "text_span",
    "unknown": "text_span",
}

OFFICIAL_EVIDENCE_SOURCE_TYPES = {
    "text_span",
    "table",
    "figure",
    "equation_algorithm",
    "citation_context",
}

HEADER_FOOTER_RECORD_TYPES = {"header", "footer", "header_footer"}

VISUAL_RECORD_TYPES = {"figure", "figure_caption", "table", "table_caption", "equation", "algorithm"}


def to_official_source_type(record_type: str | None = None, source_type: str | None = None) -> str | None:
    candidate = str(source_type or "").strip()
    if candidate in OFFICIAL_EVIDENCE_SOURCE_TYPES:
        return candidate
    mapped = RECORD_TYPE_TO_SOURCE_TYPE.get(str(record_type or "").strip())
    if mapped in OFFICIAL_EVIDENCE_SOURCE_TYPES:
        return mapped
    if str(record_type or "").strip() == "unknown":
        return "text_span"
    return None


def grounding_label_from_record(source_type: str | None, label: object) -> dict[str, str] | None:
    value = str(label or "").strip()
    if not value:
        return None
    source = str(source_type or "")
    if source == "table":
        return {"type": "table_id", "value": value}
    if source == "figure":
        return {"type": "figure_id", "value": value}
    if source == "equation_algorithm":
        lowered = value.lower()
        if "algorithm" in lowered:
            return {"type": "algorithm_id", "value": value}
        return {"type": "equation_id", "value": value}
    if source == "citation_context":
        return {"type": "citation_id", "value": value}
    return None
