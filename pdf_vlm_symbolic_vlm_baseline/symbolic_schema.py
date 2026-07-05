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

VISUAL_RECORD_TYPES = {"figure", "figure_caption", "table", "table_caption", "equation", "algorithm"}
