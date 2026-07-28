from __future__ import annotations

import re

from .pdf_layout_blocks import BBox, TextBlock, bbox_intersects


# Sections are deliberately restricted to short, canonical first-level names.
# This prevents table cells, figure labels, appendix prompts, and checklist items
# from becoming sections merely because they start with a number or capital letter.
COMMON_SECTION_TITLES = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Related Works",
    "Background",
    "Preliminaries",
    "Method",
    "Methods",
    "Methodology",
    "Approach",
    "Framework",
    "Data",
    "Dataset",
    "Data Construction",
    "Training",
    "Training Objective",
    "Optimization",
    "Experiments",
    "Experimental Setup",
    "Implementation Details",
    "Evaluation",
    "Results",
    "Analysis",
    "Ablation",
    "Discussion",
    "Conclusion",
    "Limitations",
    "Ethics Statement",
    "Acknowledgment",
    "Acknowledgments",
    "Acknowledgement",
    "Acknowledgements",
    "Broader Impact",
    "Broader Impacts",
    "References",
    "Bibliography",
    "Appendix",
    "Supplementary Material",
    "Supplementary",
)
_COMMON_SECTION_TITLES_RE = "|".join(re.escape(title) for title in COMMON_SECTION_TITLES)
FIRST_LEVEL_SECTION_RE = re.compile(
    rf"^(?:[1-9]\d*\s*[.)]?\s+)?(?:{_COMMON_SECTION_TITLES_RE})$",
    re.IGNORECASE,
)
APPENDIX_SECTION_RE = re.compile(
    r"^(?:Appendix\s+[A-Z]|[A-Z]\s+Appendix)$",
    re.IGNORECASE,
)
CITATION_MARKER_RE = re.compile(r"\[(?:\d{1,3}(?:\s*[-,;]\s*\d{1,3})*)\]|\([A-Z][A-Za-z-]+(?: et al\.)?,\s*\d{4}[a-z]?\)")
TABLE_METRIC_RE = re.compile(
    r"(?:#|%|↑|↓|±|✓|AP(?:val|50|75|s|m|l)?\b|FID\b|F1\b|Acc\.?|Params?\.?|GFLOPs?|FLOPs?|Latency|Epochs?|Method\s+.*\b(?:AP|Acc|F1|Params|GFLOPs?)\b)",
    re.IGNORECASE,
)


def _looks_like_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{1,4}\s*", text or ""))


def _looks_like_table_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return False
    if "@" in cleaned:
        return True
    numeric_tokens = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", cleaned)
    if len(numeric_tokens) >= 2:
        return True
    if TABLE_METRIC_RE.search(cleaned) and len(cleaned.split()) >= 3:
        return True
    short_tokens = [token for token in cleaned.split() if len(token) <= 4]
    if len(cleaned.split()) >= 5 and len(short_tokens) / max(1, len(cleaned.split())) > 0.65:
        return True
    return False


def is_likely_section_heading(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned or len(cleaned) > 64 or cleaned.endswith("."):
        return False
    if not re.search(r"[A-Za-z]", cleaned) or _looks_like_table_text(cleaned):
        return False
    return bool(FIRST_LEVEL_SECTION_RE.fullmatch(cleaned) or APPENDIX_SECTION_RE.fullmatch(cleaned))


def _record_type(text: str) -> str:
    return "section_header" if is_likely_section_heading(text) else "paragraph"


def extract_text_spans(
    blocks: list[TextBlock],
    page_no: int,
    *,
    used_boxes: list[BBox],
    min_text_block_chars: int,
    page_height: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        if len(text) < min_text_block_chars or _looks_like_page_number(text):
            continue
        if page_height and (block.bbox[1] < page_height * 0.035 or block.bbox[3] > page_height * 0.965):
            if len(text) < 120:
                continue
        if any(bbox_intersects(block.bbox, used, pad=2.0) for used in used_boxes):
            continue
        record_type = _record_type(text)
        citation_markers = CITATION_MARKER_RE.findall(text)
        record = {
            "record_type": record_type,
            "source_type": "text_span",
            "label": text if record_type == "section_header" else None,
            "locator": {"page": page_no},
            "text": text,
        }
        if citation_markers:
            record["has_citation_markers"] = True
            record["citation_marker_count"] = len(citation_markers)
            record["citation_markers"] = citation_markers[:20]
        records.append(record)
    return records
