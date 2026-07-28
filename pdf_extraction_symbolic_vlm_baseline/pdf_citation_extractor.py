from __future__ import annotations

import re

from .pdf_layout_blocks import BBox, TextBlock, bbox_list


REFERENCE_HEADING_RE = re.compile(r"^\s*(References|Bibliography)\s*$", re.IGNORECASE)
REFERENCE_ENTRY_RE = re.compile(r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+(.+)")
AUTHOR_YEAR_ENTRY_RE = re.compile(r"^[A-Z].{8,600}\b(?:19|20)\d{2}[a-z]?\b")
BIBLIOGRAPHIC_CUE_RE = re.compile(
    r"\b(?:arXiv|preprint|Proceedings|Conference|Workshop|Transactions|Association for Computational Linguistics|pages?|volume|journal)\b",
    re.IGNORECASE,
)
NON_REFERENCE_HEADING_RE = re.compile(
    r"^(?:Limitations?|Ethics Statement|Availability Statement|Acknowledgements?|Appendix|[A-Z]\.\d+)\b",
    re.IGNORECASE,
)
DATA_ROW_RE = re.compile(r"^(?:Number of|Total steps|Warm-up steps|Learning rate|Batch size|Optimizer|Source of)\b", re.IGNORECASE)


def _initial_state(references_active: bool | dict[str, object]) -> dict[str, object]:
    if isinstance(references_active, dict):
        return {"active": bool(references_active.get("active")), "next_id": int(references_active.get("next_id") or 1)}
    return {"active": bool(references_active), "next_id": 1}


def _next_reference_id(state: dict[str, object]) -> int:
    ref_id = int(state.get("next_id") or 1)
    state["next_id"] = ref_id + 1
    return ref_id


def extract_citations(blocks: list[TextBlock], page_no: int, *, references_active: bool | dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[BBox], dict[str, object]]:
    runtime_like: list[dict[str, object]] = []
    debug: list[dict[str, object]] = []
    used_boxes: list[BBox] = []
    state = _initial_state(references_active)
    active = bool(state.get("active"))
    for block in blocks:
        text = re.sub(r"\s+", " ", block.text).strip()
        if not text:
            continue
        if REFERENCE_HEADING_RE.match(text):
            active = True
            runtime_like.append(
                {
                    "record_type": "section_header",
                    "source_type": "text_span",
                    "label": text,
                    "locator": {"page": page_no},
                    "text": text,
                }
            )
            debug.append(
                {
                    "record_type": "section_header",
                    "source_type": "text_span",
                    "label": text,
                    "bbox": bbox_list(block.bbox),
                    "extraction_method": "references_section_heading",
                }
            )
            used_boxes.append(block.bbox)
            continue
        author_year_match = AUTHOR_YEAR_ENTRY_RE.match(text)
        if DATA_ROW_RE.match(text):
            continue
        if not active and not (page_no >= 5 and author_year_match and BIBLIOGRAPHIC_CUE_RE.search(text)):
            continue
        if NON_REFERENCE_HEADING_RE.match(text):
            continue
        match = REFERENCE_ENTRY_RE.match(text)
        if match:
            ref_id = int(match.group(1) or match.group(2))
            body = match.group(3).strip()
        elif author_year_match:
            ref_id = _next_reference_id(state)
            body = text
        elif runtime_like and runtime_like[-1].get("record_type") == "reference_entry":
            previous_text = str(runtime_like[-1].get("text") or "")
            if text[:1].islower() or (len(previous_text) < 90 and not NON_REFERENCE_HEADING_RE.match(text)):
                runtime_like[-1]["text"] = f"{previous_text} {text}".strip()
                debug[-1]["bbox_continuation"] = [*debug[-1].get("bbox_continuation", []), bbox_list(block.bbox)]
                used_boxes.append(block.bbox)
            continue
        else:
            continue
        label = f"Reference {ref_id}"
        runtime_like.append(
            {
                "record_type": "reference_entry",
                "source_type": "citation_context",
                "label": label,
                "locator": {"page": page_no, "citation_id": ref_id},
                "text": f"[{ref_id}] {body}",
            }
        )
        debug.append(
            {
                "record_type": "reference_entry",
                "source_type": "citation_context",
                "label": label,
                "bbox": bbox_list(block.bbox),
                "extraction_method": "numbered_references_section_entry" if match else "author_year_references_section_entry",
            }
        )
        used_boxes.append(block.bbox)
    state["active"] = active
    return runtime_like, debug, used_boxes, state
