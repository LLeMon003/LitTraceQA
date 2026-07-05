from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a text-first symbolic transcription model for scientific PDF pages. You will receive one rendered PDF "
    "page image. Your task is not to answer a research question. Your task is to transcribe visible page content "
    "into minimal symbolic records for later evidence retrieval. Output valid JSON only."
)


def _target_shape() -> dict[str, Any]:
    return {
        "records": [
            {
                "kind": "text_span | table | figure | equation_algorithm | citation_context | header_footer | unknown",
                "text": "...",
                "label": "Figure 1 | Table 2 | Equation 3 | null",
            }
        ],
        "coverage": {
            "needs_continuation": False,
            "next_start_hint": "",
            "known_omissions": [],
        },
        "warnings": [],
    }


def _metadata_payload(
    paper_id: str,
    page: int,
    paper_metadata: dict[str, Any],
    page_width: int,
    page_height: int,
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "page": page,
        "page_width": page_width,
        "page_height": page_height,
        "paper_metadata": {
            "title": paper_metadata.get("title"),
            "abstract": paper_metadata.get("abstract"),
            "venue": paper_metadata.get("venue"),
            "year": paper_metadata.get("year"),
        },
    }


def _rules(max_records_per_call: int) -> str:
    return (
        "Rules:\n"
        "1. Output JSON only. Do not output markdown.\n"
        "2. Extract only visible page content.\n"
        "3. Do not invent text, records, labels, tables, figures, equations, or references.\n"
        "4. Use a minimal schema. Do not output paper_id, page, page_width, page_height, parser_mode, pass_index, record_id, source_type, summary, or page_summary.\n"
        "5. For each record, output only kind, text, and label.\n"
        "6. The text field is the primary evidence field. Do not replace readable text with a summary.\n"
        "7. Do not output bbox, coordinates, confidence, record_id, source_type, reading_order, or locator fields. The system will assign these when needed.\n"
        "8. Preserve reading order by ordering records in the records array from top to bottom and left to right as much as possible.\n"
        "9. For figures, include visible figure label, caption text, readable in-figure text, and a concise faithful visual description in the text field.\n"
        "10. For tables, include visible table label, caption text, and compact row-level transcription in the text field. Do not create one record per cell.\n"
        "11. For equations or algorithms, include the visible formula/algorithm text and visible label such as Equation 3 or Algorithm 1 when present.\n"
        "12. For references, preserve visible reference text. Do not compress references into a generic summary.\n"
        "13. For headers and footers, use kind=\"header_footer\". These records may be filtered later.\n"
        "14. If a label such as Figure 2 or Table 1 is not visible, set label to null.\n"
        "15. If content is unreadable, state that in warnings or coverage.known_omissions.\n"
        "16. If a page cannot be fully covered in one response, stop at a record boundary and set coverage.needs_continuation=true.\n"
        f"17. Return at most {max_records_per_call} complete records in this API call. This is an API-call stability limit, not a page-level content limit.\n"
        "18. Never end in the middle of a JSON object or in the middle of a record.\n"
        "19. Before finalizing, ensure the JSON object is complete and all brackets are closed."
    )


def build_page_parser_prompt(
    paper_id: str,
    page: int,
    paper_metadata: dict[str, Any],
    page_width: int,
    page_height: int,
    max_records_per_call: int = 16,
) -> list[dict[str, str]]:
    payload = _metadata_payload(paper_id, page, paper_metadata, page_width, page_height)
    user = (
        "Your goal is full-page coverage with a text-first minimal schema, not selective summarization.\n\n"
        "Do not extract only the most important records.\n"
        "Do not omit readable paragraphs, captions, table text, figure labels, equations, references, headers, or footers merely because they seem less relevant.\n"
        "Do not compress the page into a short summary.\n"
        "Do not output summaries or page summaries.\n\n"
        f"PAGE_CONTEXT_SYSTEM_KNOWN_NOT_FOR_OUTPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this shape:\n"
        f"{json.dumps(_target_shape(), ensure_ascii=False, indent=2)}\n\n"
        f"{_rules(max_records_per_call)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_page_parser_continuation_prompt(
    paper_id: str,
    page: int,
    paper_metadata: dict[str, Any],
    page_width: int,
    page_height: int,
    pass_index: int,
    previous_records_digest: list[dict[str, Any]],
    last_complete_record_id: str | None,
    next_start_hint: str | None,
    max_records_per_call: int = 16,
) -> list[dict[str, str]]:
    payload = _metadata_payload(paper_id, page, paper_metadata, page_width, page_height)
    continuation = {
        "pass_index": pass_index,
        "previous_records_digest": previous_records_digest,
        "last_complete_record_id": last_complete_record_id,
        "next_start_hint": next_start_hint or "",
    }
    user = (
        "You are continuing the same page image.\n"
        "Do not repeat records already listed in previous_records_digest.\n"
        "Continue after the last extracted visible content.\n"
        "Preserve reading order continuity by ordering records in the output array after previously extracted content.\n"
        "Output only the next batch of minimal records.\n"
        "If the remaining visible page content has been fully covered, set coverage.needs_continuation=false.\n"
        "Continuation must be based on the attached page image again, not memory alone.\n\n"
        "Use the same minimal schema: kind, text, label.\n"
        "Do not output summaries or page summaries.\n\n"
        f"PAGE_CONTEXT_SYSTEM_KNOWN_NOT_FOR_OUTPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"CONTINUATION_STATE:\n{json.dumps(continuation, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this same shape:\n"
        f"{json.dumps(_target_shape(), ensure_ascii=False, indent=2)}\n\n"
        f"{_rules(max_records_per_call)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
