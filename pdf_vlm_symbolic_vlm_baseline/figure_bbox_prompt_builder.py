from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a visual grounding model for scientific PDF pages. You will receive one rendered PDF page image "
    "and a list of figure records previously transcribed from the same page. Your task is to locate each visible "
    "figure on the page and return approximate bounding boxes. Output valid JSON only."
)


def _target_shape() -> dict[str, Any]:
    return {
        "figures": [
            {
                "record_id": "p001_r0001",
                "label": "Figure 1 | Fig. 2 | null",
                "bbox_1000": [0, 0, 1000, 1000],
                "confidence": 0.0,
                "notes": "",
            }
        ],
        "warnings": [],
    }


def build_figure_bbox_prompt(
    *,
    paper_id: str,
    page: int,
    page_width: int,
    page_height: int,
    figure_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "paper_id": paper_id,
        "page": page,
        "page_width": page_width,
        "page_height": page_height,
        "figure_records_to_locate": [
            {
                "record_id": record.get("record_id"),
                "label": record.get("label"),
                "locator": record.get("locator"),
                "text_preview": str(record.get("text") or "")[:700],
            }
            for record in figure_records
        ],
    }
    user = (
        "Locate only the figures listed in figure_records_to_locate on the attached page image.\n"
        "Return one item per located figure. If a listed record is not visibly locatable, omit it and explain in warnings.\n\n"
        "Bounding box rules:\n"
        "1. Use bbox_1000 as normalized coordinates [x1, y1, x2, y2] on a 0-1000 page scale.\n"
        "2. The bbox should enclose the full visible figure region, including panels, axes, legends, in-figure labels, and adjacent caption when it is visually part of the figure block.\n"
        "3. Do not include unrelated body paragraphs, neighboring tables, or other figures.\n"
        "4. Preserve the input record_id exactly so the system can attach the bbox to the existing symbolic record.\n"
        "5. Do not invent figures or labels. Do not output text transcription. Do not output markdown.\n"
        "6. If multiple listed records refer to the same visible figure block, return the same bbox for each matching record_id.\n"
        "7. Output complete valid JSON only.\n\n"
        f"PAGE_AND_RECORD_CONTEXT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON with this shape:\n"
        f"{json.dumps(_target_shape(), ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
