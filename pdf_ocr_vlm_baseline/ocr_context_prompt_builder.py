from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a LitTraceQA OCR-context baseline model. You will receive a research question, "
    "candidate paper metadata, and selected page-aware contexts extracted from candidate PDFs by an OCR module. "
    "Your job is to answer the question using only the provided metadata and selected contexts. "
    "Output valid JSON only. Do not include markdown or explanations outside JSON."
)


def build_ocr_context_answer_prompt(
    input_example: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    selected_contexts: dict[str, Any],
    answer_model_supports_images: bool = False,
) -> list[dict[str, str]]:
    payload = {
        "query_id": input_example.get("query_id"),
        "task_family": input_example.get("task_family"),
        "primary_evidence_type": input_example.get("primary_evidence_type"),
        "question": input_example.get("question"),
        "answer_types": input_example.get("answer_types", []),
        "table_schema": input_example.get("table_schema"),
        "candidate_papers": [
            {
                "paper_id": c.get("paper_id"),
                "title": c.get("title"),
                "abstract": c.get("abstract"),
                "retrieval_score": c.get("score", c.get("bm25_score")),
            }
            for c in candidate_records
        ],
        "selected_text_contexts": selected_contexts.get("selected_text_contexts", []),
        "selected_visual_contexts": selected_contexts.get("selected_visual_contexts", []),
        "answer_model_supports_images": answer_model_supports_images,
    }
    user = (
        "Use only the provided candidate metadata and selected OCR contexts. The context units are page-aware "
        "and identified by paper_id, page, and chunk_id. If visual contexts are provided and the answer model "
        "supports image input, use the attached images as additional evidence. If the answer model does not "
        "support image input, use the visual context metadata only and do not claim to have inspected the images.\n\n"
        "Current baseline uses OCR-converted contexts and does not use native PDF input. Return valid JSON only. "
        "Use only candidate paper IDs. Do not invent bbox, table_id, figure_id, or equation_id.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "TARGET_JSON_SHAPE:\n"
        "{\n"
        '  "query_id": "<same query id>",\n'
        '  "gold_papers": [{"paper_id": "<predicted paper id>"}],\n'
        '  "evidence": [{"paper_id": "<paper id>", "source_type": "text_span", "locator": {"page": 1, "chunk_id": "p001_c001"}}],\n'
        '  "answer": {"freeform": {"text": "..."}, "multiple_choice": {"gold": "A"}, "table": {"rows": []}},\n'
        '  "confidence": {"paper_retrieval": 0.0, "context_grounding": 0.0, "answer": 0.0},\n'
        '  "notes": {"baseline_type": "pdf_ocr_context_vlm", "ocr_model": "deepseek-ai/DeepSeek-OCR", "context_selection": "ocr_chunk_lexical_without_embedding", "limitations": ""}\n'
        "}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
