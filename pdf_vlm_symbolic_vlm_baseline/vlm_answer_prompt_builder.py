from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a LitTraceQA symbolic-context answer model. You will receive a research question, candidate paper "
    "metadata, and selected structured symbolic records extracted from rendered PDF pages by a separate VLM parser. "
    "Your task is to answer the question using only the provided metadata and selected symbolic records. Output valid "
    "JSON only. Do not include markdown or explanations outside JSON."
)


def _project_candidate_for_prompt(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": candidate.get("paper_id"),
        "title": candidate.get("title"),
        "abstract": candidate.get("abstract"),
    }


def _project_evidence_for_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "paper_id": evidence.get("paper_id"),
        "page": evidence.get("page"),
        "source_type": evidence.get("source_type"),
        "locator": evidence.get("locator") or {"page": evidence.get("page")},
        "text": evidence.get("text"),
    }
    if evidence.get("image_ref"):
        projected["image_ref"] = evidence.get("image_ref")
    return projected


def build_symbolic_answer_prompt(
    input_example: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    selected_contexts: dict[str, Any],
    answer_model_supports_images: bool = False,
    parser_model: str = "",
    answer_model: str = "",
) -> list[dict[str, str]]:
    payload = {
        "query_id": input_example.get("query_id"),
        "task_family": input_example.get("task_family"),
        "primary_evidence_type": input_example.get("primary_evidence_type"),
        "question": input_example.get("question"),
        "answer_types": input_example.get("answer_types", []),
        "table_schema": input_example.get("table_schema"),
        "candidate_papers": [_project_candidate_for_prompt(c) for c in candidate_records],
        "selected_evidence": [
            _project_evidence_for_prompt(evidence)
            for evidence in selected_contexts.get("selected_evidence", [])
            if isinstance(evidence, dict)
        ],
        "has_partial_artifacts": bool(selected_contexts.get("has_partial_artifacts") or selected_contexts.get("partial_artifacts_present")),
        "attached_image_refs": selected_contexts.get("attached_image_refs", []),
    }
    has_partial = bool(selected_contexts.get("has_partial_artifacts") or selected_contexts.get("partial_artifacts_present"))
    has_images = answer_model_supports_images and bool(selected_contexts.get("attached_image_refs"))
    image_note = (
        "Some selected images are attached. image_ref identifies an attached image in this request. Do not treat local file paths as accessible evidence."
        if has_images
        else "No images are attached to this answer model call. Use only the symbolic records and metadata. Do not claim to have inspected page images directly."
    )
    partial_note = (
        "Some parser artifacts are marked partial. Use them cautiously and avoid overclaiming unsupported evidence."
        if has_partial
        else "Selected parser artifacts are not marked partial."
    )
    user = (
        "Use only the provided candidate metadata and selected symbolic evidence records. These records were generated from rendered PDF page images by a separate VLM parser "
        "and validated by a symbolic layer.\n\n"
        "You will receive selected symbolic evidence records. Each record contains only answer-facing fields: paper_id, page, source_type, locator, text, "
        "and optionally image_ref. locator is assigned by the symbolic system from visible labels/text and should be copied when it supports the answer. "
        "Ranking scores, retrieval scores, selector scores, parser confidence values, bbox, and internal record IDs are intentionally withheld from this prompt. "
        "Do not invent page numbers, table_id, figure_id, equation_id, algorithm_id, citation_id, image references, or hidden record IDs. Use only the provided evidence.\n\n"
        "This baseline does not use native PDF input and does not access online paper links, DOI pages, arXiv, OpenReview, or conference webpages during answer generation.\n"
        f"{image_note}\n"
        f"{partial_note}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "TARGET_JSON_SHAPE:\n"
        "{\n"
        '  "query_id": "<same query id>",\n'
        '  "gold_papers": [{"paper_id": "<predicted paper id>"}],\n'
        '  "evidence": [{"paper_id": "<paper id>", "source_type": "table | figure | text_span | equation_algorithm | citation_context", "locator": {"page": 1, "table_id": "Table 1"}}],\n'
        '  "answer": {"freeform": {"text": "..."}, "multiple_choice": {"gold": "A"}, "table": {"rows": []}}\n'
        "}\n\n"
        "Rules:\n"
        "1. Output JSON only.\n"
        "2. Do not output markdown.\n"
        "3. query_id must match input.\n"
        "4. gold_papers must only use candidate paper_ids.\n"
        "5. evidence paper_id must be from candidate papers.\n"
        "6. evidence locator should copy the provided locator for supporting evidence.\n"
        "7. Do not invent table_id, figure_id, equation_id, algorithm_id, citation_id, bbox, record_id, or page.\n"
        "8. For table evidence, include locator.page and locator.table_id when provided. For figure evidence, include locator.page and locator.figure_id when provided. Other evidence types need locator.page.\n"
        "9. If answer_types does not include an answer type, omit that answer field.\n"
        "10. For table answers, use table_schema column names exactly.\n"
        "11. Numeric table values should be JSON numbers when possible.\n"
        "12. If selected evidence is insufficient, keep evidence sparse and avoid unsupported claims."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
