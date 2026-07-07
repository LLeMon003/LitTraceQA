from __future__ import annotations

import json
from typing import Any

from .data_io import extract_answer_contract


SYSTEM_PROMPT = (
    "You are a LitTraceQA answer model. You must answer using only the provided candidate papers, answer contract, "
    "and selected evidence records. The selected evidence records use official source_type values. Some records match "
    "the primary_evidence_type, and some records provide supporting context. Use supporting context when it helps, "
    "but do not invent evidence. Output valid JSON only."
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
        "label": evidence.get("label"),
        "text": evidence.get("text"),
    }
    if isinstance(evidence.get("grounding_label"), dict):
        projected["grounding_label"] = evidence.get("grounding_label")
    if evidence.get("image_ref"):
        projected["image_ref"] = evidence.get("image_ref")
    return projected


def _required_answer_shape(contract: dict[str, Any]) -> dict[str, Any]:
    shape: dict[str, Any] = {}
    answer_types = [str(item) for item in contract.get("answer_types", [])]
    if "freeform" in answer_types:
        shape["freeform"] = {"text": "<concise answer text>"}
    if "multiple_choice" in answer_types:
        option_keys = [
            str(option.get("key") or "")
            for option in (contract.get("multiple_choice") or {}).get("options", [])
            if isinstance(option, dict) and option.get("key")
        ]
        shape["multiple_choice"] = {"gold": f"<one of {option_keys}>" if option_keys else "<option key>"}
    if "table" in answer_types:
        columns = (contract.get("table") or {}).get("table_schema") or []
        if columns:
            shape["table"] = {"rows": [{str(column): "<value>" for column in columns}]}
        else:
            shape["table"] = {"rows": []}
    return shape


def build_symbolic_answer_prompt(
    input_example: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    selected_contexts: dict[str, Any],
    answer_model_supports_images: bool = False,
    parser_model: str = "",
    answer_model: str = "",
    answer_contract: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    contract = answer_contract or extract_answer_contract(input_example)
    required_answer_fields = [str(item) for item in contract.get("answer_types", [])]
    required_answer_shape = _required_answer_shape(contract)
    payload = {
        "query_id": input_example.get("query_id"),
        "task_family": input_example.get("task_family"),
        "primary_evidence_type": input_example.get("primary_evidence_type"),
        "question": input_example.get("question"),
        "answer_contract": contract,
        "required_answer_fields": required_answer_fields,
        "required_answer_shape": required_answer_shape,
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
        "You will receive selected symbolic evidence records. Each record contains only answer-facing fields: paper_id, page, source_type, label, text, "
        "optional grounding_label, and optional image_ref. source_type must be one of text_span, table, figure, equation_algorithm, citation_context. "
        "Ranking scores, retrieval scores, selector scores, parser confidence values, bbox, and internal record IDs are intentionally withheld from this prompt. "
        "Do not invent page numbers, table_id, figure_id, equation_id, algorithm_id, citation_id, image references, or hidden record IDs. Use only the provided evidence.\n\n"
        "You must follow answer_contract exactly. Output every answer field listed in required_answer_fields using required_answer_shape. Missing any field in required_answer_fields is invalid. If required_answer_fields includes both freeform and multiple_choice, output both fields. Do not treat multiple_choice as a replacement for freeform. Do not output freeform, multiple_choice, or table fields unless that answer type is explicitly listed. "
        "For multiple_choice, choose exactly one key from the provided options. Do not invent option keys. Do not choose a key that is not listed. Use the option text when reasoning, but output only the option key in answer.multiple_choice.gold. If multiple_choice is required but no options are provided, set answer.multiple_choice.gold to an empty string instead of guessing. "
        "For table answers, output rows using exactly the provided table_schema column names. Do not add, rename, or omit columns unless the schema explicitly allows it. "
        "For freeform, always use the object shape answer.freeform.text, for example \"freeform\": {\"text\": \"<concise answer>\"}. Do not output freeform as a bare string. Do not read or assume gold answers.\n\n"
        "This baseline does not use native PDF input and does not access online paper links, DOI pages, arXiv, OpenReview, or conference webpages during answer generation.\n"
        f"{image_note}\n"
        f"{partial_note}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "TARGET_JSON_SHAPE:\n"
        "{\n"
        '  "query_id": "<same query id>",\n'
        '  "gold_papers": [{"paper_id": "<predicted paper id>"}],\n'
        '  "evidence": [{"paper_id": "<paper id>", "source_type": "table | figure | text_span | equation_algorithm | citation_context", "locator": {"page": 1, "table_id": "Table 1"}}],\n'
        f'  "answer": {json.dumps(required_answer_shape, ensure_ascii=False)}\n'
        "}\n\n"
        "Rules:\n"
        "1. Output JSON only.\n"
        "2. Do not output markdown.\n"
        "3. query_id must match input.\n"
        "4. gold_papers must only use candidate paper_ids.\n"
        "5. evidence paper_id must be from candidate papers.\n"
        "6. evidence locator.page must come from selected_evidence for the same paper_id and source_type.\n"
        "7. Do not invent table_id, figure_id, equation_id, algorithm_id, citation_id, bbox, record_id, or page.\n"
        "8. For table/figure/equation/algorithm/citation labels, only output locator IDs that match selected_evidence.grounding_label.value. If label is null or no grounding_label is provided, do not output an ID.\n"
        "9. If required_answer_fields includes an answer type, include that answer field. If required_answer_fields does not include an answer type, omit that answer field.\n"
        "10. For table answers, use table_schema column names exactly.\n"
        "11. Numeric table values should be JSON numbers when possible.\n"
        "12. If selected evidence is insufficient, keep evidence sparse and avoid unsupported claims."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
