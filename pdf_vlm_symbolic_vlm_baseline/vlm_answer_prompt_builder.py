from __future__ import annotations

import json
import re
from typing import Any

from .data_io import extract_answer_contract
from .table_structure import table_text_to_structure


SYSTEM_PROMPT = (
    "You are a LitTraceQA answer model. You must answer using only the provided candidate papers, answer contract, "
    "and selected evidence packets. The selected evidence packets use official source_type values. Some packets match "
    "the primary_evidence_type, and some packets provide supporting context. Use supporting context when it helps, "
    "but do not invent evidence. Output valid JSON only."
)


ANSWER_STYLE_GUIDE = (
    "Answer style guide inferred from the public validation answer format, expressed only as generic formatting rules:\n"
    "- Default freeform style is extractive and short. Most freeform answers are a number, an integer count, a yes/no value, an entity name, a method name, a dataset name, an author name, a paper title, or a short phrase.\n"
    "- For numeric questions such as how much, by how much, score, accuracy, F1, AP, NRMSE, standard deviation, count, number of references, number of panels, or number of parentheses, answer.freeform.text should contain only the final number unless the question explicitly asks for units or a full sentence.\n"
    "- For which/who/what entity questions, answer.freeform.text should contain only the entity or short span, not a sentence explaining where it appeared.\n"
    "- For yes/no questions, answer.freeform.text should usually be Yes or No. Add a short clause only if the question asks for a comparison statement rather than a bare yes/no.\n"
    "- When both freeform and multiple_choice are required, answer.freeform.text should normally be the short answer content corresponding to answer.multiple_choice.gold, not a rationale.\n"
    "- Use sentence-length freeform only for broad synthesis questions asking across papers, among methods, what happens/trend, what base model each method uses, or requests that naturally require a list or mapping. Even then, keep it concise and answer-first.\n"
    "- For table answers, use the exact table_schema columns. Row keys should be concise canonical labels; numeric cells should be JSON numbers when possible.\n"
)


def _query_answer_style_guidance(input_example: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    question = str(input_example.get("question") or "")
    lowered = question.lower()
    answer_types = [str(item) for item in contract.get("answer_types", [])]
    guidance: dict[str, Any] = {
        "default_freeform_granularity": "minimal_answer_span",
        "freeform_rule": "Use the shortest answer text that would still be correct under exact string matching.",
    }
    if "freeform" not in answer_types:
        return guidance

    numeric_patterns = [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bby how much\b",
        r"\bnumber of\b",
        r"\bindex of\b",
        r"\bwhat (?:is|was|are|were)\b.*\b(?:score|accuracy|f1|ap|nrmse|standard deviation|success rate)\b",
        r"\b(?:score|accuracy|f1|ap|nrmse|standard deviation|success rate)\b.*\b(?:achieved by|of|for)\b",
        r"\bparentheses\b",
        r"\bsubfigures?\b",
        r"\bpanels?\b",
        r"\bnumber of\s+references?\b",
    ]
    entity_patterns = [
        r"^\s*who\b",
        r"^\s*which\b",
        r"^\s*what (?:is|are|was|were|kind|dataset|model|method|feature|base model|backbone)\b",
        r"\bwhich (?:system|model|method|dataset|task|paper|feature|author)\b",
        r"\bwhat (?:dataset|model|method|feature|hardware|index|base model|backbone)\b",
    ]
    yes_no_patterns = [
        r"(?:^|[,;]\s*)(?:does|do|did|is|are|was|were|can|could|has|have|had)\b",
    ]
    broad_patterns = [
        r"\bacross all\b",
        r"\bamong\b.*\bwhat\b.*\beach\b",
        r"\bwhat base model does each\b",
        r"\bwhat happens\b",
        r"\bas .* increases\b",
        r"\btrend\b",
        r"\bwhich .* papers\b",
        r"\blist\b",
        r"\beach method\b",
        r"\brespectively\b",
    ]

    if any(re.search(pattern, lowered) for pattern in broad_patterns):
        guidance.update(
            {
                "freeform_granularity": "concise_sentence_or_list",
                "freeform_rule": "Use an answer-first concise sentence or compact list/mapping. Include only information requested by the question.",
                "avoid": "Do not include reasoning, provenance narration, or unrelated evidence details.",
            }
        )
    elif any(re.search(pattern, lowered) for pattern in yes_no_patterns):
        guidance.update(
            {
                "freeform_granularity": "yes_no_or_short_comparison",
                "freeform_rule": "Prefer only Yes or No. If the question asks a comparison with named entities, a short answer-first clause is acceptable.",
                "examples": ["Yes", "No"],
            }
        )
    elif any(re.search(pattern, lowered) for pattern in numeric_patterns):
        guidance.update(
            {
                "freeform_granularity": "number_or_integer_only",
                "freeform_rule": "Output only the final numeric value. For counts, output only the integer. Do not add units or explanatory words unless the question explicitly asks for them.",
                "examples": ["14.70", "8", "0.09"],
            }
        )
    elif any(re.search(pattern, lowered) for pattern in entity_patterns):
        guidance.update(
            {
                "freeform_granularity": "entity_or_short_phrase",
                "freeform_rule": "Output only the requested entity, method, model, dataset, task, author, paper title, feature, or short phrase.",
                "avoid": "Do not wrap the span in a sentence such as 'The method is ...'.",
            }
        )
    else:
        guidance.update(
            {
                "freeform_granularity": "short_phrase_by_default",
                "freeform_rule": "Use a short phrase unless the question clearly requires a sentence.",
            }
        )

    if "multiple_choice" in answer_types:
        guidance["multiple_choice_alignment"] = "If multiple_choice is also required, choose the option key and make freeform match the option's answer content at the same granularity."
    if "table" in answer_types:
        guidance["table_alignment"] = "If table is also required, keep freeform to the single requested answer or compact summary; put structured rows in answer.table.rows."
    return guidance


def _is_multi_paper_task(input_example: dict[str, Any], contract: dict[str, Any]) -> bool:
    task_family = str(input_example.get("task_family") or "").lower()
    if "multi" in task_family:
        return True
    question = str(input_example.get("question") or "").lower()
    if "table" in [str(item) for item in contract.get("answer_types", [])]:
        return True
    patterns = [
        r"\bacross (?:all|the) papers\b",
        r"\bwhich .* papers\b",
        r"\bwhat .* each\b",
        r"\blist\b.*\bpapers?\b",
        r"\bcompare[sd]?\b",
        r"\bamong\b.*\bpapers?\b",
        r"\brespectively\b",
    ]
    return any(re.search(pattern, question) for pattern in patterns)


def _project_candidate_for_prompt(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": candidate.get("paper_id"),
        "title": candidate.get("title"),
        "abstract": candidate.get("abstract"),
    }


def _project_evidence_for_prompt(evidence: dict[str, Any], contract: dict[str, Any] | None = None) -> dict[str, Any]:
    projected = {
        "paper_id": evidence.get("paper_id"),
        "page": evidence.get("page"),
        "source_type": evidence.get("source_type"),
        "label": evidence.get("label"),
        "text": evidence.get("text"),
    }
    if projected["source_type"] == "table":
        table_schema = ((contract or {}).get("table") or {}).get("table_schema") or []
        table_structure = table_text_to_structure(str(evidence.get("text") or ""), table_schema)
        if table_structure:
            projected["table_structure"] = table_structure
    if isinstance(evidence.get("grounding_label"), dict):
        projected["grounding_label"] = evidence.get("grounding_label")
    if isinstance(evidence.get("source_type_hints"), list) and evidence.get("source_type_hints"):
        projected["source_type_hints"] = evidence.get("source_type_hints")
    if evidence.get("image_ref"):
        projected["image_ref"] = evidence.get("image_ref")
    return projected


def _clip_text(text: Any, max_chars: int = 1800) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _packet_label(evidence: dict[str, Any]) -> str:
    label = str(evidence.get("label") or "").strip()
    if label:
        return label
    grounding = evidence.get("grounding_label")
    if isinstance(grounding, dict):
        value = str(grounding.get("value") or "").strip()
        if value:
            return value
    return ""


def _packet_key(evidence: dict[str, Any]) -> tuple[str, int, str]:
    try:
        page = int(evidence.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    return (str(evidence.get("source_type") or "text_span"), page, _packet_label(evidence))


def _project_supporting_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "source_type": evidence.get("source_type"),
        "page": evidence.get("page"),
        "label": _packet_label(evidence) or evidence.get("label"),
        "text": _clip_text(evidence.get("text"), 700),
    }
    if isinstance(evidence.get("grounding_label"), dict):
        projected["grounding_label"] = evidence.get("grounding_label")
    if isinstance(evidence.get("source_type_hints"), list) and evidence.get("source_type_hints"):
        projected["source_type_hints"] = evidence.get("source_type_hints")
    return projected


def _build_evidence_packets(
    selected_evidence: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    contract: dict[str, Any],
    primary_evidence_type: str,
) -> list[dict[str, Any]]:
    candidate_by_id = {str(candidate.get("paper_id") or ""): _project_candidate_for_prompt(candidate) for candidate in candidate_records}
    records_by_paper: dict[str, list[dict[str, Any]]] = {}
    for evidence in selected_evidence:
        if not isinstance(evidence, dict):
            continue
        paper_id = str(evidence.get("paper_id") or "")
        if not paper_id:
            continue
        records_by_paper.setdefault(paper_id, []).append(evidence)

    paper_packets: list[dict[str, Any]] = []
    for paper_id, records in records_by_paper.items():
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for evidence in records:
            grouped.setdefault(_packet_key(evidence), []).append(evidence)

        packet_rows: list[dict[str, Any]] = []
        for (source_type, page, label), group in grouped.items():
            primary = group[0]
            same_page_support = [
                item
                for item in records
                if item is not primary
                and item not in group
                and item.get("page") == page
                and str(item.get("source_type") or "") != source_type
            ]
            packet: dict[str, Any] = {
                "paper_id": paper_id,
                "primary_source_type": source_type,
                "page": page,
                "label": label or primary.get("label"),
                "primary_text": _clip_text(primary.get("text"), 2200 if source_type == "table" else 1500),
                "supporting_text": [_clip_text(item.get("text"), 650) for item in same_page_support[:4] if str(item.get("text") or "").strip()],
                "supporting_evidence": [_project_supporting_evidence(item) for item in same_page_support[:4]],
            }
            if isinstance(primary.get("grounding_label"), dict):
                packet["grounding_label"] = primary.get("grounding_label")
            if isinstance(primary.get("source_type_hints"), list) and primary.get("source_type_hints"):
                packet["source_type_hints"] = primary.get("source_type_hints")
            if source_type == "table":
                table_schema = (contract.get("table") or {}).get("table_schema") or []
                table_structure = table_text_to_structure(str(primary.get("text") or ""), table_schema)
                if table_structure:
                    packet["table_structure"] = table_structure
            packet["_rank_source_type_match"] = source_type == primary_evidence_type
            packet_rows.append(packet)

        packet_rows.sort(
            key=lambda item: (
                not bool(item.get("_rank_source_type_match")),
                int(item.get("page") or 0),
                str(item.get("primary_source_type") or ""),
                str(item.get("label") or ""),
            )
        )
        for packet in packet_rows:
            packet.pop("_rank_source_type_match", None)
        paper_packets.append(
            {
                "paper_id": paper_id,
                "title": (candidate_by_id.get(paper_id) or {}).get("title"),
                "abstract": (candidate_by_id.get(paper_id) or {}).get("abstract"),
                "evidence_packets": packet_rows,
            }
        )

    candidate_order = {str(candidate.get("paper_id") or ""): index for index, candidate in enumerate(candidate_records)}
    paper_packets.sort(key=lambda item: candidate_order.get(str(item.get("paper_id") or ""), 10**6))
    return paper_packets


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
    answer_style_guidance = _query_answer_style_guidance(input_example, contract)
    multi_paper_task = _is_multi_paper_task(input_example, contract)
    selected_evidence = [
        evidence
        for evidence in selected_contexts.get("selected_evidence", [])
        if isinstance(evidence, dict)
    ]
    paper_evidence_packets = _build_evidence_packets(
        selected_evidence,
        candidate_records,
        contract,
        str(input_example.get("primary_evidence_type") or ""),
    )
    payload = {
        "query_id": input_example.get("query_id"),
        "task_family": input_example.get("task_family"),
        "primary_evidence_type": input_example.get("primary_evidence_type"),
        "question": input_example.get("question"),
        "answer_contract": contract,
        "answer_style_guidance": answer_style_guidance,
        "multi_paper_contribution_required": multi_paper_task,
        "required_answer_fields": required_answer_fields,
        "required_answer_shape": required_answer_shape,
        "candidate_papers": [_project_candidate_for_prompt(c) for c in candidate_records],
        "selected_evidence_record_count": len(selected_evidence),
        "evidence_packet_count": sum(len(item.get("evidence_packets", [])) for item in paper_evidence_packets),
        "paper_evidence_packets": paper_evidence_packets,
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
        "Use only the provided candidate metadata and selected symbolic evidence packets. These packets were generated from rendered PDF page images by a separate VLM parser "
        "and validated by a symbolic layer.\n\n"
        "You will receive paper_evidence_packets grouped by paper_id. Each evidence packet contains primary_source_type, page, label, primary_text, optional supporting_text, optional supporting_evidence, optional grounding_label, optional source_type_hints, and for some table packets an optional table_structure derived from the table text. primary_source_type must be one of text_span, table, figure, equation_algorithm, citation_context. source_type_hints mark cases where the same packet text visibly contains another official evidence type, for example a text_span that contains a table caption or equation. "
        "Ranking scores, retrieval scores, selector scores, parser confidence values, bbox, and internal record IDs are intentionally withheld from this prompt. "
        "Do not invent page numbers, table_id, figure_id, equation_id, algorithm_id, citation_id, image references, or hidden record IDs. Use only the provided evidence.\n\n"
        + (
            "This is a multi-paper or cross-paper task. Identify all papers that contribute evidence to the final answer. Do not output only the single most relevant paper if multiple papers support the answer. "
            "In addition to official fields, include an internal contributing_papers array. Each item should contain paper_id, supporting_evidence, and a short contribution. Use it to make gold_papers complete across all contributing papers.\n\n"
            if multi_paper_task
            else ""
        )
        + "You must follow answer_contract exactly. Output every answer field listed in required_answer_fields using required_answer_shape. Missing any field in required_answer_fields is invalid. If required_answer_fields includes both freeform and multiple_choice, output both fields. Do not treat multiple_choice as a replacement for freeform. Do not output freeform, multiple_choice, or table fields unless that answer type is explicitly listed. "
        "For multiple_choice, choose exactly one key from the provided options. Multiple-choice means single-choice in this benchmark: output one option key, not several keys and not option text. Do not invent option keys. Do not choose a key that is not listed. Before choosing, compare every provided option against the selected evidence and the question: reject options contradicted by evidence, prefer options directly supported by evidence, and among partially supported options choose the one that is most specific and most directly answers the question. If options are provided, do not leave answer.multiple_choice.gold empty; choose the best-supported or closest option even when evidence is imperfect. Use the option text when reasoning, but output only the final option key in answer.multiple_choice.gold. If multiple_choice is required but no options are provided, return only a letter A/B/C/D when genuinely confident; otherwise set answer.multiple_choice.gold to an empty string. "
        "For table answers, do not rely on freely writing a final table from memory. Select relevant table packets, rows, or cells and output an internal table_answer_plan. Each table_answer_plan item must have row_source with paper_id, page, and label, plus values keyed by the requested table_schema column names. The system will assemble answer.table.rows from table_answer_plan. You may also include answer.table.rows as a best-effort copy, but table_answer_plan is the authoritative table plan. If a table packet has table_structure, use its header_rows, columns, rows, and cells to resolve row/column alignment while keeping the original text as the source of truth. "
        "For freeform, always use the object shape answer.freeform.text, for example \"freeform\": {\"text\": \"<concise answer>\"}. Do not output freeform as a bare string. Freeform must be the shortest final answer that directly satisfies the question, not a rationale or explanatory sentence. If the answer is a number, output only the number unless the question explicitly asks for units. If the answer is a count, output only the integer. If the answer is an entity, method, dataset, author, paper title, option text, or short phrase, output only that exact short span. If the answer is yes/no, output only Yes or No. If both freeform and multiple_choice are required, make freeform the short answer content corresponding to the chosen option, not an explanation of why that option is correct. Do not read or assume gold answers.\n\n"
        f"{ANSWER_STYLE_GUIDE}\n"
        "This baseline does not use native PDF input and does not access online paper links, DOI pages, arXiv, OpenReview, or conference webpages during answer generation.\n"
        f"{image_note}\n"
        f"{partial_note}\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "TARGET_JSON_SHAPE:\n"
        "{\n"
        '  "query_id": "<same query id>",\n'
        '  "gold_papers": [{"paper_id": "<predicted paper id>"}],\n'
        + (
            '  "contributing_papers": [{"paper_id": "<candidate paper id>", "supporting_evidence": [{"page": 1, "source_type": "table", "label": "Table 1"}], "contribution": "<how this paper supports the final answer>"}],\n'
            if multi_paper_task
            else ""
        )
        + (
            '  "table_answer_plan": [{"row_source": {"paper_id": "<paper id>", "page": 1, "label": "Table 1"}, "values": {"<table_schema column>": "<value>"}}],\n'
            if "table" in required_answer_fields
            else ""
        )
        + '  "evidence": [{"paper_id": "<paper id>", "source_type": "table | figure | text_span | equation_algorithm | citation_context", "locator": {"page": 1, "table_id": "Table 1"}}],\n'
        f'  "answer": {json.dumps(required_answer_shape, ensure_ascii=False)}\n'
        "}\n\n"
        "Rules:\n"
        "1. Output JSON only.\n"
        "2. Do not output markdown.\n"
        "3. query_id must match input.\n"
        "4. gold_papers must only use candidate paper_ids.\n"
        + (
            "4a. For multi-paper tasks, contributing_papers must list every candidate paper that contributes evidence to the final answer. gold_papers must include every contributing_papers.paper_id. Do not collapse multiple supporting papers into only one paper.\n"
            if multi_paper_task
            else ""
        )
        + "5. evidence paper_id must be from candidate papers.\n"
        "6. evidence locator.page must come from paper_evidence_packets for the same paper_id. evidence.source_type should normally equal primary_source_type, but it may use a source_type_hints.source_type from the same packet when the hint better matches the evidence needed by the question.\n"
        "7. Do not invent table_id, figure_id, equation_id, algorithm_id, citation_id, bbox, record_id, or page.\n"
        "8. For table/figure/equation/algorithm/citation labels, only output locator IDs that match evidence packet grounding_label.value or a locator value inside source_type_hints for the same packet. If label is null and no grounding_label or source_type_hints locator is provided, do not output an ID.\n"
        "9. If required_answer_fields includes an answer type, include that answer field. If required_answer_fields does not include an answer type, omit that answer field.\n"
        "10. For table answers, use table_schema column names exactly.\n"
        "11. For table evidence packets with table_structure, align values by table_structure.columns and table_structure.rows before answering.\n"
        "12. For multiple_choice with options, internally evaluate all options first, then output exactly one option key. Never output an empty key when options are present.\n"
        "13. For freeform answers, obey answer_style_guidance.freeform_granularity and answer_style_guidance.freeform_rule. Output the minimal gold-style answer span unless answer_style_guidance explicitly permits a concise sentence/list.\n"
        "14. If a question asks for a single value but both freeform and table are required, keep freeform to the single value or short span and put structured details in answer.table.rows.\n"
        "15. Numeric table values should be JSON numbers when possible.\n"
        "16. If table is required, output table_answer_plan. Each plan row must cite row_source.paper_id, row_source.page, row_source.label, and values for the requested columns. Do not invent columns outside table_schema.\n"
        "17. If selected evidence is insufficient, keep evidence sparse and avoid unsupported claims, but still choose the closest multiple-choice option when options are provided."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
