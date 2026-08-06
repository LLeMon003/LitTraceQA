from __future__ import annotations

import json
import re
from typing import Any

from .data_io import extract_answer_contract
from .evidence_hierarchy import hierarchy_prompt_projection, keyed_hierarchy_prompt_projection
from .table_structure import table_text_to_structure


SYSTEM_PROMPT = (
    "Answer only from INPUT evidence and candidate papers. Do not invent facts or evidence. Return valid JSON only."
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
        table_structure = evidence.get("table_structure") if isinstance(evidence.get("table_structure"), dict) else table_text_to_structure(str(evidence.get("text") or ""), table_schema)
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
        "section_title": evidence.get("section_title"),
        "section_type": evidence.get("section_type"),
        "section_path": evidence.get("section_path"),
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
                "section_title": primary.get("section_title"),
                "section_type": primary.get("section_type"),
                "section_path": primary.get("section_path"),
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
                table_structure = primary.get("table_structure") if isinstance(primary.get("table_structure"), dict) else table_text_to_structure(str(primary.get("text") or ""), table_schema)
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


def _compact_evidence_ledger(selected_evidence: list[dict[str, Any]]) -> str:
    """Serialize selected evidence once, with provenance kept in the caller.

    The answer model needs a stable evidence reference and enough local context
    to answer the query.  It does not need the full repeated package/section
    JSON because the parser restores the official locator from ``evidence_ref``
    after generation.  This ledger is deliberately extractive: text is never
    rewritten here, so a tighter serialization cannot introduce a new fact.
    """
    papers: dict[str, list[dict[str, Any]]] = {}
    for item in selected_evidence:
        paper_id = str(item.get("paper_id") or "unknown")
        papers.setdefault(paper_id, []).append(item)

    lines: list[str] = []
    for paper_id, rows in papers.items():
        lines.append(f"PAPER {paper_id}")
        current_section: tuple[str, str] | None = None
        for item in rows:
            section = (str(item.get("section_id") or ""), str(item.get("section_title") or ""))
            if section != current_section:
                current_section = section
                section_text = section[1] or section[0] or "unsectioned"
                lines.append(f"SECTION {section_text}")
            ref = str(item.get("evidence_ref") or "")
            page = item.get("page")
            source_type = str(item.get("source_type") or "text_span")
            label = str(item.get("label") or "")
            header = f"{ref}\tp{page}\t{source_type}"
            if label:
                header += f"\t{label}"
            text = str(item.get("text") or "").strip().replace("\x00", "")
            lines.append(f"{header}\t{text}")
    return "\n".join(lines)


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
    # A long multi-paper response previously repeated provenance in both
    # evidence and contributing_papers, which can exhaust the completion budget
    # before JSON closes. Evidence is an audit trail, not a transcript.
    evidence_output_limit = 16 if multi_paper_task else 12
    table_plan_limit = 16 if multi_paper_task else 12
    selected_evidence = [
        evidence
        for evidence in selected_contexts.get("selected_evidence", [])
        if isinstance(evidence, dict)
    ]
    compact_chunk_packets = [
        packet
        for packet in selected_contexts.get("compact_chunk_packets", [])
        if isinstance(packet, dict)
    ]
    evidence_ledger = str(selected_contexts.get("evidence_ledger") or "").strip()
    evidence_hierarchy = selected_contexts.get("evidence_hierarchy") if isinstance(selected_contexts.get("evidence_hierarchy"), dict) else None
    keyed_mode = bool(evidence_hierarchy and str(evidence_hierarchy.get("prompt_mode") or "") == "keyed_l2_only")
    if evidence_hierarchy or evidence_ledger:
        # The ledger is a deliberately flat, provenance-preserving projection.
        # Full packet objects remain available in debug artifacts but would use
        # most of the context window on repeated JSON keys and inherited fields.
        paper_evidence_packets: list[dict[str, Any]] = []
    elif compact_chunk_packets:
        candidate_by_id = {
            str(candidate.get("paper_id") or ""): candidate for candidate in candidate_records
        }
        packets_by_paper: dict[str, list[dict[str, Any]]] = {}
        for packet in compact_chunk_packets:
            packets_by_paper.setdefault(str(packet.get("paper_id") or ""), []).append(packet)
        paper_evidence_packets = []
        for paper_id, packets in packets_by_paper.items():
            sections: dict[str, dict[str, Any]] = {}
            for packet in packets:
                section_id = str(packet.get("section_id") or "")
                section = sections.setdefault(
                    section_id,
                    {
                        "section_id": section_id,
                        "section_title": packet.get("section_title"),
                        "section_type": packet.get("section_type"),
                        "section_path": packet.get("section_path"),
                        "chunks": [],
                    },
                )
                section["chunks"].append(
                    {
                        "chunk_ref": packet.get("chunk_ref"),
                        "package_id": packet.get("package_id"),
                        "anchor_record_id": packet.get("anchor_record_id"),
                        "package_source_type": packet.get("package_source_type"),
                        "package_label": packet.get("package_label"),
                        "record_defaults": packet.get("record_defaults", {}),
                        "records": packet.get("records", []),
                    }
                )
            paper_evidence_packets.append(
                {
                    "paper_id": paper_id,
                    "title": candidate_by_id.get(paper_id, {}).get("title"),
                    "sections": list(sections.values()),
                }
            )
    else:
        paper_evidence_packets = _build_evidence_packets(
            selected_evidence,
            candidate_records,
            contract,
            str(input_example.get("primary_evidence_type") or ""),
        )
    payload = {
        "question": input_example.get("question"),
        "answer_contract": {key: value for key, value in contract.items() if key != "query_id"},
        "answer_style_guidance": answer_style_guidance,
        "multi_paper_contribution_required": multi_paper_task,
        # Candidate titles are not L2 evidence. In keyed mode the model sees
        # only immutable paper IDs; a table cell requiring a formal title is
        # resolved from its C-key's paper ID after factual grounding.
        "candidate_papers": ([{"paper_id": c.get("paper_id")} for c in candidate_records] if keyed_mode else [_project_candidate_for_prompt(c) for c in candidate_records]),
        "evidence_output_limit": evidence_output_limit,
        "table_plan_limit": table_plan_limit,
    }
    if evidence_hierarchy:
        projection = keyed_hierarchy_prompt_projection(evidence_hierarchy) if keyed_mode else hierarchy_prompt_projection(evidence_hierarchy)
        projection.pop("_key_index", None)
        projection.pop("_card_support_keys", None)
        # Verification status/propositions are runtime-only proof metadata.
        # Serializing this map duplicates every L2 card and can consume half of
        # a fixed context window without adding information for generation.
        projection.pop("_card_metadata", None)
        payload["evidence_hierarchy"] = projection
        payload["evidence_hierarchy_format"] = "L2 cards and micro rows are factual; L3 is navigation only. Cite triple/expansion facts through support_card_keys. image_ref maps a card to its attached crop." if keyed_mode else "L2 cards are factual; L1 disambiguates; L3 is navigation only."
    elif evidence_ledger:
        payload["evidence_ledger_format"] = "PAPER <paper_id>; SECTION <title>; <evidence_ref> TAB p<page> TAB <source_type> TAB [label] TAB <extractive text>"
        payload["evidence_ledger"] = evidence_ledger
    else:
        payload["paper_evidence_packets"] = paper_evidence_packets
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
    if keyed_mode:
        # Keyed L2 uses neither legacy ledgers nor raw evidence packets. Keep
        # this instruction surface deliberately small: every repeated legacy
        # rule directly displaces a provenance-preserving L2 micro card under
        # the fixed prompt budget.
        target = {
            "gold_papers": [{"paper_id": "<candidate paper id>"}],
            "claim_to_support_keys": {"<claim_id>": ["C001"]},
            "answer": required_answer_shape,
        }
        if multi_paper_task:
            target["contributing_papers"] = [{"paper_id": "<candidate paper id>"}]
        if "table" in required_answer_fields:
            target["table_answer_plan"] = [{"row_support_key": "C001", "values": {"<table_schema column>": "<value>"}}]
        keyed_instructions = [
            "Use only L2 cards/micro rows, verified triples or requested expansions, and matching image_ref crops. Cite triple or expansion facts through their support_card_keys.",
            "Match TARGET_JSON_SHAPE and answer_contract exactly. Every factual claim needs one to four visible Cxxx keys; never emit raw refs, locators, page numbers, labels, R/P/S keys, or invented keys.",
            "Use only candidate paper IDs. For multi-paper answers list only supported contributors. For tables use row_support_key and exact schema columns; output the source C-card paper_id for Paper Title.",
            "For multiple choice output one listed key. Freeform is the shortest supported answer span; prefer sparse grounded output to unsupported claims.",
        ]
        user = (
            "\n".join(keyed_instructions)
            + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\nTARGET_JSON_SHAPE:\n" + json.dumps(target, ensure_ascii=False, separators=(",", ":"))
        )
        return [{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": user}]
    user = (
        "Use only INPUT evidence. Cite direct support by echoing evidence_refs; the runtime restores locators. Do not invent IDs, pages, or evidence.\n"
        + (
            "List every supported contributor in contributing_papers and gold_papers; keep evidence_refs top-level.\n"
            if multi_paper_task
            else ""
        )
        + "Match answer_contract and TARGET_JSON_SHAPE exactly; include only its answer types and use candidate paper IDs only. "
        "For multiple choice choose exactly one listed option key. For freeform follow answer_style_guidance and give the shortest supported span. "
        + (
            f"For tables output at most {table_plan_limit} table_answer_plan rows with a direct Cxxx row_support_key and exact schema columns; the runtime restores row sources. "
            if evidence_hierarchy and str(evidence_hierarchy.get("prompt_mode") or "") == "keyed_l2_only"
            else f"For tables output at most {table_plan_limit} table_answer_plan rows with an input row_evidence_ref, row_source, and exact schema columns. Use table_structure for alignment. "
        )
        + f"{image_note} {partial_note}\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "TARGET_JSON_SHAPE:\n"
        "{\n"
        '  "gold_papers": [{"paper_id": "<candidate paper id>"}],\n'
        + (
            '  "contributing_papers": [{"paper_id": "<candidate paper id>"}],\n'
            if multi_paper_task
            else ""
        )
        + (
            ('  "table_answer_plan": [{"row_support_key": "C001", "values": {"<table_schema column>": "<value>"}}],\n' if evidence_hierarchy and str(evidence_hierarchy.get("prompt_mode") or "") == "keyed_l2_only" else '  "table_answer_plan": [{"row_evidence_ref": "<an L2 support ref>", "row_source": {"paper_id": "<paper id>", "page": 1, "label": "Table 1"}, "values": {"<table_schema column>": "<value>"}}],\n')
            if "table" in required_answer_fields
            else ""
        )
        + ('  "claim_to_support_keys": {"<claim_id>": ["C001"]},\n' if evidence_hierarchy and str(evidence_hierarchy.get("prompt_mode") or "") == "keyed_l2_only" else '  "evidence_refs": ["<an evidence_ref copied exactly from an input record>"],\n')
        + f'  "answer": {json.dumps(required_answer_shape, ensure_ascii=False)}\n'
        "}\n"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
