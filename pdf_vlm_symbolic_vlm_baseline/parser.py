from __future__ import annotations

import json
import os
import re
from typing import Any

from .data_io import extract_answer_contract
from .symbolic_schema import OFFICIAL_EVIDENCE_SOURCE_TYPES, to_official_source_type


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    fence_match = FENCE_RE.search(content)
    if fence_match:
        content = fence_match.group(1).strip()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippet = content[start : index + 1]
                    obj = json.loads(snippet)
                    if not isinstance(obj, dict):
                        raise ValueError("Extracted JSON is not an object")
                    return obj
    raise ValueError("No complete JSON object found")


def _requested_answer(input_example: dict[str, Any], source_answer: dict[str, Any] | None = None) -> dict[str, Any]:
    answer_types = input_example.get("answer_types") or []
    source_answer = source_answer or {}
    answer: dict[str, Any] = {}
    if "freeform" in answer_types:
        freeform = source_answer.get("freeform") if isinstance(source_answer.get("freeform"), dict) else {}
        answer["freeform"] = {"text": str(freeform.get("text", ""))}
    if "multiple_choice" in answer_types:
        multiple = source_answer.get("multiple_choice") if isinstance(source_answer.get("multiple_choice"), dict) else {}
        gold = str(multiple.get("gold", "") or "").strip()
        answer["multiple_choice"] = {"gold": gold}
    if "table" in answer_types:
        table = source_answer.get("table") if isinstance(source_answer.get("table"), dict) else {}
        rows = table.get("rows", [])
        answer["table"] = {"rows": rows if isinstance(rows, list) else []}
    return answer


def validate_prediction_against_answer_contract(
    prediction: dict[str, Any],
    answer_contract: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    allowed_types = [str(item) for item in answer_contract.get("answer_types", []) if str(item) in {"freeform", "multiple_choice", "table"}]
    answer = prediction.get("answer") if isinstance(prediction.get("answer"), dict) else {}
    cleaned: dict[str, Any] = {}
    option_text_by_key = {
        str(option.get("key") or "").strip(): str(option.get("text") or "")
        for option in (answer_contract.get("multiple_choice") or {}).get("options", [])
        if isinstance(option, dict) and str(option.get("key") or "").strip()
    }
    for key in list(answer.keys()):
        if key not in allowed_types:
            errors.append("answer_type_extra_fields_removed")
    if "freeform" in allowed_types:
        raw_freeform = answer.get("freeform")
        raw = raw_freeform if isinstance(raw_freeform, dict) else {}
        if isinstance(raw_freeform, str):
            text = raw_freeform
            errors.append("freeform_string_normalized")
        elif isinstance(raw, dict):
            text = raw.get("text") or raw.get("answer") or raw.get("value") or ""
        else:
            text = ""
        if "freeform" not in answer:
            errors.append("missing_required_answer_type")
        cleaned["freeform"] = {"text": str(text or "")}
    if "multiple_choice" in allowed_types:
        raw = answer.get("multiple_choice") if isinstance(answer.get("multiple_choice"), dict) else {}
        if "multiple_choice" not in answer:
            errors.append("missing_required_answer_type")
        gold = str(raw.get("gold", "") if isinstance(raw, dict) else "").strip()
        option_keys = set(option_text_by_key)
        if gold and option_keys and gold not in option_keys:
            errors.append("invalid_multiple_choice_option")
            gold = ""
        elif gold and not option_keys:
            errors.append("missing_multiple_choice_options")
        cleaned["multiple_choice"] = {"gold": gold}
        if (
            "freeform" in allowed_types
            and not str((cleaned.get("freeform") or {}).get("text") or "").strip()
            and gold
            and option_text_by_key.get(gold)
        ):
            cleaned["freeform"] = {"text": option_text_by_key[gold]}
            errors.append("freeform_filled_from_multiple_choice_option")
    if "table" in allowed_types:
        raw = answer.get("table") if isinstance(answer.get("table"), dict) else {}
        if "table" not in answer:
            errors.append("missing_required_answer_type")
        rows = raw.get("rows", []) if isinstance(raw, dict) else []
        rows = rows if isinstance(rows, list) else []
        schema = (answer_contract.get("table") or {}).get("table_schema")
        if isinstance(schema, list) and schema:
            normalized_rows = []
            for row in rows:
                source_row = row if isinstance(row, dict) else {}
                normalized_rows.append({str(column): source_row.get(str(column)) for column in schema})
            rows = normalized_rows
        cleaned["table"] = {"rows": rows}
    prediction["answer"] = cleaned
    return prediction, errors


def _selected_evidence_map(selected_evidence: list[dict[str, Any]] | None) -> dict[tuple[str, str], dict[str, set[Any]]]:
    mapping: dict[tuple[str, str], dict[str, set[Any]]] = {}
    for evidence in selected_evidence or []:
        if not isinstance(evidence, dict):
            continue
        paper_id = str(evidence.get("paper_id") or "")
        source_type = str(evidence.get("source_type") or "")
        if not paper_id or source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            continue
        key = (paper_id, source_type)
        entry = mapping.setdefault(key, {"pages": set(), "table_id": set(), "figure_id": set(), "equation_id": set(), "algorithm_id": set(), "citation_id": set(), "reference_id": set()})
        try:
            entry["pages"].add(int(evidence.get("page")))
        except (TypeError, ValueError):
            pass
        grounding = evidence.get("grounding_label")
        if isinstance(grounding, dict):
            label_type = str(grounding.get("type") or "")
            value = str(grounding.get("value") or "").strip()
            if label_type in entry and value:
                entry[label_type].add(value)
                if label_type == "citation_id":
                    entry["reference_id"].add(value)
    return mapping


def validate_prediction_locators_against_selected_evidence(
    prediction: dict[str, Any],
    selected_evidence: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    evidence_map = _selected_evidence_map(selected_evidence)
    normalized: list[dict[str, Any]] = []
    raw_evidence = prediction.get("evidence") if isinstance(prediction.get("evidence"), list) else []
    for item in raw_evidence:
        if not isinstance(item, dict):
            errors.append("locator_validation_error")
            continue
        paper_id = str(item.get("paper_id") or "")
        source_type = to_official_source_type(source_type=str(item.get("source_type") or "")) or "text_span"
        if source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            errors.append("locator_validation_error")
            continue
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        key = (paper_id, source_type)
        allowed = evidence_map.get(key)
        if not allowed:
            errors.append("locator_validation_error")
            continue
        try:
            page = int(locator.get("page"))
        except (TypeError, ValueError):
            errors.append("locator_validation_error")
            continue
        if page not in allowed.get("pages", set()):
            errors.append("locator_validation_error")
            continue
        clean_locator: dict[str, Any] = {"page": page}
        for id_key in ("table_id", "figure_id", "equation_id", "algorithm_id", "citation_id", "reference_id"):
            if id_key not in locator:
                continue
            value = str(locator.get(id_key) or "").strip()
            if value and value in allowed.get(id_key, set()):
                clean_locator[id_key] = value
            else:
                errors.append("invented_grounding_label_removed")
        normalized.append({"paper_id": paper_id, "source_type": source_type, "locator": clean_locator})
    prediction["evidence"] = normalized
    return prediction, errors


def make_fallback_prediction(input_example: dict[str, Any], top1_candidate: dict[str, Any] | None) -> dict[str, Any]:
    paper_id = (top1_candidate or {}).get("paper_id", "")
    gold_papers = [{"paper_id": paper_id}] if paper_id else []
    return {
        "query_id": input_example.get("query_id"),
        "gold_papers": gold_papers,
        "evidence": [],
        "answer": _requested_answer(input_example),
    }


def normalize_prediction(
    obj: dict[str, Any],
    input_example: dict[str, Any],
    candidate_paper_ids: list[str],
    answer_contract: dict[str, Any] | None = None,
    selected_evidence: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_id = input_example.get("query_id")
    errors: list[dict[str, Any]] = []
    allow_noncandidate = os.environ.get("ALLOW_NONCANDIDATE_PAPERS", "").lower() == "true"
    candidate_set = set(candidate_paper_ids)
    top1 = candidate_paper_ids[0] if candidate_paper_ids else ""

    raw_papers = obj.get("gold_papers") or obj.get("papers") or []
    normalized_papers: list[dict[str, str]] = []
    if isinstance(raw_papers, list):
        for item in raw_papers:
            paper_id = item.get("paper_id") if isinstance(item, dict) else item
            paper_id = str(paper_id or "")
            if not paper_id:
                continue
            if paper_id not in candidate_set and not allow_noncandidate:
                errors.append({"query_id": query_id, "type": "noncandidate_paper_id", "paper_id": paper_id, "replacement": top1})
                paper_id = top1
            if paper_id and {"paper_id": paper_id} not in normalized_papers:
                normalized_papers.append({"paper_id": paper_id})
    if not normalized_papers and top1:
        errors.append({"query_id": query_id, "type": "fallback_paper_id", "replacement": top1})
        normalized_papers = [{"paper_id": top1}]

    evidence = _normalize_official_evidence(obj.get("evidence", []))
    if not isinstance(evidence, list):
        errors.append({"query_id": query_id, "type": "invalid_evidence_replaced"})
        evidence = []
    contract = answer_contract or extract_answer_contract(input_example)
    answer = obj.get("answer") if isinstance(obj.get("answer"), dict) else {}
    pred = {"query_id": query_id, "gold_papers": normalized_papers, "evidence": evidence, "answer": answer}
    pred, locator_errors = validate_prediction_locators_against_selected_evidence(pred, selected_evidence)
    pred, answer_errors = validate_prediction_against_answer_contract(pred, contract)
    for error_type in locator_errors + answer_errors:
        errors.append({"query_id": query_id, "type": error_type})
    return pred, errors


def validate_prediction_shape(pred: dict[str, Any]) -> bool:
    return (
        isinstance(pred.get("query_id"), str)
        and isinstance(pred.get("gold_papers"), list)
        and isinstance(pred.get("evidence"), list)
        and isinstance(pred.get("answer"), dict)
    )


def _normalize_official_evidence(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized: list[Any] = []
    for item in value:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        row = dict(item)
        source_type = to_official_source_type(source_type=str(row.get("source_type") or ""), record_type=str(row.get("record_type") or "")) or "text_span"
        row["source_type"] = source_type
        normalized.append(row)
    return normalized


def strip_internal_grounding(value: Any) -> Any:
    internal_keys = {
        "confidence",
        "notes",
        "global_record_id",
        "record_type",
        "bbox_1000",
        "bbox_pixel",
        "validation_status",
        "record_id",
        "label",
        "image_ref",
        "vlm_parse_confidence",
    }
    if isinstance(value, list):
        return [strip_internal_grounding(item) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in internal_keys:
                continue
            if key == "locator" and isinstance(item, dict):
                locator = {k: strip_internal_grounding(v) for k, v in item.items() if k not in internal_keys}
                cleaned[key] = locator
            else:
                cleaned[key] = strip_internal_grounding(item)
        return cleaned
    return value
