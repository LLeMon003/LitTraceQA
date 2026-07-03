from __future__ import annotations

import json
import os
import re
from typing import Any


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
        gold = str(multiple.get("gold", "A") or "A").strip()[:1].upper()
        answer["multiple_choice"] = {"gold": gold or "A"}
    if "table" in answer_types:
        table = source_answer.get("table") if isinstance(source_answer.get("table"), dict) else {}
        rows = table.get("rows", [])
        answer["table"] = {"rows": rows if isinstance(rows, list) else []}
    return answer


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
    obj: dict[str, Any], input_example: dict[str, Any], candidate_paper_ids: list[str]
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

    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        errors.append({"query_id": query_id, "type": "invalid_evidence_replaced"})
        evidence = []
    answer = _requested_answer(input_example, obj.get("answer") if isinstance(obj.get("answer"), dict) else {})
    pred = {"query_id": query_id, "gold_papers": normalized_papers, "evidence": evidence, "answer": answer}
    return pred, errors


def validate_prediction_shape(pred: dict[str, Any]) -> bool:
    return (
        isinstance(pred.get("query_id"), str)
        and isinstance(pred.get("gold_papers"), list)
        and isinstance(pred.get("evidence"), list)
        and isinstance(pred.get("answer"), dict)
    )

