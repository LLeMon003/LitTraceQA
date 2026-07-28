from __future__ import annotations

import json
from typing import Any

from .parser import extract_json_object


def is_topic_profile_candidate(candidate: dict[str, Any]) -> bool:
    retrieval_method = str(candidate.get("retrieval_method") or "")
    score_components = candidate.get("retrieval_score_components") if isinstance(candidate.get("retrieval_score_components"), dict) else {}
    return retrieval_method == "hybrid_alias_topic_optin" or bool(candidate.get("topic_profile") or score_components.get("topic_signature"))


def selection_candidates_for_metadata_vlm(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topic_candidates = [candidate for candidate in candidates if is_topic_profile_candidate(candidate)]
    if topic_candidates:
        return topic_candidates, {
            "selection_candidate_policy": "topic_profile_candidates_only",
            "selection_candidate_count": len(topic_candidates),
            "full_candidate_count": len(candidates),
        }
    return candidates, {
        "selection_candidate_policy": "all_retrieved_candidates",
        "selection_candidate_count": len(candidates),
        "full_candidate_count": len(candidates),
    }


def empty_answer_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for answer_type in sample.get("answer_types") or []:
        if answer_type == "freeform":
            answer["freeform"] = {"text": ""}
        elif answer_type == "multiple_choice":
            answer["multiple_choice"] = {"gold": ""}
        elif answer_type == "table":
            answer["table"] = {"rows": []}
    return answer


def empty_metadata_prediction(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": sample.get("query_id"),
        "gold_papers": [],
        "evidence": [],
        "answer": empty_answer_for_sample(sample),
    }


def project_metadata_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    score_components = candidate.get("retrieval_score_components") if isinstance(candidate.get("retrieval_score_components"), dict) else {}
    topic_profile = candidate.get("topic_profile") or score_components.get("topic_signature")
    is_topic_candidate = is_topic_profile_candidate(candidate)
    return {
        "rank": candidate.get("rank") or candidate.get("retrieval_rank"),
        "paper_id": candidate.get("paper_id"),
        "title": candidate.get("title"),
        "abstract": candidate.get("abstract"),
        "authors": candidate.get("authors", []),
        "venue": candidate.get("venue"),
        "year": candidate.get("year"),
        "selection_prior": "topic_profile_candidate" if is_topic_candidate else "retrieval_candidate",
        "topic_profile": topic_profile or None,
        "topic_score": candidate.get("topic_score") or score_components.get("topic_score"),
        "matched_aliases": candidate.get("matched_aliases", []),
    }


def build_metadata_selection_messages(sample: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload = {
        "query_id": sample.get("query_id"),
        "task_family": sample.get("task_family"),
        "primary_evidence_type": sample.get("primary_evidence_type"),
        "question": sample.get("question"),
        "answer_types": sample.get("answer_types", []),
        "candidate_papers": [project_metadata_candidate(candidate) for candidate in candidates],
    }
    system = (
        "You are a LitTraceQA metadata-only paper selection model. "
        "Select the paper_id values from candidate_papers that are relevant to the question using only candidate metadata and retrieval annotations. "
        "Do not answer the question. Do not provide evidence. Output valid JSON only."
    )
    user = (
        "Select relevant papers from candidate_papers.\n\n"
        "Rules:\n"
        "1. Output JSON only.\n"
        "2. query_id must match input.\n"
        "3. gold_papers must contain only paper_id values present in candidate_papers.\n"
        "4. For hidden_source_single_paper or other single-paper tasks, prefer exactly one best paper.\n"
        "5. For multi_paper tasks, include all and only papers that match the query topic; avoid broad topical neighbors.\n"
        "6. If candidate_papers include selection_prior=topic_profile_candidate, treat those as explicit topic-profile annotations for this query. Prefer selecting within that set; select non-topic retrieval_candidate papers only when the topic-profile set is clearly incomplete.\n"
        "7. Do not output answer, evidence, confidence, scores, explanations, or markdown.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "TARGET_JSON_SHAPE:\n"
        '{\n  "query_id": "<same query id>",\n  "gold_papers": [{"paper_id": "<candidate paper id>"}]\n}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_metadata_selection(
    obj: dict[str, Any],
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_id = sample.get("query_id")
    candidate_ids = [str(candidate.get("paper_id") or "") for candidate in candidates if candidate.get("paper_id")]
    candidate_set = set(candidate_ids)
    errors: list[dict[str, Any]] = []
    papers: list[dict[str, str]] = []
    for item in obj.get("gold_papers") or obj.get("papers") or []:
        paper_id = item.get("paper_id") if isinstance(item, dict) else item
        paper_id = str(paper_id or "")
        if not paper_id:
            continue
        if paper_id not in candidate_set:
            errors.append({"query_id": query_id, "type": "metadata_vlm_non_candidate_paper_removed", "paper_id": paper_id})
            continue
        row = {"paper_id": paper_id}
        if row not in papers:
            papers.append(row)
    task_family = str(sample.get("task_family") or "").strip().lower().replace("-", "_")
    if "multi" not in task_family and len(papers) > 1:
        errors.append(
            {
                "query_id": query_id,
                "type": "metadata_vlm_single_paper_extra_candidates_removed",
                "removed_paper_ids": [paper["paper_id"] for paper in papers[1:]],
            }
        )
        papers = papers[:1]
    if not papers:
        errors.append({"query_id": query_id, "type": "metadata_vlm_no_valid_papers_selected"})
    return {
        "query_id": query_id,
        "gold_papers": papers,
        "evidence": [],
        "answer": empty_answer_for_sample(sample),
    }, errors


def select_papers_with_metadata_vlm(
    answer_client: Any,
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selection_candidates, selection_policy = selection_candidates_for_metadata_vlm(candidates)
    messages = build_metadata_selection_messages(sample, selection_candidates)
    result = answer_client.generate_prediction(messages, image_paths=None)
    selected = extract_json_object(str(result["content"]))
    prediction, errors = normalize_metadata_selection(selected, sample, selection_candidates)
    return prediction, errors, {"messages": messages, "result": result, "selection_policy": selection_policy}
