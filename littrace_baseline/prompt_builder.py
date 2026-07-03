from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are a LitTraceQA metadata-only baseline model. Your job is to answer "
    "literature-grounded research questions using only the provided candidate paper "
    "titles and abstracts. You must output valid JSON only. Do not include markdown, "
    "commentary, or explanations outside JSON."
)


def build_littraceqa_prompt(input_example: dict[str, Any], candidate_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    answer_types = input_example.get("answer_types") or []
    payload = {
        "query_id": input_example.get("query_id"),
        "task_family": input_example.get("task_family"),
        "primary_evidence_type": input_example.get("primary_evidence_type"),
        "question": input_example.get("question"),
        "answer_types": answer_types,
        "table_schema": input_example.get("table_schema"),
        "candidate_papers": [
            {
                "paper_id": candidate.get("paper_id"),
                "title": candidate.get("title"),
                "venue": candidate.get("venue"),
                "year": candidate.get("year"),
                "bm25_score": candidate.get("bm25_score", candidate.get("score")),
                "abstract": candidate.get("abstract"),
            }
            for candidate in candidate_records
        ],
    }
    requested_answer: dict[str, Any] = {}
    if "freeform" in answer_types:
        requested_answer["freeform"] = {"text": ""}
    if "multiple_choice" in answer_types:
        requested_answer["multiple_choice"] = {"gold": "A"}
    if "table" in answer_types:
        requested_answer["table"] = {"rows": []}

    example_shape = {
        "query_id": "<same query id>",
        "gold_papers": [{"paper_id": "<predicted paper id from candidates>"}],
        "evidence": [],
        "answer": requested_answer,
        "confidence": {"paper_retrieval": 0.0, "evidence_grounding": 0.0, "answer": 0.0},
        "notes": {
            "baseline_type": "metadata_only_title_abstract",
            "used_online_access": False,
            "accessed_links": [],
            "limitations": "This baseline used only paper titles and abstracts, without PDF or full-text access.",
        },
    }
    user_prompt = (
        "This is a metadata-only baseline. You do not have access to full papers, PDFs, "
        "DOI links, arXiv pages, OpenReview pages, tables, figures, equations, or "
        "page-level evidence in this run. Do not claim that you opened or inspected "
        "any online paper. Make the best possible prediction from the provided titles "
        "and abstracts only. Evidence grounding will necessarily be low confidence and "
        "should not invent precise page numbers, table IDs, or figure IDs unless they "
        "are explicitly available in the provided metadata.\n\n"
        "Return valid JSON only. Use only paper_id values from candidate_papers. "
        "Use the official field name gold_papers for predicted relevant papers. "
        "Do not include answer types that were not requested. If table rows are needed, "
        "use exactly the table_schema column names and JSON numbers for numeric cells.\n\n"
        "INPUT:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "TARGET_JSON_SHAPE:\n"
        f"{json.dumps(example_shape, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
