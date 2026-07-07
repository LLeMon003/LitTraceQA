from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .data_io import read_jsonl
from .metadata_index import BM25Okapi, tokenize


TABLE_ID_RE = re.compile(r"\b(?:Table|Tab\.)\s+([A-Za-z]*\d+[A-Za-z0-9.\-]*)", re.IGNORECASE)
FIGURE_ID_RE = re.compile(r"\b(?:Figure|Fig\.)\s+([A-Za-z]*\d+[A-Za-z0-9.\-]*)", re.IGNORECASE)
ALGORITHM_ID_RE = re.compile(r"\bAlgorithm\s+([A-Za-z]*\d+[A-Za-z0-9.\-]*)", re.IGNORECASE)
TABLE_RE = re.compile(r"\b(?:Table|Tab\.)\s+[A-Za-z]*\d+[A-Za-z0-9.\-]*|\bablation\b|\bresults?\b|\bcomparison\b", re.IGNORECASE)
FIGURE_RE = re.compile(r"\b(?:Figure|Fig\.)\s+[A-Za-z]*\d+[A-Za-z0-9.\-]*", re.IGNORECASE)
EQUATION_RE = re.compile(r"\b(?:Equation|Eq\.|Algorithm|Theorem|loss|objective|argmax|sum|minimize)\b|[=∑∏∫≤≥]", re.IGNORECASE)
CITATION_RE = re.compile(r"\[[0-9,\-\s]+\]|\([A-Z][A-Za-z\-]+(?: et al\.)?,\s*20[0-9]{2}\)|\bReferences\b|\bRelated Work\b", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(?:[0-9]+\.?\s+)?(?:Abstract|Introduction|Method|Approach|Experiment|Results|Ablation|Analysis|Conclusion|References)\b", re.IGNORECASE | re.MULTILINE)

ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}


def _query_text(query_example: dict[str, Any]) -> str:
    parts = [
        str(query_example.get("question") or ""),
        str(query_example.get("task_family") or ""),
        str(query_example.get("primary_evidence_type") or ""),
        " ".join(str(x) for x in query_example.get("answer_types") or []),
        json.dumps(query_example.get("table_schema") or "", ensure_ascii=False),
    ]
    return " ".join(part for part in parts if part)


def _effective_source_type(sample: dict[str, Any]) -> str:
    question = str(sample.get("question") or "")
    primary_type = str(sample.get("primary_evidence_type") or "")
    if FIGURE_ID_RE.search(question):
        return "figure"
    if TABLE_ID_RE.search(question) or re.search(r"\btable\b|\bcomparison table\b", question, re.IGNORECASE):
        return "table"
    if re.search(r"\b(?:equation|eq\.|algorithm)\s+\(?[A-Za-z0-9.\-]+\)?", question, re.IGNORECASE):
        return "equation_algorithm"
    return primary_type


def _normalized_object_id(prefix: str, value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().rstrip(".,;:")
    if not re.search(r"\d", value):
        return None
    return f"{prefix} {value}"


def _object_pattern(label: str, object_id: str | None, caption_only: bool = False) -> re.Pattern[str] | None:
    if not object_id:
        return None
    _, _, value = object_id.partition(" ")
    if not value:
        return None
    suffix = r"\s*:" if caption_only else r"\b"
    return re.compile(rf"\b{re.escape(label)}\s*{re.escape(value)}{suffix}", re.IGNORECASE)


def _question_targets(sample: dict[str, Any]) -> dict[str, Any]:
    question = str(sample.get("question") or "")
    lower = question.lower()
    reference_id: int | None = None
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\s+reference\b|\breference\s*(?:number|#)?\s*(\d+)\b", lower)
    if match:
        reference_id = int(match.group(1) or match.group(2))
    else:
        for word, value in ORDINAL_WORDS.items():
            if re.search(rf"\b{word}\s+reference\b", lower):
                reference_id = value
                break

    equation_id: str | None = None
    equation_match = re.search(r"\b(?:equation|eq\.)\s*\(?([A-Za-z0-9.\-]+)\)?", question, re.IGNORECASE)
    if equation_match:
        equation_id = equation_match.group(1).strip().rstrip(".,;:")

    algorithm_id: str | None = None
    algorithm_match = ALGORITHM_ID_RE.search(question)
    if algorithm_match:
        algorithm_id = algorithm_match.group(1).strip().rstrip(".,;:")

    table_match = TABLE_ID_RE.search(question)
    figure_match = FIGURE_ID_RE.search(question)
    return {
        "table_id": _normalized_object_id("Table", table_match.group(1) if table_match else None),
        "figure_id": _normalized_object_id("Figure", figure_match.group(1) if figure_match else None),
        "equation_id": equation_id,
        "algorithm_id": _normalized_object_id("Algorithm", algorithm_id),
        "reference_id": reference_id,
        "last_reference": bool(re.search(r"\blast\s+reference\b|\bindex\s+of\s+the\s+last\s+reference\b", lower)),
        "hardware": bool(re.search(r"\bhardware\b|\bgpu\b|\bconfigure\b|\bconfiguration\b", lower)),
        "subfigure_count": bool(re.search(r"\bhow many\s+(?:subfigures|sub-figures|panels)\b|\bnumber of\s+(?:subfigures|sub-figures|panels)\b", lower)),
    }


def _reference_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in re.finditer(r"(?:^|\n|\s)\[(\d{1,3})\]\s+", text or ""):
        try:
            numbers.append(int(match.group(1)))
        except ValueError:
            continue
    return numbers


def _evidence_page_bonus(sample: dict[str, Any], text: str, page_number: int, total_pages: int) -> float:
    source_type = _effective_source_type(sample)
    targets = _question_targets(sample)
    bonus = 0.0
    if source_type == "table":
        exact_caption = _object_pattern("Table", targets.get("table_id"), caption_only=True)
        exact_mention = _object_pattern("Table", targets.get("table_id"), caption_only=False)
        if exact_caption and exact_caption.search(text):
            bonus += 70.0
        elif exact_mention and exact_mention.search(text):
            bonus += 35.0
        elif TABLE_RE.search(text):
            bonus += 8.0
    elif source_type == "figure":
        exact_caption = _object_pattern("Figure", targets.get("figure_id"), caption_only=True)
        exact_mention = _object_pattern("Figure", targets.get("figure_id"), caption_only=False)
        if exact_caption and exact_caption.search(text):
            bonus += 80.0
        elif exact_mention and exact_mention.search(text):
            bonus += 30.0
        elif FIGURE_RE.search(text):
            bonus += 8.0
        if targets.get("subfigure_count") and re.search(r"\([a-z]\)", text):
            bonus += 8.0
    elif source_type == "citation_context":
        reference_numbers = _reference_numbers(text)
        target_ref = targets.get("reference_id")
        if isinstance(target_ref, int) and target_ref in reference_numbers:
            bonus += 90.0
        if targets.get("last_reference") and reference_numbers:
            bonus += min(80.0, max(reference_numbers) * 1.15) + min(20.0, len(reference_numbers))
        if re.search(r"\bReferences\b", text, re.IGNORECASE):
            bonus += 12.0
        elif reference_numbers:
            bonus += 8.0
        if total_pages:
            bonus += 2.0 * (page_number / total_pages)
    elif source_type == "text_span":
        if targets.get("hardware") and re.search(r"\b(?:gpu|rtx|a100|h100|cuda|nvidia)\b", text, re.IGNORECASE):
            bonus += 60.0
        if re.search(r"\b(?:implementation|experiment|setup|configuration|hardware|overhead|efficiency)\b", text, re.IGNORECASE):
            bonus += 6.0
    elif source_type == "equation_algorithm":
        algorithm_pattern = _object_pattern("Algorithm", targets.get("algorithm_id"), caption_only=False)
        if algorithm_pattern and algorithm_pattern.search(text):
            bonus += 70.0
        equation_id = targets.get("equation_id")
        if equation_id and re.search(rf"\(\s*{re.escape(str(equation_id))}\s*\)", text):
            bonus += 70.0
        elif re.search(r"\bAlgorithm\b|\bequation\b|\bloss\b|\bobjective\b|=", text, re.IGNORECASE):
            bonus += 8.0
    return bonus


def _candidate_score_prior(candidate: dict[str, Any]) -> float:
    rank = int(candidate.get("rank") or candidate.get("retrieval_rank") or 999)
    if rank == 1:
        return 3.0
    if rank == 2:
        return 2.0
    if rank == 3:
        return 1.0
    return 0.0


def _page_position_prior(page: int, page_count: int, primary_type: str) -> float:
    if page_count <= 0:
        return 0.0
    frac = page / max(1, page_count)
    if page <= 2:
        return 0.8
    if primary_type in {"table", "figure"} and 0.35 <= frac <= 0.85:
        return 0.7
    if primary_type == "citation_context" and frac >= 0.75:
        return 0.9
    if primary_type == "text_span" and 0.15 <= frac <= 0.7:
        return 0.5
    return 0.0


def _rule_boosts(text: str, primary_type: str) -> dict[str, float]:
    boosts = {
        "primary_evidence_type_boost": 0.0,
        "label_match_boost": 0.0,
        "section_heading_boost": 0.0,
        "page_position_prior": 0.0,
    }
    if primary_type == "figure" and FIGURE_RE.search(text):
        boosts["primary_evidence_type_boost"] += 8.0
        boosts["label_match_boost"] += 3.0
    elif primary_type == "table" and TABLE_RE.search(text):
        boosts["primary_evidence_type_boost"] += 8.0
        boosts["label_match_boost"] += 3.0
    elif primary_type == "equation_algorithm" and EQUATION_RE.search(text):
        boosts["primary_evidence_type_boost"] += 6.0
        boosts["label_match_boost"] += 2.0
    elif primary_type == "citation_context" and CITATION_RE.search(text):
        boosts["primary_evidence_type_boost"] += 5.0
        boosts["label_match_boost"] += 2.0
    if SECTION_RE.search(text):
        boosts["section_heading_boost"] += 1.2
    return boosts


def build_global_page_pool(candidates: list[dict[str, Any]], paper_page_texts: dict[str, list[dict[str, Any]]], query_id: str) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for candidate in candidates:
        paper_id = str(candidate.get("paper_id") or "")
        for row in paper_page_texts.get(paper_id, []):
            pool.append(
                {
                    "query_id": query_id,
                    "candidate_rank": int(candidate.get("rank") or candidate.get("retrieval_rank") or 0),
                    "candidate_score": float(candidate.get("score") or candidate.get("hybrid_score") or candidate.get("bm25_score") or 0.0),
                    "paper_id": paper_id,
                    "page": int(row.get("page") or 0),
                    "native_text": str(row.get("text") or ""),
                    "native_text_char_count": int(row.get("char_count") or 0),
                    "has_native_text": bool(row.get("has_native_text")),
                    "paper_title": candidate.get("title", ""),
                    "paper_venue": candidate.get("venue", ""),
                    "paper_year": candidate.get("year", ""),
                }
            )
    return pool


def rank_global_pages_for_query(
    query_example: dict[str, Any],
    candidates: list[dict[str, Any]],
    paper_page_texts: dict[str, list[dict[str, Any]]],
    top_p_global: int,
    max_pages_global: int | None = None,
    *,
    fallback_on_empty_text: bool = True,
    empty_text_parse_first_n_pages: int = 4,
) -> dict[str, Any]:
    query_id = str(query_example.get("query_id") or "")
    primary_type = _effective_source_type(query_example)
    query = _query_text(query_example)
    pool = build_global_page_pool(candidates, paper_page_texts, query_id)
    if not pool:
        return {
            "query_id": query_id,
            "ranking_source": "native_text",
            "ranking_method": "global_native_text_bm25_rules",
            "total_candidate_pages": 0,
            "ranked_pages": [],
            "selected_pages_initial": [],
            "selected_pages_final": [],
            "fallback_reason": "empty_global_page_pool",
        }
    all_empty = not any(row.get("has_native_text") for row in pool)
    if all_empty and fallback_on_empty_text:
        ranked_fallback = sorted(pool, key=lambda row: (int(row.get("candidate_rank") or 999), int(row.get("page") or 999)))
        ranked_pages = []
        empty_limit = max(0, int(empty_text_parse_first_n_pages))
        for rank, row in enumerate(ranked_fallback, start=1):
            selected = rank <= empty_limit
            ranked_pages.append(
                {
                    "global_page_rank": rank,
                    "paper_id": row["paper_id"],
                    "candidate_rank": row["candidate_rank"],
                    "page": row["page"],
                    "score": 0.0,
                    "score_components": {
                        "native_text_bm25": 0.0,
                        "query_overlap": 0.0,
                        "candidate_rank_prior": _candidate_score_prior(row),
                        "primary_evidence_type_boost": 0.0,
                        "label_match_boost": 0.0,
                        "section_heading_boost": 0.0,
                        "page_position_prior": 0.0,
                    },
                    "native_text_char_count": row["native_text_char_count"],
                    "selected_for_initial_parse": selected,
                }
            )
        selected_initial = [
            {
                "paper_id": row["paper_id"],
                "page": row["page"],
                "global_page_rank": row["global_page_rank"],
                "candidate_rank": row["candidate_rank"],
            }
            for row in ranked_pages
            if row["selected_for_initial_parse"]
        ]
        return {
            "query_id": query_id,
            "ranking_source": "native_text",
            "ranking_method": "global_native_text_bm25_rules",
            "top_k_papers": len(candidates),
            "total_candidate_pages": len(pool),
            "ranked_pages": ranked_pages,
            "selected_pages_initial": selected_initial,
            "selected_pages_final": selected_initial,
            "fallback_reason": "all_native_text_empty_global_fallback",
        }
    corpus = [tokenize(f"{row.get('native_text') or ''} {row.get('paper_title') or ''}") for row in pool]
    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(pool)
    query_token_set = set(query_tokens)
    page_count_by_paper = Counter(str(row["paper_id"]) for row in pool)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row, bm25_score in zip(pool, bm25_scores):
        text = str(row.get("native_text") or "")
        tokens = set(tokenize(text))
        overlap = len(query_token_set & tokens) / max(1.0, math.sqrt(len(query_token_set) + 1))
        rule = _rule_boosts(text, primary_type)
        prior = _candidate_score_prior(row)
        position = _page_position_prior(int(row.get("page") or 0), page_count_by_paper[str(row.get("paper_id"))], primary_type)
        evidence_bonus = _evidence_page_bonus(
            query_example,
            text,
            page_number=int(row.get("page") or 0),
            total_pages=page_count_by_paper[str(row.get("paper_id"))],
        )
        components = {
            "native_text_bm25": float(bm25_score),
            "query_overlap": float(overlap),
            "candidate_rank_prior": prior,
            "primary_evidence_type_boost": rule["primary_evidence_type_boost"],
            "label_match_boost": rule["label_match_boost"],
            "section_heading_boost": rule["section_heading_boost"],
            "page_position_prior": position,
            "target_locator_bonus": evidence_bonus,
        }
        score = sum(float(value) for value in components.values())
        scored.append((score, {**row, "score": score, "score_components": components}))
    ranked_raw = sorted(scored, key=lambda item: item[0], reverse=True)
    ranked_pages: list[dict[str, Any]] = []
    initial_limit = max(0, int(top_p_global))
    for rank, (_, row) in enumerate(ranked_raw, start=1):
        ranked_pages.append(
            {
                "global_page_rank": rank,
                "paper_id": row["paper_id"],
                "candidate_rank": row["candidate_rank"],
                "page": row["page"],
                "score": round(float(row["score"]), 6),
                "score_components": row["score_components"],
                "native_text_char_count": row["native_text_char_count"],
                "has_native_text": row["has_native_text"],
                "selected_for_initial_parse": rank <= initial_limit,
            }
        )
    selected_initial = [
        {
            "paper_id": row["paper_id"],
            "page": row["page"],
            "global_page_rank": row["global_page_rank"],
            "candidate_rank": row["candidate_rank"],
        }
        for row in ranked_pages[:initial_limit]
    ]
    final_limit = max(initial_limit, int(max_pages_global or 0)) or len(ranked_pages)
    selected_final = [
        {
            "paper_id": row["paper_id"],
            "page": row["page"],
            "global_page_rank": row["global_page_rank"],
            "candidate_rank": row["candidate_rank"],
        }
        for row in ranked_pages[:final_limit]
    ]
    return {
        "query_id": query_id,
        "ranking_source": "native_text",
        "ranking_method": "global_native_text_bm25_rules",
        "top_k_papers": len(candidates),
        "total_candidate_pages": len(pool),
        "ranked_pages": ranked_pages,
        "selected_pages_initial": selected_initial,
        "selected_pages_final": selected_final,
        "fallback_reason": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank query-level global PDF pages from native text.")
    parser.add_argument("--query-id", default="query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--primary-evidence-type", default="")
    parser.add_argument("--candidate-page-texts", required=True)
    parser.add_argument("--top-p-global", type=int, default=8)
    args = parser.parse_args()
    rows = read_jsonl(args.candidate_page_texts)
    candidates: list[dict[str, Any]] = []
    paper_page_texts: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        paper_page_texts.setdefault(paper_id, []).append(
            {
                "paper_id": paper_id,
                "page": row.get("page"),
                "text": row.get("native_text", row.get("text", "")),
                "char_count": row.get("native_text_char_count", row.get("char_count", 0)),
                "has_native_text": row.get("has_native_text", False),
            }
        )
        if paper_id not in seen:
            seen.add(paper_id)
            candidates.append(
                {
                    "paper_id": paper_id,
                    "rank": row.get("candidate_rank", len(candidates) + 1),
                    "score": row.get("candidate_score", 0.0),
                    "title": row.get("paper_title", ""),
                    "venue": row.get("paper_venue", ""),
                    "year": row.get("paper_year", ""),
                }
            )
    result = rank_global_pages_for_query(
        {"query_id": args.query_id, "question": args.query, "primary_evidence_type": args.primary_evidence_type},
        candidates,
        paper_page_texts,
        args.top_p_global,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
