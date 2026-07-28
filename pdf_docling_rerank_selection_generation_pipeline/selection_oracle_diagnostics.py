"""Measure the maximum strict-gold coverage reachable by current packages.

This is deliberately an oracle, not an alternative selector.  It holds the
candidate papers, Docling records, package construction, and configured package
budget fixed, then solves maximum gold coverage exactly for each query.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from .data_io import find_official_file, read_jsonl, write_jsonl
from .evidence_matcher import gold_source_type, match_gold_to_record
from .evidence_packages import _anchor_score_tracks, _order, _package_records, canonical_records, record_id
from .extraction_selection_evaluation import _candidate_records, _gold_evidence, _load_extraction_records
from .metadata_index import BM25Okapi, tokenize
from .symbolic_context_selector import _is_header_footer_record, _query_needs_header_footer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle diagnostics for evidence-package selection.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", default="outputs/pdf_docling_rerank_selection_generation_pipeline")
    parser.add_argument("--processed-output-dir", default="processed_pdfs/pdf_docling_rerank_selection_generation_pipeline")
    parser.add_argument("--diagnostic-output-dir", default="")
    parser.add_argument("--single-package-budget", type=int, default=12)
    parser.add_argument("--multi-package-budget", type=int, default=36)
    parser.add_argument("--text-neighbors", type=int, default=1)
    parser.add_argument("--max-packages-per-page", type=int, default=2)
    parser.add_argument("--only-query-ids", default="")
    return parser.parse_args()


def _split_ids(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def _is_multi_paper(validation_input: dict[str, Any]) -> bool:
    return "multi" in str(validation_input.get("task_family") or "").lower()


def _valid_records(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    allow_header_footer = _query_needs_header_footer(query)
    return [
        record
        for record in records
        if record.get("validation_status") != "rejected"
        and (allow_header_footer or not _is_header_footer_record(record))
    ]


def _gold_mask(gold_rows: list[dict[str, Any]], package: dict[str, Any]) -> int:
    mask = 0
    package_records = package.get("records") or []
    for index, gold in enumerate(gold_rows):
        if match_gold_to_record(gold, package_records, "strict").get("matched"):
            mask |= 1 << index
    return mask


def _reduced_masks(packages: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Retain one package per coverage set and remove dominated sets."""
    representative: dict[int, int] = {}
    for index, package in enumerate(packages):
        mask = _gold_mask(gold_rows, package)
        if mask:
            representative.setdefault(mask, index)
    masks = sorted(representative.items(), key=lambda item: (-item[0].bit_count(), item[1]))
    kept: list[tuple[int, int]] = []
    for mask, index in masks:
        if any(mask | existing == existing for existing, _ in kept):
            continue
        kept.append((mask, index))
    return kept


def maximum_coverage(packages: list[dict[str, Any]], gold_rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    """Solve unweighted maximum coverage exactly with a bit-mask dynamic program."""
    masks = _reduced_masks(packages, gold_rows)
    if budget <= 0 or not masks or not gold_rows:
        return {"covered_mask": 0, "covered_count": 0, "package_indexes": [], "effective_package_count": len(masks)}
    states: list[dict[int, tuple[int, ...]]] = [{0: ()} for _ in range(min(budget, len(masks)) + 1)]
    for mask, package_index in masks:
        for used in range(len(states) - 1, 0, -1):
            for covered, selected in list(states[used - 1].items()):
                merged = covered | mask
                existing = states[used].get(merged)
                candidate = (*selected, package_index)
                if existing is None or candidate < existing:
                    states[used][merged] = candidate
    candidates = [
        (covered.bit_count(), -len(selected), covered, selected)
        for state in states
        for covered, selected in state.items()
    ]
    _, _, covered_mask, selected = max(candidates, default=(0, 0, 0, ()))
    return {
        "covered_mask": covered_mask,
        "covered_count": covered_mask.bit_count(),
        "package_indexes": list(selected),
        "effective_package_count": len(masks),
    }


def greedy_page_capped_coverage(
    packages: list[dict[str, Any]], gold_rows: list[dict[str, Any]], budget: int, max_per_page: int
) -> int:
    """A feasible lower bound under the selector's existing page-diversity cap."""
    masks = _reduced_masks(packages, gold_rows)
    page_counts: Counter[tuple[str, Any]] = Counter()
    selected: set[int] = set()
    covered = 0
    for _ in range(max(0, budget)):
        eligible = [
            (mask, index)
            for mask, index in masks
            if index not in selected
            and page_counts[(str(packages[index].get("paper_id") or ""), packages[index].get("page"))] < max(1, max_per_page)
        ]
        if not eligible:
            break
        mask, index = max(
            eligible,
            key=lambda item: (
                (item[0] & ~covered).bit_count(),
                item[0].bit_count(),
                float(packages[item[1]].get("qwen") or -1.0),
                -item[1],
            ),
        )
        if not (mask & ~covered):
            break
        selected.add(index)
        page_counts[(str(packages[index].get("paper_id") or ""), packages[index].get("page"))] += 1
        covered |= mask
    return covered.bit_count()


def _focused_packages(
    records: list[dict[str, Any]], trace: dict[str, Any], gold_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Build only anchors that can match a strict gold on its paper/page/type.

    Other anchors cannot add evaluator-visible coverage: object packages include
    same-page peers and nearby text, and text-span matching is page/type based.
    Restricting construction here makes the oracle feasible on the full corpus
    without changing the coverage search space relevant to gold evidence.
    """
    canonical = canonical_records(records)
    focused_ids: set[str] = set()
    for gold in gold_rows:
        paper_id = str(gold.get("paper_id") or "")
        source_type = gold_source_type(gold)
        page = (gold.get("locator") or {}).get("page") if isinstance(gold.get("locator"), dict) else gold.get("page")
        for record in canonical:
            if (
                str(record.get("paper_id") or "") == paper_id
                and str(record.get("page") or "") == str(page or "")
                and str(record.get("source_type") or "") == source_type
            ):
                focused_ids.add(record_id(record))
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in canonical:
        by_paper[str(record.get("paper_id") or "")].append(record)
    scores = _anchor_score_tracks(trace)
    packages: list[dict[str, Any]] = []
    for anchor in canonical:
        anchor_id = record_id(anchor)
        if anchor_id not in focused_ids:
            continue
        contained = _package_records(anchor, by_paper[str(anchor.get("paper_id") or "")], text_neighbors=1)
        unique = {record_id(record): record for record in contained if record_id(record)}
        packages.append(
            {
                "package_id": f"pkg::{anchor_id}",
                "anchor_record_id": anchor_id,
                "paper_id": anchor.get("paper_id"),
                "page": anchor.get("page"),
                "section_id": anchor.get("section_id"),
                "source_type": anchor["source_type"],
                **scores.get(anchor_id, {}),
                "records": sorted(unique.values(), key=_order),
            }
        )
    return packages, canonical, scores


def _best_text_anchor_ranks(canonical: list[dict[str, Any]], scores: dict[str, dict[str, float]], gold: dict[str, Any]) -> tuple[int | None, int | None]:
    ranked = sorted(canonical, key=lambda record: (-float(scores.get(record_id(record), {}).get("qwen") or -1.0), record_id(record)))
    text_ranked = [record for record in ranked if record.get("source_type") == "text_span"]
    match = match_gold_to_record(gold, canonical, "strict")
    matching_id = str(match.get("matched_global_record_id") or match.get("matched_record_id") or "")
    global_rank = next((rank for rank, record in enumerate(ranked, start=1) if record_id(record) == matching_id), None)
    text_rank = next((rank for rank, record in enumerate(text_ranked, start=1) if record_id(record) == matching_id), None)
    return global_rank, text_rank


def _anchor_rankings(canonical: list[dict[str, Any]], scores: dict[str, dict[str, float]], query: str) -> dict[str, dict[str, int]]:
    texts = [" ".join(str(record.get(key) or "") for key in ("source_type", "label", "section_title", "text")) for record in canonical]
    bm25_scores = BM25Okapi([tokenize(text) for text in texts]).get_scores(tokenize(query)) if canonical else []
    qwen_order = sorted(range(len(canonical)), key=lambda index: (-float(scores.get(record_id(canonical[index]), {}).get("qwen") or -1.0), index))
    bm25_order = sorted(range(len(canonical)), key=lambda index: (-float(bm25_scores[index]), index))
    qwen_rank = {record_id(canonical[index]): rank for rank, index in enumerate(qwen_order, start=1)}
    bm25_rank = {record_id(canonical[index]): rank for rank, index in enumerate(bm25_order, start=1)}
    rrf_order = sorted(
        range(len(canonical)),
        key=lambda index: (
            -(1.0 / (60 + qwen_rank[record_id(canonical[index])]) + 1.0 / (60 + bm25_rank[record_id(canonical[index])])),
            index,
        ),
    )
    rrf_rank = {record_id(canonical[index]): rank for rank, index in enumerate(rrf_order, start=1)}
    return {"qwen": qwen_rank, "bm25": bm25_rank, "rrf": rrf_rank}


def _oracle_for_source(packages: list[dict[str, Any]], gold_rows: list[dict[str, Any]], budget: int, source_type: str) -> int:
    subset = [gold for gold in gold_rows if gold_source_type(gold) == source_type]
    return maximum_coverage(packages, subset, budget)["covered_count"] if subset else 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    diagnostic_dir = Path(args.diagnostic_output_dir or output_dir / "selection_oracle_diagnostics")
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    only_ids = _split_ids(args.only_query_ids)
    inputs = {
        str(row.get("query_id") or ""): row
        for row in read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    }
    validation_rows = read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))
    if only_ids:
        validation_rows = [row for row in validation_rows if str(row.get("query_id") or "") in only_ids]
    candidates_by_query = _candidate_records(read_jsonl(output_dir / "candidate_papers.jsonl"))
    traces = {
        str(row.get("query_id") or ""): row
        for row in read_jsonl(output_dir / "section_relevance.trace.jsonl")
    }
    all_papers = {str(row.get("paper_id") or "") for rows in candidates_by_query.values() for row in rows if row.get("paper_id")}
    records_by_paper, _, _ = _load_extraction_records(Path(args.processed_output_dir), all_papers)

    per_query: list[dict[str, Any]] = []
    source_totals: Counter[str] = Counter()
    source_oracle_budget: Counter[str] = Counter()
    source_oracle_actual: Counter[str] = Counter()
    aggregate = Counter()
    text_rank_counts = Counter()
    text_rank_counts_by_route: dict[str, Counter[str]] = defaultdict(Counter)
    crowding = Counter()

    for validation in validation_rows:
        query_id = str(validation.get("query_id") or "")
        sample = inputs.get(query_id, {})
        query = str(sample.get("question") or "")
        candidate_records = [
            record
            for candidate in candidates_by_query.get(query_id, [])
            for record in records_by_paper.get(str(candidate.get("paper_id") or ""), [])
        ]
        valid = _valid_records(candidate_records, query)
        gold_rows = _gold_evidence(validation)
        extracted_gold = [gold for gold in gold_rows if match_gold_to_record(gold, valid, "strict").get("matched")]
        is_multi = _is_multi_paper(sample)
        configured_budget = max(1, args.multi_package_budget if is_multi else args.single_package_budget)
        trace = traces.get(query_id, {})
        actual_budget = int(((trace.get("package_selection") or {}).get("selected_package_count") or configured_budget))
        packages, canonical, score_by_record = _focused_packages(valid, trace, extracted_gold)
        anchor_rankings = _anchor_rankings(canonical, score_by_record, query)
        budget_oracle = maximum_coverage(packages, extracted_gold, configured_budget)
        actual_oracle = maximum_coverage(packages, extracted_gold, actual_budget)
        page_capped_greedy = greedy_page_capped_coverage(
            packages, extracted_gold, configured_budget, args.max_packages_per_page
        )

        per_type = {}
        for source_type in sorted({gold_source_type(gold) for gold in extracted_gold}):
            total = sum(gold_source_type(gold) == source_type for gold in extracted_gold)
            budget_count = _oracle_for_source(packages, extracted_gold, configured_budget, source_type)
            actual_count = _oracle_for_source(packages, extracted_gold, actual_budget, source_type)
            per_type[source_type] = {"extracted_gold_count": total, "oracle_at_budget": budget_count, "oracle_at_actual_selected_packages": actual_count}
            source_totals[source_type] += total
            source_oracle_budget[source_type] += budget_count
            source_oracle_actual[source_type] += actual_count

        for gold in extracted_gold:
            if gold_source_type(gold) != "text_span":
                continue
            global_rank, text_rank = _best_text_anchor_ranks(canonical, score_by_record, gold)
            matched = match_gold_to_record(gold, canonical, "strict")
            matched_id = str(matched.get("matched_global_record_id") or matched.get("matched_record_id") or "")
            for threshold in (5, 10, 20):
                text_rank_counts[f"top_{threshold}"] += int(global_rank is not None and global_rank <= threshold)
                for route, ranks in anchor_rankings.items():
                    text_rank_counts_by_route[route][f"top_{threshold}"] += int(ranks.get(matched_id, 10**9) <= threshold)
            if global_rank is not None and text_rank is not None and global_rank > configured_budget and text_rank <= configured_budget:
                crowding["text_gold_with_nontext_qwen_crowding"] += 1

        aggregate["gold_count"] += len(gold_rows)
        aggregate["strict_extracted_gold_count"] += len(extracted_gold)
        aggregate["oracle_at_budget_count"] += budget_oracle["covered_count"]
        aggregate["oracle_at_actual_selected_packages_count"] += actual_oracle["covered_count"]
        aggregate["page_capped_greedy_at_budget_count"] += page_capped_greedy
        per_query.append(
            {
                "query_id": query_id,
                "task_family": sample.get("task_family"),
                "configured_package_budget": configured_budget,
                "actual_selected_package_count": actual_budget,
                "gold_count": len(gold_rows),
                "strict_extracted_gold_count": len(extracted_gold),
                "logical_package_count": len(packages),
                "oracle_effective_package_masks": budget_oracle["effective_package_count"],
                "oracle_at_budget_count": budget_oracle["covered_count"],
                "oracle_at_budget_witness_package_ids": [packages[index]["package_id"] for index in budget_oracle["package_indexes"]],
                "oracle_at_actual_selected_packages_count": actual_oracle["covered_count"],
                "page_capped_greedy_at_budget_count": page_capped_greedy,
                "by_source_type": per_type,
            }
        )

    strict_extracted = aggregate["strict_extracted_gold_count"]
    summary = {
        "query_count": len(per_query),
        "gold_count": aggregate["gold_count"],
        "strict_extracted_gold_count": strict_extracted,
        "oracle_selection_given_extracted_at_budget": round(aggregate["oracle_at_budget_count"] / strict_extracted, 6) if strict_extracted else 0.0,
        "oracle_selection_over_gold_at_budget": round(aggregate["oracle_at_budget_count"] / aggregate["gold_count"], 6) if aggregate["gold_count"] else 0.0,
        "oracle_selection_given_extracted_at_actual_selected_packages": round(aggregate["oracle_at_actual_selected_packages_count"] / strict_extracted, 6) if strict_extracted else 0.0,
        "oracle_selection_over_gold_at_actual_selected_packages": round(aggregate["oracle_at_actual_selected_packages_count"] / aggregate["gold_count"], 6) if aggregate["gold_count"] else 0.0,
        "page_capped_greedy_selection_given_extracted_at_budget": round(aggregate["page_capped_greedy_at_budget_count"] / strict_extracted, 6) if strict_extracted else 0.0,
        "by_source_type": {
            source_type: {
                "strict_extracted_gold_count": total,
                "oracle_recall_at_budget": round(source_oracle_budget[source_type] / total, 6) if total else 0.0,
                "oracle_recall_at_actual_selected_packages": round(source_oracle_actual[source_type] / total, 6) if total else 0.0,
            }
            for source_type, total in sorted(source_totals.items())
        },
        "text_span_anchor_global_rank": {
            key: round(value / source_totals["text_span"], 6) if source_totals["text_span"] else 0.0
            for key, value in sorted(text_rank_counts.items())
        },
        "text_span_anchor_rank_by_route": {
            route: {
                threshold: round(count / source_totals["text_span"], 6) if source_totals["text_span"] else 0.0
                for threshold, count in sorted(counts.items())
            }
            for route, counts in sorted(text_rank_counts_by_route.items())
        },
        "package_crowding_loss": dict(crowding),
        "definition": {
            "oracle": "Exact maximum strict-evaluator coverage over current packages, with no Qwen ranking or page-cap constraint.",
            "page_capped_greedy": "Feasible greedy coverage with the current max-packages-per-page constraint; it is a lower bound, not an oracle.",
            "actual_selected_packages": "Uses each query's realized package count after adaptive stopping, not its configured maximum budget.",
            "text_span_rank": "Rank of the best matching text-span anchor package by generic Qwen package score across all package types.",
            "crowding": "Extracted text gold whose best text anchor is within the configured text-only top-K but falls below K globally because non-text anchors rank ahead.",
        },
    }
    write_jsonl(diagnostic_dir / "per_query.jsonl", per_query)
    (diagnostic_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
