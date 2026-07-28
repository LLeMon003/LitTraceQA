"""Paired fixed-budget evaluation for Qwen baseline versus Qwen + HyDE."""
from __future__ import annotations

import argparse
import json
import statistics
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

from .data_io import find_official_file, read_jsonl
from .extraction_selection_evaluation import run as run_extraction_selection_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--hyde-output-dir", required=True)
    parser.add_argument("--processed-output-dir", required=True)
    parser.add_argument("--eval-output-dir", required=True)
    return parser.parse_args()


def _run_official(official_dir: str, output_dir: str, processed_dir: str, eval_dir: Path) -> None:
    run_extraction_selection_evaluation(
        Namespace(
            official_dir=official_dir,
            gold_path="",
            output_dir=output_dir,
            processed_output_dir=processed_dir,
            eval_output_dir=str(eval_dir),
            candidate_papers_path="",
            selected_contexts_path="",
            paper_metadata_path="",
            match_mode="both",
            only_query_ids="",
            max_queries=None,
            show_examples=False,
            write_failure_cases=True,
        )
    )


def _trace_index(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    record_index: dict[tuple[str, str], dict[str, Any]] = {}
    query_trace: dict[str, dict[str, Any]] = {}
    for trace in read_jsonl(path):
        query_id = str(trace.get("query_id") or "")
        query_trace[query_id] = trace
        for section in trace.get("sections") or []:
            section_row = {
                "section_key": f"{section.get('paper_id')}::{section.get('section_id')}",
                "paper_id": section.get("paper_id"),
                "section_id": section.get("section_id"),
                "rank": int((section.get("assessment") or {}).get("relevance", {}).get("rank") or 0),
                "selected": bool(section.get("selected")),
                "source_type": None,
            }
            for record_id in section.get("record_ids") or []:
                record_index[(query_id, str(record_id))] = section_row
    return record_index, query_trace


def _percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return float(ordered[index])


def _system_metrics(
    rows: list[dict[str, Any]],
    index: dict[tuple[str, str], dict[str, Any]],
    traces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence_rows = [row for row in rows if row.get("locator_extracted")]
    sections: dict[tuple[str, str], dict[str, Any]] = {}
    for row in evidence_rows:
        matched = str(row.get("matched_extraction_record") or "")
        section = index.get((str(row.get("query_id") or ""), matched))
        if not section:
            continue
        key = (str(row.get("query_id") or ""), section["section_key"])
        sections[key] = section
    ranks = [section["rank"] for section in sections.values() if section["rank"] > 0]
    selected_sections = sum(section["selected"] for section in sections.values())
    hyde_audits = [trace.get("hyde") for trace in traces.values() if isinstance(trace.get("hyde"), dict)]
    return {
        "extractable_gold_evidence": len(evidence_rows),
        "selected_gold_evidence": sum(bool(row.get("selected")) for row in evidence_rows),
        "R_select_given_extracted": (
            sum(bool(row.get("selected")) for row in evidence_rows) / len(evidence_rows) if evidence_rows else 0.0
        ),
        "extractable_unique_gold_sections": len(sections),
        "selected_unique_gold_sections": selected_sections,
        "gold_section_recall_at_budget": selected_sections / len(sections) if sections else 0.0,
        "MRR": statistics.fmean(1.0 / rank for rank in ranks) if ranks else 0.0,
        "median_gold_section_rank": statistics.median(ranks) if ranks else None,
        "p90_gold_section_rank": _percentile(ranks, 0.9),
        "selected_sections_per_query": statistics.fmean(
            len(trace.get("selected_section_ids") or []) for trace in traces.values()
        ) if traces else 0.0,
        "generated_claims_per_query": statistics.fmean(
            len(audit.get("parsed_claims") or []) for audit in hyde_audits
        ) if hyde_audits else 0.0,
        "retrieval_units_evaluated": sum(int(audit.get("retrieval_unit_count") or 0) for audit in hyde_audits),
        "claim_unit_pairs_evaluated": sum(int(audit.get("claim_unit_pair_count") or 0) for audit in hyde_audits),
        "fallback_count": sum(bool(audit.get("fallback_used")) for audit in hyde_audits),
        "fallback_rate": (
            sum(bool(audit.get("fallback_used")) for audit in hyde_audits) / len(hyde_audits)
            if hyde_audits else 0.0
        ),
    }


def _by_source(
    baseline_rows: list[dict[str, Any]],
    hyde_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    source_types = sorted({str(row.get("gold_source_type") or "") for row in baseline_rows})
    for source_type in source_types:
        baseline = [row for row in baseline_rows if row.get("gold_source_type") == source_type and row.get("locator_extracted")]
        hyde = [row for row in hyde_rows if row.get("gold_source_type") == source_type and row.get("locator_extracted")]
        result[source_type] = {
            "extractable": len(baseline),
            "baseline_selected": sum(bool(row.get("selected")) for row in baseline),
            "hyde_selected": sum(bool(row.get("selected")) for row in hyde),
            "baseline_recall": sum(bool(row.get("selected")) for row in baseline) / len(baseline) if baseline else 0.0,
            "hyde_recall": sum(bool(row.get("selected")) for row in hyde) / len(hyde) if hyde else 0.0,
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.eval_output_dir)
    baseline_eval, hyde_eval = output / "baseline", output / "hyde"
    output.mkdir(parents=True, exist_ok=True)
    _run_official(args.official_dir, args.baseline_output_dir, args.processed_output_dir, baseline_eval)
    _run_official(args.official_dir, args.hyde_output_dir, args.processed_output_dir, hyde_eval)
    validation = {row["query_id"]: row for row in read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))}
    baseline_rows = [
        row for row in read_jsonl(baseline_eval / "per_gold_evidence.jsonl")
        if row.get("match_mode") == "strict"
        and validation.get(row.get("query_id"), {}).get("task_family") == "multi_paper"
    ]
    hyde_rows = [
        row for row in read_jsonl(hyde_eval / "per_gold_evidence.jsonl")
        if row.get("match_mode") == "strict"
        and validation.get(row.get("query_id"), {}).get("task_family") == "multi_paper"
    ]
    baseline_index, baseline_traces = _trace_index(Path(args.baseline_output_dir) / "section_relevance.trace.jsonl")
    hyde_index, hyde_traces = _trace_index(Path(args.hyde_output_dir) / "section_relevance.trace.jsonl")
    multi_query_ids = {
        query_id for query_id, row in validation.items() if row.get("task_family") == "multi_paper"
    }
    baseline_traces = {query_id: trace for query_id, trace in baseline_traces.items() if query_id in multi_query_ids}
    hyde_traces = {query_id: trace for query_id, trace in hyde_traces.items() if query_id in multi_query_ids}
    baseline_by_gold = {(row["query_id"], row["gold_index"]): row for row in baseline_rows}
    hyde_by_gold = {(row["query_id"], row["gold_index"]): row for row in hyde_rows}
    recovered, displaced = [], []
    for key, baseline in baseline_by_gold.items():
        hyde = hyde_by_gold.get(key, {})
        if baseline.get("locator_extracted") and not baseline.get("selected") and hyde.get("selected"):
            recovered.append({"query_id": key[0], "gold_index": key[1], "source_type": baseline.get("gold_source_type")})
        if baseline.get("selected") and not hyde.get("selected"):
            displaced.append({"query_id": key[0], "gold_index": key[1], "source_type": baseline.get("gold_source_type")})
    summary = {
        "scope": "multi_paper_strict_fixed_section_budget",
        "baseline": _system_metrics(baseline_rows, baseline_index, baseline_traces),
        "hyde": _system_metrics(hyde_rows, hyde_index, hyde_traces),
        "by_source_type": _by_source(baseline_rows, hyde_rows),
        "hyde_recovered_misses": recovered,
        "baseline_hits_displaced": displaced,
        "hyde_recovered_count": len(recovered),
        "baseline_displaced_count": len(displaced),
        "net_selected_gold_evidence": len(recovered) - len(displaced),
        "fallback_reasons": dict(Counter(
            str(trace.get("hyde", {}).get("fallback_reason") or "none")
            for trace in hyde_traces.values()
            if trace.get("hyde", {}).get("fallback_used")
        )),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Multi-Paper Qwen vs Qwen + HyDE",
        "",
        "| Metric | Baseline | HyDE |",
        "|---|---:|---:|",
    ]
    for key in (
        "gold_section_recall_at_budget", "R_select_given_extracted", "MRR",
        "median_gold_section_rank", "p90_gold_section_rank", "selected_sections_per_query",
        "generated_claims_per_query", "retrieval_units_evaluated", "claim_unit_pairs_evaluated",
        "fallback_count", "fallback_rate",
    ):
        lines.append(f"| {key} | {summary['baseline'].get(key)} | {summary['hyde'].get(key)} |")
    lines.extend([
        "",
        f"- HyDE-recovered misses: {len(recovered)}",
        f"- Baseline hits displaced: {len(displaced)}",
        f"- Net selected gold evidence: {len(recovered) - len(displaced)}",
    ])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
