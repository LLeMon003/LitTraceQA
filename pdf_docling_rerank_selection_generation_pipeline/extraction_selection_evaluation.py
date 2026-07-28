from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .data_io import find_official_file, read_jsonl, write_jsonl
from .evaluation_report import breakdown_by, metrics_from_rows, paper_source, write_examples_markdown, write_summary_markdown
from .evidence_matcher import gold_page, gold_source_type, match_gold_to_record, normalize_source_type


PARSER_SLUG = "pymupdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate symbolic extraction and selected-context coverage before VLM-2 answer generation.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--gold-path", default="")
    parser.add_argument("--output-dir", default="outputs/pdf_docling_rerank_selection_generation_pipeline")
    parser.add_argument("--processed-output-dir", default="processed_pdfs/pdf_docling_rerank_selection_generation_pipeline")
    parser.add_argument("--eval-output-dir", default="outputs/pdf_docling_rerank_selection_generation_pipeline/extraction_selection_eval")
    parser.add_argument("--candidate-papers-path", default="")
    parser.add_argument("--selected-contexts-path", default="")
    parser.add_argument("--section-relevance-trace-path", default="")
    parser.add_argument("--paper-metadata-path", default="")
    parser.add_argument("--match-mode", choices=["strict", "relaxed", "both"], default="both")
    parser.add_argument("--only-query-ids", default="")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--show-examples", action="store_true")
    parser.add_argument("--write-failure-cases", action="store_true")
    return parser.parse_args()


def _split_ids(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def _path_or_default(value: str, default: Path) -> Path:
    return Path(value) if str(value or "").strip() else default


def _load_gold_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.gold_path) if args.gold_path else find_official_file(args.official_dir, "validation.jsonl")
    rows = read_jsonl(path)
    only = _split_ids(args.only_query_ids)
    if only:
        rows = [row for row in rows if str(row.get("query_id") or "") in only]
    if args.max_queries is not None:
        rows = rows[: args.max_queries]
    return rows


def _gold_evidence(validation_row: dict[str, Any]) -> list[dict[str, Any]]:
    query_id = str(validation_row.get("query_id") or "")
    primary = validation_row.get("primary_evidence_type")
    result: list[dict[str, Any]] = []
    for index, evidence in enumerate(validation_row.get("evidence") or []):
        if not isinstance(evidence, dict):
            continue
        row = dict(evidence)
        row["query_id"] = query_id
        row["gold_index"] = index
        row["source_type"] = normalize_source_type(row.get("source_type") or primary)
        result.append(row)
    return result


def _group_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id") or "")].append(row)
    return grouped


def _iter_candidate_entries(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in candidate_rows:
        query_id = str(row.get("query_id") or "")
        nested = row.get("candidates")
        if isinstance(nested, list):
            for candidate in nested:
                if isinstance(candidate, dict):
                    copied = dict(candidate)
                    copied.setdefault("query_id", query_id)
                    entries.append(copied)
            continue
        entries.append(row)
    return entries


def _candidate_ids(candidate_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in _iter_candidate_entries(candidate_rows):
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            grouped[str(row.get("query_id") or "")].add(paper_id)
    return grouped


def _candidate_records(candidate_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_candidate_entries(candidate_rows):
        grouped[str(row.get("query_id") or "")].append(row)
    return grouped


def _load_pdf_availability(output_dir: Path) -> dict[tuple[str, str], bool]:
    path = output_dir / "pdf_availability.jsonl"
    availability: dict[tuple[str, str], bool] = {}
    if not path.exists():
        return availability
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "")
        paper_id = str(row.get("paper_id") or "")
        if query_id and paper_id:
            availability[(query_id, paper_id)] = bool(row.get("available"))
    return availability


def _candidate_union_by_query(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    output: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "")
        union = row.get("candidate_union") if isinstance(row.get("candidate_union"), dict) else {}
        record_ids = union.get("qwen_scored_record_ids") if isinstance(union.get("qwen_scored_record_ids"), list) else []
        if query_id:
            output[query_id] = {str(record_id) for record_id in record_ids if str(record_id)}
    return output


def _package_selection_by_query(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        query_id = str(row.get("query_id") or "")
        package_selection = row.get("package_selection")
        if query_id and isinstance(package_selection, dict):
            result[query_id] = package_selection
    return result


def _paper_dir(processed_root: Path, paper_id: str) -> Path:
    direct = processed_root / paper_id
    if direct.exists():
        return direct
    default = processed_root / PARSER_SLUG / paper_id
    if default.exists():
        return default
    matches = list(processed_root.glob(f"*/{paper_id}"))
    return matches[0] if matches else default


def _load_extraction_records(processed_root: Path, paper_ids: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, bool]]:
    records_by_paper: dict[str, list[dict[str, Any]]] = {}
    status_by_paper: dict[str, dict[str, Any]] = {}
    available_by_paper: dict[str, bool] = {}
    for paper_id in sorted(paper_ids):
        paper_dir = _paper_dir(processed_root, paper_id)
        status_path = paper_dir / "artifact_status.json"
        status = {}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = {}
        status_by_paper[paper_id] = status
        runtime_path = paper_dir / "symbolic_records.runtime.jsonl"
        available_by_paper[paper_id] = paper_dir.exists() and runtime_path.exists() and status.get("status") != "failed"
        records_by_paper[paper_id] = read_jsonl(runtime_path) if runtime_path.exists() else []
    return records_by_paper, status_by_paper, available_by_paper


def _selected_rows(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    origin_by_gid: dict[str, str] = {}
    for record in row.get("selected_records") or row.get("selected_evidence") or []:
        if isinstance(record, dict):
            copied = dict(record)
            copied.setdefault("_selection_hit_origin", "flat_selected_record")
            records.append(copied)
            gid = str(copied.get("global_record_id") or "")
            if gid:
                origin_by_gid.setdefault(gid, "flat_selected_record")
    for group in row.get("selected_context_groups") or []:
        if not isinstance(group, dict):
            continue
        expansion = str(group.get("expansion_mode") or "unknown")
        anchor = str(group.get("anchor_record_id") or "")
        for record in group.get("records") or []:
            if not isinstance(record, dict):
                continue
            copied = dict(record)
            gid = str(copied.get("global_record_id") or copied.get("record_id") or "")
            origin = "anchor" if anchor and gid == anchor else expansion
            copied.setdefault("_selection_hit_origin", origin)
            records.append(copied)
            if gid:
                origin_by_gid.setdefault(gid, origin)
    return records, origin_by_gid


def _selection_by_query(selected_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        query_id = str(row.get("query_id") or "")
        records, origin_by_gid = _selected_rows(row)
        result[query_id] = {"raw": row, "records": records, "origin_by_gid": origin_by_gid}
    return result


def _has_page(records: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    page = gold_page(gold)
    return any(str(record.get("paper_id") or "") == str(gold.get("paper_id") or "") and int(record.get("page") or 0) == page for record in records)


def _has_source(records: list[dict[str, Any]], gold: dict[str, Any]) -> bool:
    page = gold_page(gold)
    source = gold_source_type(gold)
    return any(
        str(record.get("paper_id") or "") == str(gold.get("paper_id") or "")
        and int(record.get("page") or 0) == page
        and normalize_source_type(record.get("source_type") or record.get("record_type")) == source
        for record in records
    )


def _failure_stage(*, candidate_hit: bool, pdf_available: bool, page_hit: bool, source_hit: bool, locator_hit: bool, selected: bool) -> str:
    if not candidate_hit:
        return "paper_not_retrieved"
    if not pdf_available:
        return "pdf_not_available"
    if not page_hit:
        return "page_not_extracted"
    if not source_hit:
        return "source_type_not_extracted"
    if not locator_hit:
        return "locator_not_extracted"
    if not selected:
        return "extracted_but_not_selected"
    return "selected"


def _coverage_row(
    query_id: str,
    selected: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
    mode: str | None = None,
    package_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = selected.get("raw") or {}
    records = selected.get("records") or []
    gold_rows = rows or []
    return {
        "query_id": query_id,
        "match_mode": mode,
        "selected_record_count": len(raw.get("selected_records") or []),
        "selected_context_group_count": len(raw.get("selected_context_groups") or []),
        "selected_source_type_distribution": dict(Counter(normalize_source_type(record.get("source_type") or record.get("record_type")) for record in records)),
        "selected_section_type_distribution": dict(Counter(str(record.get("section_type") or "unknown") for record in records)),
        "gold_evidence_count": len(gold_rows),
        "gold_selected_count": sum(1 for row in gold_rows if row.get("selected")),
        "gold_anchor_hit_count": sum(1 for row in gold_rows if row.get("selection_hit_origin") == "anchor"),
        "gold_expansion_hit_count": sum(1 for row in gold_rows if row.get("selection_hit_origin") in {"local_window", "subsection_full", "section_full"}),
        "gold_flat_selected_record_hit_count": sum(1 for row in gold_rows if row.get("selection_hit_origin") == "flat_selected_record"),
        "selection_hit_origin_distribution": dict(Counter(str(row.get("selection_hit_origin") or "unknown") for row in gold_rows if row.get("selected"))),
        "package_budget": (package_selection or {}).get("package_budget"),
        "max_context_chars": (package_selection or {}).get("max_context_chars"),
        "selected_package_count": (package_selection or {}).get("selected_package_count"),
        "selected_context_chars": (package_selection or {}).get("selected_char_count"),
        "requested_modalities": (package_selection or {}).get("requested_modalities", []),
    }


def _query_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = metrics_from_rows(rows)
    metrics["failure_summary"] = dict(Counter(str(row.get("failure_stage") or "unknown_failure") for row in rows))
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    eval_dir = Path(args.eval_output_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = _path_or_default(args.candidate_papers_path, output_dir / "candidate_papers.jsonl")
    selected_path = _path_or_default(args.selected_contexts_path, output_dir / "selected_symbolic_contexts.debug.jsonl")
    trace_path = _path_or_default(args.section_relevance_trace_path, output_dir / "section_relevance.trace.jsonl")
    metadata_path = _path_or_default(args.paper_metadata_path, find_official_file(args.official_dir, "paper_metadata.jsonl"))
    processed_root = Path(args.processed_output_dir)

    validation_rows = _load_gold_rows(args)
    candidate_rows = read_jsonl(candidate_path) if candidate_path.exists() else []
    selected_debug_rows = read_jsonl(selected_path) if selected_path.exists() else []
    metadata = {str(row.get("paper_id") or ""): row for row in read_jsonl(metadata_path)} if metadata_path.exists() else {}
    candidate_ids_by_query = _candidate_ids(candidate_rows)
    candidates_by_query = _candidate_records(candidate_rows)
    selected_by_query = _selection_by_query(selected_debug_rows)
    candidate_union_by_query = _candidate_union_by_query(trace_path)
    package_selection_by_query = _package_selection_by_query(trace_path)
    pdf_availability = _load_pdf_availability(output_dir)
    all_candidate_papers = {str(row.get("paper_id") or "") for row in _iter_candidate_entries(candidate_rows) if row.get("paper_id")}
    records_by_paper, status_by_paper, processed_available = _load_extraction_records(processed_root, all_candidate_papers)
    modes = ["strict", "relaxed"] if args.match_mode == "both" else [args.match_mode]

    per_gold: list[dict[str, Any]] = []
    for validation in validation_rows:
        query_id = str(validation.get("query_id") or "")
        selected_info = selected_by_query.get(query_id, {"records": [], "origin_by_gid": {}, "raw": {}})
        selected_records = selected_info.get("records") or []
        origin_by_gid = selected_info.get("origin_by_gid") or {}
        for gold in _gold_evidence(validation):
            paper_id = str(gold.get("paper_id") or "")
            source = gold_source_type(gold)
            page = gold_page(gold)
            candidate_hit = paper_id in candidate_ids_by_query.get(query_id, set())
            pdf_available = pdf_availability.get((query_id, paper_id), processed_available.get(paper_id, False))
            extraction_records = records_by_paper.get(paper_id, [])
            candidate_union_ids = candidate_union_by_query.get(query_id)
            candidate_union_records = (
                [record for record in extraction_records if str(record.get("global_record_id") or "") in candidate_union_ids]
                if candidate_union_ids is not None
                else []
            )
            page_hit = _has_page(extraction_records, gold)
            source_hit = _has_source(extraction_records, gold)
            for mode in modes:
                extraction_match = match_gold_to_record(gold, extraction_records, mode)
                candidate_union_match = match_gold_to_record(gold, candidate_union_records, mode) if candidate_union_ids is not None else None
                selection_match = match_gold_to_record(gold, selected_records, mode)
                locator_hit = bool(extraction_match.get("matched"))
                selected_hit = bool(selection_match.get("matched"))
                selected_gid = str(selection_match.get("matched_global_record_id") or "")
                stage = _failure_stage(
                    candidate_hit=candidate_hit,
                    pdf_available=bool(pdf_available),
                    page_hit=page_hit,
                    source_hit=source_hit,
                    locator_hit=locator_hit,
                    selected=selected_hit,
                )
                per_gold.append(
                    {
                        "query_id": query_id,
                        "match_mode": mode,
                        "gold_index": gold.get("gold_index"),
                        "gold_paper_id": paper_id,
                        "gold_page": page,
                        "gold_source_type": source,
                        "gold_locator": gold.get("locator") if isinstance(gold.get("locator"), dict) else {},
                        "gold_evidence_text_or_value": gold.get("evidence_text_or_value"),
                        "paper_source": paper_source(metadata.get(paper_id), paper_id),
                        "candidate_paper_hit": candidate_hit,
                        "pdf_available": bool(pdf_available),
                        "processed_status": (status_by_paper.get(paper_id) or {}).get("status"),
                        "page_extracted": page_hit,
                        "source_type_extracted": source_hit,
                        "locator_extracted": locator_hit,
                        "candidate_union_available": candidate_union_ids is not None,
                        "candidate_union_selected": bool(candidate_union_match and candidate_union_match.get("matched")),
                        "selected": selected_hit,
                        "failure_stage": stage,
                        "failure_reason": selection_match.get("reason") if selected_hit else extraction_match.get("reason"),
                        "matched_extraction_record": extraction_match.get("matched_global_record_id"),
                        "matched_selection_record": selection_match.get("matched_global_record_id"),
                        "selection_hit_origin": origin_by_gid.get(selected_gid) or (selection_match.get("match_level") if selected_hit else None) or "unknown",
                    }
                )

    write_jsonl(eval_dir / "per_gold_evidence.jsonl", per_gold)
    failure_rows = [row for row in per_gold if row.get("failure_stage") != "selected"]
    if args.write_failure_cases or True:
        write_jsonl(eval_dir / "failure_cases.jsonl", failure_rows)
    coverage_rows = []
    rows_by_query_mode = _group_by_query_mode(per_gold)
    for query_id in sorted({str(row.get("query_id") or "") for row in validation_rows} | set(selected_by_query)):
        selected_info = selected_by_query.get(query_id, {"records": [], "raw": {}})
        for mode in modes:
            coverage_rows.append(
                _coverage_row(
                    query_id,
                    selected_info,
                    rows_by_query_mode.get((query_id, mode), []),
                    mode,
                    package_selection_by_query.get(query_id),
                )
            )
    write_jsonl(eval_dir / "selected_context_coverage.jsonl", coverage_rows)

    per_query: list[dict[str, Any]] = []
    for (query_id, mode), rows in sorted(rows_by_query_mode.items()):
        validation = next((row for row in validation_rows if str(row.get("query_id") or "") == query_id), {})
        metrics = _query_metrics(rows)
        metrics.update(
            {
                "query_id": query_id,
                "match_mode": mode,
                "primary_evidence_type": normalize_source_type(validation.get("primary_evidence_type")),
                "gold_evidence_count": len(rows),
                "package_budget": (package_selection_by_query.get(query_id) or {}).get("package_budget"),
                "selected_package_count": (package_selection_by_query.get(query_id) or {}).get("selected_package_count"),
                "selected_context_chars": (package_selection_by_query.get(query_id) or {}).get("selected_char_count"),
            }
        )
        per_query.append(metrics)
    write_jsonl(eval_dir / "per_query_metrics.jsonl", per_query)

    summary_by_mode: dict[str, Any] = {}
    source_breakdown_by_mode: dict[str, Any] = {}
    paper_breakdown_by_mode: dict[str, Any] = {}
    for mode in modes:
        rows = [row for row in per_gold if row.get("match_mode") == mode]
        source_breakdown_by_mode[mode] = breakdown_by(rows, "gold_source_type", include_all_source_types=True)
        paper_breakdown_by_mode[mode] = breakdown_by(rows, "paper_source")
        summary_by_mode[mode] = {
            "aggregate": metrics_from_rows(rows),
            "by_source_type": source_breakdown_by_mode[mode],
            "by_paper_source": paper_breakdown_by_mode[mode],
            "failure_stage_distribution": dict(Counter(str(row.get("failure_stage") or "unknown_failure") for row in rows)),
        }
    primary_mode = "strict" if "strict" in summary_by_mode else modes[0]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "match_mode": args.match_mode,
        "gold_query_count": len(validation_rows),
        "gold_evidence_count": len([row for row in per_gold if row.get("match_mode") == primary_mode]),
        "candidate_papers_path": str(candidate_path),
        "selected_contexts_path": str(selected_path),
        "section_relevance_trace_path": str(trace_path),
        "processed_output_dir": str(processed_root),
        **summary_by_mode[primary_mode],
        "by_match_mode": summary_by_mode,
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (eval_dir / "source_type_breakdown.json").write_text(json.dumps(source_breakdown_by_mode, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (eval_dir / "paper_source_breakdown.json").write_text(json.dumps(paper_breakdown_by_mode, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    examples = failure_rows if args.show_examples else []
    write_summary_markdown(eval_dir / "summary.md", summary, examples)
    write_examples_markdown(eval_dir / "examples.md", failure_rows)
    return summary


def _group_by_query_mode(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("query_id") or ""), str(row.get("match_mode") or "strict"))].append(row)
    return grouped


def main() -> int:
    summary = run(parse_args())
    print(json.dumps({"aggregate": summary.get("aggregate"), "failure_stage_distribution": summary.get("failure_stage_distribution")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
