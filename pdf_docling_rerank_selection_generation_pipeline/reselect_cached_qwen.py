"""Re-run package selection from frozen Qwen traces without rescoring units."""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .data_io import find_official_file, read_jsonl
from .evidence_packages import EvidencePackageConfig, build_packages, select_packages
from .section_relevance import query_for_relevance_mode
from .slot_generation import plan_package_routes, plan_paper_package_routes
from .task_structure import derive_task_structure


def _compact(record: dict) -> dict:
    return {
        key: record.get(key)
        for key in (
            "paper_id", "page", "record_id", "global_record_id", "record_type", "source_type",
            "label", "locator", "text", "section_id", "section_title", "section_type", "section_path",
            "document_order", "reading_order", "crop_path", "table_structure",
        )
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline package re-selection using an existing full-Qwen trace.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument(
        "--inputs-path",
        default="",
        help="Input-only query file. Defaults to validation_inputs.jsonl so offline selection never reads gold answers/evidence.",
    )
    parser.add_argument("--source-output-dir", required=True)
    parser.add_argument(
        "--slot-plans-input",
        default="",
        help="Frozen slot_plans.jsonl. Defaults to the source output; plans add only post-Qwen package routes.",
    )
    parser.add_argument(
        "--processed-output-dir",
        default="",
        help="Load candidate-paper records lazily from processed artifacts instead of the large debug JSONL.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-budget", type=int, default=0, help="Legacy fixed budget for every query.")
    parser.add_argument("--single-package-budget", type=int, default=0)
    parser.add_argument("--multi-package-budget", type=int, default=0)
    parser.add_argument("--min-package-budget", type=int, default=4)
    parser.add_argument("--no-adaptive-stop", action="store_true")
    # Keep the offline baseline aligned with the online selector defaults.
    parser.add_argument("--min-distinct-papers", type=int, default=4)
    parser.add_argument("--modality-packages-per-paper", type=int, default=2)
    parser.add_argument("--supporting-text-packages-per-paper", type=int, default=0)
    parser.add_argument("--page-text-anchors-per-page", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=80000)
    parser.add_argument("--max-packages-per-page", type=int, default=2)
    parser.add_argument(
        "--paper-local-bm25-route-mode",
        choices=("disabled", "original", "mask_method_aliases"),
        default="disabled",
        help="Replay the online multi-paper per-candidate BM25 route when it was enabled.",
    )
    parser.add_argument(
        "--retriever-pool-budget",
        type=int,
        default=0,
        help=">0 restricts selectable records to the hybrid retriever pool (special objects always preserved).",
    )
    parser.add_argument("--only-query-ids", default="")
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSONL rows without materializing multi-hundred-MB score traces."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _processed_records(root: Path, paper_id: str, allowed_ids: set[str] | None = None) -> list[dict]:
    direct = root / paper_id
    if not direct.is_dir():
        matches = list(root.glob(f"*/{paper_id}"))
        direct = matches[0] if matches else direct
    for name in ("symbolic_records.runtime.jsonl", "symbolic_records.debug.jsonl"):
        path = direct / name
        if path.is_file():
            rows = _iter_jsonl(path)
            if not allowed_ids:
                return list(rows)
            return [
                row for row in rows
                if str(row.get("global_record_id") or row.get("record_id") or "") in allowed_ids
            ]
    return []


def _candidate_rows(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        query_id, paper_id = str(row.get("query_id") or ""), str(row.get("paper_id") or "")
        if query_id and paper_id:
            result[query_id].append(row)
    return result


def _slot_plans(path: Path) -> dict[str, dict]:
    plans: dict[str, dict] = {}
    if not path.is_file():
        return plans
    for row in _iter_jsonl(path):
        query_id, plan = str(row.get("query_id") or ""), row.get("plan")
        if query_id and isinstance(plan, dict):
            plans[query_id] = plan
    return plans


def _retriever_pool_record_ids(
    trace: dict,
    question: str,
    processed_root: Path,
    budget: int,
) -> set[str]:
    """Record ids selectable under the hybrid retriever pool for one query."""
    from .content_retriever import build_retriever_pool, hybrid_retriever_scores

    paper_ids = {str(unit.get("paper_id") or "") for unit in (trace.get("ranked_units") or [])}
    record_text: dict[str, str] = {}
    for paper_id in paper_ids:
        for record in _processed_records(processed_root, paper_id):
            gid = str(record.get("global_record_id") or record.get("record_id") or "")
            if gid:
                record_text[gid] = str(record.get("text") or "")
    units = [
        {
            "unit_type": str(unit.get("unit_type") or ""),
            "section_id": unit.get("section_id"),
            "text": " ".join(record_text.get(str(record_id), "") for record_id in (unit.get("record_ids") or [])),
        }
        for unit in (trace.get("ranked_units") or [])
    ]
    scores = hybrid_retriever_scores(question, units, trace.get("sections"))
    pool = build_retriever_pool(units, scores, budget)
    pool_ids: set[str] = set()
    for index in pool:
        for record_id in (trace.get("ranked_units") or [])[index].get("record_ids") or []:
            pool_ids.add(str(record_id))
    return pool_ids


def main() -> int:
    args = _args()
    source, output = Path(args.source_output_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs_path = Path(args.inputs_path) if args.inputs_path else find_official_file(args.official_dir, "validation_inputs.jsonl")
    validation = {str(row.get("query_id") or ""): row for row in read_jsonl(inputs_path)}
    candidate_rows = _candidate_rows(source / "candidate_papers.jsonl")
    plans = _slot_plans(Path(args.slot_plans_input) if args.slot_plans_input else source / "slot_plans.jsonl")
    processed_root = Path(args.processed_output_dir) if args.processed_output_dir else None
    records_by_paper: dict[str, list[dict]] = defaultdict(list)
    if processed_root is None:
        # Legacy mode remains available for small artifacts. Full Docling runs
        # should use --processed-output-dir to avoid holding every duplicate
        # debug projection in memory.
        for record in _iter_jsonl(source / "symbolic_records.debug.jsonl"):
            records_by_paper[str(record.get("paper_id") or "")].append(record)
    if max(args.package_budget, args.single_package_budget, args.multi_package_budget) <= 0:
        raise ValueError("Set --package-budget or both task-specific package budgets.")
    only = {item.strip() for item in args.only_query_ids.split(",") if item.strip()}
    output_path = output / "selected_symbolic_contexts.debug.jsonl"
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in _iter_jsonl(source / "section_relevance.trace.jsonl"):
            query_id = str(trace.get("query_id") or "")
            if only and query_id not in only:
                continue
            sample = validation.get(query_id)
            if not sample:
                continue
            candidate_metadata = candidate_rows.get(query_id, [])
            paper_ids = {str(row.get("paper_id") or "") for row in candidate_metadata}
            candidate_union = trace.get("candidate_union") if isinstance(trace.get("candidate_union"), dict) else {}
            scored_ids = {
                str(record_id) for record_id in candidate_union.get("qwen_scored_record_ids") or []
                if str(record_id)
            }
            if args.retriever_pool_budget > 0 and processed_root is not None:
                pool_ids = _retriever_pool_record_ids(
                    trace,
                    str(sample.get("question") or sample.get("query") or ""),
                    Path(args.processed_output_dir),
                    args.retriever_pool_budget,
                )
                scored_ids = scored_ids & pool_ids
            if processed_root is not None:
                records = [
                    record for paper_id in paper_ids
                    for record in _processed_records(processed_root, paper_id, scored_ids or None)
                ]
                # Reference paragraphs do not need a Qwen score: they are
                # context for an already scored object anchor, not selectable
                # packages themselves. Load the full per-paper catalog only
                # for explicit Table/Figure/Eq narration resolution.
                context_records = [
                    record for paper_id in paper_ids
                    for record in _processed_records(processed_root, paper_id)
                ]
            else:
                records = [record for paper_id in paper_ids for record in records_by_paper.get(paper_id, [])]
                context_records = records
            plan = plans.get(query_id)
            structure = derive_task_structure(sample, plan)
            package_budget = (
                args.multi_package_budget if structure.is_multi_paper and args.multi_package_budget > 0
                else args.single_package_budget if not structure.is_multi_paper and args.single_package_budget > 0
                else args.package_budget
            )
            if package_budget <= 0:
                raise ValueError("A package budget is required for every task family.")
            config = EvidencePackageConfig(
                package_budget=package_budget,
                min_package_budget=max(1, args.min_package_budget),
                min_distinct_papers=args.min_distinct_papers,
                adaptive_stop=not args.no_adaptive_stop,
                modality_packages_per_paper=args.modality_packages_per_paper,
                supporting_text_packages_per_paper=args.supporting_text_packages_per_paper,
                page_text_anchors_per_page=args.page_text_anchors_per_page,
                max_context_chars=args.max_context_chars,
                max_packages_per_page=args.max_packages_per_page,
            )
            global_query = str(trace.get("query") or sample.get("question") or sample.get("query") or "")
            slot_package_routes = plan_package_routes(plan, str(sample.get("question") or sample.get("query") or ""))
            slot_paper_package_routes = plan_paper_package_routes(
                plan, candidate_metadata, str(sample.get("question") or sample.get("query") or "")
            )
            fallback_source_types = {
                source_type
                for route in slot_package_routes
                if route.get("catalog_fallback")
                for source_type in route.get("record_types") or []
            }
            if fallback_source_types:
                seen_record_ids = {str(record.get("global_record_id") or record.get("record_id") or "") for record in records}
                records.extend(
                    record for record in context_records
                    if str(record.get("source_type") or "") in fallback_source_types
                    and str(record.get("global_record_id") or record.get("record_id") or "") not in seen_record_ids
                )
            paper_local_route_queries = (
                [
                    (paper_id, query_for_relevance_mode(global_query, args.paper_local_bm25_route_mode))
                    for paper_id in sorted(paper_ids)
                ]
                if structure.is_multi_paper and args.paper_local_bm25_route_mode != "disabled"
                else []
            )
            result = select_packages(
                query=global_query,
                source_hint_query=str(sample.get("question") or sample.get("query") or ""),
                packages=build_packages(records, trace, config, context_records=context_records),
                preferred_source_types=structure.preferred_source_types,
                is_multi_paper_task=structure.is_multi_paper,
                route_queries=[],
                slot_route_queries=[
                    (route["query"], tuple(route["record_types"]))
                    for route in slot_package_routes
                    if not route.get("catalog_fallback")
                ],
                catalog_fallback_slot_route_queries=[
                    (route["query"], tuple(route["record_types"]))
                    for route in slot_package_routes
                    if route.get("catalog_fallback")
                ],
                slot_paper_route_queries=[
                    (route["paper_id"], route["query"], tuple(route["record_types"]))
                    for route in slot_paper_package_routes
                ],
                paper_local_route_queries=paper_local_route_queries,
                config=config,
            )
            row = {
                "query_id": query_id,
                "selection_method": "offline_cached_qwen_package_coverage",
                "selected_record_count": len(result["records"]),
                "selected_records": [_compact(record) for record in result["records"]],
                "selected_context_groups": [],
                "package_selection": result["trace"],
                "slot_package_routes": slot_package_routes,
                "slot_paper_package_routes": slot_paper_package_routes,
                "frozen_qwen_trace": {
                    "reused": True,
                    "query": global_query,
                    "instruction_version": trace.get("instruction_version"),
                    "template_version": trace.get("template_version"),
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            # The full trace is intentionally read one query at a time. Drop
            # the package graph before moving on, since some multi-paper rows
            # have tens of thousands of raw parser records.
            del records, context_records, result
            gc.collect()
    print(json.dumps({"queries": count, "output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
