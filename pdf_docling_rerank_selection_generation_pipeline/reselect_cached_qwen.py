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


def _compact(record: dict) -> dict:
    return {
        key: record.get(key)
        for key in (
            "paper_id", "page", "record_id", "global_record_id", "record_type", "source_type",
            "label", "locator", "text", "section_id", "section_title", "section_type", "section_path",
            "document_order", "reading_order", "crop_path",
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
        "--processed-output-dir",
        default="",
        help="Load candidate-paper records lazily from processed artifacts instead of the large debug JSONL.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-budget", type=int, required=True)
    parser.add_argument("--min-distinct-papers", type=int, default=12)
    parser.add_argument("--modality-packages-per-paper", type=int, default=3)
    parser.add_argument("--supporting-text-packages-per-paper", type=int, default=0)
    parser.add_argument("--page-text-anchors-per-page", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=80000)
    parser.add_argument("--max-packages-per-page", type=int, default=2)
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


def _candidate_ids(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        query_id, paper_id = str(row.get("query_id") or ""), str(row.get("paper_id") or "")
        if query_id and paper_id:
            result[query_id].add(paper_id)
    return result


def main() -> int:
    args = _args()
    source, output = Path(args.source_output_dir), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs_path = Path(args.inputs_path) if args.inputs_path else find_official_file(args.official_dir, "validation_inputs.jsonl")
    validation = {str(row.get("query_id") or ""): row for row in read_jsonl(inputs_path)}
    candidates = _candidate_ids(source / "candidate_papers.jsonl")
    processed_root = Path(args.processed_output_dir) if args.processed_output_dir else None
    records_by_paper: dict[str, list[dict]] = defaultdict(list)
    if processed_root is None:
        # Legacy mode remains available for small artifacts. Full Docling runs
        # should use --processed-output-dir to avoid holding every duplicate
        # debug projection in memory.
        for record in _iter_jsonl(source / "symbolic_records.debug.jsonl"):
            records_by_paper[str(record.get("paper_id") or "")].append(record)
    config = EvidencePackageConfig(
        package_budget=args.package_budget,
        min_package_budget=args.package_budget,
        min_distinct_papers=args.min_distinct_papers,
        adaptive_stop=False,
        modality_packages_per_paper=args.modality_packages_per_paper,
        supporting_text_packages_per_paper=args.supporting_text_packages_per_paper,
        page_text_anchors_per_page=args.page_text_anchors_per_page,
        max_context_chars=args.max_context_chars,
        max_packages_per_page=args.max_packages_per_page,
    )
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
            paper_ids = candidates.get(query_id, set())
            candidate_union = trace.get("candidate_union") if isinstance(trace.get("candidate_union"), dict) else {}
            scored_ids = {
                str(record_id) for record_id in candidate_union.get("qwen_scored_record_ids") or []
                if str(record_id)
            }
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
            result = select_packages(
                query=str(sample.get("question") or sample.get("query") or ""),
                packages=build_packages(records, trace, config, context_records=context_records),
                primary_evidence_type=sample.get("primary_evidence_type"),
                is_multi_paper_task=str(sample.get("task_family") or "") == "multi_paper",
                route_queries=[],
                config=config,
            )
            row = {
                "query_id": query_id,
                "selection_method": "offline_cached_qwen_package_coverage",
                "selected_record_count": len(result["records"]),
                "selected_records": [_compact(record) for record in result["records"]],
                "selected_context_groups": [],
                "package_selection": result["trace"],
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
