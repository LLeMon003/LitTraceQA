from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from .config import load_pipeline_config
from .data_io import append_jsonl, append_jsonl_rows, extract_answer_contract, find_official_file, read_jsonl, write_jsonl
from .metadata_index import build_metadata_records, retrieve_candidates
from .metadata_selection import empty_metadata_prediction
from .multi_paper_hyde import MultiPaperHyDEConfig
from .paper_conditioned_claims import PaperConditionedClaimsConfig
from .openreview_filter import filter_openreview_metadata
from .parser import extract_json_object, make_fallback_prediction, normalize_prediction, strip_internal_grounding, validate_prediction_shape
from .pdf_cache import ensure_candidate_pdfs, write_pdf_availability
from .pdf_extraction_parser import ARTIFACT_VERSION, extract_pdf_symbolic_records_with_backend
from .symbolic_context_selector import select_symbolic_contexts
from .section_relevance import SectionRelevanceConfig
from .transcription_backends import BACKEND_CHOICES, normalize_backend_name
from .vlm_answer_client import VLMAnswerClient
from .vlm_answer_prompt_builder import build_symbolic_answer_prompt


BASELINE_TYPE = "pdf_docling_rerank_selection_generation"
DEFAULT_TRANSCRIPTION_BACKEND = "docling"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parser-first PDF extraction symbolic VLM LitTraceQA baseline.")
    p.add_argument("--official-dir", default="official_dev")
    p.add_argument("--output-dir", default="outputs/pdf_docling_rerank_selection_generation_pipeline")
    p.add_argument("--pdf-output-dir", default="raw_pdfs")
    p.add_argument("--processed-output-dir", default="processed_pdfs/pdf_docling_rerank_selection_generation_pipeline")
    p.add_argument("--top-k-papers", type=int, default=None)
    p.add_argument("--top-n-records", type=int, default=24)
    p.add_argument("--top-n-visual-records", type=int, default=6)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--only-query-ids", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--replace-query-results",
        action="store_true",
        help="With --resume and --only-query-ids, remove existing rows for those query ids before rerunning them.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retrieval-only", action="store_true", help="Only run metadata paper retrieval and write candidate_papers.jsonl.")
    p.add_argument(
        "--candidate-papers-input",
        default="",
        help="Reuse flat candidate_papers.jsonl from an earlier retrieval stage instead of rerunning metadata retrieval.",
    )
    p.add_argument("--skip-selection", action="store_true", help="Stop after PDF retrieval/extraction and do not run symbolic context selection.")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--env-path", default="pdf_docling_rerank_selection_generation_pipeline/.env")
    p.add_argument("--skip-openreview-papers", action="store_true")
    p.add_argument("--pdf-sleep-seconds", type=float, default=2.0)
    p.add_argument("--pdf-timeout-seconds", type=float, default=60.0)
    p.add_argument("--pdf-max-retries", type=int, default=2)
    p.add_argument("--pdf-overwrite", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--skip-extraction", action="store_true")
    p.add_argument(
        "--extraction-cache-policy",
        choices=["reuse_complete_only", "reuse_partial_allowed", "refresh", "fail_if_missing"],
        default="reuse_complete_only",
    )
    p.add_argument("--extract-all-pages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-pages-per-paper", type=int, default=None)
    p.add_argument("--enable-figure-crops", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-table-crops", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-equation-crops", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--crop-dpi", type=int, default=160)
    p.add_argument("--min-text-block-chars", type=int, default=20)
    p.add_argument("--include-debug-bbox", action="store_true")
    p.add_argument("--show-progress", action="store_true")
    p.add_argument("--vlm2-context-mode", choices=["text_only", "cropped_image"], default=None)
    p.add_argument("--transcription-backend", choices=BACKEND_CHOICES, default=None)
    return p.parse_args()


def _split_query_ids(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def _paths(output_dir: Path, resume: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "predictions": "predictions.jsonl",
        "internal_predictions": "internal_predictions.jsonl",
        "candidate_papers": "candidate_papers.jsonl",
        "pdf_availability": "pdf_availability.jsonl",
        "pdf_extraction_artifacts": "pdf_extraction_artifacts.jsonl",
        "symbolic_records_runtime": "symbolic_records.runtime.jsonl",
        "symbolic_records_debug": "symbolic_records.debug.jsonl",
        "selected_contexts_debug": "selected_symbolic_contexts.debug.jsonl",
        "selected_contexts_prompt": "selected_symbolic_contexts.prompt.jsonl",
        "section_relevance_trace": "section_relevance.trace.jsonl",
        "raw_answer": "raw_vlm_answer_responses.jsonl",
        "prompt_previews": "prompt_previews.jsonl",
        "errors": "errors.jsonl",
        "report": "run_report.md",
    }
    paths = {key: output_dir / value for key, value in names.items()}
    if resume:
        for path in paths.values():
            if path.suffix == ".jsonl":
                path.touch(exist_ok=True)
    else:
        for key, path in paths.items():
            if key != "report":
                path.write_text("", encoding="utf-8")
    return paths


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _load_reused_candidates(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load the flat candidate artifact written by this baseline, grouped by query."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"candidate papers input does not exist: {source}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(source):
        query_id = str(row.get("query_id") or "")
        paper_id = str(row.get("paper_id") or "")
        if not query_id or not paper_id:
            continue
        grouped.setdefault(query_id, []).append(dict(row))
    for candidates in grouped.values():
        candidates.sort(key=lambda row: int(row.get("rank") or sys.maxsize))
    if not grouped:
        raise ValueError(f"candidate papers input contains no flat candidate rows: {source}")
    return grouped


def _acquire_output_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"output directory is already in use: {output_dir}") from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def _remove_query_rows(paths: dict[str, Path], query_ids: set[str], show_progress: bool = False) -> None:
    query_scoped_paths = {
        key: path
        for key, path in paths.items()
        if key not in {"symbolic_records_runtime", "symbolic_records_debug", "pdf_extraction_artifacts"}
    }
    for key, path in query_scoped_paths.items():
        if path.suffix != ".jsonl" or not path.exists():
            continue
        _progress(show_progress, f"replacement filtering {path.name}")
        temp_path = path.with_name(f"{path.name}.replace.{os.getpid()}.tmp")
        try:
            with path.open(encoding="utf-8") as source, temp_path.open("w", encoding="utf-8") as target:
                for line in source:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        target.write(line)
                        continue
                    row_query_id = str(row.get("query_id") or "")
                    # A prior process-level error has no query provenance and
                    # would otherwise survive every query-level replacement.
                    if key == "errors" and not row_query_id:
                        continue
                    if row_query_id not in query_ids:
                        target.write(line)
            temp_path.replace(path)
            _progress(show_progress, f"replacement filtered {path.name}")
        finally:
            temp_path.unlink(missing_ok=True)


def _empty_answer_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return empty_metadata_prediction(sample).get("answer", {})


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_report(
    path: Path,
    stats: dict[str, Any],
    paths: dict[str, Path],
    *,
    refresh_row_counts: bool = False,
) -> None:
    if refresh_row_counts:
        for key, item in paths.items():
            if item.suffix == ".jsonl":
                stats[f"{key}_rows"] = _count_jsonl(item)
    lines = [
        "# PDF Extraction Symbolic VLM Baseline Report",
        "",
        f"- baseline_type: `{BASELINE_TYPE}`",
        f"- pipeline: metadata retrieval -> PDF cache -> {stats.get('transcription_backend', 'unknown')} transcription backend -> symbolic selector -> VLM-2",
        "- comparison note: this baseline removes VLM-1 and pre-parser page selection; remaining bottlenecks are candidate retrieval, parser quality, symbolic selection, and VLM-2.",
        "",
        "## Stats",
        "```json",
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_dist_add(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value or 0)


def _messages_for_json_retry(messages: list[dict[str, Any]], evidence_limit: int) -> list[dict[str, Any]]:
    retried = [dict(message) for message in messages]
    for message in reversed(retried):
        if message.get("role") != "user":
            continue
        message["content"] = (
            str(message.get("content") or "")
            + "\n\nRETRY REQUIREMENT: Return one complete valid JSON object only. "
            + f"Use at most {evidence_limit} directly supporting evidence_ref items. "
            + "Do not enumerate all selected records and do not include reasoning or markdown."
        )
        break
    return retried


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "] 0/0"
    done = max(0, min(done, total))
    filled = int(round(width * done / total))
    percent = 100.0 * done / total
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} {percent:5.1f}%"


def _source_dist_text(source_dist: dict[str, Any], limit: int = 6) -> str:
    items = [(str(key), int(value or 0)) for key, value in source_dist.items() if int(value or 0) > 0]
    items.sort(key=lambda item: (-item[1], item[0]))
    if not items:
        return "{}"
    shown = ", ".join(f"{key}:{value}" for key, value in items[:limit])
    if len(items) > limit:
        shown += ", ..."
    return "{" + shown + "}"


def _progress(enabled: bool, message: str) -> None:
    if not enabled:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", file=sys.stderr, flush=True)


def _load_runtime_records(paper_dir: Path, candidate_score: float) -> list[dict[str, Any]]:
    records = _read_jsonl_if_exists(paper_dir / "symbolic_records.runtime.jsonl")
    debug_by_id = {
        str(row.get("global_record_id") or ""): row
        for row in _read_jsonl_if_exists(paper_dir / "symbolic_records.debug.jsonl")
        if row.get("global_record_id")
    }
    for record in records:
        record["_candidate_bm25_score"] = candidate_score
        debug = debug_by_id.get(str(record.get("global_record_id") or "")) or {}
        for key in ("crop_path", "table_crop_path", "figure_crop_path", "equation_algorithm_crop_path"):
            if debug.get(key):
                record[key] = debug.get(key)
    return records


def _project_runtime_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "paper_id",
        "page",
        "record_id",
        "global_record_id",
        "section_id",
        "section_title",
        "section_level",
        "section_path",
        "section_type",
        "record_type",
        "source_type",
        "label",
        "locator",
        "text",
        "reading_order",
        "document_order",
    ]
    return {key: record.get(key) for key in keys}


def _primary_type(sample: dict[str, Any]) -> str:
    return str(sample.get("primary_evidence_type") or sample.get("evidence_type") or "text_span")


def _is_multi_paper_task(sample: dict[str, Any]) -> bool:
    task_family = str(sample.get("task_family") or "").lower()
    if "multi" in task_family:
        return True
    question = str(sample.get("question") or "").lower()
    return any(term in question for term in ["across papers", "across all papers", "which papers", "each paper", "among the papers"])


def _top_k_for_sample(sample: dict[str, Any], config: Any, cli_top_k: int | None) -> int:
    if cli_top_k is not None:
        return max(1, int(cli_top_k))
    if not config.task_family_budget_enabled:
        return max(1, int(config.single_paper_top_k_papers))
    if _is_multi_paper_task(sample):
        return max(1, int(config.multi_paper_top_k_papers))
    return max(1, int(config.single_paper_top_k_papers))


def main() -> int:
    run_started = time.monotonic()
    args = parse_args()
    config = load_pipeline_config(args.env_path)
    only_ids = _split_query_ids(args.only_query_ids)
    if args.replace_query_results and (not args.resume or not only_ids):
        raise ValueError("--replace-query-results requires --resume and non-empty --only-query-ids")
    if args.retrieval_only and args.candidate_papers_input:
        raise ValueError("--retrieval-only cannot be combined with --candidate-papers-input")
    transcription_backend = normalize_backend_name(
        args.transcription_backend or config.transcription_backend,
        default=DEFAULT_TRANSCRIPTION_BACKEND,
    )
    output_dir = Path(args.output_dir)
    output_lock = _acquire_output_lock(output_dir)
    paths = _paths(output_dir, args.resume)
    if args.replace_query_results:
        _remove_query_rows(paths, only_ids, args.show_progress)
    processed_root = Path(args.processed_output_dir)
    structured_root = processed_root / transcription_backend
    inputs = read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    validation_rows = read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))
    validation_by_id = {str(row.get("query_id") or ""): row for row in validation_rows if row.get("query_id")}
    if only_ids:
        inputs = [row for row in inputs if str(row.get("query_id") or "") in only_ids]
    if args.max_queries is not None:
        inputs = inputs[: args.max_queries]
    reused_candidates_by_query = (
        _load_reused_candidates(args.candidate_papers_input)
        if args.candidate_papers_input
        else {}
    )
    if reused_candidates_by_query:
        missing_candidate_queries = [
            str(row.get("query_id") or "")
            for row in inputs
            if str(row.get("query_id") or "") not in reused_candidates_by_query
        ]
        if missing_candidate_queries:
            raise ValueError(
                "candidate papers input is missing queried ids: "
                + ", ".join(missing_candidate_queries[:10])
            )
    metadata = build_metadata_records(read_jsonl(find_official_file(args.official_dir, "paper_metadata.jsonl")))
    if args.skip_openreview_papers:
        metadata, _ = filter_openreview_metadata(metadata)
    metadata_by_id = {str(row.get("paper_id") or ""): row for row in metadata}
    answer_client = VLMAnswerClient(config)
    context_mode = args.vlm2_context_mode or config.vlm2_context_mode
    if context_mode == "cropped_image" and not answer_client.supports_image_input():
        context_mode = "text_only"
    stats: dict[str, Any] = {
        "run_status": "running",
        "baseline_type": BASELINE_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "transcription_backend": transcription_backend,
        "backend_output_root": str(structured_root),
        "processed_queries": 0,
        "top_k_papers_override": args.top_k_papers,
        "candidate_papers_input": str(args.candidate_papers_input or ""),
        "task_family_budget_enabled": config.task_family_budget_enabled,
        "single_paper_top_k_papers": config.single_paper_top_k_papers,
        "multi_paper_top_k_papers": config.multi_paper_top_k_papers,
        "pdf_available": 0,
        "pdf_missing": 0,
        "parsed_paper_count": 0,
        "reused_extraction_count": 0,
        "parsed_page_count": 0,
        "failed_page_count": 0,
        "runtime_records_total": 0,
        "source_type_distribution": {},
        "selected_source_type_distribution": {},
        "vlm2_context_truncation_count": 0,
        "answer_api_calls": 0,
        "selection_skipped_queries": 0,
        "successful_predictions": 0,
        "fallback_predictions": 0,
        "errors": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    exit_code = 0
    total_queries = len(inputs)
    _progress(
        args.show_progress,
        "starting baseline "
        f"queries={total_queries} output_dir={Path(args.output_dir)} processed_output_dir={processed_root} "
        f"transcription_backend={transcription_backend} backend_output_root={structured_root} "
        f"context_mode={context_mode} retrieval_only={args.retrieval_only} "
        f"candidate_papers_input={args.candidate_papers_input or '-'} "
        f"skip_generation={args.skip_generation} dry_run={args.dry_run}",
    )
    try:
        for query_index, sample in enumerate(inputs, start=1):
            query_started = time.monotonic()
            query_id = str(sample.get("query_id") or "")
            question = str(sample.get("question") or "")
            answer_contract = extract_answer_contract(sample, validation_by_id.get(query_id))
            top_k_papers = _top_k_for_sample(sample, config, args.top_k_papers)
            task_family = str(sample.get("task_family") or "")
            is_multi_paper = _is_multi_paper_task(sample)
            query_decomposition_enabled = bool(config.retrieval_enable_query_decomposition and is_multi_paper)
            query_label = f"query {query_index}/{total_queries} {query_id}"
            _progress(
                args.show_progress,
                f"{_progress_bar(query_index - 1, total_queries)} {query_label} start "
                f"task_family={task_family or '-'} primary={_primary_type(sample)} "
                f"top_k_papers={top_k_papers} query_decomposition={query_decomposition_enabled}",
            )
            if reused_candidates_by_query:
                candidates = [dict(row) for row in reused_candidates_by_query[query_id]]
                top_k_papers = len(candidates)
                query_decomposition_enabled = any(
                    bool(row.get("query_decomposition_enabled")) for row in candidates
                )
                retrieval_status = "reused"
            else:
                candidates = retrieve_candidates(
                    question,
                    metadata,
                    top_k=top_k_papers,
                    method=config.retrieval_method,
                    enable_query_decomposition=query_decomposition_enabled,
                    subquery_top_k=config.retrieval_subquery_top_k,
                )
                retrieval_status = "computed"
            candidate_preview = ", ".join(str(c.get("paper_id") or "") for c in candidates[:5])
            _progress(
                args.show_progress,
                f"{_progress_bar(query_index - 1, total_queries)} {query_label} retrieved "
                f"source={retrieval_status} candidates={len(candidates)} top=[{candidate_preview}]",
            )
            candidate_score = {str(c.get("paper_id") or ""): float(c.get("score") or c.get("bm25_score") or 0.0) for c in candidates}
            for rank, candidate in enumerate(candidates, start=1):
                append_jsonl(
                    paths["candidate_papers"],
                    {
                        "query_id": query_id,
                        "rank": rank,
                        "configured_top_k_papers": top_k_papers,
                        "query_decomposition_enabled": query_decomposition_enabled,
                        "retrieval_source": retrieval_status,
                        **candidate,
                    },
                )
            if args.retrieval_only:
                stats["processed_queries"] += 1
                _write_report(paths["report"], stats, paths)
                _progress(
                    args.show_progress,
                    f"{_progress_bar(query_index, total_queries)} {query_label} done "
                    f"status=retrieval_only elapsed={_format_seconds(time.monotonic() - query_started)}",
                )
                continue
            if args.skip_extraction:
                rows = []
                for candidate in candidates:
                    paper_id = str(candidate.get("paper_id") or "")
                    runtime_path = structured_root / paper_id / "symbolic_records.runtime.jsonl"
                    rows.append(
                        {
                            "paper_id": paper_id,
                            "available": runtime_path.is_file(),
                            "local_path": "",
                            "status": "processed_cache" if runtime_path.is_file() else "processed_cache_missing",
                            "note": "Selection-only reuse of processed symbolic records.",
                        }
                    )
                availability = {"rows": rows, "reuse_mode": "processed_records"}
            else:
                availability = ensure_candidate_pdfs(
                    candidates,
                    args.pdf_output_dir,
                    overwrite=args.pdf_overwrite,
                    metadata_by_id=metadata_by_id,
                    sleep_seconds=args.pdf_sleep_seconds,
                    timeout_seconds=args.pdf_timeout_seconds,
                    max_retries=args.pdf_max_retries,
                    openreview_policy=config.pdf_openreview_policy,
                )
            write_pdf_availability(paths["pdf_availability"], availability, query_id)
            available_count = sum(1 for row in availability.get("rows", []) if row.get("available"))
            missing_count = sum(1 for row in availability.get("rows", []) if not row.get("available"))
            stats["pdf_available"] += available_count
            stats["pdf_missing"] += missing_count
            _progress(
                args.show_progress,
                f"{_progress_bar(query_index - 1, total_queries)} {query_label} pdf availability "
                f"available={available_count} missing={missing_count}",
            )
            candidate_records: list[dict[str, Any]] = []
            if available_count:
                extraction_action = "loading cached records" if args.skip_extraction else "extracting"
                _progress(
                    args.show_progress,
                    f"{_progress_bar(query_index - 1, total_queries)} {query_label} {extraction_action} "
                    f"available_papers={available_count}",
                )
            for row in availability.get("rows", []):
                paper_id = str(row.get("paper_id") or "")
                if not row.get("available"):
                    append_jsonl(
                        paths["errors"],
                        {"query_id": query_id, "paper_id": paper_id, "type": "processed_cache_missing" if args.skip_extraction else "pdf_missing"},
                    )
                    stats["errors"] += 1
                    continue
                paper_dir = structured_root / paper_id
                if not args.skip_extraction:
                    status = extract_pdf_symbolic_records_with_backend(
                        paper_id,
                        row["local_path"],
                        paper_dir,
                        transcription_backend=transcription_backend,
                        overwrite=args.force_extract,
                        extract_all_pages=args.extract_all_pages,
                        selected_pages=None,
                        max_pages=args.max_pages_per_paper,
                        enable_figure_crops=args.enable_figure_crops,
                        enable_table_crops=args.enable_table_crops,
                        enable_equation_crops=args.enable_equation_crops,
                        crop_dpi=args.crop_dpi,
                        min_text_block_chars=args.min_text_block_chars,
                        include_debug_bbox=args.include_debug_bbox,
                        cache_policy=args.extraction_cache_policy,
                        docling_do_ocr=config.docling_do_ocr,
                    )
                    append_jsonl(paths["pdf_extraction_artifacts"], {"query_id": query_id, **status})
                    if status.get("cache_status") == "reused":
                        stats["reused_extraction_count"] += 1
                    elif status.get("status") in {"complete", "partial"}:
                        stats["parsed_paper_count"] += 1
                    stats["parsed_page_count"] += int(status.get("parsed_pages") or 0)
                    stats["failed_page_count"] += int(status.get("failed_pages") or 0)
                    _source_dist_add(stats["source_type_distribution"], status.get("source_type_distribution") or {})
                    if status.get("status") == "failed":
                        append_jsonl(paths["errors"], {"query_id": query_id, "paper_id": paper_id, "type": "extraction_failed", "error": status.get("error", "")})
                        stats["errors"] += 1
                records = _load_runtime_records(paper_dir, candidate_score.get(paper_id, 0.0))
                candidate_records.extend(records)
                if not args.replace_query_results:
                    append_jsonl_rows(paths["symbolic_records_runtime"], (_project_runtime_record(record) for record in records))
                    append_jsonl_rows(paths["symbolic_records_debug"], _read_jsonl_if_exists(paper_dir / "symbolic_records.debug.jsonl"))
            stats["runtime_records_total"] += len(candidate_records)
            query_source_dist = Counter(str(record.get("source_type") or "unknown") for record in candidate_records)
            _progress(
                args.show_progress,
                f"{_progress_bar(query_index - 1, total_queries)} {query_label} extracted "
                f"records={len(candidate_records)} source_types={_source_dist_text(dict(query_source_dist))}",
            )
            stats["processed_queries"] += 1
            if args.skip_selection:
                stats["selection_skipped_queries"] += 1
                _write_report(paths["report"], stats, paths)
                _progress(
                    args.show_progress,
                    f"{_progress_bar(query_index, total_queries)} {query_label} done "
                    f"status=skip_selection elapsed={_format_seconds(time.monotonic() - query_started)}",
                )
                continue
            selected = select_symbolic_contexts(
                question,
                candidate_records,
                processed_root=processed_root,
                parser_model_slug=transcription_backend,
                top_n_records=args.top_n_records,
                top_n_visual_records=args.top_n_visual_records,
                primary_evidence_type=_primary_type(sample),
                query_id=query_id,
                vlm2_context_mode=context_mode,
                include_parse_confidence=False,
                context_selection_mode=config.vlm2_context_selection_mode,
                source_type_hints_enabled=config.symbolic_source_type_hints,
                section_relevance_config=SectionRelevanceConfig(
                    backend=config.section_relevance_backend,
                    unit_mode=config.section_relevance_unit_mode,
                    unit_target_tokens=config.section_relevance_unit_target_tokens,
                    unit_max_tokens=config.section_relevance_unit_max_tokens,
                    unit_overlap_records=config.section_relevance_unit_overlap_records,
                    object_units_enabled=config.section_relevance_object_units_enabled,
                    object_neighbor_records=config.section_relevance_object_neighbor_records,
                    llmrerank_api_key=config.answer_api_key,
                    llmrerank_base_url=config.answer_base_url,
                    llmrerank_model=config.llmrerank_model,
                    llmrerank_input_mode=config.llmrerank_input_mode,
                    llmrerank_batch_size=config.llmrerank_batch_size,
                    llmrerank_request_concurrency=config.llmrerank_request_concurrency,
                    llmrerank_request_timeout_seconds=config.llmrerank_request_timeout_seconds,
                    llmrerank_max_retries=config.llmrerank_max_retries,
                    llmrerank_failure_fallback=config.llmrerank_failure_fallback,
                    llmrerank_max_images_per_section=config.llmrerank_max_images_per_section,
                    llmrerank_instruction_version=config.llmrerank_instruction_version,
                    llmrerank_query_mode=config.llmrerank_query_mode,
                    llmrerank_include_paper_identity=config.llmrerank_include_paper_identity,
                ),
                multi_paper_hyde_config=MultiPaperHyDEConfig(
                    enabled=config.multi_paper_hyde_enabled,
                    model=config.multi_paper_hyde_model,
                    max_claims=config.multi_paper_hyde_max_claims,
                    cache_enabled=config.multi_paper_hyde_cache_enabled,
                    temperature=config.multi_paper_hyde_temperature,
                    max_tokens=config.multi_paper_hyde_max_tokens,
                    timeout_seconds=config.multi_paper_hyde_timeout_seconds,
                ),
                paper_conditioned_claims_config=PaperConditionedClaimsConfig(
                    enabled=config.paper_conditioned_claims_enabled,
                    model=config.paper_conditioned_claims_model,
                    max_papers=config.paper_conditioned_claims_max_papers,
                    cache_enabled=config.paper_conditioned_claims_cache_enabled,
                    temperature=config.paper_conditioned_claims_temperature,
                    max_tokens=config.paper_conditioned_claims_max_tokens,
                    timeout_seconds=config.paper_conditioned_claims_timeout_seconds,
                ),
                paper_local_bm25_route_mode=config.paper_local_bm25_route_mode,
                is_multi_paper_task=is_multi_paper,
                hyde_client=answer_client,
                candidate_paper_metadata=candidates,
                evidence_package_budget=(
                    config.multi_paper_evidence_package_budget
                    if is_multi_paper else config.single_paper_evidence_package_budget
                ),
                evidence_package_min_budget=(
                    config.multi_paper_evidence_package_min
                    if is_multi_paper else config.single_paper_evidence_package_min
                ),
                multi_paper_min_distinct_papers=config.multi_paper_min_distinct_papers,
                evidence_package_adaptive_stop=config.evidence_package_adaptive_stop,
                multi_paper_modality_packages_per_paper=config.multi_paper_modality_packages_per_paper,
                multi_paper_supporting_text_packages_per_paper=config.multi_paper_supporting_text_packages_per_paper,
                evidence_package_max_context_chars=config.evidence_package_max_context_chars,
                evidence_package_rrf_k=config.evidence_package_rrf_k,
                evidence_package_candidate_pool_per_route=config.evidence_package_candidate_pool_per_route,
                evidence_package_max_per_page=config.evidence_package_max_per_page,
                evidence_package_page_text_anchors_per_page=config.evidence_package_page_text_anchors_per_page,
            )
            selected_debug = {
                "query_id": query_id,
                "selection_method": selected.get("selection_method"),
                "source_type_distribution": selected.get("source_type_distribution", {}),
                "primary_evidence_type_count": selected.get("primary_evidence_type_count", 0),
                "supporting_evidence_count": selected.get("supporting_evidence_count", 0),
                "context_truncated": selected.get("context_truncated", False),
                "selected_record_count": selected.get("selected_record_count", 0),
                "skipped_records_count": max(0, len(candidate_records) - int(selected.get("selected_record_count") or 0)),
                "selected_records": selected.get("selected_records_debug", []),
                "selected_context_groups": selected.get("selected_context_groups", []),
                "selected_visual_records": selected.get("selected_visual_records", []),
                "attached_image_refs": selected.get("attached_image_refs", []),
                "prompt_audit": selected.get("prompt_audit"),
            }
            selected_prompt = {
                "query_id": query_id,
                "selected_evidence": selected.get("selected_evidence", []),
                "compact_chunk_packets": selected.get("compact_chunk_packets", []),
                "attached_image_refs": selected.get("attached_image_refs", []),
                "prompt_audit": selected.get("prompt_audit"),
                "selected_context_groups": selected.get("selected_context_groups", []),
                "source_type_distribution": selected.get("source_type_distribution", {}),
                "primary_evidence_type_count": selected.get("primary_evidence_type_count", 0),
                "supporting_evidence_count": selected.get("supporting_evidence_count", 0),
                "context_truncated": selected.get("context_truncated", False),
                "selected_record_count": selected.get("selected_record_count", 0),
            }
            append_jsonl(paths["selected_contexts_debug"], selected_debug)
            append_jsonl(paths["selected_contexts_prompt"], selected_prompt)
            if selected.get("section_relevance_trace"):
                append_jsonl(paths["section_relevance_trace"], selected["section_relevance_trace"])
            _source_dist_add(stats["selected_source_type_distribution"], selected.get("source_type_distribution") or {})
            if selected.get("context_truncated"):
                stats["vlm2_context_truncation_count"] += 1
            _progress(
                args.show_progress,
                f"{_progress_bar(query_index - 1, total_queries)} {query_label} selected "
                f"records={selected.get('selected_record_count', 0)} "
                f"visual={len(selected.get('selected_visual_records', []) or [])} "
                f"truncated={bool(selected.get('context_truncated'))} "
                f"source_types={_source_dist_text(selected.get('source_type_distribution') or {})}",
            )
            messages = build_symbolic_answer_prompt(sample, candidates, selected, answer_client.supports_image_input(), transcription_backend, config.answer_model, answer_contract)
            append_jsonl(paths["prompt_previews"], {"query_id": query_id, "baseline_type": BASELINE_TYPE, "messages": messages})
            if args.dry_run or args.skip_generation:
                fallback = make_fallback_prediction(sample, candidates[0] if candidates else None)
                fallback["answer"] = _empty_answer_for_sample(sample)
                append_jsonl(paths["predictions"], fallback)
                stats["fallback_predictions"] += 1
                _write_report(paths["report"], stats, paths)
                _progress(
                    args.show_progress,
                    f"{_progress_bar(query_index, total_queries)} {query_label} done "
                    f"status=fallback_skip_generation elapsed={_format_seconds(time.monotonic() - query_started)}",
                )
                continue
            if not selected.get("selected_evidence"):
                fallback = make_fallback_prediction(sample, candidates[0] if candidates else None)
                append_jsonl(paths["predictions"], fallback)
                append_jsonl(paths["errors"], {"query_id": query_id, "type": "no_selected_symbolic_contexts"})
                stats["fallback_predictions"] += 1
                stats["errors"] += 1
                _write_report(paths["report"], stats, paths)
                _progress(
                    args.show_progress,
                    f"{_progress_bar(query_index, total_queries)} {query_label} done "
                    f"status=fallback_no_selected_context elapsed={_format_seconds(time.monotonic() - query_started)}",
                )
                continue
            try:
                query_status = "success"
                evidence_limit = 64 if is_multi_paper else 12
                generation_messages = messages
                parse_retries = max(0, int(config.generation_parse_max_retries))
                for generation_attempt in range(1, parse_retries + 2):
                    result = answer_client.generate_prediction(
                        generation_messages,
                        image_paths=selected.get("attached_image_paths") if context_mode == "cropped_image" else None,
                    )
                    stats["answer_api_calls"] += 1
                    append_jsonl(
                        paths["raw_answer"],
                        {
                            "query_id": query_id,
                            "generation_attempt": generation_attempt,
                            "content": result["content"],
                            "raw_response": result["raw_response"],
                        },
                    )
                    try:
                        internal = extract_json_object(str(result["content"]))
                        prediction, errors = normalize_prediction(
                            internal,
                            sample,
                            [str(c.get("paper_id") or "") for c in candidates],
                            answer_contract=answer_contract,
                            selected_evidence=selected_prompt.get("selected_evidence", []),
                            symbolic_evidence_standardization=config.symbolic_evidence_standardization,
                            candidate_records=candidates,
                        )
                        if not validate_prediction_shape(prediction):
                            raise ValueError("invalid prediction shape")
                        break
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                        if generation_attempt > parse_retries:
                            raise
                        generation_messages = _messages_for_json_retry(messages, evidence_limit)
                append_jsonl(paths["internal_predictions"], internal)
                for error in errors:
                    append_jsonl(paths["errors"], error)
                append_jsonl(paths["predictions"], strip_internal_grounding(prediction))
                stats["successful_predictions"] += 1
            except Exception as exc:
                query_status = "fallback_answer_generation_or_parse_failure"
                fallback = make_fallback_prediction(sample, candidates[0] if candidates else None)
                append_jsonl(paths["predictions"], fallback)
                append_jsonl(paths["errors"], {"query_id": query_id, "type": "answer_generation_or_parse_failure", "error": str(exc)})
                stats["fallback_predictions"] += 1
                stats["errors"] += 1
            _write_report(paths["report"], stats, paths)
            _progress(
                args.show_progress,
                f"{_progress_bar(query_index, total_queries)} {query_label} done "
                f"status={query_status} elapsed={_format_seconds(time.monotonic() - query_started)}",
            )
        stats["run_status"] = "complete"
    except Exception as exc:
        stats["run_status"] = "failed"
        stats["failure_reason"] = str(exc)
        append_jsonl(paths["errors"], {"type": "run_failed", "error": str(exc)})
        _progress(args.show_progress, f"run failed error={exc}")
        exit_code = 1
    finally:
        stats["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_report(paths["report"], stats, paths, refresh_row_counts=True)
        _progress(
            args.show_progress,
            f"finished status={stats.get('run_status')} processed={stats.get('processed_queries', 0)}/{total_queries} "
            f"errors={stats.get('errors', 0)} elapsed={_format_seconds(time.monotonic() - run_started)}",
        )
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_UN)
        output_lock.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
