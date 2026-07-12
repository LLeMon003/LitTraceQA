from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import load_pipeline_config, model_slug
from ..data_io import append_jsonl, extract_answer_contract, find_official_file, read_jsonl
from ..parser import extract_json_object, make_fallback_prediction, normalize_prediction, strip_internal_grounding, validate_prediction_shape
from ..symbolic_context_selector import select_symbolic_contexts
from ..vlm_answer_client import AnswerVLMError, VLMAnswerClient
from ..vlm_answer_prompt_builder import build_symbolic_answer_prompt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


ANSWER_STAGE_FILES = {
    "predictions": "predictions.jsonl",
    "internal_predictions": "internal_predictions.jsonl",
    "raw_answer": "raw_vlm_answer_responses.jsonl",
    "prompt_previews": "prompt_previews.jsonl",
    "selected_contexts_prompt": "selected_symbolic_contexts.prompt.jsonl",
    "selected_contexts_debug": "selected_symbolic_contexts.debug.jsonl",
    "errors": "errors.jsonl",
    "rerun_report": "vlm2_rerun_report.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun VLM-2 using existing retrieval and selected symbolic contexts.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-selected-contexts", action="store_true", help="Rebuild selected symbolic contexts from global_page_parse_plan and a symbolic cache root instead of reusing selected_symbolic_contexts.prompt.jsonl.")
    parser.add_argument("--symbolic-cache-root", default="", help="Root containing symbolic cache data. Pass either the run root or the parser-model slug subdirectory.")
    parser.add_argument("--processed-output-dir", default="processed_pdfs/vlm_symbolic", help="Processed root used only for image path projection when rebuilding contexts.")
    parser.add_argument("--rerun-failed-queries", action="store_true", help="Only rerun query ids that have VLM generation/normalization failures in the existing errors.jsonl.")
    parser.add_argument("--only-query-ids", default="", help="Comma-separated query ids to rerun, for example q046,q047.")
    parser.add_argument("--resume-successful-predictions", action="store_true", help="Copy existing outputs for non-rerun query ids so partial reruns still produce a complete predictions.jsonl.")
    parser.add_argument("--backup-existing", action="store_true", default=True)
    parser.add_argument("--no-backup-existing", dest="backup_existing", action="store_false")
    return parser.parse_args()


def _latest_by_query_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = _normalize_query_id(str(row.get("query_id") or ""))
        if query_id:
            latest[query_id] = row
    return latest


def _split_query_ids(raw: str) -> set[str]:
    return {_normalize_query_id(item) for item in raw.split(",") if item.strip()}


def _normalize_query_id(query_id: str) -> str:
    value = query_id.strip()
    match = re.fullmatch(r"q_?(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"q_{int(match.group(1)):03d}"
    return value


def _failed_query_ids(output_dir: Path) -> set[str]:
    errors_path = output_dir / "errors.jsonl"
    if not errors_path.exists():
        return set()
    failed: set[str] = set()
    for row in read_jsonl(errors_path):
        error_type = str(row.get("type") or "")
        query_id = _normalize_query_id(str(row.get("query_id") or ""))
        if query_id and error_type.startswith("vlm2_"):
            failed.add(query_id)
    return failed


def _temp_paths(output_dir: Path) -> dict[str, Path]:
    temp_dir = output_dir / ".vlm2_rerun_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return {key: temp_dir / filename for key, filename in ANSWER_STAGE_FILES.items()}


def _final_paths(output_dir: Path) -> dict[str, Path]:
    return {key: output_dir / filename for key, filename in ANSWER_STAGE_FILES.items()}


def _backup_existing(paths: dict[str, Path], output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = output_dir / "vlm2_rerun_backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        if key == "rerun_report":
            continue
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def _replace_outputs(temp_paths: dict[str, Path], final_paths: dict[str, Path]) -> None:
    for key, temp_path in temp_paths.items():
        if key == "rerun_report":
            continue
        final_path = final_paths[key]
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_path), str(final_path))
    shutil.move(str(temp_paths["rerun_report"]), str(final_paths["rerun_report"]))


def _empty_stage_files(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        if key == "rerun_report":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _existing_rows_by_stage(final_paths: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    rows_by_stage: dict[str, dict[str, dict[str, Any]]] = {}
    for key, path in final_paths.items():
        if key == "rerun_report" or not path.exists():
            continue
        rows_by_stage[key] = _latest_by_query_id(read_jsonl(path))
    return rows_by_stage


def _copy_existing_query_outputs(
    *,
    query_id: str,
    temp_paths: dict[str, Path],
    existing_rows: dict[str, dict[str, dict[str, Any]]],
    stats: dict[str, Any],
) -> bool:
    prediction = existing_rows.get("predictions", {}).get(query_id)
    if not prediction:
        return False
    for key in ("predictions", "internal_predictions", "raw_answer", "prompt_previews", "selected_contexts_prompt", "selected_contexts_debug"):
        row = existing_rows.get(key, {}).get(query_id)
        if row:
            append_jsonl(temp_paths[key], row)
    if existing_rows.get("selected_contexts_prompt", {}).get(query_id):
        stats["selected_context_rows_written"] += 1
    if existing_rows.get("selected_contexts_debug", {}).get(query_id):
        stats["selected_context_debug_rows_written"] += 1
    stats["resumed_successful_predictions"] += 1
    stats["completed_queries"] += 1
    return True


def _candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for candidate in candidates:
        paper_id = str(candidate.get("paper_id") or "")
        if paper_id:
            ids.append(paper_id)
    return ids


def _top_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return candidates[0] if candidates and isinstance(candidates[0], dict) else None


def _resolve_symbolic_root(symbolic_cache_root: str, parser_slug: str) -> Path:
    root = Path(symbolic_cache_root or "processed_pdfs/vlm_symbolic")
    if root.name == parser_slug:
        return root
    return root / parser_slug


def _candidate_score_by_id(candidates: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        paper_id = str(candidate.get("paper_id") or "")
        if not paper_id:
            continue
        raw_score = candidate.get("score")
        if raw_score is None:
            raw_score = candidate.get("retrieval_score")
        try:
            scores[paper_id] = float(raw_score)
        except (TypeError, ValueError):
            scores[paper_id] = float(max(0, 1000 - index))
    return scores


def _read_page_records(symbolic_root: Path, paper_id: str, page: int) -> list[dict[str, Any]]:
    page_path = symbolic_root / paper_id / "page_records" / f"page_{page:03d}.records.runtime.jsonl"
    if page_path.exists():
        return read_jsonl(page_path)
    legacy_page_path = symbolic_root / paper_id / f"page_{page:03d}" / "records.runtime.jsonl"
    if legacy_page_path.exists():
        return read_jsonl(legacy_page_path)
    paper_path = symbolic_root / paper_id / "symbolic_records.runtime.jsonl"
    if paper_path.exists():
        return [record for record in read_jsonl(paper_path) if int(record.get("page") or 0) == page]
    return []


def _selected_pages_by_query_id(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows = _latest_by_query_id(read_jsonl(output_dir / "global_page_parse_plan.jsonl"))
    result: dict[str, list[dict[str, Any]]] = {}
    for query_id, row in rows.items():
        pages = row.get("final_parsed_pages") if isinstance(row.get("final_parsed_pages"), list) else []
        result[query_id] = [item for item in pages if isinstance(item, dict)]
    return result


def _rebuild_selected_context(
    *,
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_pages: list[dict[str, Any]],
    symbolic_root: Path,
    processed_output_dir: str,
    parser_slug: str,
    config: Any,
    max_context_records: int | None = None,
    max_context_chars: int | None = None,
) -> dict[str, Any]:
    candidate_scores = _candidate_score_by_id(candidates)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in selected_pages:
        paper_id = str(item.get("paper_id") or "")
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if not paper_id or page <= 0 or (paper_id, page) in seen:
            continue
        seen.add((paper_id, page))
        for record in _read_page_records(symbolic_root, paper_id, page):
            copied = dict(record)
            copied["_candidate_bm25_score"] = candidate_scores.get(str(copied.get("paper_id") or ""), 0.0)
            records.append(copied)
    return select_symbolic_contexts(
        str(sample.get("question") or ""),
        records,
        processed_output_dir,
        parser_slug,
        primary_evidence_type=str(sample.get("primary_evidence_type") or ""),
        query_id=str(sample.get("query_id") or ""),
        vlm2_context_mode="text_only",
        include_parse_confidence=False,
        evidence_total_budget=config.vlm2_evidence_total_budget,
        primary_evidence_min=config.vlm2_primary_evidence_min,
        support_text_min=config.vlm2_support_text_min,
        context_types_enabled=config.vlm2_context_types_enabled,
        context_type_budget_per_type=config.vlm2_context_type_budget_per_type,
        context_selection_mode=config.vlm2_context_selection_mode,
        max_context_records=config.vlm2_max_context_records if max_context_records is None else max_context_records,
        max_context_chars=config.vlm2_max_context_chars if max_context_chars is None else max_context_chars,
        source_type_hints_enabled=config.symbolic_source_type_hints,
    )


def _selected_context_from_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_evidence": row.get("selected_evidence") if isinstance(row.get("selected_evidence"), list) else [],
        "has_partial_artifacts": bool(row.get("has_partial_artifacts")),
        "partial_artifacts_present": bool(row.get("has_partial_artifacts")),
        "attached_image_refs": row.get("attached_image_refs") if isinstance(row.get("attached_image_refs"), list) else [],
    }


def _selected_prompt_row(query_id: str, selected_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "selected_evidence": selected_context.get("selected_evidence", []),
        "has_partial_artifacts": selected_context.get("has_partial_artifacts", False),
        "attached_image_refs": [],
        "source_type_distribution": selected_context.get("source_type_distribution", {}),
        "primary_evidence_type_count": selected_context.get("primary_evidence_type_count", 0),
        "supporting_evidence_count": selected_context.get("supporting_evidence_count", 0),
        "grounding_label_hints_by_type": selected_context.get("grounding_label_hints_by_type", {}),
        "context_selection_mode": selected_context.get("context_selection_mode"),
        "context_truncated": selected_context.get("context_truncated", False),
        "selected_record_count": selected_context.get("selected_record_count", 0),
    }


def _selected_debug_row(query_id: str, selected_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "selection_method": selected_context.get("selection_method"),
        "prompt_context_mode": selected_context.get("prompt_context_mode"),
        "has_partial_artifacts": selected_context.get("has_partial_artifacts", False),
        "source_type_distribution": selected_context.get("source_type_distribution", {}),
        "primary_evidence_type_count": selected_context.get("primary_evidence_type_count", 0),
        "supporting_evidence_count": selected_context.get("supporting_evidence_count", 0),
        "grounding_label_hints_by_type": selected_context.get("grounding_label_hints_by_type", {}),
        "context_selection_mode": selected_context.get("context_selection_mode"),
        "context_truncated": selected_context.get("context_truncated", False),
        "selected_record_count": selected_context.get("selected_record_count", 0),
        "selected_records": selected_context.get("selected_records_debug", []),
        "selected_visual_records": selected_context.get("selected_visual_records", []),
    }


def _append_selected_context_rows(
    *,
    temp_paths: dict[str, Path],
    stats: dict[str, Any],
    query_id: str,
    selected_context: dict[str, Any],
) -> None:
    append_jsonl(temp_paths["selected_contexts_prompt"], _selected_prompt_row(query_id, selected_context))
    append_jsonl(temp_paths["selected_contexts_debug"], _selected_debug_row(query_id, selected_context))
    stats["selected_context_rows_written"] += 1
    stats["selected_context_debug_rows_written"] += 1


def _record_successful_prediction(
    *,
    temp_paths: dict[str, Path],
    stats: dict[str, Any],
    query_id: str,
    result: dict[str, Any],
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    answer_contract: dict[str, Any],
    selected_context: dict[str, Any],
    config: Any,
) -> None:
    append_jsonl(temp_paths["raw_answer"], {"query_id": query_id, "content": result["content"], "raw_response": result["raw_response"]})
    internal = extract_json_object(str(result["content"]))
    append_jsonl(temp_paths["internal_predictions"], internal)
    prediction, errors = normalize_prediction(
        internal,
        sample,
        _candidate_ids(candidates),
        answer_contract=answer_contract,
        selected_evidence=selected_context.get("selected_evidence", []),
        symbolic_evidence_standardization=config.symbolic_evidence_standardization,
        candidate_records=candidates,
    )
    for error in errors:
        error_type = str(error.get("type") or "")
        if error_type == "symbolic_evidence_locator_standardized":
            stats["symbolic_evidence_locator_standardized_count"] += 1
        elif error_type == "symbolic_evidence_empty_filled":
            stats["symbolic_evidence_empty_filled_count"] += 1
        elif error_type == "symbolic_evidence_non_object_replaced":
            stats["symbolic_evidence_non_object_replaced_count"] += 1
        elif error_type == "symbolic_evidence_standardization_no_match":
            stats["symbolic_evidence_standardization_no_match_count"] += 1
        elif error_type == "symbolic_evidence_standardization_no_locator":
            stats["symbolic_evidence_standardization_no_locator_count"] += 1
        append_jsonl(temp_paths["errors"], error)
    if not validate_prediction_shape(prediction):
        raise ValueError("Normalized prediction does not match expected shape")
    append_jsonl(temp_paths["predictions"], strip_internal_grounding(prediction))
    stats["successful_predictions"] += 1


def _json_repair_messages(sample: dict[str, Any], raw_content: str) -> list[dict[str, str]]:
    query_id = str(sample.get("query_id") or "")
    return [
        {
            "role": "system",
            "content": (
                "Return valid JSON only. Repair the malformed answer into the official prediction schema. "
                "Do not add explanations, markdown fences, or text outside JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query_id": query_id,
                    "required_schema": {
                        "query_id": query_id,
                        "gold_papers": [{"paper_id": "paper_id"}],
                        "evidence": [{"paper_id": "paper_id", "source_type": "text_span|table|figure|equation_algorithm|citation_context", "locator": {"page": 1}}],
                        "answer": {},
                    },
                    "malformed_model_output": raw_content,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _should_retry_with_limited_context(exc: Exception) -> bool:
    if isinstance(exc, AnswerVLMError):
        return exc.kind in {"context_length", "timeout"}
    text = str(exc).lower()
    return "http error 400" in text or "http error 413" in text or "timeout" in text


def _write_report(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# VLM-2 Rerun Report",
        "",
        f"- created_at: {stats['created_at']}",
        f"- official_dir: `{stats['official_dir']}`",
        f"- output_dir: `{stats['output_dir']}`",
        f"- answer_model: `{stats['answer_model']}`",
        f"- dry_run: {stats['dry_run']}",
        f"- input_queries: {stats['input_queries']}",
        f"- rerun_query_count: {stats['rerun_query_count']}",
        f"- completed_queries: {stats['completed_queries']}",
        f"- resumed_successful_predictions: {stats['resumed_successful_predictions']}",
        f"- answer_api_calls: {stats['answer_api_calls']}",
        f"- successful_predictions: {stats['successful_predictions']}",
        f"- fallback_predictions: {stats['fallback_predictions']}",
        f"- generation_errors: {stats['generation_errors']}",
        f"- json_repair_attempts: {stats['json_repair_attempts']}",
        f"- json_repair_successes: {stats['json_repair_successes']}",
        f"- generation_retry_on_429: {stats['generation_retry_on_429']}",
        f"- generation_429_max_retries: {stats['generation_429_max_retries']}",
        f"- missing_candidate_rows: {stats['missing_candidate_rows']}",
        f"- missing_selected_context_rows: {stats['missing_selected_context_rows']}",
        f"- rebuild_selected_contexts: {stats['rebuild_selected_contexts']}",
        f"- symbolic_cache_root: `{stats['symbolic_cache_root']}`",
        f"- context_selection_mode: `{stats['context_selection_mode']}`",
        f"- selected_context_rows_written: {stats['selected_context_rows_written']}",
        f"- selected_context_debug_rows_written: {stats['selected_context_debug_rows_written']}",
        f"- full_context_first_enabled: {stats['full_context_first_enabled']}",
        f"- limited_context_retry_enabled: {stats['limited_context_retry_enabled']}",
        f"- limited_context_retry_attempts: {stats['limited_context_retry_attempts']}",
        f"- limited_context_retry_successes: {stats['limited_context_retry_successes']}",
        f"- multiple_choice_queries: {stats['multiple_choice_queries']}",
        f"- multiple_choice_with_options: {stats['multiple_choice_with_options']}",
        f"- table_queries: {stats['table_queries']}",
        f"- table_with_schema: {stats['table_with_schema']}",
        f"- symbolic_evidence_standardization: {stats['symbolic_evidence_standardization']}",
        f"- symbolic_evidence_locator_standardized_count: {stats['symbolic_evidence_locator_standardized_count']}",
        f"- symbolic_evidence_empty_filled_count: {stats['symbolic_evidence_empty_filled_count']}",
        f"- symbolic_evidence_non_object_replaced_count: {stats['symbolic_evidence_non_object_replaced_count']}",
        f"- symbolic_evidence_standardization_no_match_count: {stats['symbolic_evidence_standardization_no_match_count']}",
        f"- symbolic_evidence_standardization_no_locator_count: {stats['symbolic_evidence_standardization_no_locator_count']}",
        f"- backup_dir: `{stats.get('backup_dir', '')}`",
        "",
        "This rerun reuses retrieval candidates from the existing run output. It does not call retrieval, PDF rendering, or VLM-1.",
        "When rebuild_selected_contexts is true, selected symbolic evidence is rebuilt from global_page_parse_plan plus the configured symbolic cache root.",
        "The answer contract is rebuilt from validation_inputs.jsonl plus sanitized options/schema from validation.jsonl. Gold answers, gold evidence, and gold paper ids are not sent to VLM-2.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = load_pipeline_config(args.env_path)
    answer_client = VLMAnswerClient(config)
    final_paths = _final_paths(output_dir)
    temp_paths = _temp_paths(output_dir)
    _empty_stage_files(temp_paths)

    inputs = read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    if args.max_queries is not None:
        inputs = inputs[: args.max_queries]
    existing_rows = _existing_rows_by_stage(final_paths)
    only_query_ids = _split_query_ids(args.only_query_ids)
    failed_query_ids = _failed_query_ids(output_dir) if args.rerun_failed_queries else set()
    requested_query_ids = only_query_ids or failed_query_ids
    if requested_query_ids:
        inputs_to_rerun = [sample for sample in inputs if _normalize_query_id(str(sample.get("query_id") or "")) in requested_query_ids]
    else:
        inputs_to_rerun = inputs
    validation_rows = read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))
    validation_by_query_id = _latest_by_query_id(validation_rows)
    candidates_by_query_id = _latest_by_query_id(read_jsonl(output_dir / "candidate_papers.jsonl"))
    selected_by_query_id = _latest_by_query_id(read_jsonl(output_dir / "selected_symbolic_contexts.prompt.jsonl"))
    selected_pages_by_query_id = _selected_pages_by_query_id(output_dir) if args.rebuild_selected_contexts else {}
    parser_slug = model_slug(config.parser_model)
    symbolic_root = _resolve_symbolic_root(args.symbolic_cache_root, parser_slug)

    backup_dir = ""
    if args.backup_existing:
        backup_dir = str(_backup_existing(final_paths, output_dir))

    stats: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "official_dir": str(args.official_dir),
        "output_dir": str(output_dir),
        "answer_model": config.answer_model,
        "dry_run": bool(args.dry_run),
        "input_queries": len(inputs),
        "rerun_query_count": len(inputs_to_rerun),
        "completed_queries": 0,
        "resumed_successful_predictions": 0,
        "answer_api_calls": 0,
        "successful_predictions": 0,
        "fallback_predictions": 0,
        "generation_errors": 0,
        "json_repair_attempts": 0,
        "json_repair_successes": 0,
        "generation_retry_on_429": config.generation_retry_on_429,
        "generation_429_max_retries": config.generation_429_max_retries,
        "missing_candidate_rows": 0,
        "missing_selected_context_rows": 0,
        "rebuild_selected_contexts": bool(args.rebuild_selected_contexts),
        "symbolic_cache_root": str(symbolic_root) if args.rebuild_selected_contexts else "",
        "context_selection_mode": config.vlm2_context_selection_mode,
        "selected_context_rows_written": 0,
        "selected_context_debug_rows_written": 0,
        "full_context_first_enabled": bool(args.rebuild_selected_contexts),
        "limited_context_retry_enabled": bool(args.rebuild_selected_contexts and (config.vlm2_max_context_records > 0 or config.vlm2_max_context_chars > 0)),
        "limited_context_retry_attempts": 0,
        "limited_context_retry_successes": 0,
        "multiple_choice_queries": 0,
        "multiple_choice_with_options": 0,
        "table_queries": 0,
        "table_with_schema": 0,
        "symbolic_evidence_standardization": config.symbolic_evidence_standardization,
        "symbolic_evidence_locator_standardized_count": 0,
        "symbolic_evidence_empty_filled_count": 0,
        "symbolic_evidence_non_object_replaced_count": 0,
        "symbolic_evidence_standardization_no_match_count": 0,
        "symbolic_evidence_standardization_no_locator_count": 0,
        "backup_dir": backup_dir,
    }

    if args.resume_successful_predictions:
        rerun_ids = {_normalize_query_id(str(sample.get("query_id") or "")) for sample in inputs_to_rerun}
        for sample in inputs:
            query_id = _normalize_query_id(str(sample.get("query_id") or ""))
            if query_id and query_id not in rerun_ids:
                _copy_existing_query_outputs(query_id=query_id, temp_paths=temp_paths, existing_rows=existing_rows, stats=stats)

    iterator: Any = inputs_to_rerun
    if args.show_progress and tqdm is not None:
        iterator = tqdm(inputs_to_rerun, desc="vlm2-rerun", unit="query")

    for sample in iterator:
        query_id = str(sample.get("query_id") or "")
        candidate_row = candidates_by_query_id.get(query_id) or {}
        selected_row = selected_by_query_id.get(query_id) or {}
        candidates = candidate_row.get("candidates") if isinstance(candidate_row.get("candidates"), list) else []
        if args.rebuild_selected_contexts:
            selected_context = _rebuild_selected_context(
                sample=sample,
                candidates=candidates,
                selected_pages=selected_pages_by_query_id.get(query_id) or [],
                symbolic_root=symbolic_root,
                processed_output_dir=args.processed_output_dir,
                parser_slug=parser_slug,
                config=config,
                max_context_records=0,
                max_context_chars=0,
            )
        else:
            selected_context = _selected_context_from_prompt_row(selected_row)
        if not candidates:
            stats["missing_candidate_rows"] += 1
        if not selected_context["selected_evidence"]:
            stats["missing_selected_context_rows"] += 1

        answer_contract = extract_answer_contract(sample, validation_by_query_id.get(query_id))
        answer_types = answer_contract.get("answer_types") or []
        if "multiple_choice" in answer_types:
            stats["multiple_choice_queries"] += 1
            if (answer_contract.get("multiple_choice") or {}).get("options"):
                stats["multiple_choice_with_options"] += 1
        if "table" in answer_types:
            stats["table_queries"] += 1
            if (answer_contract.get("table") or {}).get("table_schema"):
                stats["table_with_schema"] += 1

        messages = build_symbolic_answer_prompt(
            sample,
            candidates,
            selected_context,
            answer_model_supports_images=False,
            parser_model=config.parser_model,
            answer_model=config.answer_model,
            answer_contract=answer_contract,
        )
        append_jsonl(temp_paths["prompt_previews"], {"query_id": query_id, "messages": messages, "baseline_type": "pdf_vlm_symbolic_vlm_vlm2_rerun"})

        if args.dry_run:
            _append_selected_context_rows(temp_paths=temp_paths, stats=stats, query_id=query_id, selected_context=selected_context)
            prediction = make_fallback_prediction(sample, _top_candidate(candidates))
            append_jsonl(temp_paths["predictions"], prediction)
            append_jsonl(temp_paths["internal_predictions"], prediction)
            stats["fallback_predictions"] += 1
            stats["completed_queries"] += 1
            continue

        limited_context_available = bool(args.rebuild_selected_contexts and (config.vlm2_max_context_records > 0 or config.vlm2_max_context_chars > 0))
        result: dict[str, Any] | None = None
        try:
            stats["answer_api_calls"] += 1
            result = answer_client.generate_prediction(messages, image_paths=None)
        except Exception as first_exc:
            if limited_context_available and _should_retry_with_limited_context(first_exc):
                append_jsonl(temp_paths["errors"], {"query_id": query_id, "type": "vlm2_full_context_generation_failure", "error": str(first_exc)})
                stats["limited_context_retry_attempts"] += 1
                selected_context = _rebuild_selected_context(
                    sample=sample,
                    candidates=candidates,
                    selected_pages=selected_pages_by_query_id.get(query_id) or [],
                    symbolic_root=symbolic_root,
                    processed_output_dir=args.processed_output_dir,
                    parser_slug=parser_slug,
                    config=config,
                )
                messages = build_symbolic_answer_prompt(
                    sample,
                    candidates,
                    selected_context,
                    answer_model_supports_images=False,
                    parser_model=config.parser_model,
                    answer_model=config.answer_model,
                    answer_contract=answer_contract,
                )
                append_jsonl(
                    temp_paths["prompt_previews"],
                    {
                        "query_id": query_id,
                        "messages": messages,
                        "baseline_type": "pdf_vlm_symbolic_vlm_vlm2_rerun_limited_context_retry",
                    },
                )
                try:
                    stats["answer_api_calls"] += 1
                    result = answer_client.generate_prediction(messages, image_paths=None)
                    stats["limited_context_retry_successes"] += 1
                except Exception as retry_exc:
                    stats["generation_errors"] += 1
                    append_jsonl(temp_paths["errors"], {"query_id": query_id, "type": "vlm2_limited_context_generation_failure", "error": str(retry_exc)})
                    fallback = make_fallback_prediction(sample, _top_candidate(candidates))
                    append_jsonl(temp_paths["predictions"], fallback)
                    append_jsonl(temp_paths["internal_predictions"], fallback)
                    stats["fallback_predictions"] += 1
            else:
                stats["generation_errors"] += 1
                append_jsonl(temp_paths["errors"], {"query_id": query_id, "type": "vlm2_rerun_generation_failure", "error": str(first_exc)})
                fallback = make_fallback_prediction(sample, _top_candidate(candidates))
                append_jsonl(temp_paths["predictions"], fallback)
                append_jsonl(temp_paths["internal_predictions"], fallback)
                stats["fallback_predictions"] += 1
        if result is not None:
            try:
                _record_successful_prediction(
                    temp_paths=temp_paths,
                    stats=stats,
                    query_id=query_id,
                    result=result,
                    sample=sample,
                    candidates=candidates,
                    answer_contract=answer_contract,
                    selected_context=selected_context,
                    config=config,
                )
            except Exception as normalization_exc:
                repaired = False
                try:
                    stats["json_repair_attempts"] += 1
                    stats["answer_api_calls"] += 1
                    repair_result = answer_client.generate_prediction(_json_repair_messages(sample, str(result.get("content") or "")), image_paths=None)
                    _record_successful_prediction(
                        temp_paths=temp_paths,
                        stats=stats,
                        query_id=query_id,
                        result=repair_result,
                        sample=sample,
                        candidates=candidates,
                        answer_contract=answer_contract,
                        selected_context=selected_context,
                        config=config,
                    )
                    stats["json_repair_successes"] += 1
                    repaired = True
                except Exception as repair_exc:
                    append_jsonl(
                        temp_paths["errors"],
                        {
                            "query_id": query_id,
                            "type": "vlm2_json_repair_failure",
                            "error": str(repair_exc),
                            "original_error": str(normalization_exc),
                        },
                    )
                if not repaired:
                    stats["generation_errors"] += 1
                    append_jsonl(temp_paths["errors"], {"query_id": query_id, "type": "vlm2_rerun_normalization_failure", "error": str(normalization_exc)})
                    fallback = make_fallback_prediction(sample, _top_candidate(candidates))
                    append_jsonl(temp_paths["predictions"], fallback)
                    append_jsonl(temp_paths["internal_predictions"], fallback)
                    stats["fallback_predictions"] += 1
        _append_selected_context_rows(temp_paths=temp_paths, stats=stats, query_id=query_id, selected_context=selected_context)
        stats["completed_queries"] += 1

    _write_report(temp_paths["rerun_report"], stats)
    _replace_outputs(temp_paths, final_paths)
    shutil.rmtree(output_dir / ".vlm2_rerun_tmp", ignore_errors=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
