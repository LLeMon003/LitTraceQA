from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import load_pipeline_config
from ..data_io import append_jsonl, extract_answer_contract, find_official_file, read_jsonl, write_jsonl
from ..parser import extract_json_object, make_fallback_prediction, normalize_prediction, strip_internal_grounding, validate_prediction_shape
from ..vlm_answer_client import VLMAnswerClient
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
    parser.add_argument("--backup-existing", action="store_true", default=True)
    parser.add_argument("--no-backup-existing", dest="backup_existing", action="store_false")
    return parser.parse_args()


def _latest_by_query_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if query_id:
            latest[query_id] = row
    return latest


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


def _candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for candidate in candidates:
        paper_id = str(candidate.get("paper_id") or "")
        if paper_id:
            ids.append(paper_id)
    return ids


def _top_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return candidates[0] if candidates and isinstance(candidates[0], dict) else None


def _selected_context_from_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_evidence": row.get("selected_evidence") if isinstance(row.get("selected_evidence"), list) else [],
        "has_partial_artifacts": bool(row.get("has_partial_artifacts")),
        "partial_artifacts_present": bool(row.get("has_partial_artifacts")),
        "attached_image_refs": row.get("attached_image_refs") if isinstance(row.get("attached_image_refs"), list) else [],
    }


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
        f"- completed_queries: {stats['completed_queries']}",
        f"- answer_api_calls: {stats['answer_api_calls']}",
        f"- successful_predictions: {stats['successful_predictions']}",
        f"- fallback_predictions: {stats['fallback_predictions']}",
        f"- generation_errors: {stats['generation_errors']}",
        f"- missing_candidate_rows: {stats['missing_candidate_rows']}",
        f"- missing_selected_context_rows: {stats['missing_selected_context_rows']}",
        f"- multiple_choice_queries: {stats['multiple_choice_queries']}",
        f"- multiple_choice_with_options: {stats['multiple_choice_with_options']}",
        f"- table_queries: {stats['table_queries']}",
        f"- table_with_schema: {stats['table_with_schema']}",
        f"- backup_dir: `{stats.get('backup_dir', '')}`",
        "",
        "This rerun reuses retrieval candidates and selected symbolic evidence from the existing run output. It does not call retrieval, PDF rendering, or VLM-1.",
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
    validation_rows = read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))
    validation_by_query_id = _latest_by_query_id(validation_rows)
    candidates_by_query_id = _latest_by_query_id(read_jsonl(output_dir / "candidate_papers.jsonl"))
    selected_by_query_id = _latest_by_query_id(read_jsonl(output_dir / "selected_symbolic_contexts.prompt.jsonl"))

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
        "completed_queries": 0,
        "answer_api_calls": 0,
        "successful_predictions": 0,
        "fallback_predictions": 0,
        "generation_errors": 0,
        "missing_candidate_rows": 0,
        "missing_selected_context_rows": 0,
        "multiple_choice_queries": 0,
        "multiple_choice_with_options": 0,
        "table_queries": 0,
        "table_with_schema": 0,
        "backup_dir": backup_dir,
    }

    iterator: Any = inputs
    if args.show_progress and tqdm is not None:
        iterator = tqdm(inputs, desc="vlm2-rerun", unit="query")

    for sample in iterator:
        query_id = str(sample.get("query_id") or "")
        candidate_row = candidates_by_query_id.get(query_id) or {}
        selected_row = selected_by_query_id.get(query_id) or {}
        candidates = candidate_row.get("candidates") if isinstance(candidate_row.get("candidates"), list) else []
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
            prediction = make_fallback_prediction(sample, _top_candidate(candidates))
            append_jsonl(temp_paths["predictions"], prediction)
            append_jsonl(temp_paths["internal_predictions"], prediction)
            stats["fallback_predictions"] += 1
            stats["completed_queries"] += 1
            continue

        try:
            stats["answer_api_calls"] += 1
            result = answer_client.generate_prediction(messages, image_paths=None)
            append_jsonl(temp_paths["raw_answer"], {"query_id": query_id, "content": result["content"], "raw_response": result["raw_response"]})
            internal = extract_json_object(str(result["content"]))
            append_jsonl(temp_paths["internal_predictions"], internal)
            prediction, errors = normalize_prediction(
                internal,
                sample,
                _candidate_ids(candidates),
                answer_contract=answer_contract,
                selected_evidence=selected_context.get("selected_evidence", []),
            )
            for error in errors:
                append_jsonl(temp_paths["errors"], error)
            if not validate_prediction_shape(prediction):
                raise ValueError("Normalized prediction does not match expected shape")
            append_jsonl(temp_paths["predictions"], strip_internal_grounding(prediction))
            stats["successful_predictions"] += 1
        except Exception as exc:
            stats["generation_errors"] += 1
            append_jsonl(temp_paths["errors"], {"query_id": query_id, "type": "vlm2_rerun_generation_failure", "error": str(exc)})
            fallback = make_fallback_prediction(sample, _top_candidate(candidates))
            append_jsonl(temp_paths["predictions"], fallback)
            append_jsonl(temp_paths["internal_predictions"], fallback)
            stats["fallback_predictions"] += 1
        stats["completed_queries"] += 1

    _write_report(temp_paths["rerun_report"], stats)
    _replace_outputs(temp_paths, final_paths)
    shutil.rmtree(output_dir / ".vlm2_rerun_tmp", ignore_errors=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
