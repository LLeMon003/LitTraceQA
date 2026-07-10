from __future__ import annotations

import argparse
import json
import re
import signal
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_pipeline_config, model_slug
from .data_io import append_jsonl, extract_answer_contract, find_official_file, read_jsonl, write_jsonl
from .metadata_index import HYBRID_SCORE_WEIGHTS, build_metadata_records, retrieve_candidates
from .metadata_selection import (
    build_metadata_selection_messages,
    empty_metadata_prediction,
    normalize_metadata_selection,
    selection_candidates_for_metadata_vlm,
)
from .openreview_filter import filter_openreview_metadata
from .page_ranker import build_global_page_pool, rank_global_pages_for_query
from .parser import extract_json_object, make_fallback_prediction, normalize_prediction, strip_internal_grounding, validate_prediction_shape
from .pdf_cache import ensure_candidate_pdfs, write_pdf_availability
from .pdf_native_text_scanner import extract_native_page_text
from .pdf_page_renderer import render_pdf_pages
from .symbolic_context_selector import select_symbolic_contexts
from .symbolic_index import build_symbolic_index
from .symbolic_validator import migrate_legacy_record_to_runtime, normalize_page_records, to_runtime_record, validate_page_structure
from .sync_symbolic_run_to_processed import sync_symbolic_run_to_processed
from .topic_expansion import expand_candidates_with_topic_profiles
from .vlm_answer_client import VLMAnswerClient
from .vlm_answer_prompt_builder import build_symbolic_answer_prompt
from .vlm_parser_client import VLMParserClient
from .vlm_parser_prompt_builder import build_page_parser_continuation_prompt, build_page_parser_prompt


BASELINE_TYPE = "pdf_vlm_symbolic_vlm"


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF VLM symbolic VLM LitTraceQA baseline.")
    p.add_argument("--official-dir", default="official_dev")
    p.add_argument("--output-dir", default="outputs/pdf_vlm_symbolic_vlm_baseline")
    p.add_argument("--pdf-output-dir", default="raw_pdfs")
    p.add_argument("--processed-output-dir", default="processed_pdfs/vlm_symbolic")
    p.add_argument("--symbolic-cache-root", default=None, help="Optional symbolic cache root override. Pass a vlm_symbolic_runs/<batch> directory to read/write <parser_slug>/<paper_id> caches there.")
    p.add_argument("--top-k-papers", type=int, default=5)
    p.add_argument("--top-n-records", type=int, default=24)
    p.add_argument("--top-n-visual-records", type=int, default=6)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--skip-parse", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--force-parse", action="store_true")
    p.add_argument("--clear-structured-cache", action="store_true")
    p.add_argument(
        "--structured-cache-policy",
        choices=["reuse_complete_only", "reuse_partial_allowed", "refresh", "fail_if_missing", "reuse", "fail-if-missing"],
        default=None,
    )
    p.add_argument("--render-dpi", type=int, default=None)
    p.add_argument("--render-format", default=None)
    p.add_argument("--max-pages-per-paper", type=int, default=None)
    p.add_argument("--pdf-sleep-seconds", type=float, default=2.0)
    p.add_argument("--pdf-timeout-seconds", type=float, default=60.0)
    p.add_argument("--pdf-max-retries", type=int, default=2)
    p.add_argument("--pdf-overwrite", action="store_true")
    p.add_argument("--env-path", default=".env")
    p.add_argument("--skip-openreview-papers", action="store_true")
    p.add_argument("--show-progress", action="store_true")
    p.add_argument("--max-parser-json-failures", type=int, default=3)
    p.add_argument("--vlm2-context-mode", choices=["text_only", "cropped_image"], default=None)
    p.add_argument("--vlm2-include-parse-confidence", dest="vlm2_include_parse_confidence", action="store_true", default=None)
    p.add_argument("--vlm2-no-parse-confidence", dest="vlm2_include_parse_confidence", action="store_false")
    p.add_argument("--metadata-only-eval-freeze", action="store_true", help="Skip PDF/VLM-1/symbolic stages and use VLM-2 to select gold_papers from top-k metadata candidates.")
    p.add_argument("--page-routing-enabled", dest="page_routing_enabled", action="store_true", default=None)
    p.add_argument("--no-page-routing", dest="page_routing_enabled", action="store_false")
    p.add_argument("--page-routing-top-pages-global", type=int, default=None)
    p.add_argument("--page-routing-max-pages-global", type=int, default=None)
    p.add_argument("--page-routing-parse-batch-size", type=int, default=None)
    p.add_argument("--page-routing-expansion-step-global", type=int, default=None)
    p.add_argument("--no-sync-processed-run-store", dest="sync_processed_run_store", action="store_false", default=True)
    return p.parse_args()


def _paths(output_dir: Path, resume: bool, dry_run: bool, skip_generation: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "predictions": "predictions.jsonl",
        "internal_predictions": "internal_predictions.jsonl",
        "candidate_papers": "candidate_papers.jsonl",
        "pdf_availability": "pdf_availability.jsonl",
        "native_page_text": "native_page_text.jsonl",
        "global_page_pool": "global_page_pool.jsonl",
        "global_page_ranking": "global_page_ranking.jsonl",
        "global_page_parse_plan": "global_page_parse_plan.jsonl",
        "page_rendering": "page_rendering_artifacts.jsonl",
        "parser_artifacts": "parser_artifacts.jsonl",
        "symbolic_records_runtime": "symbolic_records.runtime.jsonl",
        "symbolic_records_debug": "symbolic_records.debug.jsonl",
        "selected_contexts_debug": "selected_symbolic_contexts.debug.jsonl",
        "selected_contexts_prompt": "selected_symbolic_contexts.prompt.jsonl",
        "raw_parser": "raw_vlm_parser_responses.jsonl",
        "raw_answer": "raw_vlm_answer_responses.jsonl",
        "metadata_selection_prompts": "metadata_selection_prompts.jsonl",
        "raw_metadata_selection": "raw_vlm_metadata_selection.jsonl",
        "prompt_previews": "prompt_previews.jsonl",
        "errors": "errors.jsonl",
        "report": "run_report.md",
    }
    paths = {key: output_dir / name for key, name in names.items()}
    if not resume:
        for key, path in paths.items():
            if key == "report":
                continue
            if key in {"predictions", "internal_predictions", "raw_answer"} and (dry_run or skip_generation):
                if path.exists():
                    path.unlink()
            else:
                path.write_text("", encoding="utf-8")
    else:
        for path in paths.values():
            if path.suffix == ".jsonl":
                path.touch(exist_ok=True)
    return paths


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _cache_reusable(
    status_path: Path,
    symbolic_path: Path,
    parser_model: str,
    render_dpi: int,
    artifact_version: str,
    parser_mode: str,
    cache_policy: str,
) -> bool:
    status = _load_json(status_path)
    if not (
        status
        and status.get("parser_model") == parser_model
        and status.get("render_dpi") == render_dpi
        and status.get("artifact_version") == artifact_version
        and status.get("parser_mode") == parser_mode
        and symbolic_path.exists()
        and symbolic_path.stat().st_size > 0
    ):
        return False
    if cache_policy == "reuse_partial_allowed":
        return status.get("status") in {"complete", "partial"}
    return status.get("status") == "complete"


def _record_signature(record: dict[str, Any]) -> tuple[str, str, str]:
    text = " ".join(str(record.get("text") or "").lower().split())
    label = " ".join(str(record.get("label") or "").lower().split())
    return (str(record.get("record_type") or ""), text[:240], label[:80])


def _read_symbolic(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _read_page_status_by_page(paper_dir: Path) -> dict[int, str]:
    statuses: dict[int, str] = {}
    status_dir = paper_dir / "page_status"
    if not status_dir.exists():
        return statuses
    for path in status_dir.glob("page_*.status.json"):
        obj = _load_json(path) or {}
        try:
            statuses[int(obj.get("page") or 0)] = str(obj.get("page_status") or "")
        except Exception:
            continue
    return statuses


def _runtime_records_with_status(records: list[dict[str, Any]], paper_dir: Path) -> list[dict[str, Any]]:
    statuses = _read_page_status_by_page(paper_dir)
    with_status: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        copied["_page_status"] = statuses.get(int(copied.get("page") or 0), "")
        with_status.append(copied)
    return with_status


def has_valid_page_symbolic_cache(
    paper_dir: Path,
    page: int,
    parser_model: str,
    artifact_version: str,
    parser_mode: str,
    render_dpi: int,
) -> bool:
    status = _load_json(paper_dir / "page_status" / f"page_{page:03d}.status.json") or {}
    runtime_path = paper_dir / "page_records" / f"page_{page:03d}.records.runtime.jsonl"
    return bool(
        status.get("parser_model") == parser_model
        and status.get("artifact_version") == artifact_version
        and status.get("parser_mode") == parser_mode
        and int(status.get("render_dpi") or render_dpi) == render_dpi
        and status.get("page_status") in {"complete", "partial"}
        and runtime_path.exists()
        and runtime_path.stat().st_size > 0
    )


def _read_page_runtime_cache(paper_dir: Path, page: int) -> list[dict[str, Any]]:
    runtime_path = paper_dir / "page_records" / f"page_{page:03d}.records.runtime.jsonl"
    return _runtime_records_with_status(_read_symbolic(runtime_path), paper_dir)


def _resolve_structured_root(processed_root: Path, parser_model_slug: str, symbolic_cache_root: str | None) -> tuple[Path, bool, str]:
    value = str(symbolic_cache_root or "").strip()
    if not value:
        return processed_root / parser_model_slug, False, ""
    root = Path(value)
    if root.name == parser_model_slug:
        return root, True, value
    return root / parser_model_slug, True, value


def _normalize_runtime_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    normalized: list[dict[str, Any]] = []
    changed = False
    expected = {"paper_id", "page", "record_id", "global_record_id", "record_type", "source_type", "label", "locator", "text", "reading_order"}
    for record in records:
        runtime = migrate_legacy_record_to_runtime(record)
        normalized.append(runtime)
        if set(record.keys()) != expected or any(record.get(key) != runtime.get(key) for key in expected):
            changed = True
    return normalized, changed


def _infer_page_status_from_records(
    *,
    paper_id: str,
    page: int,
    records: list[dict[str, Any]],
    parser_model: str,
    artifact_version: str,
    parser_mode: str,
    render_dpi: int,
    existing_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_status = existing_status or {}
    page_status = str(existing_status.get("page_status") or "")
    if records and page_status not in {"complete", "partial"}:
        page_status = "partial" if records else "failed"
    elif not records and page_status not in {"complete", "partial", "failed"}:
        page_status = "failed"
    return {
        "paper_id": paper_id,
        "page": page,
        "parser_model": parser_model,
        "render_dpi": render_dpi,
        "artifact_version": artifact_version,
        "parser_mode": parser_mode,
        "page_status": page_status,
        "pass_count": int(existing_status.get("pass_count") or 0),
        "valid_record_count": len(records),
        "rejected_record_count": int(existing_status.get("rejected_record_count") or 0),
        "deduplicated_record_count": int(existing_status.get("deduplicated_record_count") or 0),
        "needs_continuation_final": bool(existing_status.get("needs_continuation_final", page_status == "partial")),
        "known_omissions": existing_status.get("known_omissions") if isinstance(existing_status.get("known_omissions"), list) else [],
        "failure_reason": str(existing_status.get("failure_reason") or ""),
        "created_at": str(existing_status.get("created_at") or datetime.now(timezone.utc).isoformat()),
    }


def _manifest_page_lookup(paper_dir: Path) -> dict[int, dict[str, Any]]:
    manifest = _load_json(paper_dir / "document_manifest.json") or {}
    lookup: dict[int, dict[str, Any]] = {}
    for page_info in manifest.get("pages", []) if isinstance(manifest.get("pages"), list) else []:
        if not isinstance(page_info, dict):
            continue
        try:
            lookup[int(page_info.get("page") or 0)] = page_info
        except (TypeError, ValueError):
            continue
    return lookup


def _raw_parser_content(raw_path: Path) -> str:
    raw = _load_json(raw_path) or {}
    if raw.get("content") is not None:
        return str(raw.get("content") or "")
    raw_response = raw.get("raw_response")
    if isinstance(raw_response, dict):
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else {}
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message.get("content") or "")
    return ""


def _raw_response_finish_reason(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    raw_response = result.get("raw_response")
    if not isinstance(raw_response, dict):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    return str(first.get("finish_reason") or "")


def _rebuild_page_records_from_raw(
    paper_dir: Path,
    *,
    paper_id: str,
    page: int,
    parser_model: str,
    artifact_version: str,
    parser_mode: str,
    render_dpi: int,
) -> bool:
    raw_paths = sorted((paper_dir / "page_parser_raw").glob(f"page_{page:03d}_pass_*.raw.json"))
    if not raw_paths:
        return False
    page_info = _manifest_page_lookup(paper_dir).get(page, {})
    page_width = int(page_info.get("width_px") or 0)
    page_height = int(page_info.get("height_px") or 0)
    page_records: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str, str]] = set()
    rejected = 0
    deduplicated = 0
    needs_continuation = False
    known_omissions: list[Any] = []
    failure_reason = ""
    pass_count = 0
    for raw_path in raw_paths:
        match = re.search(r"_pass_(\d+)\.raw\.json$", raw_path.name)
        pass_index = int(match.group(1)) if match else len(page_records) + 1
        pass_count = max(pass_count, pass_index)
        content = _raw_parser_content(raw_path)
        if not content.strip():
            failure_reason = "raw parser response had empty content"
            continue
        try:
            raw_obj = extract_json_object(content)
        except Exception as exc:
            failure_reason = str(exc)
            continue
        repaired = validate_page_structure(raw_obj, paper_id, page, parser_mode)
        records = normalize_page_records(
            repaired,
            parser_model,
            page_width,
            page_height,
            artifact_version=artifact_version,
            parser_mode=parser_mode,
            page_status="partial",
            start_index=len(page_records) + 1,
            pass_index=pass_index,
        )
        for record in records:
            signature = _record_signature(record)
            if signature in seen_signatures:
                deduplicated += 1
                continue
            seen_signatures.add(signature)
            page_records.append(record)
        coverage = repaired.get("coverage") if isinstance(repaired.get("coverage"), dict) else {}
        needs_continuation = bool(coverage.get("needs_continuation"))
        known_omissions.extend(coverage.get("known_omissions") if isinstance(coverage.get("known_omissions"), list) else [])
        if not needs_continuation:
            break
    valid_page_records = [record for record in page_records if record.get("validation_status") != "rejected"]
    rejected = sum(1 for record in page_records if record.get("validation_status") == "rejected")
    if not valid_page_records:
        return False
    page_status = "partial" if needs_continuation or failure_reason else "complete"
    for record in page_records:
        record["page_status"] = page_status
    page_records_dir = paper_dir / "page_records"
    page_status_dir = paper_dir / "page_status"
    page_records_dir.mkdir(parents=True, exist_ok=True)
    page_status_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(page_records_dir / f"page_{page:03d}.records.debug.jsonl", page_records)
    write_jsonl(page_records_dir / f"page_{page:03d}.records.runtime.jsonl", [to_runtime_record(record) for record in valid_page_records])
    status = _infer_page_status_from_records(
        paper_id=paper_id,
        page=page,
        records=valid_page_records,
        parser_model=parser_model,
        artifact_version=artifact_version,
        parser_mode=parser_mode,
        render_dpi=render_dpi,
        existing_status={
            "page_status": page_status,
            "pass_count": pass_count,
            "rejected_record_count": rejected,
            "deduplicated_record_count": deduplicated,
            "needs_continuation_final": needs_continuation,
            "known_omissions": known_omissions,
            "failure_reason": failure_reason,
        },
    )
    (page_status_dir / f"page_{page:03d}.status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def _normalize_existing_symbolic_cache_for_paper(
    paper_dir: Path,
    *,
    paper_id: str,
    parser_model: str,
    artifact_version: str,
    parser_mode: str,
    render_dpi: int,
    stats: dict[str, Any],
) -> None:
    if not paper_dir.exists():
        return
    page_records_dir = paper_dir / "page_records"
    page_status_dir = paper_dir / "page_status"
    page_status_dir.mkdir(parents=True, exist_ok=True)
    normalized_any = False
    page_runtime_paths = sorted(page_records_dir.glob("page_*.records.runtime.jsonl")) if page_records_dir.exists() else []
    for runtime_path in page_runtime_paths:
        match = re.search(r"page_(\d+)\.records\.runtime\.jsonl$", runtime_path.name)
        if not match:
            continue
        page = int(match.group(1))
        records, changed = _normalize_runtime_records(_read_symbolic(runtime_path))
        if changed:
            write_jsonl(runtime_path, records)
            normalized_any = True
        status_path = page_status_dir / f"page_{page:03d}.status.json"
        status = _load_json(status_path) or {}
        status_current = (
            status.get("parser_model") == parser_model
            and status.get("artifact_version") == artifact_version
            and status.get("parser_mode") == parser_mode
            and int(status.get("render_dpi") or render_dpi) == render_dpi
            and status.get("page_status") in {"complete", "partial", "failed"}
        )
        if not status_current:
            status_path.write_text(
                json.dumps(
                    _infer_page_status_from_records(
                        paper_id=paper_id,
                        page=page,
                        records=records,
                        parser_model=parser_model,
                        artifact_version=artifact_version,
                        parser_mode=parser_mode,
                        render_dpi=render_dpi,
                        existing_status=status,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            normalized_any = True
    raw_pages: set[int] = set()
    for raw_path in sorted((paper_dir / "page_parser_raw").glob("page_*_pass_*.raw.json")):
        match = re.search(r"page_(\d+)_pass_\d+\.raw\.json$", raw_path.name)
        if match:
            raw_pages.add(int(match.group(1)))
    for page in sorted(raw_pages):
        runtime_path = page_records_dir / f"page_{page:03d}.records.runtime.jsonl"
        status = _load_json(page_status_dir / f"page_{page:03d}.status.json") or {}
        runtime_empty = not runtime_path.exists() or runtime_path.stat().st_size == 0
        if runtime_empty or status.get("page_status") == "failed":
            if _rebuild_page_records_from_raw(
                paper_dir,
                paper_id=paper_id,
                page=page,
                parser_model=parser_model,
                artifact_version=artifact_version,
                parser_mode=parser_mode,
                render_dpi=render_dpi,
            ):
                normalized_any = True

    paper_records = _collect_paper_runtime_from_page_files(paper_dir)
    symbolic_path = paper_dir / "symbolic_records.runtime.jsonl"
    if symbolic_path.exists():
        records, changed = _normalize_runtime_records(_read_symbolic(symbolic_path))
        if changed:
            write_jsonl(symbolic_path, records)
            normalized_any = True
    elif paper_records:
        write_jsonl(symbolic_path, paper_records)
        normalized_any = True
    elif (paper_dir / "symbolic_records.jsonl").exists():
        migrated = _migrate_legacy_records(paper_dir)
        if migrated:
            write_jsonl(symbolic_path, migrated)
            paper_records = migrated
            normalized_any = True

    page_statuses = _read_page_status_by_page(paper_dir)
    if paper_records:
        complete_pages = sum(1 for value in page_statuses.values() if value == "complete")
        partial_pages = sum(1 for value in page_statuses.values() if value == "partial")
        failed_pages = sum(1 for value in page_statuses.values() if value == "failed")
        paper_status = "complete" if complete_pages and partial_pages == 0 and failed_pages == 0 else "partial"
    else:
        complete_pages = partial_pages = 0
        failed_pages = len(page_statuses)
        paper_status = "failed"
    status_path = paper_dir / "artifact_status.json"
    status = _load_json(status_path) or {}
    status_current = (
        status.get("parser_model") == parser_model
        and status.get("artifact_version") == artifact_version
        and status.get("parser_mode") == parser_mode
        and int(status.get("render_dpi") or render_dpi) == render_dpi
        and status.get("status") in {"complete", "partial", "failed"}
    )
    if not status_current:
        artifact_status = {
            "paper_id": paper_id,
            "parser_model": parser_model,
            "render_dpi": render_dpi,
            "artifact_version": artifact_version,
            "parser_mode": parser_mode,
            "symbolic_records_runtime_path": str(symbolic_path),
            "symbolic_records_debug_path": str(paper_dir / "symbolic_records.debug.jsonl"),
            "page_count": int(status.get("page_count") or 0),
            "parsed_pages": complete_pages + partial_pages,
            "complete_pages": complete_pages,
            "partial_pages": partial_pages,
            "failed_pages": failed_pages,
            "valid_record_count": len(paper_records),
            "rejected_record_count": int(status.get("rejected_record_count") or 0),
            "deduplicated_record_count": int(status.get("deduplicated_record_count") or 0),
            "created_at": str(status.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "status": paper_status,
            "cache_reusable_as_complete": paper_status == "complete",
            "page_level_cache_reusable": bool(paper_records),
        }
        status_path.write_text(json.dumps(artifact_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        normalized_any = True
    if normalized_any:
        stats["symbolic_cache_records_standardized"] = int(stats.get("symbolic_cache_records_standardized") or 0) + 1


def _collect_paper_runtime_from_page_files(paper_dir: Path) -> list[dict[str, Any]]:
    page_records_dir = paper_dir / "page_records"
    rows: list[dict[str, Any]] = []
    for path in sorted(page_records_dir.glob("page_*.records.runtime.jsonl")):
        rows.extend(_read_symbolic(path))
    return rows


def _selected_pages_by_paper(selected_pages: list[dict[str, Any]]) -> dict[str, set[int]]:
    by_paper: dict[str, set[int]] = {}
    for item in selected_pages:
        paper_id = str(item.get("paper_id") or "")
        try:
            page = int(item.get("page") or 0)
        except Exception:
            continue
        if paper_id and page > 0:
            by_paper.setdefault(paper_id, set()).add(page)
    return by_paper


def _evidence_sufficient(selected: dict[str, Any], primary_type: str, min_records: int, require_primary: bool) -> bool:
    records = selected.get("selected_records_debug") or selected.get("selected_records") or []
    if len(records) < max(1, min_records):
        return False
    if not any(str(record.get("text") or "").strip() for record in records):
        return False
    if require_primary and primary_type in {"figure", "table", "equation_algorithm", "citation_context"}:
        return any(str(record.get("source_type") or record.get("record_type") or "") == primary_type for record in records)
    return True


def _update_selected_page_distribution(stats: dict[str, Any], selected_pages: list[dict[str, Any]]) -> None:
    by_rank = dict(stats.get("distribution_of_selected_pages_by_candidate_rank") or {})
    by_paper = dict(stats.get("distribution_of_selected_pages_by_paper") or {})
    papers: set[str] = set()
    for item in selected_pages:
        rank = str(item.get("candidate_rank") or "")
        paper_id = str(item.get("paper_id") or "")
        if rank:
            by_rank[rank] = int(by_rank.get(rank, 0)) + 1
        if paper_id:
            by_paper[paper_id] = int(by_paper.get(paper_id, 0)) + 1
            papers.add(paper_id)
    if len(papers) == 1 and selected_pages:
        stats["queries_where_all_selected_pages_from_one_paper"] += 1
    elif len(papers) > 1:
        stats["queries_where_selected_pages_from_multiple_papers"] += 1
    stats["distribution_of_selected_pages_by_candidate_rank"] = by_rank
    stats["distribution_of_selected_pages_by_paper"] = by_paper


def _page_batches(selected_pages: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(batch_size or len(selected_pages) or 1))
    return [selected_pages[index : index + size] for index in range(0, len(selected_pages), size)]


def _progress(items: list[Any], *, enabled: bool, desc: str, unit: str, leave: bool = False) -> Any:
    if enabled and tqdm is not None:
        return tqdm(items, desc=desc, unit=unit, leave=leave)
    return items


def _parse_rows_for_pages(
    *,
    query_id: str,
    rows: list[dict[str, Any]],
    selected_pages_by_paper: dict[str, set[int]] | None,
    args: argparse.Namespace,
    config: Any,
    metadata_by_id: dict[str, dict[str, Any]],
    candidate_score_by_id: dict[str, float],
    structured_root: Path,
    render_dpi: int,
    render_format: str,
    max_pages: int,
    cache_policy: str,
    parser: VLMParserClient,
    paths: dict[str, Path],
    stats: dict[str, Any],
    paper_records_cache: dict[str, list[dict[str, Any]]],
    page_routing_enabled: bool,
) -> list[dict[str, Any]]:
    candidate_records: list[dict[str, Any]] = []
    for row in rows:
        if args.max_parser_json_failures > 0 and stats["parser_json_failures"] >= args.max_parser_json_failures:
            append_jsonl(
                paths["errors"],
                {
                    "query_id": query_id,
                    "type": "parser_json_failure_threshold_reached",
                    "max_parser_json_failures": args.max_parser_json_failures,
                    "parser_json_failures": stats["parser_json_failures"],
                },
            )
            stats["run_status"] = "failed"
            stats["interrupted_reason"] = "parser_json_failure_threshold_reached"
            _write_report_from_paths(paths, stats)
            raise RuntimeError(f"Parser JSON/page failure threshold reached: {stats['parser_json_failures']} >= {args.max_parser_json_failures}")
        paper_id = str(row["paper_id"])
        if not row.get("available"):
            continue
        selected_pages = selected_pages_by_paper.get(paper_id, set()) if selected_pages_by_paper is not None else None
        if selected_pages_by_paper is not None and not selected_pages:
            continue
        paper_dir = structured_root / paper_id
        if stats.get("symbolic_cache_root_override_enabled"):
            _normalize_existing_symbolic_cache_for_paper(
                paper_dir,
                paper_id=paper_id,
                parser_model=config.parser_model,
                artifact_version=config.symbolic_artifact_version,
                parser_mode=config.parser_extraction_mode,
                render_dpi=render_dpi,
                stats=stats,
            )
        symbolic_path = paper_dir / "symbolic_records.runtime.jsonl"
        debug_symbolic_path = paper_dir / "symbolic_records.debug.jsonl"
        status_path = paper_dir / "artifact_status.json"
        if not page_routing_enabled and paper_id in paper_records_cache:
            cached_records = [dict(record) for record in paper_records_cache[paper_id]]
            for record in cached_records:
                record["_candidate_bm25_score"] = candidate_score_by_id.get(str(record.get("paper_id")), 0.0)
            candidate_records.extend(cached_records)
            continue
        manifest_path = paper_dir / "document_manifest.json"
        manifest = _load_json(manifest_path)
        if not args.skip_render:
            manifest = render_pdf_pages(
                paper_id,
                row["local_path"],
                paper_dir,
                dpi=render_dpi,
                image_format=render_format,
                max_pages=max_pages,
                overwrite=False,
                selected_pages=selected_pages,
            )
            append_jsonl(paths["page_rendering"], manifest)
            stats["rendered_papers"] += 1
            stats["rendered_pages"] += len(manifest.get("pages", []))
        if (
            not page_routing_enabled
            and cache_policy in {"reuse_complete_only", "reuse_partial_allowed", "fail_if_missing"}
            and not args.force_parse
            and _cache_reusable(status_path, symbolic_path, config.parser_model, render_dpi, config.symbolic_artifact_version, config.parser_extraction_mode, cache_policy)
        ):
            records = _runtime_records_with_status(_read_symbolic(symbolic_path), paper_dir)
            status = _load_json(status_path) or {}
            stats["reused_structured_papers"] += 1
            if status.get("status") == "partial":
                stats["reused_partial_artifacts"] += 1
            else:
                stats["reused_complete_artifacts"] += 1
            stats["skipped_pages"] += int(status.get("parsed_pages") or 0)
            stats["parser_api_calls_saved"] += int(status.get("parsed_pages") or 0)
            stats["valid_symbolic_records"] += len(records)
            stats["runtime_records_total"] += len(records)
            stats["debug_records_total"] += _count_jsonl(debug_symbolic_path)
            stats["rejected_symbolic_records"] += int(status.get("rejected_record_count") or 0)
            stats["complete_pages"] += int(status.get("complete_pages") or 0)
            stats["partial_pages"] += int(status.get("partial_pages") or 0)
            stats["failed_pages"] += int(status.get("failed_pages") or 0)
            if status.get("status") == "complete":
                stats["papers_complete"] += 1
            elif status.get("status") == "partial":
                stats["papers_partial"] += 1
            elif status.get("status") == "failed":
                stats["papers_failed"] += 1
        elif cache_policy == "fail_if_missing":
            append_jsonl(paths["errors"], {"query_id": query_id, "paper_id": paper_id, "type": "structured_cache_missing"})
            records = []
        elif args.skip_parse:
            records = _runtime_records_with_status(_read_symbolic(symbolic_path), paper_dir)
            if not records:
                migrated = _migrate_legacy_records(paper_dir)
                if migrated:
                    records = _runtime_records_with_status(migrated, paper_dir)
                    stats["old_artifacts_migrated_count"] += 1
        else:
            legacy_status = _load_json(status_path) or {}
            if legacy_status and (
                legacy_status.get("artifact_version") != config.symbolic_artifact_version
                or legacy_status.get("parser_mode") != config.parser_extraction_mode
            ):
                stats["old_artifacts_rejected_as_cache_count"] += 1
            if not manifest:
                append_jsonl(paths["errors"], {"query_id": query_id, "paper_id": paper_id, "type": "missing_render_manifest"})
                records = []
            elif not parser.supports_image_input():
                append_jsonl(paths["errors"], {"query_id": query_id, "paper_id": paper_id, "type": "parser_image_input_unavailable"})
                records = []
            else:
                before_calls = int(stats.get("parser_api_calls") or 0)
                records = _parse_paper(
                    paper_id=paper_id,
                    metadata=metadata_by_id.get(paper_id, {}),
                    manifest=manifest,
                    paper_dir=paper_dir,
                    parser=parser,
                    parser_model=config.parser_model,
                    artifact_version=config.symbolic_artifact_version,
                    parser_mode=config.parser_extraction_mode,
                    max_records_per_call=config.parser_max_records_per_call,
                    max_continuations_per_page=config.parser_max_continuations_per_page,
                    retry_on_json_failure=config.parser_retry_on_json_failure,
                    allow_partial_page=config.parser_allow_partial_page,
                    paths=paths,
                    stats=stats,
                    selected_pages=selected_pages,
                    force_parse=args.force_parse,
                    show_progress=args.show_progress,
                )
                stats["vlm1_pages_actually_parsed"] += max(0, int(stats.get("parser_api_calls") or 0) - before_calls)
                if records:
                    status = _load_json(paper_dir / "artifact_status.json") or {}
                    if status.get("status") == "complete":
                        stats["newly_parsed_papers"] += 1
                    else:
                        stats["failed_parsed_papers"] += 1
                else:
                    stats["failed_parsed_papers"] += 1
        if not page_routing_enabled:
            paper_records_cache[paper_id] = records
        for record in records:
            record["_candidate_bm25_score"] = candidate_score_by_id.get(str(record.get("paper_id")), 0.0)
        candidate_records.extend(records)
        for record in records:
            runtime_record = {k: v for k, v in record.items() if not k.startswith("_")}
            append_jsonl(paths["symbolic_records_runtime"], runtime_record)
        if debug_symbolic_path.exists():
            for debug_record in _read_symbolic(debug_symbolic_path):
                if selected_pages is not None and int(debug_record.get("page") or 0) not in selected_pages:
                    continue
                append_jsonl(paths["symbolic_records_debug"], debug_record)
        _write_report_from_paths(paths, stats)
    return candidate_records


def _migrate_legacy_records(paper_dir: Path) -> list[dict[str, Any]]:
    legacy_path = paper_dir / "symbolic_records.jsonl"
    if not legacy_path.exists():
        return []
    return [migrate_legacy_record_to_runtime(record) for record in _read_symbolic(legacy_path)]


def _effective_vlm2_context_mode(requested_mode: str, answer_supports_images: bool, stats: dict[str, Any]) -> str:
    if requested_mode not in {"text_only", "cropped_image"}:
        stats["vlm2_context_mode_downgraded"] = True
        stats["vlm2_context_mode_downgrade_reason"] = f"unsupported_context_mode:{requested_mode}"
        return "text_only"
    if requested_mode == "cropped_image" and not answer_supports_images:
        stats["vlm2_context_mode_downgraded"] = True
        stats["vlm2_context_mode_downgrade_reason"] = "answer_model_image_input_unavailable"
        return "text_only"
    return requested_mode


def _attach_vlm2_images(
    selected: dict[str, Any],
    mode: str,
    output_dir: Path,
    paths: dict[str, Path],
    stats: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    if mode == "text_only":
        selected["attached_image_refs"] = []
        return [], selected
    image_paths: list[str] = []
    attached_refs: list[str] = []
    for evidence, debug in zip(selected.get("selected_evidence", []), selected.get("selected_records_debug", [])):
        image_ref = evidence.get("image_ref")
        if debug.get("source_type") not in {"table", "figure", "equation_algorithm"}:
            evidence.pop("image_ref", None)
            continue
        if not image_ref:
            evidence.pop("image_ref", None)
            continue
        precomputed_crop = debug.get("figure_crop_path")
        if precomputed_crop and Path(str(precomputed_crop)).exists():
            attached_refs.append(str(image_ref))
            image_paths.append(str(precomputed_crop))
            continue
        evidence.pop("image_ref", None)
        append_jsonl(paths["errors"], {"query_id": selected.get("query_id"), "type": "vlm2_precomputed_crop_missing", "record_id": debug.get("record_id")})
        stats["vlm2_crop_failures"] += 1
    selected["attached_image_refs"] = attached_refs
    stats["vlm2_attached_image_count"] += len(image_paths)
    return image_paths, selected


def _prompt_context_stats(selected_prompt: dict[str, Any], stats: dict[str, Any]) -> None:
    records = selected_prompt.get("selected_evidence", [])
    if not isinstance(records, list) or not records:
        return
    total_fields = sum(len(record) for record in records if isinstance(record, dict))
    total_chars = sum(len(str(record.get("text") or "")) for record in records if isinstance(record, dict))
    stats["vlm2_prompt_evidence_record_count"] += len(records)
    stats["vlm2_prompt_evidence_field_total"] += total_fields
    stats["vlm2_prompt_evidence_char_total"] += total_chars
    count = max(1, int(stats["vlm2_prompt_evidence_record_count"]))
    stats["average_prompt_evidence_fields_per_record"] = round(stats["vlm2_prompt_evidence_field_total"] / count, 3)
    stats["average_prompt_evidence_chars_per_record"] = round(stats["vlm2_prompt_evidence_char_total"] / count, 3)
    for source_type, count_value in (selected_prompt.get("source_type_distribution") or {}).items():
        stats["selected_evidence_source_type_distribution"][str(source_type)] = (
            int(stats["selected_evidence_source_type_distribution"].get(str(source_type), 0)) + int(count_value or 0)
        )
    stats["primary_evidence_selected_count"] += int(selected_prompt.get("primary_evidence_type_count") or 0)
    stats["supporting_evidence_selected_count"] += int(selected_prompt.get("supporting_evidence_count") or 0)
    if selected_prompt.get("context_truncated"):
        stats["vlm2_context_truncated_count"] += 1
    completed = max(1, int(stats.get("processed_queries") or 0) + 1)
    stats["average_vlm2_selected_records_per_query"] = round(
        float(stats.get("vlm2_prompt_evidence_record_count") or 0) / completed,
        3,
    )
    for label_type, count_value in (selected_prompt.get("grounding_label_hints_by_type") or {}).items():
        stats["grounding_label_hints_by_type"][str(label_type)] = (
            int(stats["grounding_label_hints_by_type"].get(str(label_type), 0)) + int(count_value or 0)
        )


def _update_answer_contract_stats(answer_contract: dict[str, Any], stats: dict[str, Any], query_id: str, paths: dict[str, Path]) -> None:
    answer_types = answer_contract.get("answer_types") or []
    if "multiple_choice" not in answer_types:
        return
    stats["multiple_choice_queries_count"] += 1
    options = (answer_contract.get("multiple_choice") or {}).get("options") or []
    if options:
        stats["multiple_choice_options_available_count"] += 1
    else:
        stats["multiple_choice_options_missing_count"] += 1
        append_jsonl(paths["errors"], {"query_id": query_id, "type": "missing_multiple_choice_options"})


def _update_normalization_error_stats(errors: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    for error in errors:
        error_type = str(error.get("type") or "")
        if error_type == "invalid_multiple_choice_option":
            stats["invalid_multiple_choice_outputs_count"] += 1
        elif error_type == "answer_type_extra_fields_removed":
            stats["answer_type_extra_fields_removed_count"] += 1
        elif error_type == "missing_required_answer_type":
            stats["answer_type_missing_required_count"] += 1
        elif error_type == "invented_grounding_label_removed":
            stats["invented_grounding_labels_removed_count"] += 1
        elif error_type == "locator_validation_error":
            stats["locator_validation_errors_count"] += 1
        elif error_type == "symbolic_evidence_locator_standardized":
            stats["symbolic_evidence_locator_standardized_count"] += 1
        elif error_type == "symbolic_evidence_empty_filled":
            stats["symbolic_evidence_empty_filled_count"] += 1
        elif error_type == "symbolic_evidence_non_object_replaced":
            stats["symbolic_evidence_non_object_replaced_count"] += 1
        elif error_type == "symbolic_evidence_standardization_no_match":
            stats["symbolic_evidence_standardization_no_match_count"] += 1
        elif error_type == "symbolic_evidence_standardization_no_locator":
            stats["symbolic_evidence_standardization_no_locator_count"] += 1


def _empty_answer_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for answer_type in sample.get("answer_types", []):
        if answer_type == "freeform":
            answer["freeform"] = {"text": ""}
        elif answer_type == "multiple_choice":
            answer["multiple_choice"] = {"gold": ""}
        elif answer_type == "table":
            answer["table"] = {"rows": []}
    return answer


def _metadata_only_vlm_prediction(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    answer_client: VLMAnswerClient,
    paths: dict[str, Path],
    stats: dict[str, Any],
    selection_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_id = str(sample.get("query_id") or "")
    messages = build_metadata_selection_messages(sample, candidates)
    append_jsonl(paths["metadata_selection_prompts"], {"query_id": query_id, "selection_policy": selection_policy or {}, "messages": messages})
    stats["vlm2_metadata_selection_calls"] += 1
    try:
        result = answer_client.generate_prediction(messages, image_paths=None)
        append_jsonl(paths["raw_metadata_selection"], {"query_id": query_id, "content": result["content"], "raw_response": result["raw_response"]})
        selected = extract_json_object(str(result["content"]))
        prediction, errors = normalize_metadata_selection(selected, sample, candidates)
        for error in errors:
            if error.get("type") == "metadata_vlm_no_valid_papers_selected":
                stats["vlm2_metadata_no_valid_selection_count"] += 1
            append_jsonl(paths["errors"], error)
        return prediction
    except Exception as exc:
        stats["vlm2_metadata_selection_failures"] += 1
        append_jsonl(paths["errors"], {"query_id": query_id, "type": "metadata_vlm_selection_failure", "error": str(exc)})
        return empty_metadata_prediction(sample)


def _update_retrieval_coverage(sample: dict[str, Any], candidates: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    gold_items = sample.get("gold_papers") or sample.get("papers") or []
    gold_ids: set[str] = set()
    if isinstance(gold_items, list):
        for item in gold_items:
            if isinstance(item, dict):
                value = item.get("paper_id")
            else:
                value = item
            if value:
                gold_ids.add(str(value))
    if not gold_ids:
        return
    stats["top_k_retrieval_coverage_denominator"] += 1
    candidate_ids = {str(candidate.get("paper_id") or "") for candidate in candidates}
    if gold_ids & candidate_ids:
        stats["top_k_retrieval_coverage_hits"] += 1
    denom = max(1, stats["top_k_retrieval_coverage_denominator"])
    stats["top_k_retrieval_candidate_coverage"] = round(stats["top_k_retrieval_coverage_hits"] / denom, 6)


def _task_family_bucket(task_family: str) -> str:
    normalized = task_family.strip().lower().replace("-", "_")
    if "multi" in normalized:
        return "multi_paper"
    return "single_paper"


def _query_budget(sample: dict[str, Any], config: PipelineConfig, fallback_top_k: int) -> dict[str, Any]:
    task_family = str(sample.get("task_family") or "")
    bucket = _task_family_bucket(task_family)
    if not config.task_family_budget_enabled:
        return {
            "task_family": task_family,
            "task_family_bucket": bucket,
            "top_k_papers": max(1, int(fallback_top_k)),
            "page_routing_top_pages_per_candidate": max(1, int(config.page_routing_top_pages_per_candidate)),
        }
    if bucket == "multi_paper":
        return {
            "task_family": task_family,
            "task_family_bucket": bucket,
            "top_k_papers": max(1, int(config.multi_paper_top_k_papers)),
            "page_routing_top_pages_per_candidate": max(1, int(config.multi_paper_page_routing_top_pages_per_candidate)),
        }
    return {
        "task_family": task_family,
        "task_family_bucket": bucket,
        "top_k_papers": max(1, int(config.single_paper_top_k_papers)),
        "page_routing_top_pages_per_candidate": max(1, int(config.single_paper_page_routing_top_pages_per_candidate)),
    }


def _increment_counter_stat(stats: dict[str, Any], name: str, key: Any, amount: int = 1) -> None:
    values = dict(stats.get(name) or {})
    text_key = str(key)
    values[text_key] = int(values.get(text_key, 0)) + int(amount)
    stats[name] = values


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _refresh_output_counts(paths: dict[str, Path], stats: dict[str, Any]) -> None:
    for key, path in paths.items():
        if path.suffix == ".jsonl":
            stats[f"{key}_rows"] = _count_jsonl(path)


def _write_report_from_paths(paths: dict[str, Path], stats: dict[str, Any]) -> None:
    _refresh_output_counts(paths, stats)
    _write_report(paths["report"], stats)


def _parse_paper(
    *,
    paper_id: str,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    paper_dir: Path,
    parser: VLMParserClient,
    parser_model: str,
    artifact_version: str,
    parser_mode: str,
    max_records_per_call: int,
    max_continuations_per_page: int,
    retry_on_json_failure: int,
    allow_partial_page: bool,
    paths: dict[str, Path],
    stats: dict[str, Any],
    selected_pages: set[int] | None = None,
    force_parse: bool = False,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    raw_dir = paper_dir / "page_parser_raw"
    page_records_dir = paper_dir / "page_records"
    page_status_dir = paper_dir / "page_status"
    raw_dir.mkdir(parents=True, exist_ok=True)
    page_records_dir.mkdir(parents=True, exist_ok=True)
    page_status_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    rejected = 0
    deduplicated_total = 0
    complete_pages = 0
    partial_pages = 0
    failed_pages = 0
    target_pages = [
        page_info
        for page_info in manifest.get("pages", [])
        if selected_pages is None or int(page_info["page"]) in selected_pages
    ]
    page_iterator = _progress(
        target_pages,
        enabled=show_progress,
        desc=f"{paper_id} VLM-1 pages",
        unit="page",
        leave=False,
    )
    for page_info in page_iterator:
        page = int(page_info["page"])
        if not force_parse and has_valid_page_symbolic_cache(paper_dir, page, parser_model, artifact_version, parser_mode, int(manifest.get("dpi") or 0)):
            stats["page_level_cache_hits"] += 1
            cached_page_records = _read_page_runtime_cache(paper_dir, page)
            all_records.extend(cached_page_records)
            continue
        stats["page_level_cache_misses"] += 1
        page_records: list[dict[str, Any]] = []
        page_rejected = 0
        page_dedup = 0
        pass_index = 1
        max_passes = 1 + max(0, max_continuations_per_page)
        needs_continuation = False
        known_omissions: list[Any] = []
        failure_reason = ""
        seen_signatures: set[tuple[str, str, str]] = set()
        last_complete_record_id: str | None = None
        next_start_hint: str | None = None
        page_needed_continuation = False
        while pass_index <= max_passes:
            if pass_index == 1:
                messages = build_page_parser_prompt(
                    paper_id,
                    page,
                    metadata,
                    int(page_info["width_px"]),
                    int(page_info["height_px"]),
                    max_records_per_call=max_records_per_call,
                )
            else:
                previous_records_digest = [
                    {
                        "record_id": r.get("record_id"),
                        "kind": r.get("raw_kind"),
                        "label": r.get("label"),
                        "text_preview": str(r.get("text") or "")[:180],
                    }
                    for r in page_records[-max_records_per_call * 2 :]
                ]
                messages = build_page_parser_continuation_prompt(
                    paper_id,
                    page,
                    metadata,
                    int(page_info["width_px"]),
                    int(page_info["height_px"]),
                    pass_index,
                    previous_records_digest,
                    last_complete_record_id,
                    next_start_hint,
                    max_records_per_call=max_records_per_call,
                )
            raw_obj: dict[str, Any] | None = None
            last_exc: Exception | None = None
            last_result: dict[str, Any] | None = None
            for attempt in range(retry_on_json_failure + 1):
                try:
                    result = parser.generate_page_structure(messages, page_info["image_path"])
                    last_result = result
                    stats["parser_api_calls"] += 1
                    stats["total_parser_calls"] += 1
                    raw_name = raw_dir / f"page_{page:03d}_pass_{pass_index:03d}.raw.json"
                    raw_name.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    append_jsonl(
                        paths["raw_parser"],
                        {
                            "paper_id": paper_id,
                            "page": page,
                            "pass_index": pass_index,
                            "attempt": attempt + 1,
                            "content": result["content"],
                            "raw_response": result["raw_response"],
                        },
                    )
                    raw_obj = extract_json_object(str(result["content"]))
                    break
                except Exception as exc:
                    last_exc = exc
                    if _raw_response_finish_reason(last_result) == "length" or attempt >= retry_on_json_failure:
                        break
            if raw_obj is None:
                failure_reason = str(last_exc) if last_exc else "parser returned no parseable JSON"
                finish_reason = _raw_response_finish_reason(last_result)
                error_type = "parser_page_failure"
                if finish_reason == "length":
                    error_type = "parser_truncated_json_failure"
                    stats["parser_truncated_json_failures"] = int(stats.get("parser_truncated_json_failures", 0)) + 1
                    failure_reason = f"{failure_reason}; parser response truncated by max_tokens"
                else:
                    stats["parser_json_failures"] += 1
                append_jsonl(
                    paths["errors"],
                    {
                        "paper_id": paper_id,
                        "page": page,
                        "pass_index": pass_index,
                        "type": error_type,
                        "finish_reason": finish_reason,
                        "error": failure_reason,
                    },
                )
                if finish_reason != "length":
                    max_failures = int(stats.get("max_parser_json_failures") or 0)
                    if max_failures > 0 and stats["parser_json_failures"] >= max_failures:
                        append_jsonl(paths["errors"], {"paper_id": paper_id, "page": page, "type": "parser_json_failure_threshold_reached", "max_parser_json_failures": max_failures, "parser_json_failures": stats["parser_json_failures"]})
                        _write_report_from_paths(paths, stats)
                        raise RuntimeError(f"Parser JSON/page failure threshold reached: {stats['parser_json_failures']} >= {max_failures}")
                break
            repaired = validate_page_structure(raw_obj, paper_id, page, parser_mode)
            start_index = len(page_records) + 1
            records = normalize_page_records(
                repaired,
                parser_model,
                int(page_info["width_px"]),
                int(page_info["height_px"]),
                artifact_version=artifact_version,
                parser_mode=parser_mode,
                page_status="partial",
                start_index=start_index,
                pass_index=pass_index,
            )
            new_records: list[dict[str, Any]] = []
            for record in records:
                signature = _record_signature(record)
                if signature in seen_signatures:
                    page_dedup += 1
                    continue
                seen_signatures.add(signature)
                new_records.append(record)
            page_records.extend(new_records)
            page_rejected += sum(1 for r in new_records if r.get("validation_status") == "rejected")
            coverage = repaired.get("coverage") if isinstance(repaired.get("coverage"), dict) else {}
            needs_continuation = bool(coverage.get("needs_continuation"))
            if needs_continuation:
                page_needed_continuation = True
            last_complete_record_id = str(page_records[-1].get("record_id")) if page_records else coverage.get("last_complete_record_id")
            next_start_hint = str(coverage.get("next_start_hint") or "")
            known_omissions.extend(coverage.get("known_omissions") if isinstance(coverage.get("known_omissions"), list) else [])
            if not new_records and needs_continuation:
                failure_reason = "continuation requested but no new non-duplicate valid records were added"
                break
            if not needs_continuation:
                break
            pass_index += 1
        if needs_continuation and pass_index >= max_passes:
            failure_reason = failure_reason or "max continuations reached before page was complete"
        valid_page_records = [r for r in page_records if r.get("validation_status") != "rejected"]
        if valid_page_records and not needs_continuation and not failure_reason:
            page_status = "complete"
            complete_pages += 1
        elif valid_page_records and allow_partial_page:
            page_status = "partial"
            partial_pages += 1
        else:
            page_status = "failed"
            failed_pages += 1
        for record in page_records:
            record["page_status"] = page_status
        runtime_page_records = [to_runtime_record(r) for r in valid_page_records]
        write_jsonl(page_records_dir / f"page_{page:03d}.records.debug.jsonl", page_records)
        write_jsonl(page_records_dir / f"page_{page:03d}.records.runtime.jsonl", runtime_page_records)
        page_status_obj = {
            "paper_id": paper_id,
            "page": page,
            "parser_model": parser_model,
            "render_dpi": manifest.get("dpi"),
            "artifact_version": artifact_version,
            "parser_mode": parser_mode,
            "page_status": page_status,
            "pass_count": min(pass_index, max_passes),
            "valid_record_count": len(valid_page_records),
            "rejected_record_count": page_rejected,
            "deduplicated_record_count": page_dedup,
            "needs_continuation_final": needs_continuation,
            "known_omissions": known_omissions,
            "failure_reason": failure_reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if page_needed_continuation:
            stats["pages_needing_continuation"] += 1
        (page_status_dir / f"page_{page:03d}.status.json").write_text(json.dumps(page_status_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_records.extend(valid_page_records)
        rejected += page_rejected
        deduplicated_total += page_dedup
    existing_runtime_records = _collect_paper_runtime_from_page_files(paper_dir)
    if existing_runtime_records:
        runtime_records = existing_runtime_records
        debug_records = []
        for path in sorted((paper_dir / "page_records").glob("page_*.records.debug.jsonl")):
            debug_records.extend(_read_symbolic(path))
        debug_records = [r for r in debug_records if r.get("validation_status") != "rejected"]
    else:
        debug_records = [r for r in all_records if r.get("validation_status") != "rejected"]
        runtime_records = [to_runtime_record(r) for r in debug_records]
    page_statuses = _read_page_status_by_page(paper_dir)
    if page_statuses:
        complete_pages = sum(1 for value in page_statuses.values() if value == "complete")
        partial_pages = sum(1 for value in page_statuses.values() if value == "partial")
        failed_pages = sum(1 for value in page_statuses.values() if value == "failed")
    write_jsonl(paper_dir / "symbolic_records.debug.jsonl", debug_records)
    write_jsonl(paper_dir / "symbolic_records.runtime.jsonl", runtime_records)
    build_symbolic_index(runtime_records, paper_dir / "symbolic_index.json")
    if selected_pages is None:
        returned_runtime_records = runtime_records
        returned_debug_records = debug_records
        returned_page_statuses = page_statuses
    else:
        returned_runtime_records = [record for record in runtime_records if int(record.get("page") or 0) in selected_pages]
        returned_debug_records = [record for record in debug_records if int(record.get("page") or 0) in selected_pages]
        returned_page_statuses = {page: status_value for page, status_value in page_statuses.items() if page in selected_pages}
    paper_status = "complete" if runtime_records and partial_pages == 0 and failed_pages == 0 else ("partial" if runtime_records else "failed")
    status = {
        "paper_id": paper_id,
        "parser_model": parser_model,
        "render_dpi": manifest.get("dpi"),
        "artifact_version": artifact_version,
        "parser_mode": parser_mode,
        "symbolic_records_runtime_path": str(paper_dir / "symbolic_records.runtime.jsonl"),
        "symbolic_records_debug_path": str(paper_dir / "symbolic_records.debug.jsonl"),
        "page_count": manifest.get("page_count", 0),
        "parsed_pages": complete_pages + partial_pages,
        "complete_pages": complete_pages,
        "partial_pages": partial_pages,
        "failed_pages": failed_pages,
        "valid_record_count": len(runtime_records),
        "rejected_record_count": rejected,
        "deduplicated_record_count": deduplicated_total,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": paper_status,
        "cache_reusable_as_complete": paper_status == "complete",
        "page_level_cache_reusable": bool(runtime_records),
    }
    (paper_dir / "artifact_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl(paths["parser_artifacts"], status)
    returned_complete_pages = sum(1 for value in returned_page_statuses.values() if value == "complete")
    returned_partial_pages = sum(1 for value in returned_page_statuses.values() if value == "partial")
    returned_failed_pages = sum(1 for value in returned_page_statuses.values() if value == "failed")
    stats["valid_symbolic_records"] += len(returned_runtime_records)
    stats["rejected_symbolic_records"] += rejected
    stats["records_produced_total"] += len(returned_runtime_records) + rejected + deduplicated_total
    stats["records_accepted_total"] += len(returned_runtime_records)
    stats["runtime_records_total"] += len(returned_runtime_records)
    stats["debug_records_total"] += len(returned_debug_records)
    stats["records_rejected_total"] += rejected
    stats["records_deduplicated_total"] += deduplicated_total
    stats["complete_pages"] += returned_complete_pages
    stats["partial_pages"] += returned_partial_pages
    stats["failed_pages"] += returned_failed_pages
    if paper_status == "complete":
        stats["papers_complete"] += 1
    elif paper_status == "partial":
        stats["papers_partial"] += 1
    else:
        stats["papers_failed"] += 1
    return _runtime_records_with_status(returned_runtime_records, paper_dir)


def _write_report(path: Path, stats: dict[str, Any]) -> None:
    parsed_page_total = int(stats.get("complete_pages", 0)) + int(stats.get("partial_pages", 0)) + int(stats.get("failed_pages", 0))
    if parsed_page_total > 0:
        stats["average_parser_passes_per_page"] = round(float(stats.get("total_parser_calls", 0)) / parsed_page_total, 3)
    else:
        stats["average_parser_passes_per_page"] = 0
    lines = [
        "# PDF VLM Symbolic VLM Baseline 报告",
        "",
        f"- baseline type: `{BASELINE_TYPE}`",
        f"- run status: `{stats.get('run_status', 'unknown')}`",
        f"- interrupted reason: `{stats.get('interrupted_reason', '')}`",
        f"- processed query 数: {stats.get('processed_queries', 0)}",
        f"- top_k_papers: {stats.get('top_k_papers')}",
        f"- task_family_budget_enabled: {stats.get('task_family_budget_enabled')}",
        f"- single_paper_budget: `{stats.get('single_paper_budget')}`",
        f"- multi_paper_budget: `{stats.get('multi_paper_budget')}`",
        f"- task_family_budget_usage: `{stats.get('task_family_budget_usage', {})}`",
        f"- retrieval method: `{stats.get('retrieval_method')}`",
        f"- retrieval enable topic expansion: {stats.get('retrieval_enable_topic_expansion')}",
        f"- retrieval enable query decomposition: {stats.get('retrieval_enable_query_decomposition')}",
        f"- retrieval subquery top-k: {stats.get('retrieval_subquery_top_k')}",
        f"- retrieval score boost settings: `{stats.get('retrieval_score_boost_settings')}`",
        f"- top-k retrieval candidate coverage: {stats.get('top_k_retrieval_candidate_coverage', 'unavailable')}",
        f"- top_n_records: {stats.get('top_n_records')}",
        f"- top_n_visual_records: {stats.get('top_n_visual_records')}",
        f"- page_routing_enabled: {stats.get('page_routing_enabled')}",
        f"- page_routing_source: `{stats.get('page_routing_source')}`",
        f"- page_routing_method: `{stats.get('page_routing_method')}`",
        f"- page_ranking_bonus_enabled: {stats.get('page_ranking_bonus_enabled')}",
        f"- page_routing_task_family_strategy: {stats.get('page_routing_task_family_strategy')}",
        f"- page_routing_single_strategy: `{stats.get('page_routing_single_strategy')}`",
        f"- page_routing_multi_strategy: `{stats.get('page_routing_multi_strategy')}`",
        f"- page_routing_single_top1_min_pages: {stats.get('page_routing_single_top1_min_pages')}",
        f"- page_routing_strategy_usage: `{stats.get('page_routing_strategy_usage', {})}`",
        f"- single_top1_quota_queries: {stats.get('single_top1_quota_queries', 0)}",
        f"- single_top1_quota_added_pages: {stats.get('single_top1_quota_added_pages', 0)}",
        f"- single_top1_quota_replaced_pages: {stats.get('single_top1_quota_replaced_pages', 0)}",
        f"- multi_candidate_primary_quota_queries: {stats.get('multi_candidate_primary_quota_queries', 0)}",
        f"- multi_candidate_primary_quota_added_pages: {stats.get('multi_candidate_primary_quota_added_pages', 0)}",
        f"- multi_candidate_primary_quota_covered_candidates: {stats.get('multi_candidate_primary_quota_covered_candidates', 0)}",
        f"- selected_pages_from_top1_candidate_count: {stats.get('selected_pages_from_top1_candidate_count', 0)}",
        f"- selected_pages_from_non_top1_candidates_count: {stats.get('selected_pages_from_non_top1_candidates_count', 0)}",
        f"- page_routing_top_pages_per_candidate: {stats.get('page_routing_top_pages_per_candidate', 0)}",
        f"- effective_page_routing_top_pages_per_candidate_distribution: `{stats.get('effective_page_routing_top_pages_per_candidate_distribution', {})}`",
        f"- page_routing_top_pages_global_override: {stats.get('page_routing_top_pages_global_override', 0)}",
        f"- total_candidate_pages_before_routing: {stats.get('total_candidate_pages_before_routing', 0)}",
        f"- global_top_p_pages_initial: {stats.get('global_top_p_pages_initial', 0)}",
        f"- effective_top_k_distribution: `{stats.get('effective_top_k_distribution', {})}`",
        f"- effective_top_p_distribution: `{stats.get('effective_top_p_distribution', {})}`",
        f"- top_p_pages_selected_total: {stats.get('top_p_pages_selected_total', 0)}",
        f"- global_parse_batch_size: {stats.get('global_parse_batch_size', 0)}",
        f"- global_parse_batches: {stats.get('global_parse_batches', 0)}",
        f"- global_max_pages_per_query: {stats.get('global_max_pages_per_query', 0)}",
        f"- global_expansion_step: {stats.get('global_expansion_step', 0)}",
        f"- vlm1_pages_selected_after_global_routing: {stats.get('vlm1_pages_selected_after_global_routing', 0)}",
        f"- vlm1_pages_actually_parsed: {stats.get('vlm1_pages_actually_parsed', 0)}",
        f"- vlm1_pages_saved_by_global_routing: {stats.get('vlm1_pages_saved_by_global_routing', 0)}",
        f"- average_global_selected_pages_per_query: {stats.get('average_global_selected_pages_per_query', 0)}",
        f"- distribution_of_selected_pages_by_candidate_rank: `{stats.get('distribution_of_selected_pages_by_candidate_rank', {})}`",
        f"- distribution_of_selected_pages_by_paper: `{stats.get('distribution_of_selected_pages_by_paper', {})}`",
        f"- queries_where_all_selected_pages_from_one_paper: {stats.get('queries_where_all_selected_pages_from_one_paper', 0)}",
        f"- queries_where_selected_pages_from_multiple_papers: {stats.get('queries_where_selected_pages_from_multiple_papers', 0)}",
        f"- native_text_scanned_papers: {stats.get('native_text_scanned_papers', 0)}",
        f"- native_text_scanned_pages: {stats.get('native_text_scanned_pages', 0)}",
        f"- native_text_empty_papers: {stats.get('native_text_empty_papers', 0)}",
        f"- native_text_empty_pages: {stats.get('native_text_empty_pages', 0)}",
        f"- empty_text_global_fallback_used_count: {stats.get('empty_text_global_fallback_used_count', 0)}",
        f"- progressive_expansion_used_count: {stats.get('progressive_expansion_used_count', 0)}",
        f"- progressive_expanded_pages_count: {stats.get('progressive_expanded_pages_count', 0)}",
        f"- page_level_cache_hits: {stats.get('page_level_cache_hits', 0)}",
        f"- page_level_cache_misses: {stats.get('page_level_cache_misses', 0)}",
        f"- PDF cache 路径: `{stats.get('pdf_output_dir')}`",
        f"- existing PDFs 数: {stats.get('existing_pdfs', 0)}",
        f"- newly downloaded PDFs 数: {stats.get('newly_downloaded_pdfs', 0)}",
        f"- failed PDF downloads 数: {stats.get('failed_pdf_downloads', 0)}",
        f"- skipped OpenReview papers 数: {stats.get('skipped_openreview_papers', 0)}",
        f"- OpenReview policy: `{stats.get('pdf_openreview_policy')}`",
        f"- proceedings candidate attempts: {stats.get('proceedings_candidate_attempts', 0)}",
        f"- proceedings match success count: {stats.get('proceedings_match_success_count', 0)}",
        f"- direct OpenReview skipped count: {stats.get('direct_openreview_skipped_count', 0)}",
        f"- direct OpenReview attempted count: {stats.get('direct_openreview_attempted_count', 0)}",
        f"- PDF source distribution: `{stats.get('pdf_source_distribution', {})}`",
        f"- rendered papers 数: {stats.get('rendered_papers', 0)}",
        f"- rendered pages 数: {stats.get('rendered_pages', 0)}",
        f"- parser model: `{stats.get('parser_model')}`",
        f"- parser extraction mode: `{stats.get('parser_extraction_mode')}`",
        f"- symbolic artifact version: `{stats.get('symbolic_artifact_version')}`",
        f"- structured cache policy: `{stats.get('structured_cache_policy')}`",
        f"- symbolic_cache_root_override_enabled: {stats.get('symbolic_cache_root_override_enabled')}",
        f"- symbolic_cache_root_requested: `{stats.get('symbolic_cache_root_requested', '')}`",
        f"- symbolic_cache_structured_root: `{stats.get('symbolic_cache_structured_root', '')}`",
        f"- symbolic_cache_records_standardized: {stats.get('symbolic_cache_records_standardized', 0)}",
        f"- parser max tokens: {stats.get('parser_max_tokens')}",
        f"- max records per call: {stats.get('parser_max_records_per_call')}",
        f"- max continuations per page: {stats.get('parser_max_continuations_per_page')}",
        f"- parser VLM image input capability: {stats.get('parser_supports_images')}",
        f"- parser API calls: {stats.get('parser_api_calls', 0)}",
        f"- total parser calls: {stats.get('total_parser_calls', 0)}",
        f"- average parser passes per page: {stats.get('average_parser_passes_per_page', 0)}",
        f"- reused structured papers: {stats.get('reused_structured_papers', 0)}",
        f"- reused complete artifacts: {stats.get('reused_complete_artifacts', 0)}",
        f"- reused partial artifacts: {stats.get('reused_partial_artifacts', 0)}",
        f"- newly parsed papers: {stats.get('newly_parsed_papers', 0)}",
        f"- failed parsed papers: {stats.get('failed_parsed_papers', 0)}",
        f"- parser JSON/page failures: {stats.get('parser_json_failures', 0)}",
        f"- parser truncated JSON/page failures: {stats.get('parser_truncated_json_failures', 0)}",
        f"- skipped pages: {stats.get('skipped_pages', 0)}",
        f"- parser API calls saved by freeze mechanism: {stats.get('parser_api_calls_saved', 0)}",
        f"- complete pages: {stats.get('complete_pages', 0)}",
        f"- partial pages: {stats.get('partial_pages', 0)}",
        f"- failed pages: {stats.get('failed_pages', 0)}",
        f"- pages needing continuation: {stats.get('pages_needing_continuation', 0)}",
        f"- records produced total: {stats.get('records_produced_total', 0)}",
        f"- records accepted total: {stats.get('records_accepted_total', 0)}",
        f"- records rejected total: {stats.get('records_rejected_total', 0)}",
        f"- records deduplicated total: {stats.get('records_deduplicated_total', 0)}",
        f"- summary_field_removed: {stats.get('summary_field_removed')}",
        f"- runtime_record_field_count: {stats.get('runtime_record_field_count', 0)}",
        f"- debug_record_field_count: {stats.get('debug_record_field_count', 0)}",
        f"- runtime_records_total: {stats.get('runtime_records_total', 0)}",
        f"- debug_records_total: {stats.get('debug_records_total', 0)}",
        f"- header_footer_filtered_count: {stats.get('header_footer_filtered_count', 0)}",
        f"- old_artifacts_migrated_count: {stats.get('old_artifacts_migrated_count', 0)}",
        f"- old_artifacts_rejected_as_cache_count: {stats.get('old_artifacts_rejected_as_cache_count', 0)}",
        f"- papers complete: {stats.get('papers_complete', 0)}",
        f"- papers partial: {stats.get('papers_partial', 0)}",
        f"- papers failed: {stats.get('papers_failed', 0)}",
        f"- valid symbolic records 数: {stats.get('valid_symbolic_records', 0)}",
        f"- rejected symbolic records 数: {stats.get('rejected_symbolic_records', 0)}",
        "- symbolic selection method: `symbolic_lexical_bm25_without_embedding`",
        f"- answer model: `{stats.get('answer_model')}`",
        f"- metadata-only retrieval eval model: `{stats.get('metadata_only_retrieval_eval_model')}`",
        f"- answer model image input capability: {stats.get('answer_supports_images')}",
        f"- metadata-only eval freeze: {stats.get('metadata_only_eval_freeze')}",
        f"- metadata-only eval freeze mode: `{stats.get('metadata_only_eval_freeze_mode')}`",
        f"- vlm2 metadata selection calls: {stats.get('vlm2_metadata_selection_calls', 0)}",
        f"- vlm2 metadata selection failures: {stats.get('vlm2_metadata_selection_failures', 0)}",
        f"- vlm2 metadata no-valid-selection count: {stats.get('vlm2_metadata_no_valid_selection_count', 0)}",
        f"- vlm2_context_mode: `{stats.get('vlm2_context_mode')}`",
        f"- vlm2_effective_context_mode: `{stats.get('vlm2_effective_context_mode')}`",
        f"- vlm2_context_selection_mode: `{stats.get('vlm2_context_selection_mode')}`",
        f"- vlm2_max_context_records: {stats.get('vlm2_max_context_records', 0)}",
        f"- vlm2_max_context_chars: {stats.get('vlm2_max_context_chars', 0)}",
        f"- vlm2_context_truncated_count: {stats.get('vlm2_context_truncated_count', 0)}",
        f"- average_vlm2_selected_records_per_query: {stats.get('average_vlm2_selected_records_per_query', 0)}",
        f"- vlm2_context_mode_downgraded: {stats.get('vlm2_context_mode_downgraded')}",
        f"- vlm2_context_mode_downgrade_reason: `{stats.get('vlm2_context_mode_downgrade_reason')}`",
        f"- vlm2_prompt_context_fields: `{stats.get('vlm2_prompt_context_fields')}`",
        f"- vlm2_debug_context_fields: `{stats.get('vlm2_debug_context_fields')}`",
        f"- fields_removed_from_vlm2_prompt: `{stats.get('fields_removed_from_vlm2_prompt')}`",
        f"- average_prompt_evidence_fields_per_record: {stats.get('average_prompt_evidence_fields_per_record', 0)}",
        f"- average_prompt_evidence_chars_per_record: {stats.get('average_prompt_evidence_chars_per_record', 0)}",
        f"- vlm2_include_parse_confidence: {stats.get('vlm2_include_parse_confidence')}",
        f"- vlm2_attached_image_count: {stats.get('vlm2_attached_image_count', 0)}",
        f"- vlm2_crop_failures: {stats.get('vlm2_crop_failures', 0)}",
        f"- answer_contract_enabled: {stats.get('answer_contract_enabled')}",
        f"- multiple_choice_queries_count: {stats.get('multiple_choice_queries_count', 0)}",
        f"- multiple_choice_options_available_count: {stats.get('multiple_choice_options_available_count', 0)}",
        f"- multiple_choice_options_missing_count: {stats.get('multiple_choice_options_missing_count', 0)}",
        f"- invalid_multiple_choice_outputs_count: {stats.get('invalid_multiple_choice_outputs_count', 0)}",
        f"- answer_type_extra_fields_removed_count: {stats.get('answer_type_extra_fields_removed_count', 0)}",
        f"- answer_type_missing_required_count: {stats.get('answer_type_missing_required_count', 0)}",
        f"- official_source_type_constraint_enabled: {stats.get('official_source_type_constraint_enabled')}",
        f"- selected_evidence_source_type_distribution: `{stats.get('selected_evidence_source_type_distribution', {})}`",
        f"- primary_evidence_selected_count: {stats.get('primary_evidence_selected_count', 0)}",
        f"- supporting_evidence_selected_count: {stats.get('supporting_evidence_selected_count', 0)}",
        f"- grounding_label_hints_enabled: {stats.get('grounding_label_hints_enabled')}",
        f"- grounding_label_hints_by_type: `{stats.get('grounding_label_hints_by_type', {})}`",
        f"- symbolic_evidence_standardization: {stats.get('symbolic_evidence_standardization')}",
        f"- symbolic_evidence_locator_standardized_count: {stats.get('symbolic_evidence_locator_standardized_count', 0)}",
        f"- symbolic_evidence_empty_filled_count: {stats.get('symbolic_evidence_empty_filled_count', 0)}",
        f"- symbolic_evidence_non_object_replaced_count: {stats.get('symbolic_evidence_non_object_replaced_count', 0)}",
        f"- symbolic_evidence_standardization_no_match_count: {stats.get('symbolic_evidence_standardization_no_match_count', 0)}",
        f"- symbolic_evidence_standardization_no_locator_count: {stats.get('symbolic_evidence_standardization_no_locator_count', 0)}",
        f"- invented_grounding_labels_removed_count: {stats.get('invented_grounding_labels_removed_count', 0)}",
        f"- locator_validation_errors_count: {stats.get('locator_validation_errors_count', 0)}",
        f"- answer API calls: {stats.get('answer_api_calls', 0)}",
        f"- partial artifacts used in answer generation: {stats.get('partial_artifacts_used_in_answer_generation', 0)}",
        f"- successful predictions: {stats.get('successful_predictions', 0)}",
        f"- parse failures: {stats.get('parse_failures', 0)}",
        f"- fallback predictions: {stats.get('fallback_predictions', 0)}",
        f"- skipped predictions: {stats.get('skipped_predictions', 0)}",
        f"- predictions 路径: `{stats.get('predictions_path')}`",
        f"- internal_predictions 路径: `{stats.get('internal_predictions_path')}`",
        f"- symbolic_records.runtime 路径: `{stats.get('symbolic_records_runtime_path')}`",
        f"- symbolic_records.debug 路径: `{stats.get('symbolic_records_debug_path')}`",
        f"- selected_symbolic_contexts.debug 路径: `{stats.get('selected_contexts_debug_path')}`",
        f"- selected_symbolic_contexts.prompt 路径: `{stats.get('selected_contexts_prompt_path')}`",
        f"- processed symbolic store 路径: `{stats.get('processed_symbolic_store_path', '')}`",
        f"- processed symbolic store runtime records: {stats.get('processed_symbolic_store_runtime_records', 0)}",
        f"- processed symbolic store debug records: {stats.get('processed_symbolic_store_debug_records', 0)}",
        f"- processed symbolic store sync error: `{stats.get('processed_symbolic_store_sync_error', '')}`",
        f"- evaluator 命令: `python -m pdf_vlm_symbolic_vlm_baseline.evaluate_local --official-dir official_dev --pred {stats.get('predictions_path')}`",
        "",
        "## VLM-2 约束说明",
        "",
        "- VLM-2 receives multiple-choice options from sanitized validation answer constraints when available.",
        "- VLM-2 does not receive gold answers, gold evidence, or gold paper ids.",
        "- VLM-2 selected evidence is constrained to official source types.",
        "- VLM-2 receives both primary evidence type records and supporting context records.",
        "- Grounding labels are passed only when extracted from visible symbolic records or deterministic native text, not invented.",
        "",
        "## 当前落盘 JSONL 行数",
        "",
        f"- candidate_papers rows: {stats.get('candidate_papers_rows', 0)}",
        f"- pdf_availability rows: {stats.get('pdf_availability_rows', 0)}",
        f"- native_page_text rows: {stats.get('native_page_text_rows', 0)}",
        f"- global_page_pool rows: {stats.get('global_page_pool_rows', 0)}",
        f"- global_page_ranking rows: {stats.get('global_page_ranking_rows', 0)}",
        f"- global_page_parse_plan rows: {stats.get('global_page_parse_plan_rows', 0)}",
        f"- page_rendering_artifacts rows: {stats.get('page_rendering_rows', 0)}",
        f"- parser_artifacts rows: {stats.get('parser_artifacts_rows', 0)}",
        f"- raw_vlm_parser_responses rows: {stats.get('raw_parser_rows', 0)}",
        f"- symbolic_records.runtime rows: {stats.get('symbolic_records_runtime_rows', 0)}",
        f"- symbolic_records.debug rows: {stats.get('symbolic_records_debug_rows', 0)}",
        f"- selected_symbolic_contexts.debug rows: {stats.get('selected_contexts_debug_rows', 0)}",
        f"- selected_symbolic_contexts.prompt rows: {stats.get('selected_contexts_prompt_rows', 0)}",
        f"- raw_vlm_answer_responses rows: {stats.get('raw_answer_rows', 0)}",
        f"- internal_predictions rows: {stats.get('internal_predictions_rows', 0)}",
        f"- predictions rows: {stats.get('predictions_rows', 0)}",
        f"- errors rows: {stats.get('errors_rows', 0)}",
        "",
        "## 关键限制",
        "",
        "- 当前 baseline 不使用 OCR。",
        "- 当前可通过 `--skip-openreview-papers` 在 metadata retrieval 阶段跳过 OpenReview papers，避免爬虫访问 OpenReview。",
        "- 当前 baseline 不让 answer model 原生读取 PDF。",
        "- 第一层 VLM 是 document-to-symbol converter。",
        "- 中间 symbolic system 负责 schema validation、locator echo、artifact indexing 和 provenance。",
        "- 第二层 VLM 只基于 selected symbolic contexts 和可选图像生成答案。",
        "- VLM-1 main transcription does not generate bbox in the evaluator-grounded minimal schema.",
        "- Figure bbox is generated only by the figure-localization pass for crop creation and debug audit; runtime symbolic records keep crop paths, not bbox.",
        "- processed_pdfs/vlm_symbolic_runs is the durable symbolic store; outputs/ is run-level audit and prediction output.",
        "- 当前 symbolic system 是最小 JSONL-based symbolic layer，不是完整 KG。",
        "- context selection 使用 lexical / BM25 over symbolic records，不使用 embedding。",
        "- metadata top-k retrieval 的 recall 限制整个 pipeline 上限。",
        "- 本 baseline 不追求 VLM-1 的效率最优，而优先追求 page-level content coverage 和 symbolic auditability。",
        "- 系统不采用无限长单次 JSON 输出，而采用 bounded-call continuation 机制。",
        "- VLM-1 no longer generates summaries.",
        "- VLM-1 no longer generates system-derived fields.",
        "- Runtime records are text-first minimal artifacts with system-derived locators.",
        "- Debug records preserve validation and provenance details.",
        "- VLM-2 receives compact answer-facing evidence records.",
        "- Audit/debug fields are retained outside the prompt.",
        "- Selector score is not passed to VLM-2.",
        "- Retrieval scores, page ranking scores, selector scores, and parser confidence values are audit-only and are not passed to VLM-1 or VLM-2.",
        "- Local image paths are not passed to VLM-2.",
        "- bbox_1000 is not passed to VLM-2 prompt; official grounding uses paper_id, source_type, page, and table_id/figure_id when applicable.",
        "- Page routing is global across all top-k candidate PDFs for each query.",
        "- Global page ranking uses native PyMuPDF text extraction only.",
        "- No VLM-generated summaries, inventories, or key terms are used for page ranking.",
        "- For multi-paper routing, the configured strategy may reserve one primary-evidence page per candidate before global-rank filling.",
        "- For single-paper routing, the configured strategy may prioritize the top-ranked candidate paper.",
        "- VLM-1 is called only on selected pages unless fallback mode is triggered.",
        "- VLM-2 still receives only selected symbolic evidence and precomputed crop images when enabled, never full page images.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    interrupted_signal: dict[str, str] = {}

    def _handle_signal(signum: int, _frame: Any) -> None:
        interrupted_signal["reason"] = signal.Signals(signum).name
        raise KeyboardInterrupt(interrupted_signal["reason"])

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    config = load_pipeline_config(args.env_path)
    output_dir = Path(args.output_dir)
    paths = _paths(output_dir, args.resume, args.dry_run, args.skip_generation)
    processed_root = Path(args.processed_output_dir)
    slug = model_slug(config.parser_model)
    symbolic_cache_root_value = args.symbolic_cache_root if args.symbolic_cache_root is not None else config.symbolic_cache_root
    structured_root, symbolic_cache_root_override_enabled, symbolic_cache_root_requested = _resolve_structured_root(
        processed_root,
        slug,
        symbolic_cache_root_value,
    )
    if args.clear_structured_cache and structured_root.exists():
        shutil.rmtree(structured_root)
    render_dpi = args.render_dpi or config.render_dpi
    render_format = args.render_format or config.render_format
    max_pages = args.max_pages_per_paper if args.max_pages_per_paper is not None else config.render_max_pages_per_paper
    cache_policy = args.structured_cache_policy or config.structured_cache_policy
    if cache_policy == "reuse":
        cache_policy = "reuse_complete_only"
    if cache_policy == "fail-if-missing":
        cache_policy = "fail_if_missing"
    page_routing_enabled = config.page_routing_enabled if args.page_routing_enabled is None else bool(args.page_routing_enabled)
    explicit_top_pages_global = args.page_routing_top_pages_global or config.page_routing_top_pages_global
    page_routing_top_pages_global = explicit_top_pages_global or max(1, config.page_routing_top_pages_per_candidate * args.top_k_papers)
    page_routing_max_pages_global = args.page_routing_max_pages_global or config.page_routing_max_pages_global
    page_routing_parse_batch_size = max(1, int(args.page_routing_parse_batch_size or config.page_routing_parse_batch_size or page_routing_max_pages_global or 1))
    page_routing_expansion_step_global = args.page_routing_expansion_step_global or config.page_routing_expansion_step_global
    requested_vlm2_context_mode = args.vlm2_context_mode or config.vlm2_context_mode
    include_parse_confidence = False
    fields_removed_from_vlm2_prompt = ["global_record_id", "record_type", "record_id", "locator", "image_path", "score", "bbox_1000", "vlm_parse_confidence"]

    inputs = read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    validation_contract_rows = read_jsonl(find_official_file(args.official_dir, "validation.jsonl"))
    validation_contract_by_query_id = {
        str(row.get("query_id") or ""): row
        for row in validation_contract_rows
        if row.get("query_id")
    }
    if args.max_queries is not None:
        inputs = inputs[: args.max_queries]
    metadata = build_metadata_records(read_jsonl(find_official_file(args.official_dir, "paper_metadata.jsonl")))
    skipped_openreview_count = 0
    if args.skip_openreview_papers:
        metadata, skipped_openreview = filter_openreview_metadata(metadata)
        skipped_openreview_count = len(skipped_openreview)
    metadata_by_id = {str(r.get("paper_id")): r for r in metadata}
    parser = VLMParserClient(config)
    answer_client = VLMAnswerClient(config)
    metadata_selection_client = VLMAnswerClient(replace(config, answer_model=config.metadata_only_retrieval_eval_model))
    stats: dict[str, Any] = {
        "processed_queries": 0,
        "run_status": "running",
        "interrupted_reason": "",
        "top_k_papers": args.top_k_papers,
        "task_family_budget_enabled": config.task_family_budget_enabled,
        "single_paper_budget": {
            "top_k_papers": config.single_paper_top_k_papers,
            "page_routing_top_pages_per_candidate": config.single_paper_page_routing_top_pages_per_candidate,
        },
        "multi_paper_budget": {
            "top_k_papers": config.multi_paper_top_k_papers,
            "page_routing_top_pages_per_candidate": config.multi_paper_page_routing_top_pages_per_candidate,
        },
        "task_family_budget_usage": {},
        "effective_top_k_distribution": {},
        "effective_top_p_distribution": {},
        "effective_page_routing_top_pages_per_candidate_distribution": {},
        "retrieval_method": config.retrieval_method,
        "retrieval_enable_topic_expansion": config.retrieval_enable_topic_expansion,
        "retrieval_enable_query_decomposition": config.retrieval_enable_query_decomposition,
        "retrieval_subquery_top_k": config.retrieval_subquery_top_k,
        "retrieval_score_boost_settings": HYBRID_SCORE_WEIGHTS if config.retrieval_method == "hybrid_alias" else {},
        "top_k_retrieval_candidate_coverage": "unavailable",
        "top_k_retrieval_coverage_hits": 0,
        "top_k_retrieval_coverage_denominator": 0,
        "top_n_records": args.top_n_records,
        "top_n_visual_records": args.top_n_visual_records,
        "page_routing_enabled": page_routing_enabled,
        "page_routing_source": config.page_routing_source,
        "page_routing_method": config.page_routing_method,
        "page_ranking_bonus_enabled": config.page_ranking_bonus_enabled,
        "page_routing_task_family_strategy": config.page_routing_task_family_strategy,
        "page_routing_single_strategy": config.page_routing_single_strategy,
        "page_routing_multi_strategy": config.page_routing_multi_strategy,
        "page_routing_single_top1_min_pages": config.page_routing_single_top1_min_pages,
        "page_routing_strategy_usage": {},
        "single_top1_quota_queries": 0,
        "single_top1_quota_added_pages": 0,
        "single_top1_quota_replaced_pages": 0,
        "multi_candidate_primary_quota_queries": 0,
        "multi_candidate_primary_quota_added_pages": 0,
        "multi_candidate_primary_quota_covered_candidates": 0,
        "selected_pages_from_top1_candidate_count": 0,
        "selected_pages_from_non_top1_candidates_count": 0,
        "page_routing_top_pages_per_candidate": config.page_routing_top_pages_per_candidate,
        "page_routing_top_pages_global_override": explicit_top_pages_global,
        "total_candidate_pages_before_routing": 0,
        "global_top_p_pages_initial": page_routing_top_pages_global,
        "global_max_pages_per_query": page_routing_max_pages_global,
        "global_parse_batch_size": page_routing_parse_batch_size,
        "global_parse_batches": 0,
        "top_p_pages_selected_total": 0,
        "global_expansion_step": page_routing_expansion_step_global,
        "vlm1_pages_selected_after_global_routing": 0,
        "vlm1_pages_actually_parsed": 0,
        "vlm1_pages_saved_by_global_routing": 0,
        "average_global_selected_pages_per_query": 0,
        "distribution_of_selected_pages_by_candidate_rank": {},
        "distribution_of_selected_pages_by_paper": {},
        "queries_where_all_selected_pages_from_one_paper": 0,
        "queries_where_selected_pages_from_multiple_papers": 0,
        "native_text_scanned_papers": 0,
        "native_text_scanned_pages": 0,
        "native_text_empty_papers": 0,
        "native_text_empty_pages": 0,
        "empty_text_global_fallback_used_count": 0,
        "progressive_expansion_used_count": 0,
        "progressive_expanded_pages_count": 0,
        "page_level_cache_hits": 0,
        "page_level_cache_misses": 0,
        "pdf_output_dir": args.pdf_output_dir,
        "existing_pdfs": 0,
        "newly_downloaded_pdfs": 0,
        "failed_pdf_downloads": 0,
        "skipped_openreview_papers": skipped_openreview_count,
        "pdf_openreview_policy": config.pdf_openreview_policy,
        "proceedings_candidate_attempts": 0,
        "proceedings_match_success_count": 0,
        "direct_openreview_skipped_count": 0,
        "direct_openreview_attempted_count": 0,
        "pdf_source_distribution": {},
        "rendered_papers": 0,
        "rendered_pages": 0,
        "parser_model": config.parser_model,
        "parser_extraction_mode": config.parser_extraction_mode,
        "symbolic_artifact_version": config.symbolic_artifact_version,
        "structured_cache_policy": cache_policy,
        "symbolic_cache_root_override_enabled": symbolic_cache_root_override_enabled,
        "symbolic_cache_root_requested": symbolic_cache_root_requested,
        "symbolic_cache_structured_root": str(structured_root),
        "symbolic_cache_records_standardized": 0,
        "parser_max_tokens": config.parser_max_tokens,
        "parser_max_records_per_call": config.parser_max_records_per_call,
        "parser_max_continuations_per_page": config.parser_max_continuations_per_page,
        "parser_supports_images": parser.supports_image_input(),
        "parser_api_calls": 0,
        "total_parser_calls": 0,
        "reused_structured_papers": 0,
        "reused_complete_artifacts": 0,
        "reused_partial_artifacts": 0,
        "newly_parsed_papers": 0,
        "failed_parsed_papers": 0,
        "parser_json_failures": 0,
        "parser_truncated_json_failures": 0,
        "max_parser_json_failures": args.max_parser_json_failures,
        "skipped_pages": 0,
        "parser_api_calls_saved": 0,
        "valid_symbolic_records": 0,
        "rejected_symbolic_records": 0,
        "complete_pages": 0,
        "partial_pages": 0,
        "failed_pages": 0,
        "pages_needing_continuation": 0,
        "records_produced_total": 0,
        "records_accepted_total": 0,
        "records_rejected_total": 0,
        "records_deduplicated_total": 0,
        "summary_field_removed": True,
        "runtime_record_field_count": 10,
        "debug_record_field_count": 18,
        "runtime_records_total": 0,
        "debug_records_total": 0,
        "header_footer_filtered_count": 0,
        "old_artifacts_migrated_count": 0,
        "old_artifacts_rejected_as_cache_count": 0,
        "papers_complete": 0,
        "papers_partial": 0,
        "papers_failed": 0,
        "partial_artifacts_used_in_answer_generation": 0,
        "answer_model": config.answer_model,
        "metadata_only_retrieval_eval_model": config.metadata_only_retrieval_eval_model,
        "answer_supports_images": answer_client.supports_image_input(),
        "metadata_only_eval_freeze": args.metadata_only_eval_freeze,
        "metadata_only_eval_freeze_mode": "vlm2_metadata_paper_selection" if args.metadata_only_eval_freeze else "",
        "vlm2_metadata_selection_calls": 0,
        "vlm2_metadata_selection_failures": 0,
        "vlm2_metadata_no_valid_selection_count": 0,
        "vlm2_context_mode": requested_vlm2_context_mode,
        "vlm2_effective_context_mode": _effective_vlm2_context_mode(requested_vlm2_context_mode, answer_client.supports_image_input(), {}),
        "vlm2_context_selection_mode": config.vlm2_context_selection_mode,
        "vlm2_max_context_records": config.vlm2_max_context_records,
        "vlm2_max_context_chars": config.vlm2_max_context_chars,
        "vlm2_context_truncated_count": 0,
        "average_vlm2_selected_records_per_query": 0,
        "vlm2_context_mode_downgraded": False,
        "vlm2_context_mode_downgrade_reason": "",
        "vlm2_include_parse_confidence": include_parse_confidence,
        "vlm2_prompt_context_fields": [],
        "vlm2_debug_context_fields": [],
        "fields_removed_from_vlm2_prompt": fields_removed_from_vlm2_prompt,
        "average_prompt_evidence_fields_per_record": 0,
        "average_prompt_evidence_chars_per_record": 0,
        "vlm2_prompt_evidence_record_count": 0,
        "vlm2_prompt_evidence_field_total": 0,
        "vlm2_prompt_evidence_char_total": 0,
        "vlm2_attached_image_count": 0,
        "vlm2_crop_failures": 0,
        "answer_contract_enabled": True,
        "multiple_choice_queries_count": 0,
        "multiple_choice_options_available_count": 0,
        "multiple_choice_options_missing_count": 0,
        "invalid_multiple_choice_outputs_count": 0,
        "answer_type_extra_fields_removed_count": 0,
        "answer_type_missing_required_count": 0,
        "official_source_type_constraint_enabled": True,
        "selected_evidence_source_type_distribution": {},
        "primary_evidence_selected_count": 0,
        "supporting_evidence_selected_count": 0,
        "grounding_label_hints_enabled": True,
        "grounding_label_hints_by_type": {},
        "symbolic_evidence_standardization": config.symbolic_evidence_standardization,
        "symbolic_evidence_locator_standardized_count": 0,
        "symbolic_evidence_empty_filled_count": 0,
        "symbolic_evidence_non_object_replaced_count": 0,
        "symbolic_evidence_standardization_no_match_count": 0,
        "symbolic_evidence_standardization_no_locator_count": 0,
        "invented_grounding_labels_removed_count": 0,
        "locator_validation_errors_count": 0,
        "answer_api_calls": 0,
        "successful_predictions": 0,
        "parse_failures": 0,
        "fallback_predictions": 0,
        "skipped_predictions": 0,
        "predictions_path": str(paths["predictions"]),
        "internal_predictions_path": str(paths["internal_predictions"]),
        "symbolic_records_runtime_path": str(paths["symbolic_records_runtime"]),
        "symbolic_records_debug_path": str(paths["symbolic_records_debug"]),
        "symbolic_records_path": str(paths["symbolic_records_runtime"]),
        "selected_contexts_debug_path": str(paths["selected_contexts_debug"]),
        "selected_contexts_prompt_path": str(paths["selected_contexts_prompt"]),
        "selected_contexts_path": str(paths["selected_contexts_prompt"]),
    }
    effective_vlm2_context_mode = _effective_vlm2_context_mode(requested_vlm2_context_mode, answer_client.supports_image_input(), stats)
    stats["vlm2_effective_context_mode"] = effective_vlm2_context_mode

    paper_records_cache: dict[str, list[dict[str, Any]]] = {}
    completed_query_ids: set[str] = set()
    if args.resume and paths["predictions"].exists():
        for row in read_jsonl(paths["predictions"]):
            query_id_value = str(row.get("query_id") or "")
            if query_id_value:
                completed_query_ids.add(query_id_value)
        stats["resume_completed_query_count"] = len(completed_query_ids)
    exit_code = 0
    try:
        iterator = inputs
        if args.show_progress and tqdm is not None:
            iterator = tqdm(inputs, desc="queries", unit="query")  # type: ignore[assignment]
        for sample in iterator:
            query_id = str(sample.get("query_id", ""))
            if query_id in completed_query_ids:
                stats["resume_skipped_completed_queries"] = int(stats.get("resume_skipped_completed_queries") or 0) + 1
                continue
            answer_contract = extract_answer_contract(sample, validation_contract_by_query_id.get(query_id))
            _update_answer_contract_stats(answer_contract, stats, query_id, paths)
            budget = _query_budget(sample, config, args.top_k_papers)
            effective_top_k = int(budget["top_k_papers"])
            effective_pages_per_candidate = int(budget["page_routing_top_pages_per_candidate"])
            _increment_counter_stat(stats, "task_family_budget_usage", budget["task_family_bucket"])
            _increment_counter_stat(stats, "effective_top_k_distribution", effective_top_k)
            _increment_counter_stat(stats, "effective_page_routing_top_pages_per_candidate_distribution", effective_pages_per_candidate)
            candidates = retrieve_candidates(
                str(sample.get("question", "")),
                metadata,
                effective_top_k,
                method=config.retrieval_method,
                enable_query_decomposition=config.retrieval_enable_query_decomposition and budget["task_family_bucket"] == "multi_paper",
                subquery_top_k=config.retrieval_subquery_top_k,
            )
            topic_info = None
            if config.retrieval_enable_topic_expansion:
                candidates, topic_info = expand_candidates_with_topic_profiles(sample, candidates, metadata, effective_top_k)
            metadata_selection_candidates, metadata_selection_policy = selection_candidates_for_metadata_vlm(candidates)
            query_top_pages_global = explicit_top_pages_global or max(1, effective_pages_per_candidate * len(candidates))
            query_max_pages_global = max(page_routing_max_pages_global, query_top_pages_global)
            _increment_counter_stat(stats, "effective_top_p_distribution", query_top_pages_global)
            _update_retrieval_coverage(sample, candidates, stats)
            candidate_score_by_id = {str(candidate.get("paper_id")): float(candidate.get("bm25_score") or candidate.get("score") or 0.0) for candidate in candidates}
            append_jsonl(
                paths["candidate_papers"],
                {
                    "query_id": query_id,
                    "question": sample.get("question"),
                    "task_family": budget["task_family"],
                    "task_family_bucket": budget["task_family_bucket"],
                    "effective_top_k_papers": effective_top_k,
                    "effective_page_routing_top_pages_per_candidate": effective_pages_per_candidate,
                    "effective_top_p_pages": query_top_pages_global,
                    "topic_expansion": topic_info,
                    "metadata_selection_policy": metadata_selection_policy,
                    "metadata_selection_candidates": metadata_selection_candidates,
                    "candidates": candidates,
                },
            )
            if args.metadata_only_eval_freeze:
                failure_count_before = int(stats.get("vlm2_metadata_selection_failures") or 0)
                prediction = _metadata_only_vlm_prediction(
                    sample,
                    metadata_selection_candidates,
                    metadata_selection_client,
                    paths,
                    stats,
                    metadata_selection_policy,
                )
                append_jsonl(paths["predictions"], prediction)
                stats["processed_queries"] += 1
                if int(stats.get("vlm2_metadata_selection_failures") or 0) == failure_count_before:
                    stats["successful_predictions"] += 1
                _write_report_from_paths(paths, stats)
                continue
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
            stats["existing_pdfs"] += availability["existing_count"]
            stats["newly_downloaded_pdfs"] += availability["newly_downloaded_count"]
            stats["failed_pdf_downloads"] += availability["failed_count"]
            stats["proceedings_candidate_attempts"] += int(availability.get("proceedings_candidate_attempts") or 0)
            stats["proceedings_match_success_count"] += int(availability.get("proceedings_match_success_count") or 0)
            stats["direct_openreview_skipped_count"] += int(availability.get("direct_openreview_skipped_count") or 0)
            stats["direct_openreview_attempted_count"] += int(availability.get("direct_openreview_attempted_count") or 0)
            merged_distribution = dict(stats.get("pdf_source_distribution") or {})
            for source_type, count in (availability.get("source_distribution") or {}).items():
                merged_distribution[str(source_type)] = int(merged_distribution.get(str(source_type), 0)) + int(count)
            stats["pdf_source_distribution"] = merged_distribution
            available_rows = [row for row in availability["rows"] if row.get("available")]
            candidate_records: list[dict[str, Any]] = []
            routing_result: dict[str, Any] | None = None
            parsed_global_pages: list[dict[str, Any]] = []
            if page_routing_enabled:
                paper_page_texts: dict[str, list[dict[str, Any]]] = {}
                for row in available_rows:
                    paper_id = str(row.get("paper_id") or "")
                    paper_dir = structured_root / paper_id
                    native_dir = paper_dir / "native_text"
                    result = extract_native_page_text(
                        paper_id,
                        row["local_path"],
                        native_dir,
                        overwrite=False,
                        extraction_method=config.pdf_native_text_extraction_method,
                        min_chars_per_page=config.pdf_native_text_min_chars_per_page,
                    )
                    rows = list(result.get("rows") or [])
                    paper_page_texts[paper_id] = rows
                    status = result.get("status") or {}
                    stats["native_text_scanned_papers"] += 1
                    stats["native_text_scanned_pages"] += len(rows)
                    stats["native_text_empty_pages"] += int(status.get("empty_pages") or 0)
                    if not status.get("native_text_available"):
                        stats["native_text_empty_papers"] += 1
                    candidate = next((c for c in candidates if str(c.get("paper_id")) == paper_id), {"paper_id": paper_id})
                    for text_row in rows:
                        append_jsonl(
                            paths["native_page_text"],
                            {
                                "query_id": query_id,
                                "candidate_rank": candidate.get("rank"),
                                "candidate_score": candidate.get("score", candidate.get("bm25_score")),
                                "paper_id": paper_id,
                                "page": text_row.get("page"),
                                "native_text": text_row.get("text", ""),
                                "native_text_char_count": text_row.get("char_count", 0),
                                "has_native_text": text_row.get("has_native_text", False),
                                "extraction_method": text_row.get("extraction_method", config.pdf_native_text_extraction_method),
                                "paper_title": candidate.get("title", ""),
                                "paper_venue": candidate.get("venue", ""),
                                "paper_year": candidate.get("year", ""),
                            },
                        )
                pool = build_global_page_pool(candidates, paper_page_texts, query_id)
                for pool_row in pool:
                    append_jsonl(paths["global_page_pool"], pool_row)
                routing_result = rank_global_pages_for_query(
                    sample,
                    candidates,
                    paper_page_texts,
                    query_top_pages_global,
                    query_max_pages_global,
                    fallback_on_empty_text=config.page_routing_fallback_on_empty_text,
                    empty_text_parse_first_n_pages=config.page_routing_empty_text_global_parse_first_n_pages,
                    page_ranking_bonus_enabled=config.page_ranking_bonus_enabled,
                    task_family_strategy_enabled=config.page_routing_task_family_strategy,
                    single_strategy=config.page_routing_single_strategy,
                    multi_strategy=config.page_routing_multi_strategy,
                    single_top1_min_pages=config.page_routing_single_top1_min_pages,
                )
                stats["total_candidate_pages_before_routing"] += int(routing_result.get("total_candidate_pages") or 0)
                _increment_counter_stat(stats, "page_routing_strategy_usage", routing_result.get("page_routing_strategy") or "unknown")
                stats["single_top1_quota_added_pages"] += int(routing_result.get("top1_quota_added_pages") or 0)
                stats["single_top1_quota_replaced_pages"] += int(routing_result.get("top1_quota_replaced_pages") or 0)
                if str(routing_result.get("page_routing_strategy") or "") == "top1_candidate_quota":
                    stats["single_top1_quota_queries"] += 1
                if routing_result.get("multi_candidate_primary_quota_enabled"):
                    stats["multi_candidate_primary_quota_queries"] += 1
                    stats["multi_candidate_primary_quota_added_pages"] += int(routing_result.get("multi_candidate_primary_quota_added_pages") or 0)
                    stats["multi_candidate_primary_quota_covered_candidates"] += int(routing_result.get("multi_candidate_primary_quota_covered_count") or 0)
                if routing_result.get("fallback_reason") == "all_native_text_empty_global_fallback":
                    stats["empty_text_global_fallback_used_count"] += 1
                append_jsonl(paths["global_page_ranking"], routing_result)
                initial_pages = list(routing_result.get("selected_pages_initial") or [])
                top1_id_for_routing = str(routing_result.get("top1_candidate_paper_id") or "")
                stats["selected_pages_from_top1_candidate_count"] += sum(
                    1 for item in initial_pages if top1_id_for_routing and str(item.get("paper_id") or "") == top1_id_for_routing
                )
                stats["selected_pages_from_non_top1_candidates_count"] += sum(
                    1 for item in initial_pages if not top1_id_for_routing or str(item.get("paper_id") or "") != top1_id_for_routing
                )
                parsed_global_pages = list(initial_pages)
                stats["top_p_pages_selected_total"] += len(initial_pages)
                stats["vlm1_pages_selected_after_global_routing"] += len(initial_pages)
                _update_selected_page_distribution(stats, initial_pages)
                initial_page_batches = _page_batches(initial_pages, page_routing_parse_batch_size)
                batch_iterator = _progress(
                    initial_page_batches,
                    enabled=args.show_progress,
                    desc=f"{query_id} top-p page batches",
                    unit="batch",
                    leave=False,
                )
                for page_batch in batch_iterator:
                    stats["global_parse_batches"] = int(stats.get("global_parse_batches") or 0) + 1
                    candidate_records.extend(
                        _parse_rows_for_pages(
                            query_id=query_id,
                            rows=available_rows,
                            selected_pages_by_paper=_selected_pages_by_paper(page_batch),
                            args=args,
                            config=config,
                            metadata_by_id=metadata_by_id,
                            candidate_score_by_id=candidate_score_by_id,
                            structured_root=structured_root,
                            render_dpi=render_dpi,
                            render_format=render_format,
                            max_pages=max_pages,
                            cache_policy=cache_policy,
                            parser=parser,
                            paths=paths,
                            stats=stats,
                            paper_records_cache=paper_records_cache,
                            page_routing_enabled=True,
                        )
                    )
            else:
                paper_iterator = available_rows
                if args.show_progress and tqdm is not None:
                    paper_iterator = tqdm(available_rows, desc=f"{query_id} candidate PDFs", unit="paper", leave=False)  # type: ignore[assignment]
                candidate_records.extend(
                    _parse_rows_for_pages(
                        query_id=query_id,
                        rows=list(paper_iterator),
                        selected_pages_by_paper=None,
                        args=args,
                        config=config,
                        metadata_by_id=metadata_by_id,
                        candidate_score_by_id=candidate_score_by_id,
                        structured_root=structured_root,
                        render_dpi=render_dpi,
                        render_format=render_format,
                        max_pages=max_pages,
                        cache_policy=cache_policy,
                        parser=parser,
                        paths=paths,
                        stats=stats,
                        paper_records_cache=paper_records_cache,
                        page_routing_enabled=False,
                    )
                )
            selected = select_symbolic_contexts(
                str(sample.get("question", "")),
                candidate_records,
                processed_root,
                slug,
                args.top_n_records,
                args.top_n_visual_records,
                str(sample.get("primary_evidence_type") or ""),
                query_id,
                effective_vlm2_context_mode,
                include_parse_confidence,
                evidence_total_budget=config.vlm2_evidence_total_budget,
                primary_evidence_min=config.vlm2_primary_evidence_min,
                support_text_min=config.vlm2_support_text_min,
                context_types_enabled=config.vlm2_context_types_enabled,
                context_type_budget_per_type=config.vlm2_context_type_budget_per_type,
                context_selection_mode=config.vlm2_context_selection_mode,
                max_context_records=config.vlm2_max_context_records,
                max_context_chars=config.vlm2_max_context_chars,
            )
            if page_routing_enabled and routing_result is not None and config.page_routing_enable_progressive_expansion:
                ranked_pages = list(routing_result.get("ranked_pages") or [])
                parsed_keys = {(str(item.get("paper_id")), int(item.get("page") or 0)) for item in parsed_global_pages}
                while (
                    not _evidence_sufficient(
                        selected,
                        str(sample.get("primary_evidence_type") or ""),
                        min(args.top_n_records, config.page_routing_min_selected_records),
                        config.page_routing_require_primary_type_match,
                    )
                    and len(parsed_global_pages) < query_max_pages_global
                ):
                    next_pages: list[dict[str, Any]] = []
                    for ranked in ranked_pages:
                        key = (str(ranked.get("paper_id")), int(ranked.get("page") or 0))
                        if key in parsed_keys:
                            continue
                        parsed_keys.add(key)
                        next_pages.append(
                            {
                                "paper_id": ranked.get("paper_id"),
                                "page": ranked.get("page"),
                                "global_page_rank": ranked.get("global_page_rank"),
                                "candidate_rank": ranked.get("candidate_rank"),
                            }
                        )
                        if len(next_pages) >= page_routing_expansion_step_global:
                            break
                    if not next_pages:
                        break
                    remaining_budget = max(0, query_max_pages_global - len(parsed_global_pages))
                    next_pages = next_pages[:remaining_budget]
                    stats["progressive_expansion_used_count"] = int(stats.get("progressive_expansion_used_count") or 0) + 1
                    stats["progressive_expanded_pages_count"] = int(stats.get("progressive_expanded_pages_count") or 0) + len(next_pages)
                    parsed_global_pages.extend(next_pages)
                    _update_selected_page_distribution(stats, next_pages)
                    expansion_page_batches = _page_batches(next_pages, page_routing_parse_batch_size)
                    expansion_batch_iterator = _progress(
                        expansion_page_batches,
                        enabled=args.show_progress,
                        desc=f"{query_id} expansion page batches",
                        unit="batch",
                        leave=False,
                    )
                    for page_batch in expansion_batch_iterator:
                        stats["global_parse_batches"] = int(stats.get("global_parse_batches") or 0) + 1
                        candidate_records.extend(
                            _parse_rows_for_pages(
                                query_id=query_id,
                                rows=available_rows,
                                selected_pages_by_paper=_selected_pages_by_paper(page_batch),
                                args=args,
                                config=config,
                                metadata_by_id=metadata_by_id,
                                candidate_score_by_id=candidate_score_by_id,
                                structured_root=structured_root,
                                render_dpi=render_dpi,
                                render_format=render_format,
                                max_pages=max_pages,
                                cache_policy=cache_policy,
                                parser=parser,
                                paths=paths,
                                stats=stats,
                                paper_records_cache=paper_records_cache,
                                page_routing_enabled=True,
                            )
                        )
                    selected = select_symbolic_contexts(
                        str(sample.get("question", "")),
                        candidate_records,
                        processed_root,
                        slug,
                        args.top_n_records,
                        args.top_n_visual_records,
                        str(sample.get("primary_evidence_type") or ""),
                        query_id,
                        effective_vlm2_context_mode,
                        include_parse_confidence,
                        evidence_total_budget=config.vlm2_evidence_total_budget,
                        primary_evidence_min=config.vlm2_primary_evidence_min,
                        support_text_min=config.vlm2_support_text_min,
                        context_types_enabled=config.vlm2_context_types_enabled,
                        context_type_budget_per_type=config.vlm2_context_type_budget_per_type,
                        context_selection_mode=config.vlm2_context_selection_mode,
                        max_context_records=config.vlm2_max_context_records,
                        max_context_chars=config.vlm2_max_context_chars,
                    )
            if page_routing_enabled and routing_result is not None:
                total_pages = int(routing_result.get("total_candidate_pages") or 0)
                skipped_pages_count = max(0, total_pages - len(parsed_global_pages))
                plan = {
                    "query_id": query_id,
                    "task_family": budget["task_family"],
                    "task_family_bucket": budget["task_family_bucket"],
                    "top_k_papers": effective_top_k,
                    "page_routing_top_pages_per_candidate": effective_pages_per_candidate,
                    "page_routing_strategy": routing_result.get("page_routing_strategy"),
                    "top1_candidate_paper_id": routing_result.get("top1_candidate_paper_id"),
                    "top1_min_pages_required": routing_result.get("top1_min_pages_required"),
                    "top1_pages_in_global_top_p_before_quota": routing_result.get("top1_pages_in_global_top_p_before_quota"),
                    "top1_pages_selected_after_quota": routing_result.get("top1_pages_selected_after_quota"),
                    "top1_quota_added_pages": routing_result.get("top1_quota_added_pages"),
                    "top1_quota_replaced_pages": routing_result.get("top1_quota_replaced_pages"),
                    "multi_candidate_primary_quota_enabled": routing_result.get("multi_candidate_primary_quota_enabled"),
                    "multi_candidate_primary_quota_candidate_count": routing_result.get("multi_candidate_primary_quota_candidate_count"),
                    "multi_candidate_primary_quota_covered_count": routing_result.get("multi_candidate_primary_quota_covered_count"),
                    "multi_candidate_primary_quota_added_pages": routing_result.get("multi_candidate_primary_quota_added_pages"),
                    "total_candidate_pages": total_pages,
                    "initial_top_p_global": query_top_pages_global,
                    "max_pages_global": query_max_pages_global,
                    "initial_selected_pages": routing_result.get("selected_pages_initial", []),
                    "expanded_pages": parsed_global_pages[len(routing_result.get("selected_pages_initial") or []):],
                    "final_parsed_pages": parsed_global_pages,
                    "skipped_pages_count": skipped_pages_count,
                }
                append_jsonl(paths["global_page_parse_plan"], plan)
                stats["vlm1_pages_saved_by_global_routing"] += skipped_pages_count
                stats["average_global_selected_pages_per_query"] = round(
                    float(stats.get("vlm1_pages_selected_after_global_routing", 0) + stats.get("progressive_expanded_pages_count", 0))
                    / max(1, int(stats.get("processed_queries", 0)) + 1),
                    3,
                )
            image_paths, selected = _attach_vlm2_images(selected, effective_vlm2_context_mode, output_dir, paths, stats)
            selected_debug = {
                "query_id": query_id,
                "selection_method": selected.get("selection_method"),
                "prompt_context_mode": selected.get("prompt_context_mode"),
                "has_partial_artifacts": selected.get("has_partial_artifacts", False),
                "source_type_distribution": selected.get("source_type_distribution", {}),
                "primary_evidence_type_count": selected.get("primary_evidence_type_count", 0),
                "supporting_evidence_count": selected.get("supporting_evidence_count", 0),
                "grounding_label_hints_by_type": selected.get("grounding_label_hints_by_type", {}),
                "context_selection_mode": selected.get("context_selection_mode"),
                "context_truncated": selected.get("context_truncated", False),
                "selected_record_count": selected.get("selected_record_count", 0),
                "selected_records": selected.get("selected_records_debug", []),
                "selected_visual_records": selected.get("selected_visual_records", []),
            }
            selected_prompt = {
                "query_id": query_id,
                "selected_evidence": selected.get("selected_evidence", []),
                "has_partial_artifacts": selected.get("has_partial_artifacts", False),
                "attached_image_refs": selected.get("attached_image_refs", []),
                "source_type_distribution": selected.get("source_type_distribution", {}),
                "primary_evidence_type_count": selected.get("primary_evidence_type_count", 0),
                "supporting_evidence_count": selected.get("supporting_evidence_count", 0),
                "grounding_label_hints_by_type": selected.get("grounding_label_hints_by_type", {}),
                "context_selection_mode": selected.get("context_selection_mode"),
                "context_truncated": selected.get("context_truncated", False),
                "selected_record_count": selected.get("selected_record_count", 0),
            }
            if selected_prompt["selected_evidence"]:
                stats["vlm2_prompt_context_fields"] = sorted(
                    {key for record in selected_prompt["selected_evidence"] for key in record.keys()}
                )
            if selected_debug["selected_records"]:
                stats["vlm2_debug_context_fields"] = sorted(selected_debug["selected_records"][0].keys())
            _prompt_context_stats(selected_prompt, stats)
            stats["header_footer_filtered_count"] += max(
                0,
                sum(1 for record in candidate_records if record.get("record_type") == "header_footer")
                - sum(1 for record in selected_debug.get("selected_records", []) if record.get("record_type") == "header_footer"),
            )
            append_jsonl(paths["selected_contexts_debug"], selected_debug)
            append_jsonl(paths["selected_contexts_prompt"], selected_prompt)
            messages = build_symbolic_answer_prompt(sample, candidates, selected, answer_client.supports_image_input(), config.parser_model, config.answer_model, answer_contract)
            append_jsonl(paths["prompt_previews"], {"query_id": query_id, "messages": messages, "baseline_type": BASELINE_TYPE})
            stats["processed_queries"] += 1
            if args.dry_run or args.skip_generation:
                _write_report_from_paths(paths, stats)
                continue
            if not selected_prompt.get("selected_evidence"):
                stats["skipped_predictions"] += 1
                append_jsonl(paths["errors"], {"query_id": query_id, "type": "no_selected_symbolic_contexts"})
                _write_report_from_paths(paths, stats)
                continue
            if selected.get("partial_artifacts_present"):
                stats["partial_artifacts_used_in_answer_generation"] += 1
            try:
                result = answer_client.generate_prediction(messages, image_paths)
                stats["answer_api_calls"] += 1
                append_jsonl(paths["raw_answer"], {"query_id": query_id, "content": result["content"], "raw_response": result["raw_response"]})
                internal = extract_json_object(str(result["content"]))
                append_jsonl(paths["internal_predictions"], internal)
                prediction, errors = normalize_prediction(
                    internal,
                    sample,
                    [str(c.get("paper_id")) for c in candidates],
                    answer_contract=answer_contract,
                    selected_evidence=selected_prompt.get("selected_evidence", []),
                    symbolic_evidence_standardization=config.symbolic_evidence_standardization,
                )
                _update_normalization_error_stats(errors, stats)
                for error in errors:
                    append_jsonl(paths["errors"], error)
                if not validate_prediction_shape(prediction):
                    raise ValueError("invalid prediction shape")
                append_jsonl(paths["predictions"], strip_internal_grounding(prediction))
                stats["successful_predictions"] += 1
            except Exception as exc:
                stats["parse_failures"] += 1
                stats["fallback_predictions"] += 1
                append_jsonl(paths["errors"], {"query_id": query_id, "type": "answer_generation_or_parse_failure", "error": str(exc)})
                fallback = make_fallback_prediction(sample, candidates[0] if candidates else None)
                append_jsonl(paths["predictions"], fallback)
            _write_report_from_paths(paths, stats)
        stats["run_status"] = "complete"
    except KeyboardInterrupt as exc:
        stats["run_status"] = "interrupted"
        stats["interrupted_reason"] = str(exc) or interrupted_signal.get("reason", "KeyboardInterrupt")
        append_jsonl(paths["errors"], {"type": "run_interrupted", "reason": stats["interrupted_reason"]})
        exit_code = 130
    except Exception as exc:
        stats["run_status"] = "failed"
        stats["interrupted_reason"] = str(exc)
        append_jsonl(paths["errors"], {"type": "run_failed", "error": str(exc)})
        exit_code = 1
    finally:
        _write_report_from_paths(paths, stats)
        if args.sync_processed_run_store and not args.metadata_only_eval_freeze and not symbolic_cache_root_override_enabled:
            try:
                processed_run_root = Path(args.processed_output_dir).parent / "vlm_symbolic_runs"
                manifest = sync_symbolic_run_to_processed(
                    baseline_output_dir=output_dir,
                    processed_output_dir=args.processed_output_dir,
                    processed_run_root=processed_run_root,
                    env_path=args.env_path,
                )
                stats["processed_symbolic_store_path"] = manifest.get("baseline_symbolic_root")
                stats["processed_symbolic_store_runtime_records"] = manifest.get("runtime_record_count")
                stats["processed_symbolic_store_debug_records"] = manifest.get("debug_record_count")
            except Exception as sync_exc:
                stats["processed_symbolic_store_sync_error"] = str(sync_exc)
                append_jsonl(paths["errors"], {"type": "processed_symbolic_store_sync_error", "error": str(sync_exc)})
            _write_report_from_paths(paths, stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
