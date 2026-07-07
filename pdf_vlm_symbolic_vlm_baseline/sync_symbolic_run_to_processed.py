from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_pipeline_config, model_slug
from .data_io import read_jsonl, write_jsonl


RUN_LEVEL_FILES = [
    "parser_artifacts.jsonl",
    "symbolic_records.runtime.jsonl",
    "symbolic_records.debug.jsonl",
    "selected_symbolic_contexts.prompt.jsonl",
    "selected_symbolic_contexts.debug.jsonl",
    "global_page_ranking.jsonl",
    "global_page_parse_plan.jsonl",
    "run_report.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync a baseline run's symbolic artifacts into processed_pdfs.")
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--processed-output-dir", default="processed_pdfs/vlm_symbolic")
    parser.add_argument("--processed-run-root", default="processed_pdfs/vlm_symbolic_runs")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("global_record_id") or f"{record.get('paper_id')}::{record.get('record_id')}")


def _load_processed_cache_records(processed_model_root: Path, paper_ids: set[str]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for paper_id in sorted(paper_ids):
        paper_dir = processed_model_root / paper_id
        for path in sorted((paper_dir / "page_records").glob("page_*.records.runtime.jsonl")):
            for record in read_jsonl(path):
                key = _record_key(record)
                if key:
                    by_key[key] = record
    return by_key


def _merge_cache_fields(record: dict[str, Any], cache_record: dict[str, Any] | None, *, include_bbox: bool) -> dict[str, Any]:
    merged = dict(record)
    if not cache_record:
        return merged
    fields = ["figure_crop_path", "page_status"]
    if include_bbox:
        fields.extend(["bbox_1000", "figure_bbox_confidence", "figure_bbox_source"])
    for field in fields:
        if cache_record.get(field) is not None:
            merged[field] = cache_record.get(field)
    return merged


def _strip_runtime_internal_fields(record: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(record)
    for field in ["bbox_1000", "figure_bbox_confidence", "figure_bbox_source"]:
        stripped.pop(field, None)
    return stripped


def _page_image_path(processed_model_root: Path, paper_id: str, page: int) -> str:
    for suffix in ("jpg", "png", "jpeg"):
        candidate = processed_model_root / paper_id / "page_images" / f"page_{page:03d}.{suffix}"
        if candidate.exists():
            return str(candidate)
    return str(processed_model_root / paper_id / "page_images" / f"page_{page:03d}.jpg")


def _localize_record_assets(record: dict[str, Any], page_dir: Path) -> dict[str, Any]:
    copied = dict(record)
    crop_path_value = copied.get("figure_crop_path")
    if not crop_path_value:
        return copied
    crop_path = Path(str(crop_path_value))
    target_name = crop_path.name
    if not target_name:
        target_name = f"{copied.get('record_id') or 'figure'}.jpg"
    target_path = page_dir / "figure_crops" / target_name
    if not crop_path.exists():
        if target_path.exists():
            copied["figure_crop_path"] = str(target_path)
        return copied
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if crop_path.resolve() != target_path.resolve():
        shutil.copy2(crop_path, target_path)
    copied["figure_crop_path"] = str(target_path)
    return copied


def _write_grouped_records(
    *,
    records: list[dict[str, Any]],
    baseline_model_root: Path,
    processed_model_root: Path,
    name: str,
) -> None:
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        paper_id = str(record.get("paper_id") or "")
        page = int(record.get("page") or 0)
        if not paper_id or page <= 0:
            continue
        page_dir = baseline_model_root / paper_id / f"page_{page:03d}"
        localized = _localize_record_assets(record, page_dir)
        by_paper[paper_id].append(localized)
        by_page[(paper_id, page)].append(localized)
    for paper_id, paper_records in by_paper.items():
        paper_dir = baseline_model_root / paper_id
        write_jsonl(paper_dir / f"symbolic_records.{name}.jsonl", paper_records)
    for (paper_id, page), page_records in by_page.items():
        page_dir = baseline_model_root / paper_id / f"page_{page:03d}"
        write_jsonl(page_dir / f"records.{name}.jsonl", page_records)
        manifest = {
            "paper_id": paper_id,
            "page": page,
            "record_count": len(page_records),
            "page_image_path": _page_image_path(processed_model_root, paper_id, page),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (page_dir / "page_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_selected_contexts_by_page(baseline_output_dir: Path, baseline_model_root: Path) -> int:
    selected_path = baseline_output_dir / "selected_symbolic_contexts.debug.jsonl"
    if not selected_path.exists():
        return 0
    rows_written = 0
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(selected_path):
        query_id = row.get("query_id")
        for record in row.get("selected_records", []) if isinstance(row.get("selected_records"), list) else []:
            if not isinstance(record, dict):
                continue
            paper_id = str(record.get("paper_id") or "")
            page = int(record.get("page") or 0)
            if not paper_id or page <= 0:
                continue
            copied = dict(record)
            copied["query_id"] = query_id
            by_page[(paper_id, page)].append(copied)
    for (paper_id, page), records in by_page.items():
        write_jsonl(baseline_model_root / paper_id / f"page_{page:03d}" / "selected_contexts.debug.jsonl", records)
        rows_written += len(records)
    return rows_written


def sync_symbolic_run_to_processed(
    *,
    baseline_output_dir: str | Path,
    processed_output_dir: str | Path = "processed_pdfs/vlm_symbolic",
    processed_run_root: str | Path = "processed_pdfs/vlm_symbolic_runs",
    env_path: str | Path = ".env",
) -> dict[str, Any]:
    config = load_pipeline_config(env_path)
    slug = model_slug(config.parser_model)
    baseline_output_dir = Path(baseline_output_dir)
    baseline_name = baseline_output_dir.name
    processed_model_root = Path(processed_output_dir) / slug
    baseline_model_root = Path(processed_run_root) / baseline_name / slug
    baseline_model_root.mkdir(parents=True, exist_ok=True)

    runtime_path = baseline_output_dir / "symbolic_records.runtime.jsonl"
    debug_path = baseline_output_dir / "symbolic_records.debug.jsonl"
    if not runtime_path.exists():
        raise FileNotFoundError(f"missing runtime symbolic records: {runtime_path}")

    runtime_records = read_jsonl(runtime_path)
    debug_records = read_jsonl(debug_path) if debug_path.exists() else []
    paper_ids = {str(record.get("paper_id")) for record in runtime_records if record.get("paper_id")}
    cache_by_key = _load_processed_cache_records(processed_model_root, paper_ids)
    runtime_records = [
        _strip_runtime_internal_fields(_merge_cache_fields(record, cache_by_key.get(_record_key(record)), include_bbox=False))
        for record in runtime_records
    ]
    debug_records = [_merge_cache_fields(record, cache_by_key.get(_record_key(record)), include_bbox=True) for record in debug_records]

    run_level_dir = baseline_model_root / "_run_level"
    copied_files = []
    for file_name in RUN_LEVEL_FILES:
        if _copy_if_exists(baseline_output_dir / file_name, run_level_dir / file_name):
            copied_files.append(file_name)
    write_jsonl(run_level_dir / "symbolic_records.runtime.jsonl", runtime_records)
    if debug_records:
        write_jsonl(run_level_dir / "symbolic_records.debug.jsonl", debug_records)
    _write_grouped_records(
        records=runtime_records,
        baseline_model_root=baseline_model_root,
        processed_model_root=processed_model_root,
        name="runtime",
    )
    if debug_records:
        _write_grouped_records(
            records=debug_records,
            baseline_model_root=baseline_model_root,
            processed_model_root=processed_model_root,
            name="debug",
        )
    selected_rows = _write_selected_contexts_by_page(baseline_output_dir, baseline_model_root)
    manifest = {
        "baseline_name": baseline_name,
        "parser_model": config.parser_model,
        "parser_model_slug": slug,
        "source_output_dir": str(baseline_output_dir),
        "processed_model_root": str(processed_model_root),
        "baseline_symbolic_root": str(baseline_model_root),
        "paper_count": len(paper_ids),
        "runtime_record_count": len(runtime_records),
        "debug_record_count": len(debug_records),
        "selected_context_records_written": selected_rows,
        "copied_run_level_files": copied_files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (baseline_model_root / "run_symbolic_store_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = sync_symbolic_run_to_processed(
        baseline_output_dir=args.baseline_output_dir,
        processed_output_dir=args.processed_output_dir,
        processed_run_root=args.processed_run_root,
        env_path=args.env_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
