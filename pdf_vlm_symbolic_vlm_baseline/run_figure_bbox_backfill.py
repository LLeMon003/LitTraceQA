from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .config import load_pipeline_config, model_slug
from .data_io import append_jsonl, read_jsonl, write_jsonl
from .figure_bbox_prompt_builder import build_figure_bbox_prompt
from .parser import extract_json_object
from .vlm_parser_client import VLMParserClient


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill figure bbox/crops for an existing symbolic VLM baseline run.")
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--processed-output-dir", default="processed_pdfs/vlm_symbolic")
    parser.add_argument("--figure-output-root", default="processed_pdfs/vlm_symbolic_runs")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--limit-pages", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-processed-cache", action="store_true", default=True)
    parser.add_argument("--no-update-processed-cache", dest="update_processed_cache", action="store_false")
    return parser.parse_args()


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value or fallback).strip() or fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._-") or fallback


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _repair_bbox(value: Any) -> tuple[list[int] | None, str]:
    if not isinstance(value, list) or len(value) != 4:
        return None, "bbox_1000 is not a 4-item list"
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return None, "bbox_1000 contains non-numeric values"
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1, x2, y2 = [max(0.0, min(1000.0, v)) for v in (x1, y1, x2, y2)]
    if not (x1 < x2 and y1 < y2):
        return None, "bbox_1000 has zero or negative area"
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], ""


def _crop_figure(image_path: Path, bbox_1000: list[int], output_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            x1, y1, x2, y2 = bbox_1000
            box = (
                max(0, min(width, int(round(x1 * width / 1000)))),
                max(0, min(height, int(round(y1 * height / 1000)))),
                max(0, min(width, int(round(x2 * width / 1000)))),
                max(0, min(height, int(round(y2 * height / 1000)))),
            )
            if box[0] >= box[2] or box[1] >= box[3]:
                return False
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.crop(box).save(output_path, format="JPEG", quality=92, optimize=True)
        return True
    except Exception:
        return False


def _figure_records_by_page(symbolic_path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for record in read_jsonl(symbolic_path):
        if record.get("source_type") != "figure" and record.get("record_type") != "figure":
            continue
        paper_id = str(record.get("paper_id") or "")
        page = int(record.get("page") or 0)
        record_id = str(record.get("record_id") or "")
        key = (paper_id, page, record_id)
        if not paper_id or page <= 0 or key in seen:
            continue
        seen.add(key)
        grouped[(paper_id, page)].append(record)
    return grouped


def _merge_enrichment_into_records(
    records: list[dict[str, Any]],
    enrichments: dict[str, dict[str, Any]],
    *,
    include_bbox: bool,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        enrichment = enrichments.get(str(copied.get("record_id") or ""))
        if enrichment:
            if include_bbox:
                copied["bbox_1000"] = enrichment.get("bbox_1000")
            copied["figure_crop_path"] = enrichment.get("crop_path")
            if include_bbox:
                copied["figure_bbox_confidence"] = enrichment.get("confidence")
                copied["figure_bbox_source"] = "vlm1_figure_bbox_backfill"
        merged.append(copied)
    return merged


def _refresh_paper_aggregate(paper_dir: Path) -> None:
    page_records_dir = paper_dir / "page_records"
    runtime_records: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    for runtime_path in sorted(page_records_dir.glob("page_*.records.runtime.jsonl")):
        runtime_records.extend(read_jsonl(runtime_path))
    for debug_path in sorted(page_records_dir.glob("page_*.records.debug.jsonl")):
        debug_records.extend(read_jsonl(debug_path))
    if runtime_records:
        write_jsonl(paper_dir / "symbolic_records.runtime.jsonl", runtime_records)
    if debug_records:
        write_jsonl(paper_dir / "symbolic_records.debug.jsonl", debug_records)


def _update_processed_page_cache(paper_dir: Path, page: int, enrichments: dict[str, dict[str, Any]]) -> None:
    page_records_dir = paper_dir / "page_records"
    runtime_path = page_records_dir / f"page_{page:03d}.records.runtime.jsonl"
    debug_path = page_records_dir / f"page_{page:03d}.records.debug.jsonl"
    changed = False
    if runtime_path.exists():
        runtime_records = _merge_enrichment_into_records(read_jsonl(runtime_path), enrichments, include_bbox=False)
        write_jsonl(runtime_path, runtime_records)
        changed = True
    if debug_path.exists():
        debug_records = _merge_enrichment_into_records(read_jsonl(debug_path), enrichments, include_bbox=True)
        write_jsonl(debug_path, debug_records)
        changed = True
    if changed:
        _refresh_paper_aggregate(paper_dir)


def _page_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def main() -> int:
    args = parse_args()
    config = load_pipeline_config(args.env_path)
    parser = VLMParserClient(config)
    baseline_output_dir = Path(args.baseline_output_dir)
    baseline_name = baseline_output_dir.name
    processed_root = Path(args.processed_output_dir)
    slug = model_slug(config.parser_model)
    paper_root = processed_root / slug
    figure_root = Path(args.figure_output_root) / baseline_name / slug
    symbolic_path = baseline_output_dir / "symbolic_records.runtime.jsonl"
    if not symbolic_path.exists():
        raise FileNotFoundError(f"missing symbolic records: {symbolic_path}")
    grouped = _figure_records_by_page(symbolic_path)
    pages = sorted(grouped.items())
    if args.limit_pages > 0:
        pages = pages[: args.limit_pages]
    summary_path = figure_root / "figure_backfill_summary.jsonl"
    if args.dry_run:
        print(json.dumps({"figure_pages": len(pages), "figure_records": sum(len(v) for _, v in pages)}, ensure_ascii=False, indent=2))
        return 0
    if not parser.supports_image_input():
        raise RuntimeError("Parser VLM image input is not configured.")
    iterator = pages
    if args.show_progress and tqdm is not None:
        iterator = tqdm(pages, desc="figure bbox pages", unit="page")  # type: ignore[assignment]
    processed_pages = 0
    skipped_pages = 0
    failed_pages = 0
    cropped_figures = 0
    for (paper_id, page), records in iterator:
        page_dir = figure_root / paper_id / f"page_{page:03d}"
        status_path = page_dir / "figure_bbox.status.json"
        if status_path.exists() and not args.force:
            status = _load_json(status_path) or {}
            if status.get("status") == "complete":
                skipped_pages += 1
                continue
        image_path = paper_root / paper_id / "page_images" / f"page_{page:03d}.jpg"
        if not image_path.exists():
            image_path = paper_root / paper_id / "page_images" / f"page_{page:03d}.png"
        if not image_path.exists():
            failed_pages += 1
            _write_json(status_path, {"paper_id": paper_id, "page": page, "status": "failed", "failure_reason": "page image missing"})
            continue
        width, height = _page_image_size(image_path)
        messages = build_figure_bbox_prompt(
            paper_id=paper_id,
            page=page,
            page_width=width,
            page_height=height,
            figure_records=records,
        )
        raw_result = parser.generate_page_structure(messages, image_path)
        raw_path = page_dir / "figure_bbox.raw.json"
        _write_json(raw_path, raw_result)
        try:
            raw_obj = extract_json_object(str(raw_result.get("content") or ""))
        except Exception as exc:
            failed_pages += 1
            _write_json(
                status_path,
                {
                    "paper_id": paper_id,
                    "page": page,
                    "status": "failed",
                    "failure_reason": str(exc),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue
        figures = raw_obj.get("figures") if isinstance(raw_obj, dict) else None
        if not isinstance(figures, list):
            figures = []
        by_record_id = {str(record.get("record_id") or ""): record for record in records}
        enriched_rows: list[dict[str, Any]] = []
        enrichments: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        for item in figures:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("record_id") or "")
            if record_id not in by_record_id:
                rejected.append({"record_id": record_id, "reason": "record_id not in page figure records"})
                continue
            bbox, bbox_error = _repair_bbox(item.get("bbox_1000"))
            if bbox is None:
                rejected.append({"record_id": record_id, "reason": bbox_error})
                continue
            source_record = by_record_id[record_id]
            crop_name = f"{_safe_name(source_record.get('label'), record_id)}_{record_id}.jpg"
            crop_path = page_dir / "figure_crops" / crop_name
            crop_ok = _crop_figure(image_path, bbox, crop_path)
            row = {
                "paper_id": paper_id,
                "page": page,
                "record_id": record_id,
                "global_record_id": source_record.get("global_record_id"),
                "label": source_record.get("label"),
                "locator": source_record.get("locator"),
                "bbox_1000": bbox,
                "confidence": item.get("confidence"),
                "notes": item.get("notes", ""),
                "page_image_path": str(image_path),
                "crop_path": str(crop_path) if crop_ok else "",
                "crop_status": "complete" if crop_ok else "failed",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if crop_ok:
                cropped_figures += 1
            enriched_rows.append(row)
            enrichments[record_id] = row
        write_jsonl(page_dir / "figure_bbox.records.jsonl", enriched_rows)
        if args.update_processed_cache and enrichments:
            _update_processed_page_cache(paper_root / paper_id, page, enrichments)
        status = {
            "paper_id": paper_id,
            "page": page,
            "status": "complete" if enriched_rows else "partial",
            "figure_record_count": len(records),
            "localized_figure_count": len(enriched_rows),
            "rejected_count": len(rejected),
            "rejected": rejected,
            "warnings": raw_obj.get("warnings") if isinstance(raw_obj.get("warnings"), list) else [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(status_path, status)
        append_jsonl(summary_path, status)
        processed_pages += 1
    print(
        json.dumps(
            {
                "baseline_name": baseline_name,
                "figure_output_root": str(figure_root),
                "processed_pages": processed_pages,
                "skipped_pages": skipped_pages,
                "failed_pages": failed_pages,
                "cropped_figures": cropped_figures,
                "total_candidate_pages": len(pages),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
