from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from .data_io import _json_safe, write_jsonl
from .pdf_citation_extractor import extract_citations
from .pdf_equation_algorithm_extractor import extract_equations_algorithms
from .pdf_figure_extractor import extract_figures
from .pdf_layout_blocks import extract_drawing_bboxes, extract_image_bboxes, extract_text_blocks, page_size
from .pdf_section_builder import build_sectioned_symbolic_layer
from .pdf_table_extractor import extract_tables
from .pdf_text_span_extractor import extract_text_spans
from .transcription_backends import get_transcription_backend, normalize_backend_name
from .transcription_backends.standardizer import STANDARDIZER_VERSION, standardize_transcribed_document


ARTIFACT_VERSION = "v10_strict_section_heading_detection"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _cache_reusable(status_path: Path, policy: str) -> bool:
    status = _load_json(status_path)
    if not status or status.get("parser") != "pymupdf" or status.get("artifact_version") != ARTIFACT_VERSION:
        return False
    if policy == "reuse_partial_allowed":
        return status.get("status") in {"complete", "partial"}
    return status.get("status") == "complete"


def _backend_cache_reusable(status_path: Path, policy: str, backend: str) -> bool:
    status = _load_json(status_path)
    if not status or (status.get("transcription_backend") or status.get("parser")) != backend:
        return False
    if backend == "docling" and status.get("standardizer_version") != STANDARDIZER_VERSION:
        return False
    if policy == "reuse_partial_allowed":
        return status.get("status") in {"complete", "partial"}
    return status.get("status") == "complete"


def _augment_backend_status(status_path: Path, backend: str, backend_version: str | None = None, backend_config: dict[str, Any] | None = None) -> dict[str, Any]:
    status = _load_json(status_path) or {}
    status["transcription_backend"] = backend
    status["transcription_backend_version"] = backend_version if backend_version is not None else status.get("parser_version", "")
    status["transcription_backend_config"] = backend_config or status.get("transcription_backend_config") or {}
    status["backend_output_dir"] = str(status_path.parent)
    _write_status(status_path, status)
    for name in ("extraction_report.json",):
        path = status_path.parent / name
        report = _load_json(path) or {}
        report["transcription_backend"] = backend
        report["transcription_backend_version"] = status["transcription_backend_version"]
        report["transcription_backend_config"] = status["transcription_backend_config"]
        report["backend_output_dir"] = str(status_path.parent)
        _write_status(path, report)
    return status


def _runtime_record(paper_id: str, page_no: int, index: int, item: dict[str, Any]) -> dict[str, Any]:
    record_id = f"p{page_no:03d}_r{index:04d}"
    locator = item.get("locator") if isinstance(item.get("locator"), dict) else {"page": page_no}
    locator = {"page": page_no, **{k: v for k, v in locator.items() if k != "page"}}
    return {
        "paper_id": paper_id,
        "page": page_no,
        "record_id": record_id,
        "global_record_id": f"{paper_id}::{record_id}",
        "record_type": item.get("record_type") or "paragraph",
        "source_type": item.get("source_type") or "text_span",
        "label": item.get("label"),
        "locator": locator,
        "text": str(item.get("text") or ""),
        "reading_order": index,
    }


def _write_status(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clear_generated_dirs(paper_dir: Path) -> None:
    for name in ("object_crops", "page_debug", "page_records", "page_status", "section_records"):
        path = paper_dir / name
        if path.exists():
            shutil.rmtree(path)


def extract_pdf_symbolic_records(
    paper_id: str,
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    extract_all_pages: bool = True,
    selected_pages: set[int] | None = None,
    max_pages: int | None = None,
    enable_figure_crops: bool = True,
    enable_table_crops: bool = False,
    enable_equation_crops: bool = False,
    crop_dpi: int = 160,
    min_text_block_chars: int = 20,
    include_debug_bbox: bool = False,
    cache_policy: str = "reuse_complete_only",
    docling_do_ocr: bool = False,
) -> dict[str, Any]:
    del include_debug_bbox
    paper_dir = Path(output_dir)
    status_path = paper_dir / "artifact_status.json"
    runtime_path = paper_dir / "symbolic_records.runtime.jsonl"
    debug_path = paper_dir / "symbolic_records.debug.jsonl"
    if not overwrite and cache_policy != "refresh" and _cache_reusable(status_path, cache_policy) and runtime_path.exists():
        status = _load_json(status_path) or {}
        status["cache_status"] = "reused"
        return status
    if cache_policy == "fail_if_missing" and not runtime_path.exists():
        return {"paper_id": paper_id, "status": "failed", "error": "extraction_cache_missing", "cache_status": "missing"}

    paper_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_dirs(paper_dir)
    page_debug_dir = paper_dir / "page_debug"
    page_status_dir = paper_dir / "page_status"
    page_debug_dir.mkdir(parents=True, exist_ok=True)
    page_status_dir.mkdir(parents=True, exist_ok=True)

    try:
        import fitz  # type: ignore
    except Exception as exc:
        status = {"paper_id": paper_id, "parser": "pymupdf", "artifact_version": ARTIFACT_VERSION, "status": "failed", "error": f"pymupdf_import_failed: {exc}"}
        _write_status(status_path, status)
        return status
    try:
        fitz.TOOLS.mupdf_display_errors(False)
        fitz.TOOLS.mupdf_display_warnings(False)
    except Exception:
        pass

    all_runtime: list[dict[str, Any]] = []
    all_debug: list[dict[str, Any]] = []
    manifest_pages: list[dict[str, Any]] = []
    complete_pages = 0
    partial_pages = 0
    failed_pages = 0
    references_active = False
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        status = {"paper_id": paper_id, "parser": "pymupdf", "artifact_version": ARTIFACT_VERSION, "status": "failed", "error": f"pdf_open_failed: {exc}"}
        _write_status(status_path, status)
        return status

    page_count = int(getattr(doc, "page_count", len(doc)) or 0)
    target_pages = list(range(1, page_count + 1))
    if not extract_all_pages and selected_pages:
        target_pages = [p for p in target_pages if p in selected_pages]
    if max_pages:
        target_pages = target_pages[: max(0, int(max_pages))]

    for page_no in target_pages:
        page_runtime: list[dict[str, Any]] = []
        page_debug: list[dict[str, Any]] = []
        page_error = ""
        try:
            page = doc.load_page(page_no - 1)
            width, height = page_size(page)
            blocks = extract_text_blocks(page)
            image_boxes = extract_image_bboxes(page)
            drawing_boxes = extract_drawing_bboxes(page)
            visual_boxes = image_boxes + drawing_boxes
            used_boxes = []

            tables, table_debug, table_boxes = extract_tables(
                page,
                blocks,
                page_no,
                paper_dir,
                enable_crops=enable_table_crops,
                crop_dpi=crop_dpi,
            )
            figures, figure_debug, figure_boxes = extract_figures(
                page,
                blocks,
                visual_boxes,
                page_no,
                paper_dir,
                enable_crops=enable_figure_crops,
                crop_dpi=crop_dpi,
            )
            equations, equation_debug, equation_boxes = extract_equations_algorithms(
                page,
                blocks,
                page_no,
                paper_dir,
                enable_crops=enable_equation_crops,
                crop_dpi=crop_dpi,
            )
            citations, citation_debug, citation_boxes, references_active = extract_citations(blocks, page_no, references_active=references_active)
            used_boxes.extend(table_boxes + figure_boxes + equation_boxes + citation_boxes)
            text_spans = extract_text_spans(blocks, page_no, used_boxes=used_boxes, min_text_block_chars=min_text_block_chars, page_height=height)
            runtime_like = tables + figures + equations + citations + text_spans
            page_debug = table_debug + figure_debug + equation_debug + citation_debug
            for index, item in enumerate(runtime_like, start=1):
                runtime = _runtime_record(paper_id, page_no, index, item)
                page_runtime.append(runtime)
                debug_source = next((d for d in page_debug if d.get("record_type") == runtime["record_type"] and d.get("label") == runtime.get("label")), {})
                debug_item = {
                    **runtime,
                    "debug": debug_source,
                }
                for key in ("has_citation_markers", "citation_marker_count", "citation_markers"):
                    if key in item:
                        debug_item[key] = item[key]
                if debug_item["debug"].get("figure_crop_path"):
                    debug_item["figure_crop_path"] = debug_item["debug"].get("figure_crop_path")
                if debug_item["debug"].get("table_crop_path"):
                    debug_item["crop_path"] = debug_item["debug"].get("table_crop_path")
                    debug_item["table_crop_path"] = debug_item["debug"].get("table_crop_path")
                if debug_item["debug"].get("equation_algorithm_crop_path"):
                    debug_item["crop_path"] = debug_item["debug"].get("equation_algorithm_crop_path")
                    debug_item["equation_algorithm_crop_path"] = debug_item["debug"].get("equation_algorithm_crop_path")
                all_debug.append(debug_item)
            manifest_pages.append(
                {
                    "page": page_no,
                    "width_pdf": width,
                    "height_pdf": height,
                    "text_block_count": len(blocks),
                    "image_box_count": len(image_boxes),
                    "drawing_box_count": len(drawing_boxes),
                    "record_count": len(page_runtime),
                }
            )
            page_status = "complete" if page_runtime else "partial"
            if page_runtime:
                complete_pages += 1
            else:
                partial_pages += 1
        except Exception as exc:
            page_status = "failed"
            page_error = str(exc)
            failed_pages += 1
        page_debug_rows = [row for row in all_debug if int(row.get("page") or 0) == page_no]
        write_jsonl(page_debug_dir / f"page_{page_no:03d}.records.debug.jsonl", page_debug_rows)
        _write_status(
            page_status_dir / f"page_{page_no:03d}.status.json",
            {
                "paper_id": paper_id,
                "page": page_no,
                "page_status": page_status,
                "valid_record_count": len(page_runtime),
                "failure_reason": page_error,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        all_runtime.extend(page_runtime)

    try:
        doc.close()
    except Exception:
        pass

    section_result = build_sectioned_symbolic_layer(paper_id, all_runtime, all_debug, paper_dir)
    all_runtime = section_result["runtime_records"]
    all_debug = section_result["debug_records"]
    source_dist = dict(Counter(str(record.get("source_type") or "") for record in all_runtime))
    section_type_dist = dict(Counter(str(section.get("section_type") or "") for section in section_result.get("sections", [])))
    status_value = "complete" if complete_pages and failed_pages == 0 else ("partial" if all_runtime else "failed")
    status = {
        "paper_id": paper_id,
        "parser": "pymupdf",
        "parser_version": getattr(fitz, "VersionBind", ""),
        "artifact_version": ARTIFACT_VERSION,
        "page_count": page_count,
        "parsed_pages": len(target_pages),
        "complete_pages": complete_pages,
        "partial_pages": partial_pages,
        "failed_pages": failed_pages,
        "valid_record_count": len(all_runtime),
        "section_count": int(section_result.get("section_count") or 0),
        "source_type_distribution": source_dist,
        "section_type_distribution": section_type_dist,
        "status": status_value,
        "cache_status": "refreshed" if overwrite or cache_policy == "refresh" else "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbolic_records_runtime_path": str(runtime_path),
        "symbolic_records_debug_path": str(debug_path),
        "sections_path": str(paper_dir / "sections.jsonl"),
        "section_tree_path": str(paper_dir / "section_tree.json"),
    }
    _write_status(status_path, status)
    _write_status(
        paper_dir / "document_manifest.json",
        {"paper_id": paper_id, "pdf_path": str(pdf_path), "page_count": page_count, "pages": manifest_pages, "created_at": datetime.now(timezone.utc).isoformat()},
    )
    _write_status(
        paper_dir / "extraction_report.json",
        {
            "paper_id": paper_id,
            "parser": "pymupdf",
            "artifact_version": ARTIFACT_VERSION,
            "record_count": len(all_runtime),
            "section_count": int(section_result.get("section_count") or 0),
            "source_type_distribution": source_dist,
            "section_type_distribution": section_type_dist,
            "status": status_value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return status


def extract_pdf_symbolic_records_with_backend(
    paper_id: str,
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    transcription_backend: str = "docling",
    overwrite: bool = False,
    extract_all_pages: bool = True,
    selected_pages: set[int] | None = None,
    max_pages: int | None = None,
    enable_figure_crops: bool = True,
    enable_table_crops: bool = False,
    enable_equation_crops: bool = False,
    crop_dpi: int = 160,
    min_text_block_chars: int = 20,
    include_debug_bbox: bool = False,
    cache_policy: str = "reuse_complete_only",
    docling_do_ocr: bool = False,
) -> dict[str, Any]:
    backend_name = normalize_backend_name(transcription_backend)
    paper_dir = Path(output_dir)
    status_path = paper_dir / "artifact_status.json"
    runtime_path = paper_dir / "symbolic_records.runtime.jsonl"
    if not overwrite and cache_policy != "refresh" and _backend_cache_reusable(status_path, cache_policy, backend_name) and runtime_path.exists():
        status = _load_json(status_path) or {}
        status["cache_status"] = "reused"
        return status
    if cache_policy == "fail_if_missing" and not runtime_path.exists():
        return {"paper_id": paper_id, "status": "failed", "error": "extraction_cache_missing", "cache_status": "missing", "transcription_backend": backend_name}

    backend = get_transcription_backend(backend_name)
    if backend_name == "pymupdf":
        document = backend.transcribe_pdf(
            paper_id,
            Path(pdf_path),
            paper_dir,
            overwrite=overwrite,
            extract_all_pages=extract_all_pages,
            selected_pages=selected_pages,
            max_pages=max_pages,
            enable_figure_crops=enable_figure_crops,
            enable_table_crops=enable_table_crops,
            enable_equation_crops=enable_equation_crops,
            crop_dpi=crop_dpi,
            min_text_block_chars=min_text_block_chars,
            include_debug_bbox=include_debug_bbox,
            cache_policy=cache_policy,
        )
        return _augment_backend_status(status_path, backend_name, document.backend_version, document.backend_config)

    document = backend.transcribe_pdf(
        paper_id,
        Path(pdf_path),
        paper_dir,
        overwrite=overwrite or cache_policy == "refresh",
        max_pages=max_pages,
        extract_all_pages=extract_all_pages,
        selected_pages=selected_pages,
        docling_do_ocr=docling_do_ocr,
    )
    return standardize_transcribed_document(document, paper_dir, overwrite=overwrite, cache_policy=cache_policy)
