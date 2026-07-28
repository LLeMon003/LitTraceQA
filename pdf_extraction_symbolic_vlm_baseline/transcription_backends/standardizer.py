from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..data_io import _json_safe, write_jsonl
from ..pdf_section_builder import build_sectioned_symbolic_layer
from ..pdf_text_span_extractor import is_likely_section_heading
from .base import TranscribedDocument, TranscribedElement


STANDARDIZER_VERSION = "v3_first_level_section_whitelist"


def _write_status(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clear_standardized_dirs(paper_dir: Path) -> None:
    for name in ("page_debug", "page_records", "page_status", "section_records"):
        path = paper_dir / name
        if path.exists():
            shutil.rmtree(path)


def _bbox_list(value: list[float] | None) -> list[float] | None:
    if not value or len(value) != 4:
        return None
    return [round(float(v), 3) for v in value]


def _normalize_label(text: str | None, source_type: str) -> str | None:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return None
    if source_type == "table":
        match = re.search(r"\b(?:Table|Tab\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)", value, re.IGNORECASE)
        return f"Table {match.group(1).rstrip('.,;:')}" if match else None
    if source_type == "figure":
        match = re.search(r"\b(?:Figure|Fig\.)\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*(?:\([a-z]\))?)", value, re.IGNORECASE)
        return f"Figure {match.group(1).rstrip('.,;:')}" if match else None
    if source_type == "equation_algorithm":
        alg = re.search(r"\bAlgorithm\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)", value, re.IGNORECASE)
        if alg:
            return f"Algorithm {alg.group(1).rstrip('.,;:')}"
        eq = re.search(r"\b(?:Equation|Eq\.)\s*\(?\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\s*\)?", value, re.IGNORECASE)
        return f"Equation {eq.group(1).rstrip('.,;:')}" if eq else None
    return None


def _locator(page: int | None, source_type: str, label: str | None) -> dict[str, Any]:
    loc: dict[str, Any] = {"page": page if page is not None else -1}
    if label:
        if source_type == "table":
            loc["table_id"] = label
        elif source_type == "figure":
            loc["figure_id"] = label
        elif source_type == "equation_algorithm":
            if label.lower().startswith("algorithm"):
                loc["algorithm_id"] = label
            else:
                loc["equation_id"] = label
        elif source_type == "citation_context":
            loc["citation_id"] = label
    return loc


def _local_docling_image_path(payload: dict[str, Any]) -> str | None:
    candidates: list[Any] = [payload.get("_littraceqa_image_path")]
    image = payload.get("image") if isinstance(payload.get("image"), dict) else {}
    candidates.append(image.get("uri"))
    for value in candidates:
        path = str(value or "").strip()
        if not path or path.startswith("data:") or re.match(r"^[a-z]+://", path, re.IGNORECASE):
            continue
        return path
    return None


def _crop_fields_for_element(source_type: str, element: TranscribedElement) -> dict[str, Any]:
    if source_type != "figure":
        return {"crop_path": None}
    image_path = _local_docling_image_path(element.raw_backend_payload)
    if not image_path:
        return {"crop_path": None}
    return {"crop_path": image_path, "figure_crop_path": image_path}


def _record_type_and_source(element: TranscribedElement) -> tuple[str, str, list[str]]:
    raw_type = str(element.element_type or element.raw_backend_type or "").lower()
    text = str(element.text or "")
    rules: list[str] = []
    if raw_type in {"table", "document_table"}:
        return "table", "table", ["docling_table_to_table_record"]
    if raw_type in {"picture", "image", "figure"}:
        return "figure", "figure", ["docling_picture_to_figure_record"]
    if raw_type in {"formula", "equation"} or _normalize_label(text, "equation_algorithm"):
        label = _normalize_label(text, "equation_algorithm") or ""
        return ("algorithm" if label.lower().startswith("algorithm") else "equation"), "equation_algorithm", ["equation_or_algorithm_label_rule"]
    if is_likely_section_heading(text):
        return "section_header", "text_span", ["heading_rule"]
    return "paragraph", "text_span", ["default_text_to_text_span"]


def _reference_label(index: int) -> str:
    return f"Reference {index}"


def _split_reference_entries(records: list[dict[str, Any]], debug_by_gid: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    new_debug = dict(debug_by_gid)
    references_active = False
    ref_index = 0
    for record in records:
        text = str(record.get("text") or "")
        if record.get("record_type") == "section_header" and re.search(r"\b(?:References|Bibliography)\b", text, re.IGNORECASE):
            references_active = True
            output.append(record)
            continue
        if references_active and record.get("source_type") == "text_span" and record.get("record_type") == "section_header" and not re.search(r"\b(?:References|Bibliography)\b", text, re.IGNORECASE):
            references_active = False
        if references_active and record.get("source_type") == "text_span" and record.get("record_type") == "paragraph" and len(text) >= 20:
            ref_index += 1
            copied = dict(record)
            label = _reference_label(ref_index)
            copied["record_type"] = "reference_entry"
            copied["source_type"] = "citation_context"
            copied["label"] = label
            copied["locator"] = {"page": copied.get("page"), "citation_id": label}
            gid = str(copied.get("global_record_id") or "")
            debug = dict(new_debug.get(gid, copied))
            debug.update({"record_type": "reference_entry", "source_type": "citation_context", "label": label, "locator": copied["locator"]})
            rules = list(debug.get("standardization_rules") or [])
            rules.append("references_section_paragraph_to_reference_entry")
            debug["standardization_rules"] = rules
            new_debug[gid] = debug
            output.append(copied)
            continue
        output.append(record)
    return output, new_debug


def _runtime_record(paper_id: str, page: int | None, index: int, item: dict[str, Any]) -> dict[str, Any]:
    page_value = page if page is not None else -1
    record_id = f"p{max(0, int(page_value)):03d}_r{index:04d}"
    return {
        "paper_id": paper_id,
        "page": page_value,
        "record_id": record_id,
        "global_record_id": f"{paper_id}::{record_id}",
        "record_type": item.get("record_type") or "paragraph",
        "source_type": item.get("source_type") or "text_span",
        "label": item.get("label"),
        "locator": item.get("locator") or {"page": page_value},
        "text": str(item.get("text") or ""),
        "reading_order": index,
    }


def standardize_transcribed_document(
    document: TranscribedDocument,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    cache_policy: str = "reuse_complete_only",
) -> dict[str, Any]:
    del overwrite, cache_policy
    paper_dir = Path(output_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    _clear_standardized_dirs(paper_dir)
    page_debug_dir = paper_dir / "page_debug"
    page_status_dir = paper_dir / "page_status"
    page_debug_dir.mkdir(parents=True, exist_ok=True)
    page_status_dir.mkdir(parents=True, exist_ok=True)

    runtime: list[dict[str, Any]] = []
    debug_by_gid: dict[str, dict[str, Any]] = {}
    missing_page_count = 0
    page_counts: Counter[int] = Counter()
    for index, element in enumerate(document.elements, start=1):
        record_type, source_type, rules = _record_type_and_source(element)
        label = element.label or _normalize_label(element.text, source_type)
        if record_type == "section_header":
            label = str(element.text or "").strip()
        page = element.page
        if page is None:
            missing_page_count += 1
        loc = _locator(page, source_type, label)
        text = str(element.text or "").strip()
        if not text:
            continue
        item = {"record_type": record_type, "source_type": source_type, "label": label, "locator": loc, "text": text}
        record = _runtime_record(document.paper_id, page, index, item)
        runtime.append(record)
        page_counts[int(record.get("page") or -1)] += 1
        debug_by_gid[str(record["global_record_id"])] = {
            **record,
            "bbox": _bbox_list(element.bbox),
            **_crop_fields_for_element(source_type, element),
            "backend": document.backend,
            "raw_backend_type": element.raw_backend_type,
            "raw_backend_payload_path": document.raw_output_path,
            "raw_backend_element_id": element.element_id,
            "raw_backend_payload": element.raw_backend_payload,
            "standardization_rules": rules,
            "warnings": list(element.warnings or []),
            "page_missing": element.page is None,
        }

    runtime, debug_by_gid = _split_reference_entries(runtime, debug_by_gid)
    debug_records = [debug_by_gid.get(str(record.get("global_record_id") or ""), record) for record in runtime]
    section_result = build_sectioned_symbolic_layer(document.paper_id, runtime, debug_records, paper_dir)
    runtime = section_result["runtime_records"]
    debug_records = section_result["debug_records"]

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in debug_records:
        by_page[int(row.get("page") or -1)].append(row)
    for page, rows in by_page.items():
        write_jsonl(page_debug_dir / f"page_{max(0, page):03d}.records.debug.jsonl", rows)
        _write_status(
            page_status_dir / f"page_{max(0, page):03d}.status.json",
            {
                "paper_id": document.paper_id,
                "page": page,
                "page_status": "complete",
                "valid_record_count": len(rows),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    source_dist = dict(Counter(str(record.get("source_type") or "") for record in runtime))
    section_type_dist = dict(Counter(str(section.get("section_type") or "") for section in section_result.get("sections", [])))
    status_value = "complete" if runtime else "failed"
    status = {
        "paper_id": document.paper_id,
        "parser": document.backend,
        "transcription_backend": document.backend,
        "transcription_backend_version": document.backend_version,
        "transcription_backend_config": document.backend_config,
        "standardizer_version": STANDARDIZER_VERSION,
        "page_count": document.page_count,
        "parsed_pages": len(page_counts),
        "complete_pages": len(page_counts),
        "partial_pages": 0,
        "failed_pages": 0,
        "missing_page_count": missing_page_count,
        "valid_record_count": len(runtime),
        "record_count": len(runtime),
        "section_count": int(section_result.get("section_count") or 0),
        "source_type_distribution": source_dist,
        "section_type_distribution": section_type_dist,
        "status": status_value,
        "cache_status": "created",
        "warnings": list(document.warnings or []),
        "backend_output_dir": str(paper_dir),
        "raw_output_path": document.raw_output_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbolic_records_runtime_path": str(paper_dir / "symbolic_records.runtime.jsonl"),
        "symbolic_records_debug_path": str(paper_dir / "symbolic_records.debug.jsonl"),
        "sections_path": str(paper_dir / "sections.jsonl"),
        "section_tree_path": str(paper_dir / "section_tree.json"),
    }
    _write_status(paper_dir / "artifact_status.json", status)
    _write_status(
        paper_dir / "document_manifest.json",
        {
            "paper_id": document.paper_id,
            "pdf_path": document.pdf_path,
            "transcription_backend": document.backend,
            "page_count": document.page_count,
            "raw_output_path": document.raw_output_path,
            "missing_page_count": missing_page_count,
            "pages": [{"page": page, "record_count": count} for page, count in sorted(page_counts.items())],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _write_status(
        paper_dir / "extraction_report.json",
        {
            "paper_id": document.paper_id,
            "transcription_backend": document.backend,
            "transcription_backend_version": document.backend_version,
            "transcription_backend_config": document.backend_config,
            "backend_output_dir": str(paper_dir),
            "record_count": len(runtime),
            "source_type_distribution": source_dist,
            "section_count": int(section_result.get("section_count") or 0),
            "section_type_distribution": section_type_dist,
            "missing_page_count": missing_page_count,
            "warnings": list(document.warnings or []),
            "status": status_value,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return status
