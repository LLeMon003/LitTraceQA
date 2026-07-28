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
from ..symbolic_schema import canonical_citation_id
from .base import TranscribedDocument, TranscribedElement


STANDARDIZER_VERSION = "v10_structured_tables_and_equation_zero_aliases"


_MATH_TEXT_FALLBACK_RE = re.compile(
    r"(?:[=≈≤≥]|∑|∏|∇|∂|∫).*(?:[α-ωΑ-Ωπθβγλσ]|\b(?:log|exp|min|max|alpha|beta|theta|gamma|lambda)\b)"
    r"|(?:[α-ωΑ-Ωπθβγλσ]|\b(?:log|exp|min|max|alpha|beta|theta|gamma|lambda)\b).*(?:[=≈≤≥]|∑|∏|∇|∂|∫)",
    re.IGNORECASE,
)


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
            loc["citation_id"] = canonical_citation_id(label)
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
    # Docling's normalized ``element_type`` is sometimes the generic ``text``
    # even when its serialized item is a formula or code block.  The backend
    # type carries the structural distinction required for source typing.
    raw_type = str(element.raw_backend_type or element.element_type or "").lower()
    text = str(element.text or "")
    rules: list[str] = []
    if raw_type in {"table", "document_table"}:
        return "table", "table", ["docling_table_to_table_record"]
    if raw_type in {"picture", "image", "figure"}:
        return "figure", "figure", ["docling_picture_to_figure_record"]
    # A prose paragraph can legitimately mention ``Equation (3)``.  Treating
    # that reference as an equation artifact produces a false visual object
    # and can displace the real formula in downstream selection.  Formula and
    # equation are structural Docling element types; an Algorithm label is
    # additionally accepted for Docling's code-block representation.
    if raw_type in {"formula", "equation"}:
        return "equation", "equation_algorithm", ["docling_formula_to_equation_record"]
    if raw_type == "code":
        label = _normalize_label(text, "equation_algorithm") or ""
        if label.lower().startswith("algorithm"):
            return "algorithm", "equation_algorithm", ["docling_code_algorithm_label_rule"]
    if is_likely_section_heading(text):
        return "section_header", "text_span", ["heading_rule"]
    return "paragraph", "text_span", ["default_text_to_text_span"]


def _is_math_text_fallback(text: str) -> bool:
    """Detect a likely display formula that Docling emitted as prose.

    Require both an operator and a mathematical identifier; ordinary prose that
    merely cites an equation never satisfies this condition.
    """
    compact = re.sub(r"\s+", " ", text or "").strip()
    return 16 <= len(compact) <= 2200 and bool(_MATH_TEXT_FALLBACK_RE.search(compact))


def _reference_label(index: int) -> str:
    return f"Reference {index}"


def _split_reference_entries(records: list[dict[str, Any]], debug_by_gid: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    new_debug = dict(debug_by_gid)
    references_active = False
    ref_index = 0
    last_reference_output_index: int | None = None
    for record in records:
        text = str(record.get("text") or "")
        if record.get("record_type") == "section_header" and re.search(r"\b(?:References|Bibliography)\b", text, re.IGNORECASE):
            references_active = True
            last_reference_output_index = None
            output.append(record)
            continue
        if references_active and record.get("source_type") == "text_span" and record.get("record_type") == "section_header" and not re.search(r"\b(?:References|Bibliography)\b", text, re.IGNORECASE):
            references_active = False
            last_reference_output_index = None
        if references_active and record.get("source_type") == "text_span" and record.get("record_type") == "paragraph" and len(text) >= 20:
            # Prefer the document's printed bibliography number. Docling often
            # splits two-column entries and page continuations into separate
            # paragraphs, so a synthetic paragraph counter is not a stable
            # citation locator.
            number = re.match(r"^\s*(?:\[\s*(\d{1,4})\s*\]|(\d{1,4})\s*[.)])\s*", text)
            if number:
                ref_index = int(next(group for group in number.groups() if group))
            elif last_reference_output_index is not None and re.match(r"^\s*(?:[a-z]|[,;:)\]])", text):
                previous = output[last_reference_output_index]
                previous["text"] = f"{str(previous.get('text') or '').rstrip()} {text}".strip()
                previous_gid = str(previous.get("global_record_id") or "")
                debug = dict(new_debug.get(previous_gid, previous))
                debug["text"] = previous["text"]
                rules = list(debug.get("standardization_rules") or [])
                rules.append("references_continuation_merged")
                debug["standardization_rules"] = rules
                new_debug[previous_gid] = debug
                continue
            else:
                ref_index += 1
            copied = dict(record)
            label = _reference_label(ref_index)
            copied["record_type"] = "reference_entry"
            copied["source_type"] = "citation_context"
            copied["label"] = label
            copied["locator"] = {"page": copied.get("page"), "citation_id": canonical_citation_id(label)}
            gid = str(copied.get("global_record_id") or "")
            debug = dict(new_debug.get(gid, copied))
            debug.update({"record_type": "reference_entry", "source_type": "citation_context", "label": label, "locator": copied["locator"]})
            rules = list(debug.get("standardization_rules") or [])
            rules.append("references_section_paragraph_to_reference_entry")
            debug["standardization_rules"] = rules
            new_debug[gid] = debug
            output.append(copied)
            last_reference_output_index = len(output) - 1
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
        "table_structure": item.get("table_structure"),
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
    figure_caption_candidates: list[tuple[int, TranscribedElement, str]] = []
    figure_labels: set[str] = set()
    table_caption_candidates: list[tuple[int, TranscribedElement, str]] = []
    table_labels: set[str] = set()
    formula_labels_by_page: dict[int, set[str]] = defaultdict(set)
    structural_equations_by_page: dict[int, list[tuple[dict[str, Any], TranscribedElement]]] = defaultdict(list)
    math_text_fallbacks_by_page: dict[int, list[tuple[int, TranscribedElement]]] = defaultdict(list)
    equation_reference_candidates: list[tuple[int, TranscribedElement, str]] = []
    missing_page_count = 0
    page_counts: Counter[int] = Counter()
    for index, element in enumerate(document.elements, start=1):
        record_type, source_type, rules = _record_type_and_source(element)
        label = element.label or _normalize_label(element.text, source_type)
        if source_type == "equation_algorithm" and record_type == "equation" and not label:
            # Official annotations use Equation 0 for a display formula with no
            # printed equation number.  This is a real formula, not a prose
            # equation mention.
            label = "Equation 0"
            rules = [*rules, "unnumbered_display_formula_to_equation_0"]
        if record_type == "section_header":
            label = str(element.text or "").strip()
        page = element.page
        if page is None:
            missing_page_count += 1
        loc = _locator(page, source_type, label)
        text = str(element.text or "").strip()
        if not text:
            continue
        item = {
            "record_type": record_type,
            "source_type": source_type,
            "label": label,
            "locator": loc,
            "text": text,
            "table_structure": (
                element.raw_backend_payload.get("_littraceqa_table_structure")
                if source_type == "table" and isinstance(element.raw_backend_payload, dict)
                else None
            ),
        }
        record = _runtime_record(document.paper_id, page, index, item)
        runtime.append(record)
        if source_type == "figure" and label:
            figure_labels.add(label)
        if source_type == "table" and label:
            table_labels.add(label)
        if source_type == "equation_algorithm" and label and page is not None:
            formula_labels_by_page[int(page)].add(label)
        if source_type == "equation_algorithm" and record_type == "equation" and page is not None:
            structural_equations_by_page[int(page)].append((record, element))
        if source_type == "text_span":
            if page is not None and _is_math_text_fallback(text):
                math_text_fallbacks_by_page[int(page)].append((index, element))
            for reference in re.findall(r"\b(?:Equation|Eq\.)\s*\(?\s*([A-Za-z]?\d+[A-Za-z0-9.\-]*)\s*\)?", text, re.IGNORECASE):
                equation_reference_candidates.append((index, element, f"Equation {reference.rstrip('.,;:')}"))
        if (
            str(element.raw_backend_type or "").lower() == "caption"
            and (caption_label := _normalize_label(text, "figure"))
        ):
            figure_caption_candidates.append((index, element, caption_label))
        if (
            str(element.raw_backend_type or "").lower() == "caption"
            and (caption_label := _normalize_label(text, "table"))
        ):
            table_caption_candidates.append((index, element, caption_label))
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

    # Evaluation uses Equation 0 as a generic locator when an annotation does
    # not identify a printed formula number.  When a page contains genuine
    # display equations but no unnumbered one, expose a page-level alias with
    # the original formula text and a warning.  The alias never overwrites the
    # printed Equation n record.
    for page, equations in structural_equations_by_page.items():
        if "Equation 0" in formula_labels_by_page.get(page, set()) or not equations:
            continue
        source_record, source_element = equations[0]
        item = {
            "record_type": "equation",
            "source_type": "equation_algorithm",
            "label": "Equation 0",
            "locator": _locator(page, "equation_algorithm", "Equation 0"),
            "text": f"Generic page-level equation locator for an unnumbered official reference. Displayed formula source: {str(source_record.get('text') or '')}",
        }
        record = _runtime_record(document.paper_id, page, len(runtime) + 1, item)
        runtime.append(record)
        formula_labels_by_page[page].add("Equation 0")
        debug_by_gid[str(record["global_record_id"])] = {
            **record,
            "bbox": _bbox_list(source_element.bbox),
            "crop_path": None,
            "backend": document.backend,
            "raw_backend_type": source_element.raw_backend_type,
            "raw_backend_payload_path": document.raw_output_path,
            "raw_backend_element_id": source_element.element_id,
            "raw_backend_payload": source_element.raw_backend_payload,
            "standardization_rules": ["equation_zero_page_alias_from_display_formula"],
            "warnings": ["generic_equation_zero_locator_alias", "printed_equation_number_preserved_separately"],
            "page_missing": False,
        }

    # Some two-column PDFs cause Docling to merge a display formula into a
    # surrounding text block.  Keep this as a last-resort Equation 0 anchor
    # only when the text itself has clear mathematical syntax.
    for page, candidates in math_text_fallbacks_by_page.items():
        if "Equation 0" in formula_labels_by_page.get(page, set()) or not candidates:
            continue
        original_index, element = candidates[0]
        item = {
            "record_type": "equation",
            "source_type": "equation_algorithm",
            "label": "Equation 0",
            "locator": _locator(page, "equation_algorithm", "Equation 0"),
            "text": f"Unnumbered mathematical expression recovered from a Docling text block. {str(element.text or '').strip()}",
        }
        record = _runtime_record(document.paper_id, page, len(runtime) + 1, item)
        runtime.append(record)
        formula_labels_by_page[page].add("Equation 0")
        debug_by_gid[str(record["global_record_id"])] = {
            **record,
            "bbox": _bbox_list(element.bbox),
            "crop_path": None,
            "backend": document.backend,
            "raw_backend_type": element.raw_backend_type,
            "raw_backend_payload_path": document.raw_output_path,
            "raw_backend_element_id": element.element_id,
            "raw_backend_payload": element.raw_backend_payload,
            "standardization_rules": ["math_rich_text_to_equation_zero_fallback"],
            "warnings": ["displayed_equation_body_not_structurally_separated"],
            "page_missing": False,
            "fallback_source_record_index": original_index,
        }

    # If Docling dropped a displayed formula but retained an explicit in-page
    # reference, expose a separate pointer record.  The original paragraph
    # stays text_span; this pointer is clearly marked as body-unavailable so
    # it can provide a strict locator without being mistaken for formula text.
    for original_index, element, label in equation_reference_candidates:
        page = element.page
        if page is None or label in formula_labels_by_page.get(int(page), set()):
            continue
        item = {
            "record_type": "equation_reference",
            "source_type": "equation_algorithm",
            "label": label,
            "locator": _locator(page, "equation_algorithm", label),
            "text": f"{label} is referenced in this passage; the displayed equation body was not emitted by Docling. {str(element.text or '').strip()}",
        }
        record = _runtime_record(document.paper_id, page, len(runtime) + 1, item)
        runtime.append(record)
        formula_labels_by_page[int(page)].add(label)
        debug_by_gid[str(record["global_record_id"])] = {
            **record,
            "bbox": _bbox_list(element.bbox),
            "crop_path": None,
            "backend": document.backend,
            "raw_backend_type": element.raw_backend_type,
            "raw_backend_payload_path": document.raw_output_path,
            "raw_backend_element_id": element.element_id,
            "raw_backend_payload": element.raw_backend_payload,
            "standardization_rules": ["explicit_equation_reference_pointer"],
            "warnings": ["displayed_equation_body_unavailable"],
            "page_missing": False,
            "reference_source_record_index": original_index,
        }

    # Docling occasionally keeps an explicit Figure n caption as text while it
    # fails to emit the associated picture object. A caption with a literal
    # figure label is independently valid symbolic evidence, so expose it as a
    # figure record rather than losing the source type entirely.
    for original_index, element, label in figure_caption_candidates:
        if label in figure_labels:
            continue
        page = element.page
        item = {
            "record_type": "figure",
            "source_type": "figure",
            "label": label,
            "locator": _locator(page, "figure", label),
            "text": str(element.text or "").strip(),
        }
        record = _runtime_record(document.paper_id, page, len(runtime) + 1, item)
        runtime.append(record)
        figure_labels.add(label)
        debug_by_gid[str(record["global_record_id"])] = {
            **record,
            "bbox": _bbox_list(element.bbox),
            "crop_path": None,
            "backend": document.backend,
            "raw_backend_type": element.raw_backend_type,
            "raw_backend_payload_path": document.raw_output_path,
            "raw_backend_element_id": element.element_id,
            "raw_backend_payload": element.raw_backend_payload,
            "standardization_rules": ["unclaimed_figure_caption_to_figure_record"],
            "warnings": list(element.warnings or []),
            "page_missing": element.page is None,
            "caption_source_record_index": original_index,
        }

    # The same Docling failure occurs for tables: a caption is emitted as
    # regular text but no structured table element is returned. Retain the
    # literal label and page as a fallback table anchor; the caption remains
    # provenance, not an invented table transcription.
    for original_index, element, label in table_caption_candidates:
        if label in table_labels:
            continue
        page = element.page
        item = {
            "record_type": "table_caption",
            "source_type": "table",
            "label": label,
            "locator": _locator(page, "table", label),
            "text": str(element.text or "").strip(),
        }
        record = _runtime_record(document.paper_id, page, len(runtime) + 1, item)
        runtime.append(record)
        table_labels.add(label)
        debug_by_gid[str(record["global_record_id"])] = {
            **record,
            "bbox": _bbox_list(element.bbox),
            "crop_path": None,
            "backend": document.backend,
            "raw_backend_type": element.raw_backend_type,
            "raw_backend_payload_path": document.raw_output_path,
            "raw_backend_element_id": element.element_id,
            "raw_backend_payload": element.raw_backend_payload,
            "standardization_rules": ["unclaimed_table_caption_to_table_record"],
            "warnings": list(element.warnings or []),
            "page_missing": element.page is None,
            "caption_source_record_index": original_index,
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
