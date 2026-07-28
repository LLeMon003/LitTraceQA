from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
from pathlib import Path
from typing import Any

from .data_io import _json_safe, write_jsonl
from .pdf_text_span_extractor import is_likely_section_heading


def _clean_title(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", value)
    value = re.sub(r"^Appendix\s+([A-Z])\s*", r"Appendix \1 ", value, flags=re.IGNORECASE).strip()
    return value


def normalize_section_type(title: str) -> str:
    lower = _clean_title(title).lower()
    if "abstract" in lower:
        return "abstract"
    if "intro" in lower:
        return "introduction"
    if any(term in lower for term in ["related", "background", "preliminar"]):
        return "related_work"
    if any(term in lower for term in ["method", "approach", "model", "framework", "training", "objective", "optimization"]):
        return "method"
    if any(term in lower for term in ["experiment", "evaluation", "implementation", "setup"]):
        return "experiments"
    if any(term in lower for term in ["result", "analysis", "ablation"]):
        return "results"
    if "discussion" in lower:
        return "discussion"
    if "conclusion" in lower:
        return "conclusion"
    if any(term in lower for term in ["references", "bibliography"]):
        return "references"
    if any(term in lower for term in ["appendix", "supplementary"]):
        return "appendix"
    if "front matter" in lower:
        return "front_matter"
    return "unknown"


def _section_level(text: str) -> int:
    del text
    # is_likely_section_heading only admits first-level whitelist headings.
    return 1


def is_section_header(record: dict[str, Any]) -> bool:
    text = re.sub(r"\s+", " ", str(record.get("text") or "")).strip()
    return is_likely_section_heading(text)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:60] or "section"


def _runtime_projection(record: dict[str, Any]) -> dict[str, Any]:
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
        "table_structure",
        "reading_order",
        "document_order",
    ]
    return {key: record.get(key) for key in keys}


def _tree(sections: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(section["section_id"]): {**section, "children": []} for section in sections}
    roots: list[dict[str, Any]] = []
    for section in by_id.values():
        parent = section.get("parent_section_id")
        if parent and parent in by_id:
            by_id[parent]["children"].append(section)
        else:
            roots.append(section)
    return {"sections": roots}


def build_sectioned_symbolic_layer(
    paper_id: str,
    records: list[dict[str, Any]],
    debug_records: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    ordered = sorted(records, key=lambda r: (int(r.get("page") or 0), int(r.get("reading_order") or 0), str(r.get("record_id") or "")))
    debug_by_gid = {str(r.get("global_record_id") or ""): dict(r) for r in debug_records}
    front = {
        "paper_id": paper_id,
        "section_id": "sec_000",
        "parent_section_id": None,
        "section_title": "Front Matter",
        "section_type": "front_matter",
        "level": 1,
        "section_path": ["Front Matter"],
        "page_start": None,
        "page_end": None,
        "document_order_start": None,
        "document_order_end": None,
        "record_count": 0,
        "source_type_counts": {},
        "labels": [],
        "created_by": "section_header_parser",
    }
    sections: list[dict[str, Any]] = [front]
    stack: list[dict[str, Any]] = [front]
    enriched_runtime: list[dict[str, Any]] = []
    enriched_debug: list[dict[str, Any]] = []
    section_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    section_debug: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for doc_order, raw_record in enumerate(ordered, start=1):
        record = dict(raw_record)
        record["document_order"] = doc_order
        header = is_section_header(record)
        if str(record.get("record_type") or "") == "section_header" and not header:
            record["record_type"] = "paragraph"
            record["label"] = None
        if header:
            record["record_type"] = "section_header"
            record["label"] = record.get("label") or record.get("text")
            title = _clean_title(record.get("text") or record.get("label") or "Untitled Section")
            level = _section_level(str(record.get("text") or ""))
            # All accepted headings are top-level by design.
            stack.clear()
            parent = None
            section_id = f"sec_{len(sections):03d}"
            section_path = [*(parent.get("section_path") if parent else []), title]
            section = {
                "paper_id": paper_id,
                "section_id": section_id,
                "parent_section_id": parent.get("section_id") if parent else None,
                "section_title": title,
                "section_type": normalize_section_type(title),
                "level": level,
                "section_path": section_path,
                "page_start": record.get("page"),
                "page_end": record.get("page"),
                "document_order_start": doc_order,
                "document_order_end": doc_order,
                "record_count": 0,
                "source_type_counts": {},
                "labels": [],
                "created_by": "section_header_parser",
            }
            sections.append(section)
            stack.append(section)
        current = stack[-1] if stack else front
        section_fields = {
            "section_id": current["section_id"],
            "section_title": current["section_title"],
            "section_level": current["level"],
            "section_path": current["section_path"],
            "section_type": current["section_type"],
        }
        record.update(section_fields)
        runtime = _runtime_projection(record)
        enriched_runtime.append(runtime)
        current["record_count"] = int(current.get("record_count") or 0) + 1
        current["page_start"] = min([p for p in [current.get("page_start"), record.get("page")] if p is not None], default=record.get("page"))
        current["page_end"] = max([p for p in [current.get("page_end"), record.get("page")] if p is not None], default=record.get("page"))
        current["document_order_start"] = min([p for p in [current.get("document_order_start"), doc_order] if p is not None], default=doc_order)
        current["document_order_end"] = max([p for p in [current.get("document_order_end"), doc_order] if p is not None], default=doc_order)
        counts = Counter(current.get("source_type_counts") or {})
        counts[str(record.get("source_type") or "")] += 1
        current["source_type_counts"] = dict(counts)
        if record.get("label"):
            labels = list(current.get("labels") or [])
            if record.get("label") not in labels:
                labels.append(record.get("label"))
            current["labels"] = labels[:50]
        debug = debug_by_gid.get(str(record.get("global_record_id") or ""), dict(record))
        debug.update(section_fields)
        debug["document_order"] = doc_order
        enriched_debug.append(debug)
        section_records[str(current["section_id"])].append(runtime)
        section_debug[str(current["section_id"])].append(debug)

    write_jsonl(out / "sections.jsonl", sections)
    (out / "section_tree.json").write_text(json.dumps(_json_safe(_tree(sections)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out / "symbolic_records.runtime.jsonl", enriched_runtime)
    write_jsonl(out / "symbolic_records.debug.jsonl", enriched_debug)
    section_root = out / "section_records"
    for section in sections:
        sec_id = str(section["section_id"])
        sec_dir = section_root / f"{sec_id}_{_slug(str(section['section_title']))}"
        write_jsonl(sec_dir / "records.runtime.jsonl", section_records.get(sec_id, []))
        write_jsonl(sec_dir / "records.debug.jsonl", section_debug.get(sec_id, []))
    return {
        "section_count": len(sections),
        "runtime_records": enriched_runtime,
        "debug_records": enriched_debug,
        "sections": sections,
    }
