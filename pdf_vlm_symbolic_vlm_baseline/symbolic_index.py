from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_symbolic_index(records: list[dict[str, Any]], output_path: str | Path | None = None) -> dict[str, Any]:
    valid = [r for r in records if r.get("validation_status") != "rejected"]
    paper_id = str(valid[0].get("paper_id", "")) if valid else ""
    pages: dict[str, Any] = {}
    objects: dict[str, Any] = {}
    edges: list[dict[str, Any]] = []
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_page[int(record.get("page") or 0)].append(record)
        label = record.get("label")
        if label:
            by_label[str(label).lower()].append(record)
    for page, page_records in sorted(by_page.items()):
        page_id = f"{paper_id}::page_{page:03d}"
        edges.append({"from": paper_id, "to": page_id, "type": "paper_has_page"})
        counter = Counter(str(r.get("record_type", "unknown")) for r in page_records)
        pages[str(page)] = {
            "records": [str(r.get("global_record_id")) for r in page_records],
            "record_types": dict(counter),
        }
        for record in sorted(page_records, key=lambda r: int(r.get("reading_order") or 0)):
            gid = str(record.get("global_record_id"))
            objects[gid] = {
                "paper_id": record.get("paper_id"),
                "page": record.get("page"),
                "record_type": record.get("record_type"),
                "source_type": record.get("source_type"),
                "label": record.get("label"),
                "text": record.get("text"),
                "bbox_1000": record.get("bbox_1000"),
                "reading_order": record.get("reading_order"),
            }
            edges.append({"from": page_id, "to": gid, "type": "page_contains_record"})
    for label_records in by_label.values():
        if len(label_records) < 2:
            continue
        ids = [str(r.get("global_record_id")) for r in label_records]
        for src in ids:
            for dst in ids:
                if src != dst:
                    edges.append({"from": src, "to": dst, "type": "same_label_family"})
    for page_records in by_page.values():
        figures = [r for r in page_records if r.get("record_type") in {"figure", "table"}]
        captions = [r for r in page_records if r.get("record_type") in {"figure_caption", "table_caption"}]
        for caption in captions:
            matches = [obj for obj in figures if obj.get("label") and obj.get("label") == caption.get("label")]
            if not matches and figures:
                matches = sorted(figures, key=lambda r: abs(int(r.get("reading_order") or 0) - int(caption.get("reading_order") or 0)))[:1]
            for obj in matches:
                edges.append(
                    {
                        "from": str(caption.get("global_record_id")),
                        "to": str(obj.get("global_record_id")),
                        "type": "caption_describes_object",
                    }
                )
    index = {"paper_id": paper_id, "pages": pages, "objects": objects, "edges": edges}
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index
