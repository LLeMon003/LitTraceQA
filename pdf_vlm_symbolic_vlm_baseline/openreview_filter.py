from __future__ import annotations

from typing import Any


def is_openreview_paper(metadata: dict[str, Any]) -> bool:
    openreview_id = str(metadata.get("openreview_id") or "").strip()
    if openreview_id:
        return True
    for key in ("source_url", "pdf_url", "url"):
        value = str(metadata.get(key) or "").lower()
        if "openreview.net" in value or "openreview" in value:
            return True
    for key, value in metadata.items():
        key_lower = str(key).lower()
        if "openreview" in key_lower and str(value or "").strip():
            return True
        if isinstance(value, str) and "openreview.net" in value.lower():
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and "openreview.net" in item.lower():
                    return True
                if isinstance(item, dict) and is_openreview_paper(item):
                    return True
        if isinstance(value, dict) and is_openreview_paper(value):
            return True
    return False


def filter_openreview_metadata(metadata_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in metadata_records:
        if is_openreview_paper(record):
            skipped.append(record)
        else:
            kept.append(record)
    return kept, skipped
