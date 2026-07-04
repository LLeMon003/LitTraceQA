from __future__ import annotations

from typing import Any

from metadata_only_baseline.parser import (
    extract_json_object,
    make_fallback_prediction,
    normalize_prediction,
    validate_prediction_shape,
)


def strip_internal_grounding(prediction: dict[str, Any]) -> dict[str, Any]:
    official = {
        "query_id": prediction.get("query_id"),
        "gold_papers": prediction.get("gold_papers", []),
        "evidence": [],
        "answer": prediction.get("answer", {}),
    }
    for item in prediction.get("evidence", []) if isinstance(prediction.get("evidence"), list) else []:
        if not isinstance(item, dict):
            continue
        locator = item.get("locator", {})
        if not isinstance(locator, dict):
            locator = {}
        official_locator = {k: v for k, v in locator.items() if k not in {"chunk_id", "visual_id"}}
        official["evidence"].append(
            {
                "paper_id": item.get("paper_id", ""),
                "source_type": item.get("source_type", ""),
                "locator": official_locator,
            }
        )
    return official


__all__ = [
    "extract_json_object",
    "make_fallback_prediction",
    "normalize_prediction",
    "strip_internal_grounding",
    "validate_prediction_shape",
]

