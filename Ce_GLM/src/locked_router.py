"""Frozen parent/challenger router for the Cycle 2 source-native solver."""

from __future__ import annotations

from typing import Any

from src.structured_challenger import StructuredSourceIndex


ROUTER_VERSION = "ver3-source-native-router.v1"
MINIMUM_CONFIDENCE = 0.90


def route_input(record: dict[str, Any], index: StructuredSourceIndex) -> dict[str, Any]:
    question = record.get("question")
    if not isinstance(question, str):
        raise ValueError("input record has no question string")
    decision = index.solve(question)
    selected = decision.status == "accepted" and decision.confidence >= MINIMUM_CONFIDENCE and decision.source is not None
    return {
        "query_id": record.get("query_id"),
        "router_version": ROUTER_VERSION,
        "selected": selected,
        "reason": "complete_unique_source_native_answer" if selected else decision.status,
        "confidence": decision.confidence,
        "source": decision.source if selected else None,
    }
