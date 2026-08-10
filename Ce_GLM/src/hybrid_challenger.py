"""Cycle 2 provenance-coalescing hybrid source retrieval."""

from __future__ import annotations

from typing import Any

from src.structured_challenger import StructuredSourceIndex


class ProvenanceCoalescingHybridIndex(StructuredSourceIndex):
    """Accept duplicate retrieval paths only when their grounded values agree."""

    @staticmethod
    def _unique(values: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not values:
            return None
        normalized = {value["text"] for value in values}
        if len(normalized) != 1:
            return None
        return sorted(values, key=lambda value: str(value.get("record_hash", "")))[0]
