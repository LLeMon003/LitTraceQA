"""Cycle 1 deterministic source-native solvers for locked synthetic recipes."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


FACT_PATTERN = re.compile(r"^What exact source text is recorded by the (.+) in paper (.+) on page (\d+)\?$")
TABLE_PATTERN = re.compile(
    r"^In paper (.+), page (\d+), (.+), what value appears at zero-based row (\d+), column (\d+)\?$"
)


def _value(row: dict[str, Any], key: str, maximum: int = 160) -> str | None:
    value = row.get(key)
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value if 2 <= len(value) <= maximum else None


@dataclass(frozen=True)
class Decision:
    status: str
    text: str | None
    confidence: float
    source: dict[str, Any] | None


class StructuredSourceIndex:
    def __init__(self, fact_rows: Iterable[dict[str, Any]], table_rows: Iterable[dict[str, Any]]):
        self.facts = self._index_facts(fact_rows)
        self.cells = self._index_cells(table_rows)

    @staticmethod
    def _unique(values: list[dict[str, Any]]) -> dict[str, Any] | None:
        if len(values) != 1:
            return None
        return values[0]

    def _index_facts(self, rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any] | None]:
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            paper, page, object_type = row.get("paper_id"), row.get("page"), row.get("object_type")
            value = _value(row, "normalized_value")
            if row.get("ambiguity_status") == "accepted" and isinstance(paper, str) and isinstance(page, int) and isinstance(object_type, str) and value:
                grouped[(paper, page, object_type)].append({
                    "text": value, "paper_id": paper, "page": page, "object_type": object_type,
                    "object_uid": row.get("object_uid"), "source_hash": row.get("source_hash"),
                    "record_hash": row.get("record_hash"),
                })
        return {key: self._unique(values) for key, values in grouped.items()}

    def _index_cells(self, rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int, str, int, int], dict[str, Any] | None]:
        grouped: dict[tuple[str, int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            paper, page, table = row.get("paper_id"), row.get("page"), row.get("evaluator_visible_table_id")
            row_index, column_index = row.get("row_index"), row.get("column_index")
            value = _value(row, "normalized_cell_value", 100)
            if (
                row.get("provenance_status") == "accepted" and not row.get("is_column_header")
                and not row.get("is_row_header") and isinstance(paper, str) and isinstance(page, int)
                and isinstance(table, str) and isinstance(row_index, int) and isinstance(column_index, int) and value
            ):
                grouped[(paper, page, table, row_index, column_index)].append({
                    "text": value, "paper_id": paper, "page": page, "table_id": table,
                    "row_index": row_index, "column_index": column_index, "source_hash": row.get("source_hash"),
                    "record_hash": row.get("record_hash"),
                })
        return {key: self._unique(values) for key, values in grouped.items()}

    def solve(self, question: str) -> Decision:
        match = FACT_PATTERN.match(question)
        if match:
            object_type, paper, page = match.groups()
            source = self.facts.get((paper, int(page), object_type))
            return self._decision(source)
        match = TABLE_PATTERN.match(question)
        if match:
            paper, page, table, row_index, column_index = match.groups()
            source = self.cells.get((paper, int(page), table, int(row_index), int(column_index)))
            return self._decision(source)
        return Decision("unsupported_question", None, 0.0, None)

    @staticmethod
    def _decision(source: dict[str, Any] | None) -> Decision:
        if source is None:
            return Decision("not_unique_or_missing", None, 0.0, None)
        return Decision("accepted", source["text"], 1.0, source)


def score(records: Iterable[dict[str, Any]], index: StructuredSourceIndex) -> dict[str, Any]:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    accepted = 0
    for record in records:
        recipe, split = record["recipe"], record["split"]
        decision = index.solve(record["question"])
        key = f"{split}:{recipe}"
        totals[key] += 1
        if decision.status == "accepted":
            accepted += 1
        if decision.text == record["answer"]["text"]:
            correct[key] += 1
    return {
        "accepted": accepted,
        "total": sum(totals.values()),
        "by_split_recipe": {
            key: {"total": total, "correct": correct[key], "exact_match": correct[key] / total}
            for key, total in sorted(totals.items())
        },
    }
