from __future__ import annotations

import re
from typing import Any


_NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[%xX])?")
_MARKDOWN_SEPARATOR_RE = re.compile(r"^\s*:?-{2,}:?\s*$")


def _clean_cell(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_clean_cell(cell) for cell in stripped.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(_MARKDOWN_SEPARATOR_RE.match(cell.replace(" ", "")) for cell in cells)


def _has_numeric_cell(cells: list[str]) -> bool:
    return any(_NUMERIC_RE.search(cell) for cell in cells)


def _looks_like_table_line(line: str) -> bool:
    return line.count("|") >= 2


def _extract_pipe_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or not _looks_like_table_line(line):
            continue
        cells = _split_pipe_row(line)
        if len(cells) < 2 or _is_separator_row(cells):
            continue
        rows.append(cells)
    return rows


def _target_width(rows: list[list[str]]) -> int:
    if not rows:
        return 0
    widths = [len(row) for row in rows]
    return max(widths, key=lambda width: (widths.count(width), width))


def _pad_row(row: list[str], width: int) -> list[str]:
    if len(row) >= width:
        return row[:width]
    return row + [""] * (width - len(row))


def _expand_group_header(cells: list[str], width: int, next_width: int | None = None) -> list[str]:
    if len(cells) == width:
        return cells
    if len(cells) <= 1:
        return _pad_row(cells, width)

    # Common compact form:
    #   Output Length | Calculations | Memory
    #   New prompt | Reused prompt | New prompt | Reused prompt
    #   0 | ...
    # The first cell is the row-header column, while the remaining cells are
    # group headers spanning the following subcolumns.
    if next_width == width - 1 and len(cells) < width:
        groups = cells[1:]
        remaining = width - 1
        expanded = [cells[0]]
        for index, group in enumerate(groups):
            groups_left = len(groups) - index
            span = max(1, remaining // max(1, groups_left))
            expanded.extend([group] * span)
            remaining -= span
        return _pad_row(expanded, width)

    expanded: list[str] = []
    remaining = width
    for index, cell in enumerate(cells):
        cells_left = len(cells) - index
        span = max(1, remaining // max(1, cells_left))
        expanded.extend([cell] * span)
        remaining -= span
    return _pad_row(expanded, width)


def _normalise_header_rows(header_rows: list[list[str]], width: int, data_width: int | None) -> list[list[str]]:
    normalised: list[list[str]] = []
    for index, row in enumerate(header_rows):
        if index > 0 and len(row) == width - 1:
            row = [""] + row
        next_width = len(header_rows[index + 1]) if index + 1 < len(header_rows) else data_width
        normalised.append(_expand_group_header(row, width, next_width))
    return normalised


def _column_names(header_rows: list[list[str]], width: int) -> list[str]:
    columns: list[str] = []
    for column_index in range(width):
        parts: list[str] = []
        for header_row in header_rows:
            value = _clean_cell(header_row[column_index]) if column_index < len(header_row) else ""
            if value and value not in parts:
                parts.append(value)
        columns.append(" / ".join(parts) if parts else f"column_{column_index + 1}")
    return columns


def _cell_records(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row in rows:
        row_index = int(row.get("row_index") or 0)
        row_label = str(row.get("row_label") or "")
        values = row.get("values") if isinstance(row.get("values"), dict) else {}
        for column_index, column in enumerate(columns):
            cells.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "row_label": row_label,
                    "column": column,
                    "value": values.get(column, ""),
                }
            )
    return cells


def table_text_to_structure(text: str, expected_output_columns: list[str] | None = None) -> dict[str, Any] | None:
    """Map compact one-dimensional table text into a prompt-facing 2D structure.

    This does not alter the stored symbolic artifact. It is a conservative
    prompt adapter for common VLM-1 table transcriptions, especially pipe-style
    markdown and compact multi-level headers.
    """

    pipe_rows = _extract_pipe_rows(text)
    if len(pipe_rows) < 2:
        return None

    width = _target_width(pipe_rows)
    if width < 2:
        return None

    first_data_index = next((index for index, row in enumerate(pipe_rows) if _has_numeric_cell(row)), None)
    if first_data_index is None:
        first_data_index = 1
    if first_data_index <= 0:
        first_data_index = 1

    raw_header_rows = pipe_rows[:first_data_index]
    raw_data_rows = pipe_rows[first_data_index:]
    data_width = len(raw_data_rows[0]) if raw_data_rows else width
    header_rows = _normalise_header_rows(raw_header_rows, width, data_width)
    columns = _column_names(header_rows, width)

    structured_rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(raw_data_rows, start=1):
        values_list = _pad_row(raw_row, width)
        values = {column: values_list[column_index] for column_index, column in enumerate(columns)}
        structured_rows.append(
            {
                "row_index": row_index,
                "row_label": values_list[0] if values_list else "",
                "values": values,
            }
        )

    expected_columns = [str(column) for column in expected_output_columns or [] if str(column)]
    return {
        "format": "derived_from_compact_pipe_text",
        "header_rows": header_rows,
        "columns": columns,
        "rows": structured_rows,
        "cells": _cell_records(structured_rows, columns),
        "expected_output_columns": expected_columns,
        "parse_warnings": [
            "Derived only for VLM-2 prompting; original symbolic text remains authoritative.",
            "Column spans are inferred heuristically when compact headers omit repeated cells.",
        ],
    }
