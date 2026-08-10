from __future__ import annotations

import re
from typing import Any, Iterable


_NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[%xX])?")
_MARKDOWN_SEPARATOR_RE = re.compile(r"^\s*:?-{2,}:?\s*$")
_UNIT_SUFFIX_RE = re.compile(
    r"^(.*?)\s*(?:%|x|s|ms|h|min|sec|secs|seconds|gb|mb|kb|tb|gib|mib|kib|g|m|k|b|tokens?|it|iters?|iterations?|epochs?|steps?)\s*$",
    re.IGNORECASE,
)


def normalize_schema_key(value: Any) -> str:
    """Compact lowercase alphanumeric form of a schema column name."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def schema_tokens(value: Any) -> set[str]:
    """Meaningful lowercase tokens of a column name (drops single letters)."""
    return {token for token in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(token) >= 2}


def match_schema_column(model_key: Any, schema_columns: Iterable[str]) -> str | None:
    """Best-matching official schema column for a model-provided key.

    Exact compact match wins; otherwise the schema column with the largest
    token overlap is chosen (e.g. schema ``ndcg_at_10`` vs model ``NDCG@10``).
    Ties are rejected so a generic key is never mis-assigned.
    """
    columns = [str(column) for column in schema_columns if str(column)]
    if not columns:
        return None
    target = normalize_schema_key(model_key)
    exact = [column for column in columns if normalize_schema_key(column) == target]
    if len(exact) == 1:
        return exact[0]
    target_tokens = schema_tokens(model_key)
    if not target_tokens:
        return None
    scored = sorted(
        ((len(target_tokens & schema_tokens(column)), column) for column in columns),
        reverse=True,
    )
    if not scored or scored[0][0] < 1:
        return None
    best_score = scored[0][0]
    tied = [column for score, column in scored if score == best_score]
    return tied[0] if len(tied) == 1 else None


def remap_row_values(values: dict[str, Any], schema_columns: Iterable[str]) -> dict[str, Any]:
    """Rebuild a model-provided table row under the official schema columns.

    Keys that match an official column (exactly or by token overlap) keep their
    value under the official name; unknown keys are dropped; official columns
    with no value are kept as ``None`` so callers can decide row completeness.
    """
    columns = [str(column) for column in schema_columns if str(column)]
    by_exact = {normalize_schema_key(column): column for column in columns}
    remapped: dict[str, Any] = {}
    consumed: set[str] = set()
    unmatched: list[tuple[str, Any]] = []
    for raw_key, value in (values or {}).items():
        column = by_exact.get(normalize_schema_key(raw_key))
        if column is not None and column not in consumed:
            remapped[column] = value
            consumed.add(column)
        else:
            unmatched.append((str(raw_key), value))
    for raw_key, value in unmatched:
        match = match_schema_column(raw_key, [column for column in columns if column not in consumed])
        if match is not None:
            remapped[match] = value
            consumed.add(match)
    return {column: remapped.get(column) for column in columns}


def coerce_number_cell(value: Any) -> Any:
    """Coerce a numeric cell to int/float, stripping a trailing unit suffix."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value or "").strip()
    if not text:
        return value
    match = _UNIT_SUFFIX_RE.match(text)
    candidate = match.group(1) if match else text
    candidate = candidate.replace(",", "").strip()
    try:
        if re.fullmatch(r"[-+]?\d+", candidate):
            return int(candidate)
        return float(candidate)
    except ValueError:
        return value


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


def _docling_cell_text(cell: dict[str, Any]) -> str:
    return _clean_cell(cell.get("text"))


def docling_table_to_structure(payload: dict[str, Any], caption: str = "") -> dict[str, Any] | None:
    """Preserve Docling's two-dimensional table model for downstream prompts.

    Markdown is useful for inspection, but it loses merged-header spans and the
    header/row flags that Docling already recovered.  This adapter keeps those
    facts in a compact, JSON-safe shape.  It deliberately does not infer cells
    that Docling did not produce.
    """
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    raw_cells = data.get("table_cells") if isinstance(data.get("table_cells"), list) else []
    try:
        row_count = int(data.get("num_rows") or 0)
        column_count = int(data.get("num_cols") or 0)
    except (TypeError, ValueError):
        return None
    if row_count <= 0 or column_count <= 0 or not raw_cells:
        return None

    grid = [["" for _ in range(column_count)] for _ in range(row_count)]
    header_grid = [[False for _ in range(column_count)] for _ in range(row_count)]
    row_header_grid = [[False for _ in range(column_count)] for _ in range(row_count)]
    cells: list[dict[str, Any]] = []
    for raw in raw_cells:
        if not isinstance(raw, dict):
            continue
        try:
            r0 = max(0, int(raw.get("start_row_offset_idx") or 0))
            r1 = min(row_count, int(raw.get("end_row_offset_idx") or r0 + 1))
            c0 = max(0, int(raw.get("start_col_offset_idx") or 0))
            c1 = min(column_count, int(raw.get("end_col_offset_idx") or c0 + 1))
        except (TypeError, ValueError):
            continue
        if r1 <= r0 or c1 <= c0:
            continue
        text = _docling_cell_text(raw)
        is_header = bool(raw.get("column_header"))
        is_row_header = bool(raw.get("row_header"))
        for row in range(r0, r1):
            for column in range(c0, c1):
                # Replication is only a display projection. ``cells`` below
                # remains the authoritative span-preserving representation.
                if not grid[row][column]:
                    grid[row][column] = text
                header_grid[row][column] = header_grid[row][column] or is_header
                row_header_grid[row][column] = row_header_grid[row][column] or is_row_header
        cells.append(
            {
                "row_index": r0,
                "column_index": c0,
                "row_span": r1 - r0,
                "column_span": c1 - c0,
                "text": text,
                "column_header": is_header,
                "row_header": is_row_header,
                "row_section": bool(raw.get("row_section")),
            }
        )

    header_indices = [index for index, row in enumerate(header_grid) if any(row)]
    header_rows = [grid[index] for index in header_indices]
    columns = _column_names(header_rows, column_count) if header_rows else [f"column_{index + 1}" for index in range(column_count)]
    data_start = max(header_indices, default=-1) + 1
    rows: list[dict[str, Any]] = []
    for row_index in range(data_start, row_count):
        values_list = grid[row_index]
        if not any(values_list):
            continue
        row_label = next(
            (values_list[column] for column in range(column_count) if row_header_grid[row_index][column] and values_list[column]),
            values_list[0],
        )
        rows.append(
            {
                "row_index": row_index,
                "row_label": row_label,
                "values": {columns[column]: values_list[column] for column in range(column_count)},
                "row_section": any(cell["row_section"] for cell in cells if cell["row_index"] == row_index),
            }
        )

    caption_text = _clean_cell(caption)
    footnotes = []
    if caption_text:
        # Keep the original caption intact and separately expose clauses that
        # carry marker/Note conditions.  This is extraction, not a summary.
        footnotes = [
            clause.strip()
            for clause in re.split(r"(?<=[.;])\s+", caption_text)
            if re.search(r"(?:\bnote\b|\*|†|‡|§|\b(?:where|denotes|means|indicates)\b)", clause, re.IGNORECASE)
        ]
    return {
        "format": "docling_table_cells_v1",
        "caption": caption_text,
        "header_rows": header_rows,
        "columns": columns,
        "rows": rows,
        "cells": cells,
        "footnotes": footnotes,
        "dimensions": {"rows": row_count, "columns": column_count},
        "parse_warnings": ["Structure is projected directly from Docling table_cells; merged-cell spans are preserved in cells."],
    }


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
