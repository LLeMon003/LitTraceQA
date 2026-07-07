from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse error in {file_path} at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row in {file_path} at line {line_no} is not an object")
            rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def find_official_file(official_dir: str | Path, filename: str) -> Path:
    root = Path(official_dir)
    candidates = [root / "data" / filename, root / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find official file {filename}. Checked: {checked}")


def _normalize_answer_types(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        raw = [str(part).strip() for part in value]
    else:
        raw = []
    allowed = {"freeform", "multiple_choice", "table"}
    return [item for item in raw if item in allowed]


def _find_first(input_example: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in input_example:
            return input_example.get(name)
    for container_name in ("answer_contract", "multiple_choice", "answer_constraints", "question_metadata"):
        container = input_example.get(container_name)
        if isinstance(container, dict):
            for name in names:
                if name in container:
                    return container.get(name)
    return None


def _normalize_options(value: Any) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, text in value.items():
            option_key = str(key).strip()
            if option_key:
                options.append({"key": option_key, "text": str(text or "")})
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                key = item.get("key") or item.get("label") or item.get("option") or item.get("id")
                text = item.get("text") or item.get("value") or item.get("content") or item.get("answer") or ""
                if key is None and len(item) == 1:
                    key, text = next(iter(item.items()))
                option_key = str(key or chr(ord("A") + index)).strip()
                options.append({"key": option_key, "text": str(text or "")})
            else:
                options.append({"key": chr(ord("A") + index), "text": str(item or "")})
    return [option for option in options if option.get("key")]


def _normalize_table_schema(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        if isinstance(value.get("columns"), list):
            value = value.get("columns")
        elif isinstance(value.get("schema"), list):
            value = value.get("schema")
    if not isinstance(value, list):
        return None
    columns: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name") or item.get("column") or item.get("key")
        else:
            name = item
        text = str(name or "").strip()
        if text:
            columns.append(text)
    return columns or None


def _find_gold_answer_container(validation_example: dict[str, Any] | None, answer_type: str) -> dict[str, Any]:
    if not isinstance(validation_example, dict):
        return {}
    answer = validation_example.get("answer")
    if not isinstance(answer, dict):
        return {}
    container = answer.get(answer_type)
    return container if isinstance(container, dict) else {}


def _find_validation_options(validation_example: dict[str, Any] | None) -> Any:
    multiple_choice = _find_gold_answer_container(validation_example, "multiple_choice")
    if "options" in multiple_choice:
        return multiple_choice.get("options")
    return _find_first(validation_example or {}, ["options", "choices", "multiple_choice_options", "answer_options"])


def _find_validation_table_schema(validation_example: dict[str, Any] | None) -> Any:
    table = _find_gold_answer_container(validation_example, "table")
    if "schema" in table:
        return table.get("schema")
    if "table_schema" in table:
        return table.get("table_schema")
    return _find_first(validation_example or {}, ["table_schema", "columns", "schema"])


def extract_answer_contract(input_example: dict[str, Any], validation_example: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract inference-available answer constraints from a validation input row.

    validation_example may be the matching row from validation.jsonl, but only
    answer-shape constraints are copied from it: multiple-choice options and
    table schema. Gold answer values, evidence, and gold paper ids must never be
    exposed through this contract.
    """
    answer_types = _normalize_answer_types(input_example.get("answer_types"))
    option_value = _find_first(input_example, ["options", "choices", "multiple_choice_options", "answer_options"])
    if option_value is None:
        option_value = _find_validation_options(validation_example)
    table_value = _find_first(input_example, ["table_schema", "columns"])
    if table_value is None:
        table_value = _find_validation_table_schema(validation_example)
    table_schema = _normalize_table_schema(table_value)
    options = _normalize_options(option_value)
    return {
        "query_id": input_example.get("query_id"),
        "answer_types": answer_types,
        "multiple_choice": {
            "options": options,
            "output_field": "gold",
            "output_rule": "choose exactly one option key from the provided options",
            "options_available": bool(options),
        },
        "table": {
            "table_schema": table_schema,
            "output_rule": "use exactly the provided column names if table answer is required",
        },
        "freeform": {
            "output_rule": "produce concise text only if freeform is listed in answer_types",
        },
    }
