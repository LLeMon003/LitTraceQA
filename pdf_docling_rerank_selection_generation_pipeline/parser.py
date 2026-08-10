from __future__ import annotations

import json
import os
import re
from html import unescape
from typing import Any

from .data_io import extract_answer_contract
from .symbolic_schema import (
    OFFICIAL_EVIDENCE_SOURCE_TYPES,
    canonical_citation_id,
    canonicalize_locator,
    to_official_source_type,
)
from .table_structure import coerce_number_cell, remap_row_values
from .task_structure import as_source_types, derive_task_structure


FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    fence_match = FENCE_RE.fullmatch(content) or (FENCE_RE.match(content) if content.startswith("```") else None)
    if fence_match:
        content = fence_match.group(1).strip()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippet = content[start : index + 1]
                    obj = json.loads(snippet)
                    if not isinstance(obj, dict):
                        raise ValueError("Extracted JSON is not an object")
                    return obj
    raise ValueError("No complete JSON object found")


def _requested_answer(input_example: dict[str, Any], source_answer: dict[str, Any] | None = None) -> dict[str, Any]:
    answer_types = input_example.get("answer_types") or []
    source_answer = source_answer or {}
    answer: dict[str, Any] = {}
    if "freeform" in answer_types:
        freeform = source_answer.get("freeform") if isinstance(source_answer.get("freeform"), dict) else {}
        answer["freeform"] = {"text": str(freeform.get("text", ""))}
    if "multiple_choice" in answer_types:
        multiple = source_answer.get("multiple_choice") if isinstance(source_answer.get("multiple_choice"), dict) else {}
        gold = str(multiple.get("gold", "") or "").strip()
        answer["multiple_choice"] = {"gold": gold}
    if "table" in answer_types:
        table = source_answer.get("table") if isinstance(source_answer.get("table"), dict) else {}
        rows = table.get("rows", [])
        answer["table"] = {"rows": rows if isinstance(rows, list) else []}
    return answer


def _canonical_table_text(value: Any) -> str:
    text = unescape(str(value or "")).strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _table_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _canonical_table_text(value).lower())


def _question_table_aliases(question: str) -> list[str]:
    aliases: list[str] = []
    stop = {
        "What",
        "Which",
        "Across",
        "Given",
        "IPC",
        "FID",
        "CIFAR",
        "Tiny",
        "ImageNet",
        "ModelNet",
        "GenEval",
        "CVPR",
        "NAACL",
        "ACL",
        "ICLR",
        "ICML",
        "NeurIPS",
    }
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b", question):
        if token in stop or len(token) < 2:
            continue
        if (
            any(char.isupper() for char in token[1:])
            or "-" in token
            or any(char.isdigit() for char in token)
            or token.isupper()
        ):
            aliases.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        key = _table_key(alias)
        if key and key not in seen:
            seen.add(key)
            deduped.append(alias)
    return deduped


def _row_matches_alias(row_key: Any, aliases: list[str]) -> bool:
    row_norm = _table_key(row_key)
    if not row_norm:
        return False
    for alias in aliases:
        alias_norm = _table_key(alias)
        if not alias_norm:
            continue
        if row_norm == alias_norm or row_norm in alias_norm or alias_norm in row_norm:
            return True
        if len(alias_norm) >= 4 and row_norm[:4] == alias_norm[:4]:
            return True
    return False


def _metadata_title_map(candidate_records: list[dict[str, Any]] | None) -> dict[str, str]:
    title_by_key: dict[str, str] = {}
    for candidate in candidate_records or []:
        if not isinstance(candidate, dict):
            continue
        title = str(candidate.get("title") or "").strip()
        if not title:
            continue
        title_by_key.setdefault(_table_key(title), title)
        paper_id = str(candidate.get("paper_id") or "").strip()
        if paper_id:
            title_by_key.setdefault(_table_key(paper_id), title)
    return title_by_key


def _canonicalize_paper_title_cell(value: Any, title_by_key: dict[str, str]) -> Any:
    if not title_by_key:
        return value
    key = _table_key(value)
    if key and key in title_by_key:
        return title_by_key[key]
    return value


def _table_row_key_columns(input_example: dict[str, Any], columns: list[str]) -> list[str]:
    """Use the public schema's full row key, not an implicit first column."""
    schema = input_example.get("table_schema") if isinstance(input_example, dict) else None
    keys = [
        str(column.get("name") or "")
        for column in schema or []
        if isinstance(column, dict) and column.get("is_row_key") and str(column.get("name") or "") in columns
    ]
    return keys or columns[:1]


def _table_row_key(row: dict[str, Any], columns: list[str]) -> tuple[str, ...]:
    return tuple(_table_key(row.get(column)) for column in columns)


def _structured_table_value(
    question: str, row_key: Any, value_column: str, records: list[dict[str, Any]] | None
) -> str | None:
    """Return one unambiguous selected-table cell for a named row."""
    ipc = re.search(r"\bipc\s*[=:]\s*(\d+)\b", question, re.IGNORECASE)
    steps = re.search(r"\b(\d+)-step\b", question, re.IGNORECASE)
    question_key = _table_key(question)
    value_terms = {term for term in re.findall(r"[a-z0-9]+", value_column.lower()) if len(term) >= 2}
    scored: list[tuple[int, str]] = []
    for record in records or []:
        structure = record.get("table_structure") if isinstance(record.get("table_structure"), dict) else {}
        columns = structure.get("columns") if isinstance(structure.get("columns"), list) else []
        source_rows = [row for row in structure.get("rows") or [] if isinstance(row, dict)]
        exact_row = any(_table_key(source_row.get("row_label")) == _table_key(row_key) for source_row in source_rows)
        for source_row in source_rows:
            source_key = _table_key(source_row.get("row_label"))
            if not (source_key == _table_key(row_key) if exact_row else _row_matches_alias(row_key, [str(source_row.get("row_label") or "")])):
                continue
            source_values = source_row.get("values") if isinstance(source_row.get("values"), dict) else {}
            if steps and not any(
                re.search(r"\b(?:nfe|steps?)\b", str(column), re.IGNORECASE)
                and str(source_values.get(column) or "").strip() == steps.group(1)
                for column in columns
            ):
                continue
            for column in columns:
                column_text = str(column)
                dataset = _table_key(re.split(r"/\s*(?:venue\s*)?ipc\b", column_text, maxsplit=1, flags=re.IGNORECASE)[0])
                if ipc and dataset and len(dataset) >= 5 and dataset in question_key and re.search(rf"\bipc\s*[=:]\s*{re.escape(ipc.group(1))}\b", column_text, re.IGNORECASE):
                    score = 100
                else:
                    column_terms = {term for term in re.findall(r"[a-z0-9]+", column_text.lower()) if len(term) >= 2}
                    overlap = len(value_terms & column_terms)
                    if not overlap:
                        continue
                    score = overlap * 10 + len(column_terms & {term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) >= 3})
                value = str(source_values.get(column) or "").strip()
                if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*±\s*\d+(?:\.\d+)?)?", value):
                    scored.append((score, value))
    if not scored:
        return None
    best = max(score for score, _value in scored)
    values = {value for score, value in scored if score == best}
    return next(iter(values)) if len(values) == 1 else None


def _structured_condition_rows(
    question: str, row_key_column: str, value_column: str, existing_keys: set[str], records: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Recover explicitly named ``w/ condition`` rows from one selected table."""
    question_key = _table_key(question)
    entities = [_table_key(alias) for alias in _question_table_aliases(question) if len(_table_key(alias)) >= 4]
    recovered: dict[str, dict[str, Any]] = {}
    for record in records or []:
        structure = record.get("table_structure") if isinstance(record.get("table_structure"), dict) else {}
        for source_row in structure.get("rows") or []:
            if not isinstance(source_row, dict):
                continue
            label = str(source_row.get("row_label") or "")
            suffix = re.search(r"\b(w/\s*.+)$", label, re.IGNORECASE)
            label_key = _table_key(label)
            if not suffix or not any(entity in label_key for entity in entities):
                continue
            output_key = suffix.group(1).strip()
            key = _table_key(output_key)
            condition_key = _table_key(re.sub(r"^w/\s*", "", output_key, flags=re.IGNORECASE))
            if not key or key in existing_keys or not condition_key or condition_key not in question_key:
                continue
            value = _structured_table_value(question, label, value_column, [record])
            if value is not None:
                recovered[key] = {row_key_column: output_key, value_column: value}
    return list(recovered.values())


def postprocess_table_rows(
    rows: list[Any],
    schema: list[str] | None,
    input_example: dict[str, Any],
    candidate_records: list[dict[str, Any]] | None = None,
    row_evidence_records: list[list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    columns = [str(column) for column in (schema or []) if str(column)]
    if not columns:
        return [row for row in rows if isinstance(row, dict)], []
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    records_by_row_id: dict[int, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        cleaned = {column: row.get(column) for column in columns}
        if any(value is not None and str(value).strip() for value in cleaned.values()):
            normalized.append(cleaned)
            if row_evidence_records and index < len(row_evidence_records):
                records_by_row_id[id(cleaned)] = row_evidence_records[index]

    if not normalized:
        return [], errors

    row_key_columns = _table_row_key_columns(input_example, columns)
    row_key_column = row_key_columns[0]
    fake_condition = re.search(r"\bfake\s*(\d+)\s*-\s*contaminated\b", str(input_example.get("question") or ""), re.IGNORECASE)
    if fake_condition:
        required_fake = f"fake{fake_condition.group(1)}"
        condition_rows = [row for row in normalized if required_fake in _table_key(row.get(row_key_column))]
        if condition_rows:
            normalized = condition_rows
            errors.append("table_rows_filtered_to_explicit_fake_condition")
    title_by_key = _metadata_title_map(candidate_records)
    if "Paper Title" in columns and title_by_key:
        for row in normalized:
            if "Paper Title" in row:
                new_title = _canonicalize_paper_title_cell(row.get("Paper Title"), title_by_key)
                if new_title != row.get("Paper Title"):
                    row["Paper Title"] = new_title
                    errors.append("table_paper_title_row_key_canonicalized")
    elif row_key_column:
        budget_labels = {
            _table_key(match.group(1)): f"{match.group(1)} ({match.group(2)})"
            for match in re.finditer(
                r"\b([A-Za-z][A-Za-z0-9-]{1,})\s*\(\s*with\s+(\d+(?:\.\d+)?[kKmMgG])\s+training\s+budget\s*\)",
                str(input_example.get("question") or ""),
            )
        }
        aliases = sorted(
            _question_table_aliases(str(input_example.get("question") or "")),
            key=lambda value: len(_table_key(value)),
            reverse=True,
        )
        for row in normalized:
            old_value = row.get(row_key_column)
            if fake_condition and required_fake in _table_key(old_value):
                continue
            # Generation occasionally emits a local paper id in a generic
            # ``paper`` column.  It is not a question alias (e.g. ``ICCV``),
            # and rewriting it before de-duplication collapses distinct rows.
            if row_key_column.casefold() == "paper" and re.fullmatch(r"[A-Za-z]+20\d{2}_\d+", str(old_value).strip()):
                detail = next((str(row.get(column) or "") for column in columns if column.casefold() == "detail"), "")
                label = re.match(r"([A-Za-z][A-Za-z0-9]*)(?=[\s-]|$)", detail)
                if label:
                    row[row_key_column] = label.group(1)
                    errors.append("table_internal_paper_id_replaced_by_detail_label")
                continue
            old_key = _table_key(old_value)
            if not old_key:
                continue
            explicit_budget = re.fullmatch(
                r"\s*(.+?)\s*\(\s*with\s+(\d+(?:\.\d+)?[kKmMgG])\s+training\s+budget\s*\)\s*",
                str(old_value),
                re.IGNORECASE,
            )
            if explicit_budget:
                row[row_key_column] = f"{explicit_budget.group(1).strip()} ({explicit_budget.group(2)})"
                errors.append("table_explicit_budget_row_key_compacted")
                continue
            if old_key in budget_labels:
                row[row_key_column] = budget_labels[old_key]
                errors.append("table_question_budget_row_key_canonicalized")
                continue
            if old_key in {_table_key(label) for label in budget_labels.values()}:
                continue
            suffix = re.search(r"\b(w/\s*.+)$", str(old_value), re.IGNORECASE)
            suffix_matches = [alias for alias in aliases if suffix and _row_matches_alias(suffix.group(1), [alias])]
            if len(suffix_matches) == 1:
                new_value = suffix.group(1).strip()
                if _table_key(new_value) != old_key:
                    row[row_key_column] = new_value
                    errors.append("table_question_condition_suffix_canonicalized")
                continue
            # A short row label ("ECM") cannot safely stand in for a more
            # specific budgeted target ("ECM-XL (102.4M)").  Only remove
            # decorations from an already-present alias in that constrained
            # case; ordinary method-name questions retain legacy matching.
            matches = (
                [alias for alias in aliases if _table_key(alias) in old_key]
                if budget_labels else [alias for alias in aliases if _row_matches_alias(old_value, [alias])]
            )
            if len(matches) != 1:
                continue
            new_value = matches[0]
            new_key = _table_key(new_value)
            if new_key in budget_labels:
                new_value = budget_labels[new_key]
                new_key = _table_key(new_value)
            if new_key != old_key or str(new_value) != str(old_value):
                row[row_key_column] = new_value
                errors.append("table_question_alias_row_key_canonicalized")

    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for row in normalized:
        key = _table_row_key(row, row_key_columns)
        if not any(key):
            errors.append("table_row_without_row_key_removed")
            continue
        if key in seen_keys:
            errors.append("table_duplicate_row_removed")
            continue
        seen_keys.add(key)
        deduped.append(row)

    column_types = _table_column_types(input_example)
    # A row's echoed evidence ref is the strongest numeric grounding signal.
    # Recover only an unambiguous structured cell; otherwise retain Qwen's
    # value and let the established global fallback below handle two-column rows.
    for row in deduped:
        records = records_by_row_id.get(id(row))
        if not records:
            continue
        for value_column in columns:
            if value_column in row_key_columns or column_types.get(value_column) != "number":
                continue
            value = _structured_table_value(
                str(input_example.get("question") or ""), row.get(row_key_column), value_column, records
            )
            if value is not None and str(row.get(value_column) or "") != value:
                row[value_column] = value
                errors.append("table_numeric_value_recovered_from_row_evidence")

    # A clean selected Docling table is more reliable than a model copying the
    # wrong adjacent IPC column.  Keep this narrow: one row-key/value schema,
    # an explicit IPC in the question, and exactly one matching table cell.
    if len(columns) == 2 and row_key_column:
        value_column = next(column for column in columns if column != row_key_column)
        for row in deduped:
            value = _structured_table_value(str(input_example.get("question") or ""), row.get(row_key_column), value_column, records_by_row_id.get(id(row)))
            if value is None:
                value = _structured_table_value(
                    str(input_example.get("question") or ""), row.get(row_key_column), value_column, candidate_records
                )
            if value is not None and str(row.get(value_column) or "") != value:
                row[value_column] = value
                errors.append("table_value_recovered_from_selected_structure")
        missing_rows = _structured_condition_rows(
            str(input_example.get("question") or ""), row_key_column, value_column,
            {_table_key(row.get(row_key_column)) for row in deduped}, candidate_records,
        )
        if missing_rows:
            deduped.extend(missing_rows)
            errors.append("table_rows_recovered_from_selected_structure")

    # Official cells must honour the schema column types: number columns are
    # coerced to JSON numbers (submission validator requires int/float) and
    # string columns keep their verbatim text so "14.70" is not silently
    # rewritten to 14.7.
    for row in deduped:
        for column, value in row.items():
            if isinstance(value, str):
                row[column] = re.sub(r"\s*±\s*", "±", value)
            if column_types:
                column_type = column_types.get(column)
                if column_type == "number":
                    row[column] = coerce_number_cell(row[column])
                elif column_type == "boolean" and isinstance(row[column], str):
                    lowered = str(row[column]).strip().lower()
                    if lowered in {"true", "false"}:
                        row[column] = lowered == "true"
    return deduped, errors


def validate_prediction_against_answer_contract(
    prediction: dict[str, Any],
    answer_contract: dict[str, Any],
    input_example: dict[str, Any] | None = None,
    candidate_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    allowed_types = [str(item) for item in answer_contract.get("answer_types", []) if str(item) in {"freeform", "multiple_choice", "table"}]
    answer = prediction.get("answer") if isinstance(prediction.get("answer"), dict) else {}
    cleaned: dict[str, Any] = {}
    option_text_by_key = {
        str(option.get("key") or "").strip(): str(option.get("text") or "")
        for option in (answer_contract.get("multiple_choice") or {}).get("options", [])
        if isinstance(option, dict) and str(option.get("key") or "").strip()
    }
    for key in list(answer.keys()):
        if key not in allowed_types:
            errors.append("answer_type_extra_fields_removed")
    if "freeform" in allowed_types:
        raw_freeform = answer.get("freeform")
        raw = raw_freeform if isinstance(raw_freeform, dict) else {}
        if isinstance(raw_freeform, str):
            text = raw_freeform
            errors.append("freeform_string_normalized")
        elif isinstance(raw, dict):
            text = raw.get("text") or raw.get("answer") or raw.get("value") or ""
        else:
            text = ""
        if "freeform" not in answer:
            errors.append("missing_required_answer_type")
        cleaned["freeform"] = {"text": unescape(str(text or ""))}
    if "multiple_choice" in allowed_types:
        raw = answer.get("multiple_choice") if isinstance(answer.get("multiple_choice"), dict) else {}
        if "multiple_choice" not in answer:
            errors.append("missing_required_answer_type")
        gold = str(
            raw.get("gold") or raw.get("answer") or raw.get("predicted_answer_id") or ""
            if isinstance(raw, dict)
            else ""
        ).strip()
        option_keys = set(option_text_by_key)
        inferred = _infer_multiple_choice_key(gold, cleaned.get("freeform"), option_text_by_key)
        if inferred:
            if inferred != gold:
                errors.append("multiple_choice_key_inferred")
            gold = inferred
        if gold and option_keys and gold not in option_keys:
            inferred = _infer_multiple_choice_key(gold, cleaned.get("freeform"), option_text_by_key)
            if inferred:
                errors.append("multiple_choice_key_inferred")
                gold = inferred
            else:
                errors.append("invalid_multiple_choice_option")
                gold = ""
        elif gold and not option_keys:
            errors.append("missing_multiple_choice_options")
        cleaned["multiple_choice"] = {"gold": gold}
        if (
            "freeform" in allowed_types
            and not str((cleaned.get("freeform") or {}).get("text") or "").strip()
            and gold
            and option_text_by_key.get(gold)
        ):
            cleaned["freeform"] = {"text": option_text_by_key[gold]}
            errors.append("freeform_filled_from_multiple_choice_option")
    if "table" in allowed_types:
        raw = answer.get("table") if isinstance(answer.get("table"), dict) else {}
        if "table" not in answer:
            errors.append("missing_required_answer_type")
        rows = raw.get("rows", []) if isinstance(raw, dict) else []
        rows = rows if isinstance(rows, list) else []
        schema = (answer_contract.get("table") or {}).get("table_schema")
        plan_rows = _rows_from_table_answer_plan(prediction.get("table_answer_plan"), schema)
        plan_row_records: list[list[dict[str, Any]]] | None = None
        if plan_rows:
            rows = plan_rows
            errors.append("table_rows_filled_from_table_answer_plan")
            records_by_ref: dict[str, list[dict[str, Any]]] = {}
            for record in candidate_records or []:
                ref = str(record.get("evidence_ref") or "") if isinstance(record, dict) else ""
                if ref:
                    records_by_ref.setdefault(ref, []).append(record)
            plan_row_records = []
            for item in prediction.get("table_answer_plan") or []:
                values = item.get("values") if isinstance(item, dict) and isinstance(item.get("values"), dict) else item
                if isinstance(values, dict) and any(value is not None and str(value).strip() for value in values.values()):
                    ref = str(item.get("row_evidence_ref") or item.get("evidence_ref") or "") if isinstance(item, dict) else ""
                    plan_row_records.append(records_by_ref.get(ref, []))
            if len(plan_row_records) != len(plan_rows):
                plan_row_records = None
        if isinstance(schema, list) and schema:
            schema_names = [str(column) for column in schema if str(column)]
            normalized_rows = []
            for row in rows:
                source_row = row if isinstance(row, dict) else {}
                normalized_rows.append(remap_row_values(source_row, schema_names))
            rows = normalized_rows
        rows, table_errors = postprocess_table_rows(
            rows, schema if isinstance(schema, list) else [], input_example or {}, candidate_records, plan_row_records
        )
        errors.extend(table_errors)
        expected_row_count = int((answer_contract.get("table") or {}).get("expected_row_count") or 0)
        if expected_row_count > 0 and len(rows) > expected_row_count:
            rows = rows[:expected_row_count]
            errors.append("table_rows_limited_to_validation_shape_count")
        cleaned["table"] = {"rows": rows}
    prediction["answer"] = cleaned
    return prediction, errors


def _normalize_text_for_choice(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    return text


def _infer_multiple_choice_key(
    raw_value: Any,
    freeform: Any,
    option_text_by_key: dict[str, str],
) -> str:
    if not option_text_by_key:
        upper = str(raw_value or "").strip().upper()
        match = re.search(r"\b(?:option|answer|choice)?\s*([A-D])\b", upper)
        return match.group(1) if match else (upper if upper in {"A", "B", "C", "D"} else "")

    allowed_by_upper = {str(key).upper(): str(key) for key in option_text_by_key}
    upper = str(raw_value or "").strip().upper()
    if upper in allowed_by_upper:
        return allowed_by_upper[upper]
    match = re.search(r"\b(?:option|answer|choice)?\s*([A-Z])\b", upper)
    if match and match.group(1) in allowed_by_upper:
        return allowed_by_upper[match.group(1)]

    candidate_texts = [raw_value]
    if isinstance(freeform, dict):
        candidate_texts.append(freeform.get("text"))
    for candidate in candidate_texts:
        normalized_candidate = _normalize_text_for_choice(candidate)
        if not normalized_candidate:
            continue
        for key, option_text in option_text_by_key.items():
            normalized_option = _normalize_text_for_choice(option_text)
            if not normalized_option:
                continue
            if (
                normalized_candidate == normalized_option
                or normalized_candidate in normalized_option
                or normalized_option in normalized_candidate
            ):
                return key
    return ""


def _structured_option_text(value: Any) -> bool:
    text = str(value or "")
    return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*=", text) or re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*:", text))


def _option_support_score(option_text: str, evidence_text: str, question: str) -> float:
    normalized_option = re.sub(r"[^a-z0-9.]+", " ", str(option_text or "").lower()).strip()
    normalized_evidence = re.sub(r"[^a-z0-9.]+", " ", str(evidence_text or "").lower()).strip()
    score = 0.0
    if normalized_option and normalized_option in normalized_evidence:
        score += 10.0 + min(5.0, len(normalized_option) / 20.0)
    option_tokens = _text_tokens(option_text)
    evidence_tokens = _text_tokens(evidence_text)
    question_tokens = _text_tokens(question)
    if option_tokens:
        score += (len(option_tokens & evidence_tokens) / len(option_tokens)) * 3.0
        non_question = option_tokens - question_tokens
        if non_question:
            score += (len(non_question & evidence_tokens) / len(non_question)) * 2.0
    for number in re.findall(r"[-+]?\d+(?:\.\d+)?", str(option_text or "")):
        if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", str(evidence_text or "")):
            score += 2.0
    return score


def _maybe_rerank_structured_multiple_choice(
    prediction: dict[str, Any],
    answer_contract: dict[str, Any],
    input_example: dict[str, Any],
    selected_evidence: list[dict[str, Any]] | None,
) -> list[str]:
    answer_types = [str(item) for item in answer_contract.get("answer_types", [])]
    if "multiple_choice" not in answer_types:
        return []
    options = [
        option
        for option in (answer_contract.get("multiple_choice") or {}).get("options", [])
        if isinstance(option, dict) and str(option.get("key") or "").strip()
    ]
    if not options or not any(_structured_option_text(option.get("text")) for option in options):
        return []
    evidence_text = " ".join(str(item.get("text") or "") for item in selected_evidence or [] if isinstance(item, dict))
    if not evidence_text.strip():
        return []
    scored = [
        (
            _option_support_score(str(option.get("text") or ""), evidence_text, str(input_example.get("question") or "")),
            str(option.get("key") or "").strip(),
            str(option.get("text") or ""),
        )
        for option in options
    ]
    scored.sort(reverse=True)
    if not scored:
        return []
    best_score, best_key, best_text = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -999.0
    if not best_key or not _structured_option_text(best_text) or best_score < 10.0 or best_score - second_score < 4.0:
        return []
    answer = prediction.setdefault("answer", {})
    mc = answer.get("multiple_choice") if isinstance(answer.get("multiple_choice"), dict) else {}
    old_key = str(mc.get("gold") or "").strip()
    if best_key == old_key:
        return []
    answer["multiple_choice"] = {"gold": best_key}
    if "freeform" in answer_types:
        answer["freeform"] = {"text": best_text}
    return ["multiple_choice_structured_option_reranked"]


def _selected_evidence_map(selected_evidence: list[dict[str, Any]] | None) -> dict[tuple[str, str], dict[str, set[Any]]]:
    mapping: dict[tuple[str, str], dict[str, set[Any]]] = {}
    def add_entry(paper_id: str, source_type: str, page: Any, grounding: Any = None, locator: dict[str, Any] | None = None) -> None:
        if not paper_id or source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            return
        key = (paper_id, source_type)
        entry = mapping.setdefault(key, {"pages": set(), "table_id": set(), "figure_id": set(), "equation_id": set(), "algorithm_id": set(), "citation_id": set(), "reference_id": set()})
        try:
            entry["pages"].add(int(page))
        except (TypeError, ValueError):
            pass
        if isinstance(grounding, dict):
            label_type = str(grounding.get("type") or "")
            value = str(grounding.get("value") or "").strip()
            if label_type in {"citation_id", "reference_id"}:
                value = canonical_citation_id(value)
            if label_type in entry and value:
                entry[label_type].add(value)
                if label_type == "citation_id":
                    entry["reference_id"].add(value)
        if isinstance(locator, dict):
            locator = canonicalize_locator(source_type, locator)
            for id_key in ("table_id", "figure_id", "equation_id", "algorithm_id", "citation_id", "reference_id"):
                value = locator.get(id_key)
                if value is None or value == "":
                    continue
                if id_key in {"citation_id", "reference_id"}:
                    value = canonical_citation_id(value)
                entry[id_key].add(value)
                if id_key == "citation_id":
                    entry["reference_id"].add(value)

    for evidence in selected_evidence or []:
        if not isinstance(evidence, dict):
            continue
        paper_id = str(evidence.get("paper_id") or "")
        source_type = str(evidence.get("source_type") or "")
        selected_locator = evidence.get("locator") if isinstance(evidence.get("locator"), dict) else {}
        add_entry(
            paper_id,
            source_type,
            evidence.get("page", selected_locator.get("page")),
            evidence.get("grounding_label"),
            selected_locator,
        )
        for hint in evidence.get("source_type_hints") or []:
            if not isinstance(hint, dict):
                continue
            hint_source = to_official_source_type(source_type=str(hint.get("source_type") or "")) or ""
            hint_locator = hint.get("locator") if isinstance(hint.get("locator"), dict) else {}
            add_entry(paper_id, hint_source, hint_locator.get("page", evidence.get("page")), None, hint_locator)
    return mapping


def validate_prediction_locators_against_selected_evidence(
    prediction: dict[str, Any],
    selected_evidence: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    evidence_map = _selected_evidence_map(selected_evidence)
    normalized: list[dict[str, Any]] = []
    raw_evidence = prediction.get("evidence") if isinstance(prediction.get("evidence"), list) else []
    for item in raw_evidence:
        if not isinstance(item, dict):
            errors.append("locator_validation_error")
            continue
        paper_id = str(item.get("paper_id") or "")
        source_type = to_official_source_type(source_type=str(item.get("source_type") or "")) or "text_span"
        if source_type not in OFFICIAL_EVIDENCE_SOURCE_TYPES:
            errors.append("locator_validation_error")
            continue
        locator = canonicalize_locator(
            source_type,
            item.get("locator") if isinstance(item.get("locator"), dict) else {},
        )
        key = (paper_id, source_type)
        allowed = evidence_map.get(key)
        if not allowed:
            errors.append("locator_validation_error")
            continue
        try:
            page = int(locator.get("page"))
        except (TypeError, ValueError):
            errors.append("locator_validation_error")
            continue
        if page not in allowed.get("pages", set()):
            errors.append("locator_validation_error")
            continue
        clean_locator: dict[str, Any] = {"page": page}
        for id_key in ("table_id", "figure_id", "equation_id", "algorithm_id", "citation_id", "reference_id"):
            if id_key not in locator:
                continue
            value = str(locator.get(id_key) or "").strip()
            if value and value in allowed.get(id_key, set()):
                clean_locator[id_key] = value
            else:
                errors.append("invented_grounding_label_removed")
        normalized.append({"paper_id": paper_id, "source_type": source_type, "locator": clean_locator})
    prediction["evidence"] = normalized
    return prediction, errors


def _text_tokens(value: object) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(value or "")) if len(token) > 1}


def _flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    return str(value)


def _answer_text(answer: object) -> str:
    if not isinstance(answer, dict):
        return ""
    parts: list[str] = []
    freeform = answer.get("freeform")
    if isinstance(freeform, dict):
        parts.append(str(freeform.get("text") or ""))
    multiple = answer.get("multiple_choice")
    if isinstance(multiple, dict):
        parts.append(str(multiple.get("gold") or ""))
    table = answer.get("table")
    if isinstance(table, dict):
        parts.append(_flatten_text(table.get("rows")))
    return " ".join(part for part in parts if part)


def _evidence_text(evidence: object) -> str:
    if not isinstance(evidence, dict):
        return _flatten_text(evidence)
    fields = [
        "text",
        "quote",
        "snippet",
        "description",
        "rationale",
        "reason",
        "support",
        "label",
        "source_type",
    ]
    parts = [str(evidence.get(field) or "") for field in fields]
    locator = evidence.get("locator")
    if isinstance(locator, dict):
        parts.append(_flatten_text(locator))
    return " ".join(part for part in parts if part)


def _is_multi_paper_task(input_example: dict[str, Any], answer_contract: dict[str, Any] | None = None) -> bool:
    """Compatibility wrapper around the shared query-visible router."""
    return derive_task_structure(input_example).is_multi_paper


def _label_to_locator(source_type: str, label: Any) -> dict[str, Any]:
    text = str(label or "").strip()
    if not text:
        return {}
    if source_type == "table":
        return {"table_id": text}
    if source_type == "figure":
        return {"figure_id": text}
    lowered = text.lower()
    if source_type == "equation_algorithm":
        if "algorithm" in lowered:
            return {"algorithm_id": text}
        if "equation" in lowered or re.search(r"\b(?:eq\.?|formula)\b", lowered):
            return {"equation_id": text}
    if source_type == "citation_context":
        return {"citation_id": canonical_citation_id(text)}
    return {}


def _evidence_from_contributing_papers(value: Any, candidate_set: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, list):
        return [], []
    evidence: list[dict[str, Any]] = []
    paper_ids: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "")
        if not paper_id or paper_id not in candidate_set:
            continue
        if paper_id not in paper_ids:
            paper_ids.append(paper_id)
        supporting = item.get("supporting_evidence") if isinstance(item.get("supporting_evidence"), list) else []
        for support in supporting:
            if not isinstance(support, dict):
                continue
            source_type = to_official_source_type(source_type=str(support.get("source_type") or "")) or "text_span"
            locator: dict[str, Any] = {}
            try:
                locator["page"] = int(support.get("page"))
            except (TypeError, ValueError):
                continue
            locator.update(_label_to_locator(source_type, support.get("label")))
            evidence.append({"paper_id": paper_id, "source_type": source_type, "locator": locator})
    return evidence, paper_ids


def _normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _coerce_table_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except ValueError:
                return text
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", text):
            try:
                return float(text)
            except ValueError:
                return text
        return text
    return value


def _table_column_types(input_example: dict[str, Any] | None) -> dict[str, str]:
    """Map official schema column name -> type from the raw input row."""
    result: dict[str, str] = {}
    schema = (input_example or {}).get("table_schema")
    if isinstance(schema, list):
        for column in schema:
            if isinstance(column, dict) and str(column.get("name") or ""):
                result[str(column["name"])] = str(column.get("type") or "string")
    return result


def _rows_from_table_answer_plan(value: Any, schema: list[str] | None) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    columns = [str(column) for column in (schema or []) if str(column)]
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        values = item.get("values") if isinstance(item.get("values"), dict) else item
        if not isinstance(values, dict):
            continue
        if columns:
            remapped = remap_row_values(values, columns)
            row = {column: remapped.get(column) for column in columns}
        else:
            row = {str(key): _coerce_table_value(val) for key, val in values.items() if key not in {"row_source", "source", "evidence"}}
        if any(value is not None and str(value).strip() for value in row.values()):
            rows.append(row)
    return rows


def _evidence_from_table_answer_plan(value: Any, candidate_set: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = item.get("row_source") if isinstance(item.get("row_source"), dict) else item.get("source")
        if not isinstance(source, dict):
            continue
        paper_id = str(source.get("paper_id") or "")
        if not paper_id or paper_id not in candidate_set:
            continue
        locator: dict[str, Any] = {}
        try:
            locator["page"] = int(source.get("page"))
        except (TypeError, ValueError):
            continue
        # `answer.table` is a response shape. Its rows may be supported by a
        # figure, equation, citation, or prose record, so preserve the exact
        # source type emitted by the keyed L2 -> L0 resolver rather than
        # coercing every row into a PDF table locator.
        source_type = to_official_source_type(source_type=str(source.get("source_type") or "")) or "text_span"
        locator.update(_label_to_locator(source_type, source.get("label") or source.get("table_id")))
        evidence.append({"paper_id": paper_id, "source_type": source_type, "locator": locator})
    return evidence


def _table_plan_evidence_refs(value: Any) -> list[str]:
    """Extract model-echoed row refs so plans bind to selected evidence."""
    if not isinstance(value, list):
        return []
    refs: list[str] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        ref = str(row.get("row_evidence_ref") or row.get("evidence_ref") or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _selected_evidence_text(evidence: dict[str, Any]) -> str:
    grounding = evidence.get("grounding_label")
    grounding_text = ""
    if isinstance(grounding, dict):
        grounding_text = f"{grounding.get('type') or ''} {grounding.get('value') or ''}"
    hint_text = ""
    if isinstance(evidence.get("source_type_hints"), list):
        hint_parts: list[str] = []
        for hint in evidence.get("source_type_hints") or []:
            if not isinstance(hint, dict):
                continue
            locator = hint.get("locator") if isinstance(hint.get("locator"), dict) else {}
            hint_parts.append(
                " ".join(
                    str(part or "")
                    for part in [
                        hint.get("source_type"),
                        hint.get("label"),
                        hint.get("reason"),
                        " ".join(f"{key} {value}" for key, value in locator.items() if key != "page"),
                    ]
                )
            )
        hint_text = " ".join(hint_parts)
    return " ".join(
        str(part or "")
        for part in [
            evidence.get("text"),
            evidence.get("label"),
            evidence.get("source_type"),
            evidence.get("page"),
            grounding_text,
            hint_text,
        ]
    )


def _token_overlap_score(query_text: str, candidate_text: str) -> float:
    query_tokens = _text_tokens(query_text)
    candidate_tokens = _text_tokens(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    precision = overlap / max(1, len(candidate_tokens))
    recall = overlap / max(1, len(query_tokens))
    return (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0


def _hint_for_source_type(evidence: dict[str, Any], source_type: str) -> dict[str, Any] | None:
    wanted = to_official_source_type(source_type=source_type) or ""
    for hint in evidence.get("source_type_hints") or []:
        if not isinstance(hint, dict):
            continue
        hinted = to_official_source_type(source_type=str(hint.get("source_type") or "")) or ""
        if hinted == wanted:
            return hint
    return None


def _locator_from_selected_evidence(evidence: dict[str, Any], source_type: str | None = None) -> dict[str, Any]:
    if source_type:
        hint = _hint_for_source_type(evidence, source_type)
        if hint:
            locator = hint.get("locator") if isinstance(hint.get("locator"), dict) else {}
            try:
                page = int(locator.get("page", evidence.get("page")))
            except (TypeError, ValueError):
                page = 0
            if page > 0:
                clean = {"page": page}
                for id_key in ("table_id", "figure_id", "equation_id", "algorithm_id", "citation_id", "reference_id"):
                    if locator.get(id_key) not in {None, ""}:
                        clean[id_key] = locator.get(id_key)
                return clean
    selected_source_type = to_official_source_type(source_type=str(evidence.get("source_type") or "")) or ""
    if not source_type or selected_source_type == to_official_source_type(source_type=source_type):
        locator = evidence.get("locator") if isinstance(evidence.get("locator"), dict) else {}
        try:
            page = int(locator.get("page", evidence.get("page")))
        except (TypeError, ValueError):
            page = 0
        if page > 0:
            clean = {"page": page}
            for id_key in ("table_id", "figure_id", "equation_id", "algorithm_id", "citation_id", "reference_id"):
                if locator.get(id_key) not in {None, ""}:
                    clean[id_key] = locator[id_key]
            return clean
    locator: dict[str, Any] = {}
    try:
        locator["page"] = int(evidence.get("page"))
    except (TypeError, ValueError):
        return {}
    grounding = evidence.get("grounding_label")
    if isinstance(grounding, dict):
        label_type = str(grounding.get("type") or "").strip()
        value = str(grounding.get("value") or "").strip()
        if label_type in {"table_id", "figure_id", "equation_id", "algorithm_id", "citation_id", "reference_id"} and value:
            locator[label_type] = value
    return locator


def _evidence_locator_is_allowed(evidence: dict[str, Any], selected_evidence: list[dict[str, Any]]) -> bool:
    probe = {"evidence": [evidence]}
    validated, errors = validate_prediction_locators_against_selected_evidence(probe, selected_evidence)
    return not errors and bool(validated.get("evidence"))


def _best_symbolic_evidence(
    query_text: str,
    predicted_paper_ids: list[str],
    raw_evidence: dict[str, Any] | None,
    selected_evidence: list[dict[str, Any]],
    preferred_source_types: Any,
) -> dict[str, Any] | None:
    if not selected_evidence:
        return None
    preferred = set(as_source_types(preferred_source_types))
    predicted_set = {paper_id for paper_id in predicted_paper_ids if paper_id}
    raw_paper_id = str((raw_evidence or {}).get("paper_id") or "")
    raw_source_type = to_official_source_type(source_type=str((raw_evidence or {}).get("source_type") or "")) or ""
    raw_locator = (raw_evidence or {}).get("locator") if isinstance((raw_evidence or {}).get("locator"), dict) else {}
    raw_page = None
    try:
        raw_page = int(raw_locator.get("page")) if isinstance(raw_locator, dict) and raw_locator.get("page") is not None else None
    except (TypeError, ValueError):
        raw_page = None

    candidates = [item for item in selected_evidence if isinstance(item, dict)]
    if raw_paper_id:
        same_paper = [item for item in candidates if str(item.get("paper_id") or "") == raw_paper_id]
        if not same_paper:
            return None
        candidates = same_paper
    elif predicted_set:
        same_prediction = [item for item in candidates if str(item.get("paper_id") or "") in predicted_set]
        if not same_prediction:
            return None
        candidates = same_prediction

    evidence_query = " ".join([query_text, _evidence_text(raw_evidence or {})]).strip()
    best: tuple[float, dict[str, Any]] | None = None
    for index, item in enumerate(candidates):
        score = _token_overlap_score(evidence_query, _selected_evidence_text(item))
        if str(item.get("source_type") or "") in preferred:
            score += 1.2
        if any(_hint_for_source_type(item, source_type) for source_type in preferred):
            score += 0.9
        if raw_source_type and str(item.get("source_type") or "") == raw_source_type:
            score += 1.0
        if raw_source_type and _hint_for_source_type(item, raw_source_type):
            score += 0.8
        if raw_page is not None and item.get("page") == raw_page:
            score += 0.7
        if raw_paper_id and str(item.get("paper_id") or "") == raw_paper_id:
            score += 0.5
        score -= index * 0.001
        if best is None or score > best[0]:
            best = (score, item)
    return best[1] if best else None


def standardize_symbolic_evidence(
    prediction: dict[str, Any],
    input_example: dict[str, Any],
    selected_evidence: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Map model evidence claims to the closest selected symbolic evidence locator.

    This is a deterministic postprocess over already-selected symbolic records.
    It does not expose scores to VLM-2 and does not introduce evidence outside
    the selected symbolic context.
    """
    selected = [item for item in (selected_evidence or []) if isinstance(item, dict)]
    if not selected:
        return prediction, []
    errors: list[str] = []
    predicted_ids = [
        str(item.get("paper_id") or "")
        for item in prediction.get("gold_papers", [])
        if isinstance(item, dict) and item.get("paper_id")
    ]
    preferred_types = derive_task_structure(input_example).preferred_source_types
    query_text = " ".join([str(input_example.get("question") or ""), _answer_text(prediction.get("answer"))]).strip()
    raw_evidence = prediction.get("evidence") if isinstance(prediction.get("evidence"), list) else []
    standardized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, tuple[tuple[str, str], ...]]] = set()

    for item in raw_evidence:
        if not isinstance(item, dict):
            errors.append("symbolic_evidence_non_object_replaced")
            item = {}
        raw_item_source_type = to_official_source_type(source_type=str(item.get("source_type") or "")) or ""
        if item and _evidence_locator_is_allowed(item, selected):
            paper_id = str(item.get("paper_id") or "")
            source_type = to_official_source_type(source_type=str(item.get("source_type") or "")) or "text_span"
            locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
            try:
                page = int(locator.get("page"))
            except (TypeError, ValueError):
                page = 0
            id_items = tuple(sorted((str(k), str(v)) for k, v in locator.items() if k != "page" and v))
            key = (paper_id, source_type, page, id_items)
            if key not in seen:
                seen.add(key)
                standardized.append({"paper_id": paper_id, "source_type": source_type, "locator": {"page": page, **{k: v for k, v in locator.items() if k != "page" and v}}})
            continue

        best = _best_symbolic_evidence(query_text, predicted_ids, item, selected, preferred_types)
        if not best:
            best = _best_symbolic_evidence(query_text, [], None, selected, preferred_types)
        if not best:
            errors.append("symbolic_evidence_standardization_no_match")
            continue
        best_source_type = str(best.get("source_type") or (preferred_types[0] if preferred_types else "text_span"))
        if raw_item_source_type and _hint_for_source_type(best, raw_item_source_type):
            best_source_type = raw_item_source_type
        elif preferred_types:
            for source_type in preferred_types:
                if _hint_for_source_type(best, source_type):
                    best_source_type = source_type
                    break
        locator = _locator_from_selected_evidence(best, best_source_type)
        if not locator:
            errors.append("symbolic_evidence_standardization_no_locator")
            continue
        paper_id = str(best.get("paper_id") or "")
        source_type = best_source_type
        id_items = tuple(sorted((str(k), str(v)) for k, v in locator.items() if k != "page" and v))
        key = (paper_id, source_type, int(locator["page"]), id_items)
        if key in seen:
            continue
        seen.add(key)
        standardized.append({"paper_id": paper_id, "source_type": source_type, "locator": locator})
        errors.append("symbolic_evidence_locator_standardized")

    if not standardized:
        best = _best_symbolic_evidence(query_text, predicted_ids, None, selected, preferred_types)
        if not best:
            best = _best_symbolic_evidence(query_text, [], None, selected, preferred_types)
        if best:
            best_source_type = str(best.get("source_type") or (preferred_types[0] if preferred_types else "text_span"))
            for source_type in preferred_types:
                if _hint_for_source_type(best, source_type):
                    best_source_type = source_type
                    break
            locator = _locator_from_selected_evidence(best, best_source_type)
            if locator:
                standardized.append(
                    {
                        "paper_id": str(best.get("paper_id") or ""),
                        "source_type": best_source_type,
                        "locator": locator,
                    }
                )
                errors.append("symbolic_evidence_empty_filled")
    prediction["evidence"] = standardized
    return prediction, errors


def make_fallback_prediction(input_example: dict[str, Any], top1_candidate: dict[str, Any] | None) -> dict[str, Any]:
    paper_id = (top1_candidate or {}).get("paper_id", "")
    gold_papers = [{"paper_id": paper_id}] if paper_id else []
    return {
        "query_id": input_example.get("query_id"),
        "gold_papers": gold_papers,
        "evidence": [],
        "answer": _requested_answer(input_example),
    }


def _resolve_evidence_ref_echo(
    obj: dict[str, Any],
    selected_evidence: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected_by_ref = {
        str(item.get("evidence_ref") or ""): item
        for item in selected_evidence or []
        if isinstance(item, dict) and str(item.get("evidence_ref") or "")
    }
    raw_evidence = obj.get("evidence") if isinstance(obj.get("evidence"), list) else []
    raw_refs = obj.get("evidence_refs") if isinstance(obj.get("evidence_refs"), list) else []
    refs = [str(item) for item in raw_refs if str(item or "")]
    passthrough: list[dict[str, Any]] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("evidence_ref") or "")
        if ref:
            refs.append(ref)
        else:
            passthrough.append(item)
    if not refs:
        return passthrough, []

    resolved: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        selected = selected_by_ref.get(ref)
        if selected is None:
            errors.append("unknown_evidence_ref_removed")
            continue
        locator = canonicalize_locator(
            str(selected.get("source_type") or ""),
            selected.get("locator") if isinstance(selected.get("locator"), dict) else {},
            selected.get("label"),
        )
        if not locator:
            locator = _locator_from_selected_evidence(selected, str(selected.get("source_type") or ""))
        resolved.append(
            {
                "paper_id": str(selected.get("paper_id") or ""),
                "source_type": str(selected.get("source_type") or ""),
                "locator": locator,
            }
        )
    if resolved:
        errors.append("evidence_ref_echo_resolved")
    return [*resolved, *passthrough], errors


def normalize_prediction(
    obj: dict[str, Any],
    input_example: dict[str, Any],
    candidate_paper_ids: list[str],
    answer_contract: dict[str, Any] | None = None,
    selected_evidence: list[dict[str, Any]] | None = None,
    symbolic_evidence_standardization: bool = True,
    candidate_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_id = input_example.get("query_id")
    errors: list[dict[str, Any]] = []
    allow_noncandidate = os.environ.get("ALLOW_NONCANDIDATE_PAPERS", "").lower() == "true"
    candidate_set = set(candidate_paper_ids)
    top1 = candidate_paper_ids[0] if candidate_paper_ids else ""
    contract = answer_contract or extract_answer_contract(input_example)
    is_multi_paper = _is_multi_paper_task(input_example, contract)
    contribution_evidence, contribution_paper_ids = _evidence_from_contributing_papers(obj.get("contributing_papers"), candidate_set)
    raw_answer = obj.get("answer") if isinstance(obj.get("answer"), dict) else {}
    table_answer_plan = obj.get("table_answer_plan")
    if not isinstance(table_answer_plan, list) and isinstance(raw_answer.get("table_answer_plan"), list):
        table_answer_plan = raw_answer.get("table_answer_plan")
    table_plan_evidence = _evidence_from_table_answer_plan(table_answer_plan, candidate_set)

    raw_papers = obj.get("gold_papers") or obj.get("papers") or []
    normalized_papers: list[dict[str, str]] = []
    if isinstance(raw_papers, list):
        for item in raw_papers:
            paper_id = item.get("paper_id") if isinstance(item, dict) else item
            paper_id = str(paper_id or "")
            if not paper_id:
                continue
            if paper_id not in candidate_set and not allow_noncandidate:
                errors.append({"query_id": query_id, "type": "noncandidate_paper_id", "paper_id": paper_id, "replacement": top1})
                paper_id = top1
            if paper_id and {"paper_id": paper_id} not in normalized_papers:
                normalized_papers.append({"paper_id": paper_id})
    if is_multi_paper:
        for paper_id in contribution_paper_ids:
            if paper_id and {"paper_id": paper_id} not in normalized_papers:
                normalized_papers.append({"paper_id": paper_id})
                errors.append({"query_id": query_id, "type": "gold_paper_filled_from_contributing_papers", "paper_id": paper_id})
    if not normalized_papers and top1:
        errors.append({"query_id": query_id, "type": "fallback_paper_id", "replacement": top1})
        normalized_papers = [{"paper_id": top1}]

    # A table row's evidence ref is an answer claim just like a top-level
    # evidence ref.  Previously it was ignored unless the model redundantly
    # echoed it at top level, leaving valid table plans ungrounded.
    table_plan_refs = _table_plan_evidence_refs(table_answer_plan)
    if table_plan_refs:
        evidence_obj = dict(obj)
        existing_refs = obj.get("evidence_refs") if isinstance(obj.get("evidence_refs"), list) else []
        evidence_obj["evidence_refs"] = [*existing_refs, *table_plan_refs]
    else:
        evidence_obj = obj
    raw_evidence, echo_errors = _resolve_evidence_ref_echo(evidence_obj, selected_evidence)
    for error_type in echo_errors:
        errors.append({"query_id": query_id, "type": error_type})
    if isinstance(raw_evidence, list) and contribution_evidence:
        raw_evidence = [*raw_evidence, *contribution_evidence]
        errors.append({"query_id": query_id, "type": "evidence_extended_from_contributing_papers", "count": len(contribution_evidence)})
    if isinstance(raw_evidence, list) and table_plan_evidence:
        raw_evidence = [*raw_evidence, *table_plan_evidence]
        errors.append({"query_id": query_id, "type": "evidence_extended_from_table_answer_plan", "count": len(table_plan_evidence)})
    evidence = _normalize_official_evidence(raw_evidence)
    if not isinstance(evidence, list):
        errors.append({"query_id": query_id, "type": "invalid_evidence_replaced"})
        evidence = []
    answer = raw_answer
    pred = {
        "query_id": query_id,
        "gold_papers": normalized_papers,
        "evidence": evidence,
        "answer": answer,
        "contributing_papers": obj.get("contributing_papers") if isinstance(obj.get("contributing_papers"), list) else [],
        "table_answer_plan": table_answer_plan if isinstance(table_answer_plan, list) else [],
    }
    if symbolic_evidence_standardization:
        pred, standardization_errors = standardize_symbolic_evidence(pred, input_example, selected_evidence)
        for error_type in standardization_errors:
            errors.append({"query_id": query_id, "type": error_type})
    pred, locator_errors = validate_prediction_locators_against_selected_evidence(pred, selected_evidence)
    if is_multi_paper and isinstance(pred.get("evidence"), list):
        for item in pred.get("evidence") or []:
            paper_id = str(item.get("paper_id") or "") if isinstance(item, dict) else ""
            if paper_id and paper_id in candidate_set and {"paper_id": paper_id} not in pred["gold_papers"]:
                pred["gold_papers"].append({"paper_id": paper_id})
                errors.append({"query_id": query_id, "type": "gold_paper_filled_from_evidence", "paper_id": paper_id})
    pred, answer_errors = validate_prediction_against_answer_contract(
        pred, contract, input_example, [*(candidate_records or []), *(selected_evidence or [])]
    )
    answer_errors.extend(_maybe_rerank_structured_multiple_choice(pred, contract, input_example, selected_evidence))
    for error_type in locator_errors + answer_errors:
        errors.append({"query_id": query_id, "type": error_type})
    return pred, errors


def validate_prediction_shape(pred: dict[str, Any]) -> bool:
    return (
        isinstance(pred.get("query_id"), str)
        and isinstance(pred.get("gold_papers"), list)
        and isinstance(pred.get("evidence"), list)
        and isinstance(pred.get("answer"), dict)
    )


def _normalize_official_evidence(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    normalized: list[Any] = []
    for item in value:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        row = dict(item)
        source_type = to_official_source_type(source_type=str(row.get("source_type") or ""), record_type=str(row.get("record_type") or "")) or "text_span"
        row["source_type"] = source_type
        normalized.append(row)
    return normalized


def strip_internal_grounding(value: Any) -> Any:
    internal_keys = {
        "confidence",
        "notes",
        "global_record_id",
        "record_type",
        "bbox_1000",
        "bbox_pixel",
        "validation_status",
        "record_id",
        "label",
        "image_ref",
        "vlm_parse_confidence",
        "contributing_papers",
        "table_answer_plan",
    }
    if isinstance(value, list):
        return [strip_internal_grounding(item) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in internal_keys:
                continue
            if key == "locator" and isinstance(item, dict):
                locator = {k: strip_internal_grounding(v) for k, v in item.items() if k not in internal_keys}
                cleaned[key] = locator
            else:
                cleaned[key] = strip_internal_grounding(item)
        return cleaned
    return value
