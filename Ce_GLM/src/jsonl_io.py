from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class JSONLLocation:
    path: Path
    line: int
    column: int | None = None


class JSONLParseError(ValueError):
    def __init__(self, location: JSONLLocation, message: str) -> None:
        suffix = f":{location.column}" if location.column is not None else ""
        super().__init__(f"{location.path}:{location.line}{suffix}: {message}")
        self.location = location
        self.message = message


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8", errors="strict", newline="")
    except UnicodeDecodeError as exc:  # pragma: no cover - decoding occurs during iteration
        raise JSONLParseError(JSONLLocation(source, 1), f"invalid UTF-8: {exc}") from exc
    with handle:
        try:
            for line_number, physical_line in enumerate(handle, 1):
                if not physical_line.strip():
                    raise JSONLParseError(JSONLLocation(source, line_number), "blank physical line")
                try:
                    value = json.loads(physical_line)
                except json.JSONDecodeError as exc:
                    raise JSONLParseError(
                        JSONLLocation(source, line_number, exc.colno), exc.msg
                    ) from exc
                if not isinstance(value, dict):
                    raise JSONLParseError(
                        JSONLLocation(source, line_number), "record must be a JSON object"
                    )
                yield value
        except UnicodeDecodeError as exc:
            raise JSONLParseError(
                JSONLLocation(source, line_number if "line_number" in locals() else 1),
                f"invalid UTF-8: {exc}",
            ) from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", errors="strict", newline="\n") as handle:
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise TypeError(f"JSONL record {index} must be a dictionary")
            serialized = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if "\n" in serialized or "\r" in serialized:
                raise AssertionError("serializer emitted a physical newline inside a record")
            handle.write(serialized)
            handle.write("\n")


def inspect_jsonl(path: str | Path, id_key: str = "query_id") -> dict[str, Any]:
    count = 0
    identifiers: set[str] = set()
    duplicates: list[str] = []
    schemas: dict[tuple[str, ...], int] = {}
    for record in iter_jsonl(path):
        count += 1
        schema = tuple(sorted(record))
        schemas[schema] = schemas.get(schema, 0) + 1
        if record.get(id_key) is not None:
            identifier = str(record[id_key])
            if identifier in identifiers and identifier not in duplicates:
                duplicates.append(identifier)
            identifiers.add(identifier)
    return {
        "records": count,
        "unique_query_ids": len(identifiers),
        "duplicate_query_ids": duplicates,
        "schema_counts": {"|".join(keys): value for keys, value in sorted(schemas.items())},
    }
