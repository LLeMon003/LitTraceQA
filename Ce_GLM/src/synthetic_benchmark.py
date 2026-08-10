"""Build source-only, mechanically labeled synthetic QA benchmarks for Ver3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_OUTPUT_KEYS = {"query_id", "gold_papers", "evidence", "prediction"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _synthetic_id(recipe: str, source_id: str) -> str:
    return f"syn_{recipe}_{_sha256_bytes(f'{recipe}:{source_id}'.encode())[:20].lower()}"


def split_for(synthetic_id: str) -> str:
    bucket = int(hashlib.sha256(synthetic_id.encode()).hexdigest()[:2], 16)
    if bucket <= 0xB2:
        return "train"
    if bucket <= 0xD8:
        return "calibration"
    return "holdout"


def _clean_value(value: Any, maximum: int = 160) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if not 2 <= len(value) <= maximum:
        return None
    return value


def _source_fact_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    location_count = Counter(
        (row.get("paper_id"), row.get("page"), row.get("object_type"))
        for row in rows
        if row.get("ambiguity_status") == "accepted"
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        paper_id, page, object_type, object_uid = (
            row.get("paper_id"), row.get("page"), row.get("object_type"), row.get("object_uid")
        )
        value = _clean_value(row.get("normalized_value"))
        if (
            row.get("ambiguity_status") != "accepted"
            or not all(isinstance(item, str) and item for item in (paper_id, object_type, object_uid))
            or not isinstance(page, int)
            or not value
            or location_count[(paper_id, page, object_type)] != 1
        ):
            continue
        synthetic_id = _synthetic_id("fact_exact_lookup", str(object_uid))
        output.append(
            {
                "synthetic_id": synthetic_id,
                "recipe": "fact_exact_lookup",
                "split": split_for(synthetic_id),
                "question": (
                    f"What exact source text is recorded by the {object_type} in paper "
                    f"{paper_id} on page {page}?"
                ),
                "answer": {"text": value},
                "source": {
                    "paper_id": paper_id,
                    "page": page,
                    "object_type": object_type,
                    "object_uid": object_uid,
                    "source_hash": row.get("source_hash"),
                    "record_hash": row.get("record_hash"),
                },
            }
        )
    return output


def _table_lookup_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        paper_id, page, table_id, record_hash = (
            row.get("paper_id"), row.get("page"), row.get("evaluator_visible_table_id"), row.get("record_hash")
        )
        value = _clean_value(row.get("normalized_cell_value"), maximum=100)
        row_index, column_index = row.get("row_index"), row.get("column_index")
        if (
            row.get("provenance_status") != "accepted"
            or row.get("is_column_header")
            or row.get("is_row_header")
            or not all(isinstance(item, str) and item for item in (paper_id, table_id, record_hash))
            or not isinstance(page, int)
            or not isinstance(row_index, int)
            or not isinstance(column_index, int)
            or not value
        ):
            continue
        synthetic_id = _synthetic_id("table_coordinate_lookup", record_hash)
        output.append(
            {
                "synthetic_id": synthetic_id,
                "recipe": "table_coordinate_lookup",
                "split": split_for(synthetic_id),
                "question": (
                    f"In paper {paper_id}, page {page}, {table_id}, what value appears at "
                    f"zero-based row {row_index}, column {column_index}?"
                ),
                "answer": {"text": value},
                "source": {
                    "paper_id": paper_id,
                    "page": page,
                    "table_id": table_id,
                    "row_index": row_index,
                    "column_index": column_index,
                    "source_hash": row.get("source_hash"),
                    "record_hash": record_hash,
                },
            }
        )
    return output


def build_records(fact_rows: Iterable[dict[str, Any]], table_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = _source_fact_records(fact_rows) + _table_lookup_records(table_rows)
    deduplicated = {record["synthetic_id"]: record for record in records}
    ordered = [deduplicated[key] for key in sorted(deduplicated)]
    for record in ordered:
        forbidden = FORBIDDEN_OUTPUT_KEYS.intersection(record)
        if forbidden:
            raise ValueError(f"forbidden synthetic output keys: {sorted(forbidden)}")
        if "q_" in _canonical_json(record.get("source", {})):
            raise ValueError("synthetic record leaked a held-out query identifier")
    return ordered


def _read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number} is not an object")
                    yield value


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def materialize(facts: Path, table_ledgers: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fact_rows, table_rows = list(_read_jsonl([facts])), list(_read_jsonl(table_ledgers))
    records = build_records(fact_rows, table_rows)
    by_recipe = Counter(record["recipe"] for record in records)
    by_split = Counter(record["split"] for record in records)
    output = output_dir / "synthetic_benchmark.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")
    temporary.replace(output)
    split_hashes = {
        split: _sha256_bytes("\n".join(record["synthetic_id"] for record in records if record["split"] == split).encode())
        for split in ("train", "calibration", "holdout")
    }
    manifest = {
        "schema_version": "ver3.synthetic-benchmark.v1",
        "policy_status": "synthetic_source_only_no_official_fields",
        "source_inputs": [
            {"role": "figure_equation_facts", "path": str(facts), "sha256": sha256_file(facts)},
            *[{"role": "table_cell_ledger", "path": str(path), "sha256": sha256_file(path)} for path in table_ledgers],
        ],
        "record_count": len(records),
        "recipe_counts": dict(sorted(by_recipe.items())),
        "split_counts": dict(sorted(by_split.items())),
        "split_id_sha256": split_hashes,
        "benchmark_path": str(output),
        "benchmark_sha256": sha256_file(output),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    _atomic_write_json(output_dir / "status.json", {"status": "complete", **manifest})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--table-ledger", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(_canonical_json(materialize(args.facts, args.table_ledger, args.output_dir)))


if __name__ == "__main__":
    main()
