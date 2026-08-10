"""Mechanically construct target-shaped Ver3 development records from source structures."""
from __future__ import annotations

import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def canon(value: Any) -> str: return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def digest(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest().upper()
def file_hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def split(paper: str) -> str:
    bucket = int(digest(paper)[:2], 16)
    return "train" if bucket <= 0xB2 else "calibration" if bucket <= 0xD8 else "holdout"
def clean(value: Any) -> str | None:
    text = " ".join(str(value or "").split())
    return text if 1 < len(text) <= 100 else None
def number(value: str) -> float | None:
    found = re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", value.replace(",", ""))
    return float(value.rstrip("%").replace(",", "")) if found else None
def jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip(): yield json.loads(line)


def record(family: str, shape: str, operator: str, question: str, answer: Any, sources: list[dict[str, Any]], paper: str, trace: dict[str, Any]) -> dict[str, Any]:
    identity = digest(canon([family, shape, operator, question, sources]))[:24].lower()
    return {"record_id": f"ta_{identity}", "answer_family": family, "answer_shape": shape, "reasoning_operator": operator, "question": question, "answer": answer, "source_paper": paper, "source_objects": sources, "verification_trace": trace, "difficulty_features": {"source_count": len(sources), "question_words": len(question.split()), "operator": operator}, "split": split(paper), "generation_method": "deterministic_source_structure", "record_hash": digest(canon([question, answer, sources, trace]))}


def build(facts: Iterable[dict[str, Any]], cells: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = [x for x in cells if x.get("provenance_status") == "accepted" and not x.get("is_column_header") and not x.get("is_row_header") and clean(x.get("normalized_cell_value"))]
    by_table: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for x in cells:
        paper, page, table, value = x.get("paper_id"), x.get("page"), x.get("evaluator_visible_table_id"), clean(x.get("normalized_cell_value"))
        if not isinstance(paper, str) or not isinstance(page, int) or not isinstance(table, str) or not value: continue
        source = {"object_id": x.get("record_hash"), "source_hash": x.get("source_hash"), "page": page, "table_id": table, "row": x.get("row_index"), "column": x.get("column_index"), "value": value}
        if int(digest(str(x.get("record_hash")))[:2], 16) % 4 == 0:
            output.append(record("freeform", "number" if number(value) is not None else "short_phrase", "structured_lookup", f"What value is reported in {table} of paper {paper} at row {x.get('row_index')} and column {x.get('column_index')}?", {"text": value}, [source], paper, {"expected": value, "method": "exact_cell"}))
        by_table[(paper, page, table)].append({"value": value, "source": source})
    for (paper, page, table), rows in by_table.items():
        unique = []
        for row in rows:
            if row["value"] not in [x["value"] for x in unique]: unique.append(row)
        if len(unique) >= 4:
            choices = unique[:4]; correct_index = int(digest(paper + table)[:2], 16) % 4
            options = {chr(65+i): row["value"] for i, row in enumerate(choices)}
            output.append(record("multiple_choice", "option", "structured_lookup", f"Which option gives the value reported by {table} in paper {paper} at page {page}?", {"options": options, "gold": chr(65+correct_index)}, [choices[correct_index]["source"]], paper, {"expected": choices[correct_index]["value"], "method": "table_option"}))
            for selected in unique:
                alternatives = [row for row in unique if row["value"] != selected["value"]][:3]
                if len(alternatives) < 3:
                    continue
                option_rows = alternatives + [selected]
                offset = int(digest(str(selected["source"].get("object_id")))[:2], 16) % 4
                option_rows = option_rows[offset:] + option_rows[:offset]
                option_map = {chr(65 + index): row["value"] for index, row in enumerate(option_rows)}
                letter = next(letter for letter, candidate in option_map.items() if candidate == selected["value"])
                output.append(record("multiple_choice", "option", "structured_lookup", f"Which option gives the value reported in {table} of paper {paper} at row {selected['source']['row']} and column {selected['source']['column']}?", {"options": option_map, "gold": letter}, [selected["source"]], paper, {"expected": selected["value"], "method": "coordinate_option"}))
            absent = next((x for x in cells if x.get("paper_id") == paper and clean(x.get("normalized_cell_value")) not in options.values()), None)
            if absent:
                options[chr(65+correct_index)] = clean(absent.get("normalized_cell_value"))
                output.append(record("multiple_choice", "option", "negation_except", f"Which option is NOT a value reported in {table} of paper {paper} on page {page}?", {"options": options, "gold": chr(65+correct_index)}, [r["source"] for r in choices] + [{"object_id": absent.get("record_hash"), "source_hash": absent.get("source_hash")}], paper, {"expected": options[chr(65+correct_index)], "method": "membership_complement"}))
        numeric = [(r, number(r["value"])) for r in unique if number(r["value"]) is not None]
        if len(numeric) >= 2:
            left, right = numeric[0], numeric[1]; delta = round(abs(left[1] - right[1]), 6); answer = str(int(delta)) if delta.is_integer() else str(delta)
            output.append(record("freeform", "number", "comparison", f"What is the absolute difference between the two reported values in {table} of paper {paper} on page {page}?", {"text": answer}, [left[0]["source"], right[0]["source"]], paper, {"left": left[0]["value"], "right": right[0]["value"], "operation": "absolute_difference", "expected": answer}))
        if len(rows) >= 2:
            output.append(record("freeform", "count", "multi_object_count", f"How many provenance-complete values are reported in {table} of paper {paper} on page {page}?", {"text": str(len(rows))}, [r["source"] for r in rows], paper, {"operation": "count", "expected": len(rows)}))
    for x in facts:
        paper, page, kind, value = x.get("paper_id"), x.get("page"), x.get("object_type"), clean(x.get("normalized_value"))
        if x.get("ambiguity_status") != "accepted" or not isinstance(paper, str) or not isinstance(page, int) or not isinstance(kind, str) or not value: continue
        source = {"object_id": x.get("object_uid"), "source_hash": x.get("source_hash"), "page": page, "object_type": kind, "value": value}
        output.append(record("freeform", "short_phrase", "direct_extraction", f"According to the {kind} in paper {paper} on page {page}, what text is stated?", {"text": value}, [source], paper, {"expected": value, "method": "exact_fact"}))
    dedup = {x["record_id"]: x for x in output}
    return [dedup[k] for k in sorted(dedup)]


def materialize(facts: Path, ledgers: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()): raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True); rows = build(jsonl([facts]), jsonl(ledgers))
    benchmark = output_dir / "benchmark.jsonl"; temporary = benchmark.with_suffix(".tmp")
    temporary.write_text("".join(canon(x)+"\n" for x in rows), encoding="utf-8"); temporary.replace(benchmark)
    counts = {key: sum(x["split"] == key for x in rows) for key in ("train", "calibration", "holdout")}
    manifest = {"schema_version": "ver3.target-aligned-benchmark.v1", "record_count": len(rows), "benchmark_sha256": file_hash(benchmark), "split_counts": counts, "split_hashes": {key: digest("\n".join(x["record_id"] for x in rows if x["split"] == key)) for key in counts}, "family_counts": {key: sum(x["answer_family"] == key for x in rows) for key in ("multiple_choice", "freeform")}, "operator_counts": {key: sum(x["reasoning_operator"] == key for x in rows) for key in sorted({x["reasoning_operator"] for x in rows})}, "inputs": [{"role": "facts", "sha256": file_hash(facts)}, *[{"role": "table_ledger", "sha256": file_hash(x)} for x in ledgers]]}
    for name in ("manifest.json", "status.json"):
        target = output_dir / name; temp = target.with_suffix(".tmp"); temp.write_text(canon({"status": "complete", **manifest})+"\n", encoding="utf-8"); temp.replace(target)
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--facts", type=Path, required=True); parser.add_argument("--table-ledger", action="append", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args(); print(canon(materialize(args.facts, args.table_ledger, args.output_dir)))
