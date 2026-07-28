"""Create inference-safe answer contracts from evaluator-only validation data.

The benchmark keeps multiple-choice options and table schemas beside gold
answers.  This one-way conversion retains only those public output constraints
so generation never opens the validation file or receives a gold value.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .data_io import read_jsonl, write_jsonl


FORBIDDEN_KEYS = {"gold", "evidence", "gold_papers", "rows"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write redacted answer contracts for inference.")
    parser.add_argument("--validation-input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _contract(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    result: dict[str, Any] = {"query_id": str(row.get("query_id") or ""), "answer": {}}
    multiple_choice = answer.get("multiple_choice") if isinstance(answer.get("multiple_choice"), dict) else {}
    if isinstance(multiple_choice.get("options"), (dict, list)):
        result["answer"]["multiple_choice"] = {"options": multiple_choice["options"]}
    table = answer.get("table") if isinstance(answer.get("table"), dict) else {}
    if isinstance(table.get("schema"), list):
        result["answer"]["table"] = {"schema": table["schema"]}
    return result


def _assert_redacted(value: Any) -> None:
    if isinstance(value, dict):
        leaked = FORBIDDEN_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"Sanitized answer contract contains forbidden keys: {sorted(leaked)}")
        for item in value.values():
            _assert_redacted(item)
    elif isinstance(value, list):
        for item in value:
            _assert_redacted(item)


def main() -> int:
    args = _args()
    contracts = [_contract(row) for row in read_jsonl(args.validation_input)]
    for contract in contracts:
        _assert_redacted(contract)
    output = Path(args.output)
    write_jsonl(output, contracts)
    print({"contracts": len(contracts), "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
