"""Route complete cached predictions by public answer contract, without gold data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_io import find_official_file, read_jsonl, write_jsonl


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one complete cached prediction per public answer type.")
    parser.add_argument("--official-dir", default="official_dev")
    parser.add_argument("--default-predictions", required=True)
    parser.add_argument(
        "--table-predictions",
        required=True,
        action="append",
        help="Table prediction JSONL; repeat the flag to merge independently generated shards.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    inputs = read_jsonl(find_official_file(args.official_dir, "validation_inputs.jsonl"))
    default = {str(row.get("query_id") or ""): row for row in read_jsonl(args.default_predictions)}
    table = {}
    for path in args.table_predictions:
        for row in read_jsonl(path):
            query_id = str(row.get("query_id") or "")
            if query_id:
                table[query_id] = row
    predictions = []
    routing = []
    for sample in inputs:
        query_id = str(sample.get("query_id") or "")
        use_table = "table" in set(sample.get("answer_types") or [])
        source = table if use_table else default
        if query_id not in source:
            raise ValueError(f"Missing cached prediction for {query_id} in {'table' if use_table else 'default'} route")
        predictions.append(source[query_id])
        routing.append({"query_id": query_id, "route": "table" if use_table else "default"})
    output = Path(args.output_dir)
    write_jsonl(output / "predictions.jsonl", predictions)
    write_jsonl(output / "routing_manifest.jsonl", routing)
    print(json.dumps({"predictions": len(predictions), "table_routed": sum(row["route"] == "table" for row in routing), "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
