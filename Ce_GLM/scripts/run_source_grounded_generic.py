from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.jsonl_io import read_jsonl, write_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def mc_letter(record: dict[str, Any]) -> str | None:
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    mc = answer.get("multiple_choice") if isinstance(answer.get("multiple_choice"), dict) else {}
    value = mc.get("gold")
    return str(value) if value is not None else None


def contains_gold_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() == "gold" or contains_gold_key(child) for key, child in value.items())
    if isinstance(value, list):
        return any(contains_gold_key(child) for child in value)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic accepted source-grounded MC replay")
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    release = args.release_root.resolve()
    module_path = release / "src" / "source_grounded_mc_replay.py"
    spec = importlib.util.spec_from_file_location("ver2_source_grounded_mc_replay", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    SourceCorpus = module.SourceCorpus
    derive_semantic_answer = module.derive_semantic_answer
    map_to_option = module.map_to_option
    permutation_check = module.permutation_check
    replace_mc_answer = module.replace_mc_answer
    semantic_to_text = module.semantic_to_text

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    parent_rows = read_jsonl(args.parent)
    option_rows = read_jsonl(args.options)
    if len(parent_rows) != 55 or len({row.get("query_id") for row in parent_rows}) != 55:
        raise ValueError("Parent must contain 55 unique query IDs")
    if any(contains_gold_key(row) for row in option_rows):
        raise ValueError("Option input contains a forbidden gold key")
    option_by_id = {str(row["query_id"]): row for row in option_rows}
    corpus = SourceCorpus(args.source_root.resolve())

    semantic_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    frozen: list[dict[str, Any]] = []
    for record in parent_rows:
        query_id = str(record["query_id"])
        input_row = option_by_id.get(query_id, {})
        options = dict((input_row.get("multiple_choice") or {}).get("options") or {})
        if not options:
            frozen.append(copy.deepcopy(record))
            continue
        question = str(input_row.get("question") or "")
        current = mc_letter(record)
        current_records = corpus.query_current_evidence_text(record.get("evidence") or [])
        semantic = derive_semantic_answer(question, options, record.get("evidence") or [], current_records)
        semantic_row = {
            "query_id": query_id,
            "question": question,
            "status": semantic.status,
            "semantic_answer": semantic.semantic_answer,
            "semantic_answer_text": semantic_to_text(semantic.semantic_answer) if semantic.semantic_answer else None,
            "source_paper": semantic.source_paper,
            "source_type": semantic.source_type,
            "source_object": semantic.source_object,
            "raw_support": semantic.raw_support,
            "rule": semantic.rule,
            "reason": semantic.reason,
        }
        semantic_rows.append(semantic_row)
        mapping = {"unique_option_match": False, "matches": [], "selected": None}
        permutation = {"passed": False, "results": []}
        selected = current
        decision = f"PRESERVE_{semantic.status}"
        if semantic.status == "SOURCE_SUPPORTED" and semantic.semantic_answer:
            mapping = map_to_option(semantic.semantic_answer, options)
            if mapping["unique_option_match"]:
                selected = str(mapping["selected"]["letter"])
                permutation = permutation_check(semantic.semantic_answer, str(mapping["selected"]["text"]), options)
                if selected != current and permutation["passed"]:
                    decision = "REPLACE"
                elif selected == current:
                    decision = "PRESERVE_SOURCE_SUPPORTED_CURRENT"
                else:
                    decision = "PRESERVE_PERMUTATION_GATE"
            else:
                decision = "PRESERVE_NO_UNIQUE_OPTION"
        result = replace_mc_answer(record, selected or "") if decision == "REPLACE" else copy.deepcopy(record)
        if result != record:
            changes.append({"query_id": query_id, "old_option": current, "new_option": selected})
        mapping_rows.append({"query_id": query_id, "current_letter": current, "selected_letter": selected, "mapping": mapping, "decision": decision})
        audits.append({"query_id": query_id, "status": semantic.status, "decision": decision, "permutation_passed": bool(permutation["passed"])})
        frozen.append(result)

    prediction = output / "frozen_predictions.jsonl"
    write_jsonl(output / "semantic_answers.jsonl", semantic_rows)
    write_jsonl(output / "option_mapping.jsonl", mapping_rows)
    write_jsonl(output / "answer_changes.jsonl", changes)
    write_jsonl(output / "audit.jsonl", audits)
    write_jsonl(prediction, frozen)
    manifest = {
        "parent_path": str(args.parent.resolve()),
        "parent_sha256": sha256(args.parent),
        "options_sha256": sha256(args.options),
        "prediction_path": str(prediction),
        "prediction_sha256": sha256(prediction),
        "records": len(frozen),
        "semantic_records": len(semantic_rows),
        "changed_count": len(changes),
        "status_counts": dict(Counter(row["status"] for row in semantic_rows)),
        "gold_used_in_generation": False,
        "query_specific_branches": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
