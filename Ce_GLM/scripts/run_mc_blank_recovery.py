from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ID = "V23_MC_BLANK_RECOVERY_001"
DEFAULT_ENDPOINT = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
SYSTEM_PROMPT = """Solve the multiple-choice research-paper question from the supplied excerpts and ordered options. Select exactly one option. Do not abstain. Return JSON only: {"letter":"A","semantic_answer":"the option meaning","reason":"brief source-grounded reasoning"}. The letter must be one of the supplied option letters."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank JSONL line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def unique_by_query(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in result:
            raise ValueError(f"{label}: missing or duplicate query_id {query_id!r}")
        result[query_id] = row
    return result


def blank_mc_query_ids(predictions: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in predictions:
        answer = row.get("answer") or {}
        mc = answer.get("multiple_choice")
        if isinstance(mc, dict) and not str(mc.get("gold") or "").strip():
            result.append(str(row["query_id"]))
    return result


def selected_paper_ids(value: Any) -> set[str]:
    result: set[str] = set()
    for item in value if isinstance(value, list) else []:
        paper_id = item.get("paper_id") if isinstance(item, dict) else item
        if paper_id:
            result.add(str(paper_id))
    return result


def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def preferred_object_type(object_type: str) -> int:
    order = [
        "table_caption", "table_row", "table", "figure_caption", "algorithm_context",
        "equation_context", "equation_block", "object_window", "paragraph", "raw_block",
        "table_cell", "section_header",
    ]
    try:
        return order.index(object_type)
    except ValueError:
        return len(order) + 1


def load_relevant_objects(index_path: Path, query_ids: set[str], selected_papers: dict[str, set[str]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with index_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{index_path}:{line_no}: blank JSONL line")
            row = json.loads(line)
            paper_id = str(row.get("paper_id") or "")
            for query_id in set(map(str, row.get("query_ids") or [])) & query_ids:
                if paper_id in selected_papers[query_id]:
                    result[query_id].append(row)
    return result


def render_context(question: str, options: dict[str, Any], evidence: list[dict[str, Any]], objects: list[dict[str, Any]], limit: int = 14000) -> str:
    query_terms = tokens(question + " " + " ".join(map(str, options.values())))
    evidence_pages = {
        (str(item.get("paper_id") or ""), int((item.get("locator") or {}).get("page") or 0))
        for item in evidence
    }
    ranked: list[tuple[int, int, float, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for obj in objects:
        text = str(obj.get("text") or obj.get("cell_value") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        overlap = len(tokens(text) & query_terms)
        on_evidence_page = (str(obj.get("paper_id") or ""), int(obj.get("page") or 0)) in evidence_pages
        ranked.append((-int(on_evidence_page), -overlap, -float(obj.get("confidence") or 0), text, obj))
    ranked.sort(key=lambda item: (item[0], item[1], preferred_object_type(str(item[4].get("object_type") or "")), item[2]))
    chunks: list[str] = []
    used = 0
    for _, _, _, text, obj in ranked:
        header = f"paper={obj.get('paper_id')} page={obj.get('page')} type={obj.get('object_type')} label={obj.get('object_label') or ''}"
        chunk = f"[{header}]\n{text}"
        remaining = limit - used
        if remaining <= 0:
            break
        chunks.append(chunk[:remaining])
        used += len(chunk) + 2
    return "\n\n".join(chunks) if chunks else "[No source excerpt resolved]"


def build_messages(question: str, options: dict[str, Any], context: str) -> list[dict[str, str]]:
    ordered_options = "\n".join(f"{letter}. {value}" for letter, value in options.items())
    user = f"Question:\n{question}\n\nOrdered options:\n{ordered_options}\n\nRetrieved source excerpts:\n{context}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prepare(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.predictions)
    prediction_by_id = unique_by_query(predictions, "predictions")
    inputs = unique_by_query(read_jsonl(args.inputs), "inputs")
    query_ids = blank_mc_query_ids(predictions)
    selected_papers = {query_id: selected_paper_ids(prediction_by_id[query_id].get("gold_papers")) for query_id in query_ids}
    relevant = load_relevant_objects(args.object_index, set(query_ids), selected_papers)
    context_counts: dict[str, int] = {}
    for query_id in query_ids:
        input_row = inputs.get(query_id)
        if input_row is None:
            raise ValueError(f"missing input for {query_id}")
        options = (input_row.get("multiple_choice") or {}).get("options")
        if not isinstance(options, dict) or not options:
            raise ValueError(f"missing ordered options for {query_id}")
        objects = relevant.get(query_id, [])
        context = render_context(str(input_row.get("question") or ""), options, prediction_by_id[query_id].get("evidence") or [], objects)
        manifest = {
            "query_id": query_id,
            "model": args.model,
            "temperature": 0,
            "option_letters": list(options),
            "messages": build_messages(str(input_row.get("question") or ""), options, context),
            "source_object_count": len(objects),
            "gold_used": False,
        }
        context_counts[query_id] = len(objects)
        write_json(args.artifact_dir / "mc_prompt_manifests" / f"{query_id}.json", manifest)
    lock = {
        "created_at_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "predictions": {"path": str(args.predictions.resolve()), "sha256": sha256(args.predictions)},
        "inputs": {"path": str(args.inputs.resolve()), "sha256": sha256(args.inputs)},
        "object_index": {"path": str(args.object_index.resolve()), "sha256": sha256(args.object_index)},
        "model": args.model,
        "endpoint": args.endpoint,
        "blank_query_ids": query_ids,
        "source_object_counts": context_counts,
        "gold_used": False,
    }
    write_json(args.artifact_dir / "MC_SOURCE_LOCK.json", lock)
    print(json.dumps({"status": "PREPARED", "blank_mc_queries": len(query_ids), "zero_context_queries": sum(not v for v in context_counts.values())}))


def load_env(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path and path.is_file():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key.startswith("SILICONFLOW_") and value})
    return values


def extract_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def validate_answer(value: dict[str, Any], allowed_letters: list[str]) -> dict[str, str]:
    letter = str(value.get("letter") or "").strip().upper()
    if letter not in allowed_letters:
        raise ValueError(f"invalid option letter {letter!r}; allowed={allowed_letters}")
    return {"letter": letter, "semantic_answer": str(value.get("semantic_answer") or ""), "reason": str(value.get("reason") or "")}


def run(args: argparse.Namespace) -> None:
    lock = json.loads((args.artifact_dir / "MC_SOURCE_LOCK.json").read_text(encoding="utf-8"))
    if lock["predictions"]["sha256"] != sha256(args.predictions):
        raise ValueError("prediction hash no longer matches MC_SOURCE_LOCK.json")
    env = load_env(args.env_file)
    token = env.get("SILICONFLOW_API_KEY") or env.get("SILICONFLOW_TOKEN")
    if not token:
        raise RuntimeError("missing SILICONFLOW_API_KEY/SILICONFLOW_TOKEN")
    output_path = args.artifact_dir / "mc_recovery_decisions.jsonl"
    rows = read_jsonl(output_path) if output_path.is_file() else []
    existing = unique_by_query(rows, "MC decisions") if rows else {}
    generated = 0
    for manifest_path in sorted((args.artifact_dir / "mc_prompt_manifests").glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        query_id = manifest["query_id"]
        if query_id in existing:
            continue
        payload = json.dumps({
            "model": args.model,
            "temperature": 0,
            "messages": manifest["messages"],
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            args.endpoint.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code} for {query_id}: {body[:500]}") from exc
        content = response_value["choices"][0]["message"]["content"]
        answer = validate_answer(extract_json(content), manifest["option_letters"])
        rows.append({"query_id": query_id, **answer, "model": args.model, "usage": response_value.get("usage"), "created_at_utc": utc_now()})
        write_jsonl(output_path, rows)
        generated += 1
        print(json.dumps({"query_id": query_id, "letter": answer["letter"], "status": "RECORDED"}), flush=True)
        if args.max_records is not None and generated >= args.max_records:
            break


def freeze(args: argparse.Namespace) -> None:
    lock = json.loads((args.artifact_dir / "MC_SOURCE_LOCK.json").read_text(encoding="utf-8"))
    if lock["predictions"]["sha256"] != sha256(args.predictions):
        raise ValueError("prediction hash no longer matches MC_SOURCE_LOCK.json")
    predictions = read_jsonl(args.predictions)
    decisions = unique_by_query(read_jsonl(args.artifact_dir / "mc_recovery_decisions.jsonl"), "MC decisions")
    expected = set(lock["blank_query_ids"])
    if set(decisions) != expected:
        raise ValueError(f"decision coverage mismatch: missing={sorted(expected-set(decisions))}, extra={sorted(set(decisions)-expected)}")
    output_rows: list[dict[str, Any]] = []
    changed = 0
    for row in predictions:
        copied = json.loads(json.dumps(row, ensure_ascii=False))
        query_id = str(row["query_id"])
        if query_id in decisions:
            mc = copied["answer"]["multiple_choice"]
            if str(mc.get("gold") or "").strip():
                raise ValueError(f"refusing to overwrite nonblank MC answer: {query_id}")
            mc["gold"] = decisions[query_id]["letter"]
            changed += 1
        output_rows.append(copied)
    output = args.artifact_dir / "mc_candidate_predictions.jsonl"
    write_jsonl(output, output_rows)
    record = {
        "created_at_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_PRE_EVALUATION",
        "source_prediction_sha256": lock["predictions"]["sha256"],
        "decision_sha256": sha256(args.artifact_dir / "mc_recovery_decisions.jsonl"),
        "candidate_prediction_sha256": sha256(output),
        "record_count": len(output_rows),
        "mc_answers_filled": changed,
        "gold_used": False,
    }
    write_json(args.artifact_dir / "MC_CANDIDATE_FREEZE.json", record)
    print(json.dumps(record, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "freeze"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--object-index", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    if args.stage == "prepare" and (args.inputs is None or args.object_index is None):
        parser.error("prepare requires --inputs and --object-index")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    {"prepare": prepare, "run": run, "freeze": freeze}[arguments.stage](arguments)
