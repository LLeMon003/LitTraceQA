"""Constrained, provenance-grounded model extraction for target-aligned development."""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable

from src.target_retrieval import retrieve


def answer_text(record: dict[str, Any]) -> str:
    answer = record["answer"]
    if record["answer_family"] == "multiple_choice":
        return str(answer["gold"])
    return str(answer["text"])


def evidence_for(record: dict[str, Any], corpus: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    table_wide = record["reasoning_operator"] in {"comparison", "negation_except", "multi_object_count"}
    return retrieve(record["question"], record["source_paper"], corpus, limit=512 if table_wide else 16)


def request_payload(records: list[dict[str, Any]], bundles: list[list[dict[str, Any]]]) -> str:
    items = []
    for record, evidence in zip(records, bundles, strict=True):
        options = record["answer"].get("options") if record["answer_family"] == "multiple_choice" else None
        items.append({"record_id": record["record_id"], "question": record["question"], "options": options, "evidence": [{"object_id": item["object_id"], "content": item["content"]} for item in evidence]})
    return json.dumps(
        {
            "task": "Answer only from evidence. Return exact option letter for multiple choice, otherwise exact text. Cite every supporting object id. Include evidence_quote: a nonempty exact substring copied from one cited source object that supports the answer.",
            "records": items,
        },
        ensure_ascii=False,
    )


def grounded(record: dict[str, Any], evidence: list[dict[str, Any]], result: dict[str, Any]) -> dict[str, Any] | None:
    prediction = result.get("answer")
    cited = result.get("source_object_ids")
    if not isinstance(prediction, str) or not isinstance(cited, list) or not cited:
        return None
    evidence_by_id = {item["object_id"]: item for item in evidence}
    if any(not isinstance(item, str) or item not in evidence_by_id for item in cited):
        return None
    quote = result.get("evidence_quote")
    if not isinstance(quote, str) or not quote.strip():
        return None
    cited_content = [str(evidence_by_id[item]["content"]) for item in cited]
    quote_supported = any(quote in content for content in cited_content)
    if not quote_supported:
        return None
    if record["answer_family"] == "multiple_choice":
        options = record["answer"].get("options", {})
        if prediction not in options:
            return None
        value = str(options[prediction])
    else:
        value = prediction
    value_supported = any(value in content for content in cited_content)
    if record["answer_family"] != "multiple_choice" and not value_supported:
        return None
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    return {
        "record_id": record["record_id"],
        "prediction": prediction,
        "source_object_ids": cited,
        "grounded": True,
        "grounding_mode": "option_value" if value_supported else "source_quote",
        "confidence": confidence,
    }


def assess(
    records: list[dict[str, Any]],
    corpus_by_paper: dict[str, list[dict[str, Any]]],
    call: Callable[[str], str],
    *,
    reused_raw: dict[int, str] | None = None,
    on_batch: Callable[[int, str, list[dict[str, Any]], bool], None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    proposals: list[dict[str, Any]] = []
    raws: list[str] = []
    for batch_index, start in enumerate(range(0, len(records), 2)):
        batch = records[start : start + 2]
        bundles = [evidence_for(record, corpus_by_paper.get(str(record["source_paper"]), [])) for record in batch]
        reused = reused_raw is not None and batch_index in reused_raw
        raw = reused_raw[batch_index] if reused else call(request_payload(batch, bundles)
        )
        raws.append(raw)
        try:
            results = json.loads(raw).get("results", [])
        except json.JSONDecodeError:
            if on_batch is not None:
                on_batch(batch_index, raw, [], reused)
            continue
        indexed = {item.get("record_id"): item for item in results if isinstance(item, dict)}
        batch_proposals: list[dict[str, Any]] = []
        for record, evidence in zip(batch, bundles, strict=True):
            proposal = grounded(record, evidence, indexed.get(record["record_id"], {}))
            if proposal:
                batch_proposals.append(proposal)
        proposals.extend(batch_proposals)
        if on_batch is not None:
            on_batch(batch_index, raw, batch_proposals, reused)
    return proposals, raws


def metrics(records: Iterable[dict[str, Any]], proposals: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records_by_id = {record["record_id"]: record for record in records}
    accepted = list(proposals)
    correct = [proposal for proposal in accepted if answer_text(records_by_id[proposal["record_id"]]) == proposal["prediction"]]
    return {"accepted": len(accepted), "correct": len(correct), "selective_exact_match": len(correct) / len(accepted) if accepted else 0.0, "all_grounded": all(item["grounded"] for item in accepted), "families": sorted({records_by_id[item["record_id"]]["answer_family"] for item in accepted}), "operators": sorted({records_by_id[item["record_id"]]["reasoning_operator"] for item in accepted})}
