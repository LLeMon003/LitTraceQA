from src.model_challenger import assess, grounded, metrics


def test_grounded_requires_a_cited_evidence_value():
    record = {"record_id": "r", "answer_family": "multiple_choice", "answer": {"options": {"A": "alpha"}, "gold": "A"}, "reasoning_operator": "structured_lookup"}
    evidence = [{"object_id": "o", "content": "alpha"}]
    proposal = grounded(record, evidence, {"answer": "A", "source_object_ids": ["o"], "evidence_quote": "alpha", "confidence": 0.95})
    assert proposal and proposal["grounded"]
    assert proposal["grounding_mode"] == "option_value"
    assert grounded(record, evidence, {"answer": "A", "source_object_ids": ["missing"], "evidence_quote": "alpha"}) is None
    assert grounded(record, evidence, {"answer": "A", "source_object_ids": ["o"], "evidence_quote": "not present"}) is None
    assert grounded(record, evidence, {"answer": "A", "source_object_ids": ["o"], "evidence_quote": "alpha", "confidence": "not-a-number"}) is None


def test_mc_can_use_an_exact_source_quote_when_option_is_semantic():
    record = {"record_id": "r", "answer_family": "multiple_choice", "answer": {"options": {"A": "semantic option"}, "gold": "A"}, "reasoning_operator": "direct_extraction"}
    evidence = [{"object_id": "o", "content": "The apparatus reports a directly relevant physical observation."}]
    proposal = grounded(record, evidence, {"answer": "A", "source_object_ids": ["o"], "evidence_quote": "directly relevant physical observation", "confidence": 0.95})
    assert proposal and proposal["grounding_mode"] == "source_quote"


def test_metrics_score_only_after_grounding():
    record = {"record_id": "r", "answer_family": "freeform", "answer": {"text": "alpha"}, "reasoning_operator": "direct_extraction"}
    result = metrics([record], [{"record_id": "r", "prediction": "alpha", "source_object_ids": ["o"], "grounded": True}])
    assert result["accepted"] == result["correct"] == 1
    assert result["selective_exact_match"] == 1.0


def test_assess_reuses_raw_batches_and_persists_accepted_proposals():
    record = {"record_id": "r", "source_paper": "p", "answer_family": "multiple_choice", "answer": {"options": {"A": "alpha"}, "gold": "A"}, "reasoning_operator": "structured_lookup", "question": "alpha"}
    raw = '{"results":[{"record_id":"r","answer":"A","source_object_ids":["o"],"evidence_quote":"alpha","confidence":0.9}]}'
    batches = []
    proposals, raws = assess([record], {"p": [{"object_uid": "o", "paper_id": "p", "content": "alpha", "normalized_cell_value": "alpha", "raw_cell_value": "alpha", "source_hash": "s"}]}, lambda _: (_ for _ in ()).throw(AssertionError("must reuse")), reused_raw={0: raw}, on_batch=lambda index, value, accepted, reused: batches.append((index, value == raw, len(accepted), reused)))
    assert len(proposals) == len(raws) == 1
    assert batches == [(0, True, 1, True)]
