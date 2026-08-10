from src.target_retrieval import retrieve, score


def test_paper_constraint_and_answer_bearing_recall():
    good = {"paper_id": "p", "record_hash": "wanted", "source_hash": "s", "page": 1, "evaluator_visible_table_id": "Table 1", "row_index": 2, "column_index": 3, "normalized_cell_value": "42"}
    other = {**good, "paper_id": "other", "record_hash": "wrong"}
    record = {"question": "Which value is reported in Table 1 at row 2 and column 3?", "source_paper": "p", "reasoning_operator": "structured_lookup", "source_objects": [{"object_id": "wanted"}]}
    assert retrieve(record["question"], "p", [good, other])[0]["object_id"] == "wanted"
    assert score([record], [good, other])["answer_bearing_recall"] == 1.0


def test_fact_object_uid_is_the_retrieval_identity():
    fact = {"paper_id": "p", "object_uid": "fact-wanted", "record_hash": "audit-hash", "source_hash": "s", "page": 3, "object_type": "equation_block", "normalized_value": "x"}
    record = {"question": "According to the equation_block in paper p on page 3, what text is stated?", "source_paper": "p", "reasoning_operator": "direct_extraction", "source_objects": [{"object_id": "fact-wanted"}]}
    assert score([record], [fact])["answer_bearing_recall"] == 1.0
