from src.target_aligned_benchmark import build


def cell(value, row, column):
    return {"provenance_status": "accepted", "is_column_header": False, "is_row_header": False, "paper_id": "paper_x", "page": 1, "evaluator_visible_table_id": "Table 1", "row_index": row, "column_index": column, "normalized_cell_value": value, "record_hash": f"h{row}{column}", "source_hash": "source"}


def test_target_builder_covers_required_mechanical_families():
    facts = [{"ambiguity_status": "accepted", "paper_id": "paper_y", "page": 2, "object_type": "figure_caption", "object_uid": "fact", "normalized_value": "caption text", "source_hash": "source"}]
    records = build(facts, [cell("11", 1, 1), cell("22", 1, 2), cell("33", 2, 1), cell("44", 2, 2), cell("absent", 3, 1)])
    operators = {x["reasoning_operator"] for x in records}
    families = {x["answer_family"] for x in records}
    assert {"multiple_choice", "freeform"} <= families
    assert {"structured_lookup", "comparison", "negation_except", "multi_object_count", "direct_extraction"} <= operators
    assert all(x["record_hash"] and x["source_objects"] and x["split"] for x in records)
