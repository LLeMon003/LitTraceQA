import json
import tempfile
import unittest
from pathlib import Path

from src.synthetic_benchmark import build_records, materialize, split_for


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_source_only_records_are_deterministic_and_strip_query_fields(self):
        facts = [
            {
                "ambiguity_status": "accepted", "paper_id": "paper_a", "page": 1,
                "object_type": "equation", "object_uid": "object_a", "normalized_value": "x = y",
                "source_hash": "source", "record_hash": "fact_hash", "query_ids": ["q_001"],
            },
            {
                "ambiguity_status": "accepted", "paper_id": "paper_a", "page": 1,
                "object_type": "equation", "object_uid": "object_b", "normalized_value": "z = 2",
            },
        ]
        tables = [
            {
                "provenance_status": "accepted", "paper_id": "paper_b", "page": 3,
                "evaluator_visible_table_id": "Table 2", "record_hash": "cell_hash",
                "normalized_cell_value": "42", "row_index": 2, "column_index": 4,
                "is_column_header": False, "is_row_header": False, "source_path": "q_002/private",
            }
        ]
        records = build_records(facts, tables)
        self.assertEqual(1, len(records))
        self.assertEqual("table_coordinate_lookup", records[0]["recipe"])
        self.assertNotIn("query_id", json.dumps(records[0]))
        self.assertEqual(records[0]["split"], split_for(records[0]["synthetic_id"]))

    def test_materialize_writes_hash_locked_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            facts = root / "facts.jsonl"
            ledger = root / "ledger.jsonl"
            facts.write_text(json.dumps({
                "ambiguity_status": "accepted", "paper_id": "paper_a", "page": 1,
                "object_type": "figure_caption", "object_uid": "only_object", "normalized_value": "Caption",
                "source_hash": "source", "record_hash": "hash",
            }) + "\n", encoding="utf-8")
            ledger.write_text(json.dumps({
                "provenance_status": "accepted", "paper_id": "paper_b", "page": 2,
                "evaluator_visible_table_id": "Table 1", "record_hash": "cell_hash",
                "normalized_cell_value": "value", "row_index": 1, "column_index": 1,
                "is_column_header": False, "is_row_header": False,
            }) + "\n", encoding="utf-8")
            manifest = materialize(facts, [ledger], root / "output")
            self.assertEqual(2, manifest["record_count"])
            self.assertTrue((root / "output" / "manifest.json").exists())
            self.assertEqual("complete", json.loads((root / "output" / "status.json").read_text())["status"])


if __name__ == "__main__":
    unittest.main()
