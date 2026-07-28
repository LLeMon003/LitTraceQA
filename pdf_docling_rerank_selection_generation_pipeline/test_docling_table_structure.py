import unittest

from .table_structure import docling_table_to_structure


class DoclingTableStructureTests(unittest.TestCase):
    def test_preserves_merged_headers_row_labels_and_caption_footnotes(self):
        payload = {
            "data": {
                "num_rows": 3,
                "num_cols": 3,
                "table_cells": [
                    {"start_row_offset_idx": 0, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Method", "column_header": True},
                    {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 1, "end_col_offset_idx": 3, "text": "Score", "column_header": True},
                    {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "Acc", "column_header": True},
                    {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 2, "end_col_offset_idx": 3, "text": "F1", "column_header": True},
                    {"start_row_offset_idx": 2, "end_row_offset_idx": 3, "start_col_offset_idx": 0, "end_col_offset_idx": 1, "text": "Ours", "row_header": True},
                    {"start_row_offset_idx": 2, "end_row_offset_idx": 3, "start_col_offset_idx": 1, "end_col_offset_idx": 2, "text": "91.2"},
                    {"start_row_offset_idx": 2, "end_row_offset_idx": 3, "start_col_offset_idx": 2, "end_col_offset_idx": 3, "text": "89.4"},
                ],
            }
        }
        structure = docling_table_to_structure(payload, "Table 1. Main result. † means tuned with validation data.")
        assert structure is not None
        self.assertEqual(structure["header_rows"], [["Method", "Score", "Score"], ["Method", "Acc", "F1"]])
        self.assertEqual(structure["rows"][0]["row_label"], "Ours")
        self.assertEqual(structure["cells"][1]["column_span"], 2)
        self.assertTrue(structure["footnotes"])


if __name__ == "__main__":
    unittest.main()
