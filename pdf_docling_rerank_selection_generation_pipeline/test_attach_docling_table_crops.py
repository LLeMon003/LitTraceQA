from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .attach_docling_table_crops import _docling_table_index, _table_number


class DoclingTableCropTests(unittest.TestCase):
    def test_table_number_requires_explicit_table_label(self) -> None:
        self.assertEqual(_table_number("Table 12"), 12)
        self.assertEqual(_table_number("see table 3 for results"), 3)
        self.assertIsNone(_table_number("Figure 3"))

    def test_docling_table_index_is_one_based(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.docling.json"
            path.write_text(json.dumps({"tables": [{"label": "table"}, {"label": "table"}]}), encoding="utf-8")
            self.assertEqual(sorted(_docling_table_index(path)), [1, 2])
