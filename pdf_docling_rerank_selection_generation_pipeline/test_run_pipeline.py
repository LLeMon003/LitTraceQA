import json
import tempfile
import unittest
from pathlib import Path

from .run_pipeline import _completed_query_ids


class RunPipelineTests(unittest.TestCase):
    def test_completed_query_ids_uses_predictions_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                json.dumps({"query_id": "q_001"}) + "\n" + json.dumps({"query_id": "q_002"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_completed_query_ids(path), {"q_001", "q_002"})
