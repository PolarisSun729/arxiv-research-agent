import json
import tempfile
import unittest
from pathlib import Path

from services.document.table_structure_service import TableStructureService


class TableStructureServiceTests(unittest.TestCase):
    def test_prefers_asset_json_path_and_normalizes_numeric_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            table_path = Path(temp_dir) / "table.json"
            table_path.write_text(
                json.dumps(
                    [
                        {"Method": "Baseline", "Accuracy": "82.5%", "BLEU": "18.4"},
                        {"Method": "Ours", "Accuracy": "91.0%", "BLEU": "22.1"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            chunks = [
                {
                    "content": "Table evidence",
                    "metadata": {
                        "chunk_type": "table",
                        "chunk_id": 3,
                        "page_start": 4,
                        "section_path": "Experiments > Results",
                        "asset_section_match_type": "heuristic",
                        "asset_section_match_confidence": 0.82,
                        "asset_section_match_reason": "basis=page_range+order_index",
                        "asset_section_match_is_heuristic": True,
                        "asset_section_match_allow_embedding": True,
                        "asset_json_path": str(table_path),
                        "asset_caption": "Table 1: Main results",
                        "order_index": 1,
                    },
                }
            ]

            result = TableStructureService().build_structured_tables("2401.00001", {}, chunks)

        self.assertEqual(result["debug"]["table_count"], 1)
        self.assertEqual(result["debug"]["structured_table_count"], 1)
        self.assertEqual(result["debug"]["failed_table_parse_count"], 0)
        self.assertIn("table_id", chunks[0]["metadata"])
        table_object = result["structured_tables"][0]
        self.assertEqual(table_object["columns"], ["Method", "Accuracy", "BLEU"])
        self.assertEqual(table_object["source_chunk_id"], 3)
        self.assertEqual(table_object["asset_section_match_type"], "heuristic")
        self.assertTrue(table_object["asset_section_match_allow_embedding"])
        accuracy_cell = next(
            cell
            for cell in table_object["cells"]
            if cell["row_label"] == "Ours" and cell["col_name"] == "Accuracy"
        )
        self.assertAlmostEqual(accuracy_cell["normalized_value"], 0.91)
        self.assertEqual(accuracy_cell["unit"], "percent")
        bleu_cell = next(
            cell
            for cell in table_object["cells"]
            if cell["row_label"] == "Ours" and cell["col_name"] == "BLEU"
        )
        self.assertEqual(bleu_cell["normalized_value"], 22.1)
        self.assertEqual(bleu_cell["unit"], "bleu")

    def test_fallback_parses_only_regular_preview_text(self) -> None:
        chunks = [
            {
                "content": "Table evidence",
                "metadata": {
                    "chunk_type": "table",
                    "chunk_id": 4,
                    "page_start": 6,
                    "asset_preview_text": "Method: Baseline; F1: 0.71 | Method: Ours; F1: 0.83",
                    "order_index": 2,
                },
            }
        ]

        result = TableStructureService().build_structured_tables("2401.00001", {}, chunks)

        self.assertEqual(result["debug"]["structured_table_count"], 1)
        table_object = result["structured_tables"][0]
        self.assertEqual(table_object["parse_source"], "fallback_text")
        self.assertEqual(table_object["columns"], ["Method", "F1"])
        ours_f1 = next(
            cell
            for cell in table_object["cells"]
            if cell["row_label"] == "Ours" and cell["col_name"] == "F1"
        )
        self.assertEqual(ours_f1["normalized_value"], 0.83)
        self.assertEqual(ours_f1["unit"], "f1")

    def test_irregular_preview_counts_as_failed_without_raising(self) -> None:
        chunks = [
            {
                "content": "Table evidence",
                "metadata": {
                    "chunk_type": "table",
                    "chunk_id": 5,
                    "page_start": 7,
                    "asset_preview_text": "Method Baseline F1 0.71 | Ours 0.83",
                    "order_index": 3,
                },
            }
        ]

        result = TableStructureService().build_structured_tables("2401.00001", {}, chunks)

        self.assertEqual(result["debug"]["table_count"], 1)
        self.assertEqual(result["debug"]["structured_table_count"], 0)
        self.assertEqual(result["debug"]["failed_table_parse_count"], 1)
        self.assertNotIn("table_id", chunks[0]["metadata"])


if __name__ == "__main__":
    unittest.main()
