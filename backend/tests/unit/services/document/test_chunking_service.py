import unittest

from services.document.chunking_service import ChunkingService


class ChunkingServiceAssetSectionTests(unittest.TestCase):
    def test_extract_asset_reference_labels_normalizes_common_caption_forms(self) -> None:
        service = ChunkingService()

        cases = [
            ({"asset_kind": "picture", "caption": "Figure 1: Overview"}, [{"kind": "figure", "label": "1"}]),
            ({"asset_kind": "picture", "caption": "Fig. 2(a) Pipeline"}, [{"kind": "figure", "label": "2a"}]),
            ({"asset_kind": "table", "caption": "Table 3: Scores"}, [{"kind": "table", "label": "3"}]),
            ({"asset_kind": "table", "caption": "Tab. 4 Ablation"}, [{"kind": "table", "label": "4"}]),
        ]

        for asset, expected in cases:
            with self.subTest(caption=asset["caption"]):
                references = service._extract_asset_reference_labels(asset)
                self.assertEqual(
                    [{"kind": item["kind"], "label": item["label"]} for item in references],
                    expected,
                )

    def test_caption_reference_match_wins_over_same_page_position_fallback(self) -> None:
        service = ChunkingService()
        sections = [
            {
                "title": "Method",
                "level": 1,
                "path": "1 Method",
                "page_start": 2,
                "page_end": 2,
                "heading_item": {"order_index": 1},
                "body_items": [
                    {"text": "Figure 1 shows the full architecture before the implementation details.", "order_index": 2}
                ],
            },
            {
                "title": "Results",
                "level": 1,
                "path": "2 Results",
                "page_start": 5,
                "page_end": 5,
                "heading_item": {"order_index": 10},
                "body_items": [
                    {"text": "The following page contains additional qualitative examples.", "order_index": 11},
                    {"text": "Figure 1 is revisited when discussing the ablation trend.", "order_index": 12},
                ],
            },
        ]
        document = {
            "docling_picture_items": [
                {
                    "asset_kind": "picture",
                    "asset_path": "figure-1.png",
                    "page_start": 5,
                    "order_index": 14,
                    "caption": "Fig. 1: Architecture overview.",
                    "asset_summary": "Architecture overview.",
                }
            ]
        }

        chunks = service._build_docling_asset_chunks(document, [], "paper.pdf", sections)

        metadata = chunks[0]["metadata"]
        self.assertEqual(metadata["asset_section_match_type"], "caption_reference")
        self.assertFalse(metadata["asset_section_match_is_heuristic"])
        self.assertGreaterEqual(metadata["asset_section_match_confidence"], 0.80)
        self.assertEqual(metadata["section_path"], "1 Method")
        self.assertIn("basis=caption_reference", metadata["asset_section_match_reason"])
        self.assertIn("matched_ref=Figure 1", metadata["asset_section_match_reason"])
        self.assertIn("1 Method", chunks[0]["content"])

    def test_docling_asset_section_match_records_heuristic_metadata_and_gates_content(self) -> None:
        service = ChunkingService()
        sections = [
            {
                "title": "Method",
                "level": 1,
                "path": "1 Method",
                "page_start": 1,
                "page_end": 1,
                "heading_item": {"order_index": 1},
            },
            {
                "title": "Results",
                "level": 1,
                "path": "2 Results",
                "page_start": 5,
                "page_end": 5,
                "heading_item": {"order_index": 10},
            },
        ]
        document = {
            "docling_picture_items": [
                {
                    "asset_kind": "picture",
                    "asset_path": "figure-low.png",
                    "page_start": 3,
                    "order_index": 80,
                    "asset_summary": "Floating figure summary.",
                }
            ],
            "docling_table_items": [
                {
                    "asset_kind": "table",
                    "asset_path": "table-high.csv",
                    "page_start": 5,
                    "order_index": 12,
                    "asset_summary": "Table 1 reports the main results.",
                    "asset_preview": [{"Method": "Ours", "Accuracy": "91.0%"}],
                }
            ],
        }

        chunks = service._build_docling_asset_chunks(document, [], "paper.pdf", sections)

        by_asset_path = {chunk["metadata"]["asset_path"]: chunk for chunk in chunks}
        low_chunk = by_asset_path["figure-low.png"]
        low_metadata = low_chunk["metadata"]
        high_chunk = by_asset_path["table-high.csv"]
        high_metadata = high_chunk["metadata"]

        self.assertEqual(low_metadata["asset_section_match_type"], "heuristic")
        self.assertTrue(low_metadata["asset_section_match_is_heuristic"])
        self.assertFalse(low_metadata["asset_section_match_allow_embedding"])
        self.assertIn("basis=page_range+order_index", low_metadata["asset_section_match_reason"])
        self.assertEqual(low_metadata["section_path"], "1 Method")
        self.assertNotIn("1 Method", low_chunk["content"])
        self.assertIn("page 3", low_chunk["content"])

        self.assertEqual(high_metadata["asset_section_match_type"], "heuristic")
        self.assertTrue(high_metadata["asset_section_match_is_heuristic"])
        self.assertTrue(high_metadata["asset_section_match_allow_embedding"])
        self.assertGreaterEqual(high_metadata["asset_section_match_confidence"], 0.70)
        self.assertEqual(high_metadata["section_path"], "2 Results")
        self.assertIn("2 Results", high_chunk["content"])


class ChunkingServiceDoclingNoiseTests(unittest.TestCase):
    def test_page_header_and_footer_are_dropped_from_docling_items(self) -> None:
        # 字段值取自真实 Docling 输出：文档对象导出的层是 "ContentLayer.FURNITURE"，原始 JSON 里是 "furniture"。
        items = [
            {"text": "Private RAG via Random Projection", "label": "page_header", "content_layer": "ContentLayer.FURNITURE", "role": "text", "page_start": 2, "order_index": 1},
            {"text": "A PREPRINT", "label": "page_header", "content_layer": "ContentLayer.FURNITURE", "role": "heading", "page_start": 2, "order_index": 2},
            {"text": "Body paragraph about the threat model and attacker capabilities.", "label": "text", "content_layer": "ContentLayer.BODY", "page_start": 2, "order_index": 3},
            {"text": "2", "label": "page_footer", "content_layer": "furniture", "page_start": 2, "order_index": 4},
        ]

        collected = ChunkingService()._collect_docling_items({"docling_text_items": items}, [])

        self.assertEqual([item["text"] for item in collected], ["Body paragraph about the threat model and attacker capabilities."])



class ChunkingServiceDoclingHeadingTests(unittest.TestCase):
    """章节标题只认 Docling 的 section_header / title 标签，不按文字长相判断。"""

    def test_section_header_label_is_trusted_and_text_is_never_promoted(self) -> None:
        items = [
            {"text": "1. Introduction", "label": "section_header", "content_layer": "body", "page_start": 1, "order_index": 1},
            {"text": "Offline preprocessing", "label": "text", "content_layer": "body", "page_start": 1, "order_index": 2},
            {"text": "A.1 Hardware Infrastructure", "label": "section_header", "content_layer": "body", "page_start": 9, "order_index": 3},
            {"text": "User query q", "label": "list_item", "content_layer": "body", "page_start": 9, "order_index": 4},
            {"text": "Encoder", "label": "section_header", "content_layer": "body", "parent": "#/pictures/0", "page_start": 9, "order_index": 5},
        ]

        service = ChunkingService()
        collected = service._collect_docling_items({"docling_text_items": items}, [])
        headings = [item["text"] for item in collected if service._docling_item_is_heading(item)]

        self.assertEqual(headings, ["1. Introduction", "A.1 Hardware Infrastructure"])

if __name__ == "__main__":
    unittest.main()
