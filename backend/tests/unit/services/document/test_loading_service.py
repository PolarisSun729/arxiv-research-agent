import unittest

from services.document.loading_service import LoadingService


class LoadingServiceDoclingFurnitureTests(unittest.TestCase):
    """页眉页脚在 Docling 里是 furniture 层的 page_header / page_footer，不能当正文或标题。"""

    def test_page_header_and_footer_are_classified_as_noise(self) -> None:
        service = LoadingService()
        cases = [
            ("page_header", "A PREPRINT", "ContentLayer.FURNITURE"),
            ("page_footer", "5", "ContentLayer.FURNITURE"),
            ("page_header", "Private RAG via Random Projection", "furniture"),
        ]
        for label, text, layer in cases:
            with self.subTest(label=label, text=text, layer=layer):
                kind = service._classify_docling_item_kind(label, text, None, layer, "{'cref': '#/body'}")
                self.assertEqual(kind, "noise")

        body = "Another strong attack is the multi-round adaptive attack, which refines prompts iteratively."
        self.assertEqual(service._classify_docling_item_kind("text", body, None, "ContentLayer.BODY", "{'cref': '#/body'}"), "paragraph")

    def test_annotation_marks_page_furniture_separately_from_titles(self) -> None:
        service = LoadingService()
        header = {"label": "page_header", "content_layer": "furniture", "text": "Private RAG via Random Projection"}
        heading = {"label": "section_header", "content_layer": "body", "text": "4.3 Multi-Round Adaptive Attacks"}

        self.assertEqual(service._docling_annotation_category(header), "furniture")
        self.assertEqual(service._docling_annotation_category(heading), "title")
        self.assertEqual(service._get_docling_annotation_style("furniture")["kind"], "furniture")


if __name__ == "__main__":
    unittest.main()
