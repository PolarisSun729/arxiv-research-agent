import unittest
from unittest import mock

from tests.helpers import build_retrieval_service, build_sample_chunks


class EnhancedRetrievalServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service, self.collection_name, *_ = build_retrieval_service()
        self.sample_chunks = build_sample_chunks()
        self.options_cls = type(self.service).enhanced_retrieve.__globals__["RetrievalOptions"]

    def test_enhanced_retrieve_top_k_boundaries_and_debug_snapshot(self) -> None:
        default_result = self.service.enhanced_retrieve(
            "What is the method of the paper?",
            self.collection_name,
            options=self.options_cls(top_k=None, debug=True, enable_llm_rerank=False),
        )
        one_result = self.service.enhanced_retrieve(
            "What is the method of the paper?",
            self.collection_name,
            options=self.options_cls(top_k=1, debug=True, enable_llm_rerank=False),
        )
        capped_result = self.service.enhanced_retrieve(
            "What is the method of the paper?",
            self.collection_name,
            options=self.options_cls(top_k=99, debug=True, enable_llm_rerank=False),
        )

        self.assertEqual(len(default_result["chunks"]), 3)
        self.assertEqual(len(one_result["chunks"]), 1)
        self.assertEqual(len(capped_result["chunks"]), 4)
        debug = default_result["debug"]
        self.assertIn("raw_retrieval_top30", debug["stages"])
        self.assertIn("fused_top30", debug["stages"])
        self.assertIn("final_context_top15", debug["stages"])
        self.assertIn("query_profile", debug)

    def test_enhanced_retrieve_fuses_routes_dedupes_and_preserves_source_fields(self) -> None:
        result = self.service.enhanced_retrieve(
            "What does Figure 2 and Table 3 show?",
            self.collection_name,
            options=self.options_cls(debug=True, enable_llm_rerank=False),
        )

        first_chunk = result["chunks"][0]
        debug = result["debug"]
        raw_ids = [item["chunk_id"] for item in debug["stages"]["raw_retrieval_top30"]]
        fused_ids = [item["chunk_id"] for item in debug["stages"]["fused_top30"]]

        self.assertIn("page_number", first_chunk)
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("parent_chunk_id", first_chunk)
        self.assertIn("section_path", first_chunk.get("metadata", {}))
        self.assertEqual(len(fused_ids), len(set(fused_ids)))
        self.assertTrue(any(len(item.get("matched_routes", [])) > 1 for item in result["chunks"]))
        self.assertIn("mode", debug["llm_rerank"])

    def test_enhanced_retrieve_rerank_failure_falls_back_to_fused_order(self) -> None:
        with mock.patch.object(self.service, "_load_llm_reranker", return_value=None):
            self.service.llm_rerank_provider = "local"
            self.service._llm_reranker_error = "unavailable"
            result = self.service.enhanced_retrieve(
                "What is the method of the paper?",
                self.collection_name,
                options=self.options_cls(debug=True, enable_llm_rerank=True, top_k=2),
            )

        fused_ids = [item["chunk_id"] for item in result["debug"]["stages"]["fused_top30"][:2]]
        final_ids = [item["chunk_id"] for item in result["chunks"]]

        self.assertFalse(result["debug"]["llm_rerank"]["applied"])
        self.assertEqual(final_ids, fused_ids)

    def test_debug_snapshot_contains_required_trace_sections(self) -> None:
        result = self.service.enhanced_retrieve(
            "Which dataset and results are most important?",
            self.collection_name,
            options=self.options_cls(debug=True, enable_llm_rerank=False),
        )
        debug = result["debug"]

        self.assertIn("query_profile", debug)
        self.assertIn("llm_rerank", debug)
        self.assertIn("raw_retrieval_top30", debug["stages"])
        self.assertIn("fused_top30", debug["stages"])
        self.assertIn("final_context_top15", debug["stages"])
        self.assertIn("routes", debug)


if __name__ == "__main__":
    unittest.main()
