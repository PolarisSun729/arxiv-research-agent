import unittest
from unittest import mock

from tests.helpers import build_retrieval_service, build_sample_chunks


class _FakeReranker:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"pairs": list(pairs), "kwargs": dict(kwargs)})
        return list(self.scores)


class RerankServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service, self.collection_name, *_ = build_retrieval_service()
        self.sample_chunks = build_sample_chunks()
        self.query_bundle = self.service.query_planner.build_query_bundle(
            user_query="What does the method of the paper do?",
            collection_name=self.collection_name,
            enable_query_rewrite=True,
        )
        self.query_profile = self.query_bundle["query_profile"]

    def test_build_rerank_document_text_uses_asset_metadata_for_figure_table(self) -> None:
        figure_chunk = self.service._normalize_chunk(self.sample_chunks[5])

        text = self.service.rerank_service.build_rerank_document_text(figure_chunk)

        self.assertIn("Figure 2 compares model performance", text)
        self.assertIn("page 6", text)

    def test_rerank_success_changes_chunk_order(self) -> None:
        chunks = [
            dict(self.service._normalize_chunk(self.sample_chunks[0]), score=0.2, route_rank=1),
            dict(self.service._normalize_chunk(self.sample_chunks[2]), score=0.1, route_rank=2),
        ]
        reranker = _FakeReranker(scores=[0.1, 5.0])
        self.service.llm_rerank_provider = "local"
        self.service._llm_reranker_path = "fake-reranker"
        self.service._llm_reranker_device = "cpu"

        with mock.patch.object(self.service, "_load_llm_reranker", return_value=reranker):
            result = self.service.rerank_service.llm_rerank(
                "method evidence",
                chunks,
                top_k=2,
                query_profile=self.query_profile,
                original_question="What does the method do?",
                candidate_limit=2,
            )

        reranked_chunks = result["reranked_chunks"]
        self.assertTrue(result["debug"]["applied"])
        self.assertEqual(reranked_chunks[0]["chunk_id"], "chunk-results")
        self.assertEqual(reranked_chunks[1]["chunk_id"], "chunk-method")

    def test_rerank_failure_falls_back_to_original_order(self) -> None:
        chunks = [
            dict(self.service._normalize_chunk(self.sample_chunks[0]), score=0.2, route_rank=1),
            dict(self.service._normalize_chunk(self.sample_chunks[2]), score=0.1, route_rank=2),
        ]
        self.service.llm_rerank_provider = "local"
        self.service._llm_reranker_error = "offline"

        with mock.patch.object(self.service, "_load_llm_reranker", return_value=None):
            result = self.service.rerank_service.llm_rerank(
                "method evidence",
                chunks,
                top_k=2,
                query_profile=self.query_profile,
                original_question="What does the method do?",
                candidate_limit=2,
            )

        self.assertFalse(result["debug"]["applied"])
        self.assertEqual([item["chunk_id"] for item in result["chunks"]], ["chunk-method", "chunk-results"])

    def test_noisy_section_penalty_penalizes_appendix_and_references(self) -> None:
        appendix_chunk = self.service._normalize_chunk(self.sample_chunks[6])
        references_chunk = self.service._normalize_chunk(self.sample_chunks[7])
        method_chunk = self.service._normalize_chunk(self.sample_chunks[0])

        appendix_bonus = self.service._compute_structural_bonus(appendix_chunk, self.query_profile)
        references_bonus = self.service._compute_structural_bonus(references_chunk, self.query_profile)
        method_bonus = self.service._compute_structural_bonus(method_chunk, self.query_profile)

        self.assertLess(appendix_bonus, 0.0)
        self.assertLess(references_bonus, 0.0)
        self.assertGreater(method_bonus, appendix_bonus)


if __name__ == "__main__":
    unittest.main()
