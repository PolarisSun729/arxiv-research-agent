from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from services.retrieval.contracts import RetrievalOptions
from tests.helpers.retrieval import build_retrieval_service


class RagGoldenPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service, self.collection_name, *_ = build_retrieval_service()
        self.service.trace_export_dir = Path(self.temp_dir.name)
        self.service.trace_builder.trace_export_dir = Path(self.temp_dir.name)
        self.paper_context = {
            "arxiv_id": "2401.00001",
            "title": "Fixed RAG Golden Paper",
            "abstract": "A deterministic paper fixture for retrieval regression.",
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _retrieve(self, question: str, *, top_k: int = 4) -> Dict[str, Any]:
        options = RetrievalOptions(
            top_k=top_k,
            enable_query_rewrite=True,
            enable_hyde=False,
            enable_keyword_search=True,
            enable_llm_rerank=False,
            debug=True,
        )
        return self.service.retrieval_pipeline.retrieve(
            question,
            self.collection_name,
            paper_context=self.paper_context,
            options=options,
        )

    @staticmethod
    def _chunk_ids(chunks: Iterable[Mapping[str, Any]]) -> List[str]:
        return [str(chunk.get("chunk_id") or "") for chunk in chunks]

    @staticmethod
    def _field(chunk: Mapping[str, Any], name: str) -> Any:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        return chunk.get(name, metadata.get(name))

    def _assert_evidence_hit(
        self,
        *,
        question: str,
        expected_chunk_id: str,
        expected_section: str,
        expected_page: int,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        result = self._retrieve(question, top_k=top_k)
        chunks = list(result["chunks"])
        chunk_ids = self._chunk_ids(chunks)
        debug = result["debug"]

        self.assertIn(expected_chunk_id, chunk_ids, msg={"question": question, "chunk_ids": chunk_ids})
        matched = next(chunk for chunk in chunks if chunk.get("chunk_id") == expected_chunk_id)
        self.assertEqual(int(self._field(matched, "page_number")), expected_page)
        self.assertIn(expected_section.lower(), str(self._field(matched, "section_path") or "").lower())

        # golden 测试关注检索证据和可排查性，不要求生成答案逐字一致。
        self.assertIn("raw_retrieval_top30", debug["stages"])
        self.assertIn("fused_top30", debug["stages"])
        self.assertIn("reranked_top30", debug["stages"])
        self.assertIn("final_context_top15", debug["stages"])
        self.assertIn("llm_rerank", debug)
        self.assertFalse(debug["llm_rerank"]["applied"])
        self.assertIn("trace_export", result)
        self.assertTrue(Path(result["trace_export"]["json"]).exists())
        self.assertTrue(Path(result["trace_export"]["md"]).exists())
        return result

    def test_method_experiment_and_dataset_questions_hit_expected_evidence(self) -> None:
        cases = [
            {
                "question": "What is the method framework of the paper?",
                "expected_chunk_id": "chunk-method",
                "expected_section": "Method",
                "expected_page": 2,
            },
            {
                "question": "Which dataset and metrics are used in the experiments?",
                "expected_chunk_id": "chunk-experiment",
                "expected_section": "Experiments",
                "expected_page": 4,
            },
            {
                "question": "How is the dataset described and split?",
                "expected_chunk_id": "chunk-dataset",
                "expected_section": "Dataset",
                "expected_page": 3,
            },
        ]

        for case in cases:
            with self.subTest(question=case["question"]):
                result = self._assert_evidence_hit(**case)
                trace_payload = json.loads(Path(result["trace_export"]["json"]).read_text(encoding="utf-8"))
                self.assertEqual(trace_payload["paper_context"]["arxiv_id"], "2401.00001")
                self.assertTrue(trace_payload["steps"])
                self.assertIn(trace_payload["sparse_index"]["load_source"], {"persistent_sparse_artifact", "runtime_build_fallback"})
                self.assertIn("keyword_search", trace_payload)
                self.assertIn("sparse_index", [step["step"] for step in trace_payload["steps"]])

    def test_figure_table_question_keeps_visual_asset_in_candidates_and_trace(self) -> None:
        result = self._assert_evidence_hit(
            question="What does Figure 2 and Table 3 show?",
            expected_chunk_id="chunk-figure-table",
            expected_section="Figure 2",
            expected_page=6,
            top_k=4,
        )

        asset_counts = result["debug"]["asset_type_counts"]
        final_counts = asset_counts["final_context_top15"]
        self.assertGreaterEqual(final_counts.get("figure", 0), 1)

        figure_chunk = next(chunk for chunk in result["chunks"] if chunk.get("chunk_id") == "chunk-figure-table")
        self.assertEqual(self._field(figure_chunk, "chunk_type"), "figure")
        self.assertIn("figure-2.png", str(self._field(figure_chunk, "asset_path") or ""))
        self.assertIn("chunk-figure-table", Path(result["trace_export"]["md"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
