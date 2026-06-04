import os
import tempfile
import unittest
from pathlib import Path

from routers.qa_utils import build_qa_diagnostic, get_latest_retrieval_trace, sanitize_trace_slug


class _FakeEnhancedRetrievalService:
    def __init__(self, trace_export_dir: str) -> None:
        self.trace_export_dir = trace_export_dir


class _FakeDbService:
    def __init__(self, qa_index):
        self.qa_index = qa_index

    def get_paper_qa_index(self, arxiv_id: str):
        return self.qa_index


class _FakeVectorStoreService:
    def __init__(self) -> None:
        self._sample_chunks = [{"chunk_id": "c1", "content": "chunk one"}]

    def list_collections(self, _provider: str):
        return ["paper_123"]

    def collection_exists(self, _provider: str, collection_name: str) -> bool:
        return collection_name == "paper_123"

    def get_collection_info(self, _provider: str, collection_name: str):
        return {"collection_name": collection_name, "num_entities": 1}

    def get_all_chunks(self, collection_name: str, limit: int = 1):
        return self._sample_chunks[:limit]


class QaUtilsUnitTests(unittest.TestCase):
    def test_sanitize_trace_slug_normalizes_text(self) -> None:
        self.assertEqual(sanitize_trace_slug(" RAG / QA ??? "), "RAG_QA")
        self.assertEqual(sanitize_trace_slug(""), "query")

    def test_get_latest_retrieval_trace_returns_latest_matching_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paper_dir = Path(temp_dir) / sanitize_trace_slug("2401.00001")
            paper_dir.mkdir(parents=True, exist_ok=True)
            first = paper_dir / "trace-1.md"
            second = paper_dir / "trace-2.md"
            first.write_text("older", encoding="utf-8")
            second.write_text("newer", encoding="utf-8")
            os.utime(first, (1, 1))
            os.utime(second, (2, 2))

            result = get_latest_retrieval_trace(_FakeEnhancedRetrievalService(temp_dir), "2401.00001")

        self.assertEqual(result.name, "trace-2.md")

    def test_get_latest_retrieval_trace_returns_none_when_directory_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = get_latest_retrieval_trace(_FakeEnhancedRetrievalService(temp_dir), "missing")

        self.assertIsNone(result)

    def test_build_qa_diagnostic_reports_collection_health(self) -> None:
        diagnostic = build_qa_diagnostic(
            db_service=_FakeDbService({"collection_name": "paper_123", "status": "indexed", "chunk_count": 1}),
            vector_store_service=_FakeVectorStoreService(),
            arxiv_id="2401.00001",
            sample_limit=2,
        )

        self.assertTrue(diagnostic["checks"]["has_qa_index"])
        self.assertTrue(diagnostic["checks"]["collection_exists"])
        self.assertTrue(diagnostic["checks"]["entity_count_matches_metadata"])
        self.assertEqual(len(diagnostic["sample_chunks"]), 1)

    def test_build_qa_diagnostic_handles_missing_index(self) -> None:
        diagnostic = build_qa_diagnostic(
            db_service=_FakeDbService(None),
            vector_store_service=_FakeVectorStoreService(),
            arxiv_id="2401.00001",
        )

        self.assertFalse(diagnostic["checks"]["has_qa_index"])
        self.assertFalse(diagnostic["checks"]["collection_exists"])


if __name__ == "__main__":
    unittest.main()
