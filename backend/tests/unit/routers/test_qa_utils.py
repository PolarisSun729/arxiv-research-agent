import os
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


def test_sanitize_trace_slug_normalizes_text_and_applies_fallback() -> None:
    assert sanitize_trace_slug(" RAG / QA ??? ") == "RAG_QA"
    assert sanitize_trace_slug("") == "query"


def test_sanitize_trace_slug_preserves_chinese_and_truncates() -> None:
    assert sanitize_trace_slug("论文 / RAG / 检索") == "论文_RAG_检索"
    assert sanitize_trace_slug("a" * 50, max_length=12) == "a" * 12


def test_get_latest_retrieval_trace_returns_latest_markdown_file(tmp_path: Path) -> None:
    paper_dir = tmp_path / sanitize_trace_slug("2401.00001")
    paper_dir.mkdir(parents=True, exist_ok=True)
    first = paper_dir / "trace-1.md"
    second = paper_dir / "trace-2.md"
    first.write_text("older", encoding="utf-8")
    second.write_text("newer", encoding="utf-8")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    result = get_latest_retrieval_trace(_FakeEnhancedRetrievalService(str(tmp_path)), "2401.00001")

    assert result == second


def test_get_latest_retrieval_trace_filters_requested_json_format(tmp_path: Path) -> None:
    paper_dir = tmp_path / sanitize_trace_slug("2401.00002")
    paper_dir.mkdir(parents=True, exist_ok=True)
    markdown = paper_dir / "trace-1.md"
    older_json = paper_dir / "trace-1.json"
    newer_json = paper_dir / "trace-2.json"
    markdown.write_text("md", encoding="utf-8")
    older_json.write_text("old-json", encoding="utf-8")
    newer_json.write_text("new-json", encoding="utf-8")
    os.utime(markdown, (1, 1))
    os.utime(older_json, (2, 2))
    os.utime(newer_json, (3, 3))

    result = get_latest_retrieval_trace(
        _FakeEnhancedRetrievalService(str(tmp_path)),
        "2401.00002",
        format_name="json",
    )

    assert result == newer_json


def test_get_latest_retrieval_trace_returns_none_when_directory_missing(tmp_path: Path) -> None:
    result = get_latest_retrieval_trace(_FakeEnhancedRetrievalService(str(tmp_path)), "missing")

    assert result is None


def test_build_qa_diagnostic_reports_collection_health() -> None:
    diagnostic = build_qa_diagnostic(
        db_service=_FakeDbService({"collection_name": "paper_123", "status": "indexed", "chunk_count": 1}),
        vector_store_service=_FakeVectorStoreService(),
        arxiv_id="2401.00001",
        sample_limit=2,
    )

    assert diagnostic["checks"]["has_qa_index"] is True
    assert diagnostic["checks"]["collection_exists"] is True
    assert diagnostic["checks"]["entity_count_matches_metadata"] is True
    assert len(diagnostic["sample_chunks"]) == 1


def test_build_qa_diagnostic_handles_missing_index() -> None:
    diagnostic = build_qa_diagnostic(
        db_service=_FakeDbService(None),
        vector_store_service=_FakeVectorStoreService(),
        arxiv_id="2401.00001",
    )

    assert diagnostic["checks"]["has_qa_index"] is False
    assert diagnostic["checks"]["collection_exists"] is False
