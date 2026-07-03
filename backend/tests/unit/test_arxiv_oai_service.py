import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import pytest


def _reload_real_config_module() -> None:
    backend_dir = Path(__file__).resolve().parents[2]
    utils_dir = backend_dir / "utils"
    utils_package = sys.modules.get("utils")
    if utils_package is None or not getattr(utils_package, "__path__", None):
        utils_package = types.ModuleType("utils")
        utils_package.__path__ = [str(utils_dir)]
        sys.modules["utils"] = utils_package

    spec = importlib.util.spec_from_file_location("utils.config", utils_dir / "config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["utils.config"] = module
    spec.loader.exec_module(module)


_saved_embedding_module = sys.modules.get("services.embedding.embedding_service")
_saved_vector_store_module = sys.modules.get("services.storage.vector_store_service")

# OAI 单测只验证 SQLite 查询逻辑，导入阶段用轻量桩挡住 embedding/vector-store 的重型依赖。
embedding_module = types.ModuleType("services.embedding.embedding_service")
embedding_module.EmbeddingService = object
sys.modules["services.embedding.embedding_service"] = embedding_module

vector_store_module = types.ModuleType("services.storage.vector_store_service")
vector_store_module.VectorStoreService = object
sys.modules["services.storage.vector_store_service"] = vector_store_module

# 前序 Agent 单测会改写 utils.config 的运行时函数；这里恢复真实配置模块，
# 既保证 OAI 单测拿到完整配置，也避免半截桩影响后续 arXiv 查询测试收集。
_reload_real_config_module()

# 前序集成测试会为轻量导入写入同名空桩模块；本单测验证真实 OAI 查询逻辑，
# 因此导入前必须清理残留桩，避免拿到没有查询方法的占位类。
sys.modules.pop("services.arxiv.arxiv_oai_service", None)

try:
    from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
finally:
    # 轻量桩只服务本模块导入；恢复公共模块，避免 pytest 混跑时污染后续真实服务测试。
    if _saved_embedding_module is None:
        sys.modules.pop("services.embedding.embedding_service", None)
    else:
        sys.modules["services.embedding.embedding_service"] = _saved_embedding_module
    if _saved_vector_store_module is None:
        sys.modules.pop("services.storage.vector_store_service", None)
    else:
        sys.modules["services.storage.vector_store_service"] = _saved_vector_store_module


def _make_service_without_init() -> ArxivOaiDatabaseService:
    return ArxivOaiDatabaseService.__new__(ArxivOaiDatabaseService)


def _make_temp_service() -> ArxivOaiDatabaseService:
    db_dir = Path(__file__).resolve().parents[2] / "temp" / "oai-tests"
    db_dir.mkdir(parents=True, exist_ok=True)
    return ArxivOaiDatabaseService(db_path=str(db_dir / f"{uuid.uuid4().hex}.db"), check_same_thread=False)


def _skip_if_fts5_unavailable(service: ArxivOaiDatabaseService) -> None:
    with service._get_connection() as conn:
        if not service._is_fts5_available(conn.cursor()):
            pytest.skip("SQLite FTS5 is not available in this Python runtime")


def _paper(
    arxiv_id: str,
    *,
    title: str,
    abstract: str,
    categories: list[str],
    authors: list[str] | None = None,
    created: str = "2024-01-01",
    updated: str = "2024-01-02",
) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": json.dumps(authors or ["Alice Example"], ensure_ascii=False),
        "categories": json.dumps(categories, ensure_ascii=False),
        "categories_list": categories,
        "primary_category": categories[0] if categories else "",
        "created": created,
        "updated": updated,
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "oai_datestamp": created,
    }


def test_local_search_uses_fts_token_matching_without_substring_false_positive() -> None:
    service = _make_temp_service()
    _skip_if_fts5_unavailable(service)
    assert service.upsert_arxiv_oai_papers(
        [
            _paper(
                "2401.00001",
                title="Retrieval-Augmented Generation for Scientific QA",
                abstract="This paper studies RAG systems.",
                categories=["cs.CL"],
            ),
            _paper(
                "2401.00002",
                title="Operation-Guided Text Transformation Benchmark",
                abstract="The benchmark studies progressive editing at multiple granularities.",
                categories=["cs.CL"],
                authors=["Marius Dragomir"],
            ),
        ]
    ) == 2

    result = service.search(search_query="all:RAG", max_results=10)

    assert [paper["arxiv_id"] for paper in result["papers"]] == ["2401.00001"]
    assert result["source"] == "local_oai"
    assert result["query_capability"]["mode"] == "local_oai_sqlite_fts"
    assert result["query_capability"]["full_arxiv_syntax_supported"] is False


def test_local_search_respects_phrase_quotes_and_does_not_split_boolean_inside_phrase() -> None:
    service = _make_temp_service()
    _skip_if_fts5_unavailable(service)
    service.upsert_arxiv_oai_paper(
        _paper(
            "2401.00003",
            title="Large AND Small Models for Retrieval",
            abstract="A controlled study.",
            categories=["cs.CL"],
        )
    )

    result = service.search(search_query='ti:"Large AND Small Models" AND cat:cs.CL', max_results=10)

    assert [paper["arxiv_id"] for paper in result["papers"]] == ["2401.00003"]


def test_local_search_uses_category_table_for_exact_category_matching() -> None:
    service = _make_temp_service()
    service.upsert_arxiv_oai_papers(
        [
            _paper("2401.00004", title="Language Paper", abstract="RAG", categories=["cs.CL"]),
            _paper("2401.00005", title="Systems Paper", abstract="RAG", categories=["cs.CV"]),
        ]
    )

    result = service.search(search_query="cat:cs.CL", max_results=10)

    assert [paper["arxiv_id"] for paper in result["papers"]] == ["2401.00004"]
    assert result["query_capability"]["mode"] == "local_oai_sqlite_index"


def test_local_search_supports_id_field_without_scanning_all_text() -> None:
    service = _make_temp_service()
    service.upsert_arxiv_oai_paper(
        _paper("2401.00006", title="ID Lookup", abstract="metadata", categories=["cs.CL"])
    )

    result = service.search(search_query="id:2401.00006", max_results=10)

    assert [paper["arxiv_id"] for paper in result["papers"]] == ["2401.00006"]


def test_recent_papers_filters_categories_with_mapping_table() -> None:
    service = _make_temp_service()
    service.upsert_arxiv_oai_papers(
        [
            _paper("2401.00007", title="CL Paper", abstract="metadata", categories=["cs.CL"]),
            _paper("2401.00008", title="CV Paper", abstract="metadata", categories=["cs.CV"]),
        ]
    )

    papers = service.get_recent_papers(categories=["cs.CL"], max_age_months=120, max_results=10)

    assert [paper["arxiv_id"] for paper in papers] == ["2401.00007"]


def test_rebuild_oai_search_index_rehydrates_mapping_and_fts() -> None:
    service = _make_temp_service()
    _skip_if_fts5_unavailable(service)
    service.upsert_arxiv_oai_paper(
        _paper("2401.00009", title="Rebuild Index Paper", abstract="RAG", categories=["cs.CL"])
    )
    with service._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM arxiv_oai_paper_categories")
        cursor.execute("DELETE FROM arxiv_oai_papers_fts")
        conn.commit()

    result = service.rebuild_oai_search_index()

    assert result["category_rows"] == 1
    assert result["fts_rows"] == 1
    assert service.search(search_query="all:RAG", max_results=10)["papers"][0]["arxiv_id"] == "2401.00009"


def test_local_search_rejects_unsupported_wildcard_query() -> None:
    service = _make_temp_service()

    with pytest.raises(Exception) as exc_info:
        service.search(search_query="all:RAG*", max_results=10)

    assert getattr(exc_info.value, "code", None) == "unsupported_local_arxiv_query"
    assert exc_info.value.query_capability["unsupported_reason"] == "unsupported_wildcard_query"


def test_local_text_search_fails_when_fts5_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _make_temp_service()
    monkeypatch.setattr(service, "_is_fts5_available", lambda _cursor: False)

    with pytest.raises(Exception) as exc_info:
        service.search(search_query="all:RAG", max_results=10)

    assert getattr(exc_info.value, "code", None) == "local_search_index_unavailable"
    assert exc_info.value.query_capability["unsupported_reason"] == "fts5_unavailable"
