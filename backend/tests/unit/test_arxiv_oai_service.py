import sys
import types


embedding_module = types.ModuleType("services.embedding.embedding_service")
embedding_module.EmbeddingService = object
sys.modules["services.embedding.embedding_service"] = embedding_module

vector_store_module = types.ModuleType("services.storage.vector_store_service")
vector_store_module.VectorStoreService = object
sys.modules["services.storage.vector_store_service"] = vector_store_module

from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService


def _make_service_without_init() -> ArxivOaiDatabaseService:
    return ArxivOaiDatabaseService.__new__(ArxivOaiDatabaseService)


def test_short_english_query_matches_complete_token_only() -> None:
    service = _make_service_without_init()

    relevant_paper = {
        "title": "Retrieval-Augmented Generation for Scientific QA",
        "abstract": "This paper studies RAG systems for multi-hop question answering.",
        "authors": "Alice Example",
        "categories": "cs.CL",
        "primary_category": "cs.CL",
    }
    substring_only_paper = {
        "title": "Operation-Guided Progressive Human-to-AI Text Transformation Benchmark",
        "abstract": "The benchmark studies progressive editing at multiple granularities.",
        "authors": "Marius Dragoi, Alexandra Dragomir",
        "categories": "cs.CL",
        "primary_category": "cs.CL",
    }

    # RAG 是短英文主题词，必须按完整 token 匹配，不能因为作者名或普通单词包含 rag 子串而误召回。
    assert service._matches_query(relevant_paper, "all:RAG") is True
    assert service._matches_query(substring_only_paper, "all:RAG") is False


def test_relevance_score_prioritizes_topic_fields_over_author_substrings() -> None:
    service = _make_service_without_init()

    topic_paper = {
        "title": "IA-RAG for Dynamic Knowledge Retrieval",
        "abstract": "The method improves retrieval-augmented generation over temporal knowledge.",
        "authors": "Alice Example",
        "categories": "cs.CL",
        "primary_category": "cs.CL",
    }
    incidental_paper = {
        "title": "Continual Learning with Spectral Updates",
        "abstract": "The method studies parameter-efficient adaptation.",
        "authors": "Marius Dragoi, Alexandra Dragomir",
        "categories": "cs.LG",
        "primary_category": "cs.LG",
    }

    assert service._score_relevance_query(topic_paper, "all:RAG") > service._score_relevance_query(incidental_paper, "all:RAG")
