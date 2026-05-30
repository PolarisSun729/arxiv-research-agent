from __future__ import annotations

from functools import lru_cache
from typing import Any

from utils.config import CORE_CONFIG, get_recommendation_clustering_runtime_config

DATA_SOURCE = CORE_CONFIG["arxiv_data_source"]
ARXIV_PROXY_URL = CORE_CONFIG.get("arxiv_proxy_url", "")


@lru_cache(maxsize=1)
def get_database_service() -> DatabaseService:
    from services.storage.database_service import DatabaseService

    return DatabaseService()


@lru_cache(maxsize=1)
def get_oai_database_service() -> ArxivOaiDatabaseService:
    from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService

    return ArxivOaiDatabaseService()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    from services.embedding.embedding_service import EmbeddingService

    return EmbeddingService()


@lru_cache(maxsize=1)
def get_vector_store_service() -> VectorStoreService:
    from services.storage.vector_store_service import VectorStoreService

    return VectorStoreService()


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    from services.llm.generation_service import GenerationService

    return GenerationService()


@lru_cache(maxsize=1)
def get_enhanced_retrieval_service() -> EnhancedRetrievalService:
    from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService

    return EnhancedRetrievalService(
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        generation_service=get_generation_service(),
    )


@lru_cache(maxsize=1)
def get_local_arxiv_service() -> LocalArxivService:
    from services.arxiv.local_arxiv_service import LocalArxivService

    return LocalArxivService()


@lru_cache(maxsize=1)
def get_arxiv_api_service() -> ArxivSearchService:
    from services.arxiv.arxiv_search_service import ArxivSearchService

    return ArxivSearchService(proxy_url=ARXIV_PROXY_URL)


@lru_cache(maxsize=1)
def get_arxiv_service():
    if DATA_SOURCE == "api":
        return get_arxiv_api_service()
    return get_local_arxiv_service()


def get_current_embedding_config() -> EmbeddingConfig:
    return get_embedding_service().get_default_embedding_config()


def get_current_recommendation_clustering_config() -> dict:
    return get_recommendation_clustering_runtime_config()


@lru_cache(maxsize=1)
def get_paper_qa_index_builder() -> PaperQAIndexBuilder:
    from services.document.chunking_service import ChunkingService
    from services.document.loading_service import LoadingService
    from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder

    return PaperQAIndexBuilder(
        db_service=get_database_service(),
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        generation_service=get_generation_service(),
        arxiv_service_factory=lambda: get_arxiv_api_service(),
        get_embedding_config=get_current_embedding_config,
        loading_service_factory=LoadingService,
        chunking_service_factory=ChunkingService,
    )


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    from services.recommendation.recommendation_service import RecommendationService

    return RecommendationService(
        db_service=get_database_service(),
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        get_embedding_config=get_current_embedding_config,
        get_clustering_config=get_current_recommendation_clustering_config,
        arxiv_service_factory=lambda: get_arxiv_service(),
        oai_db_service=get_oai_database_service(),
    )


@lru_cache(maxsize=1)
def get_paper_qa_service() -> PaperQAService:
    from services.paper_qa.paper_qa_service import PaperQAService

    return PaperQAService(
        db_service=get_database_service(),
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        generation_service=get_generation_service(),
        enhanced_retrieval_service=get_enhanced_retrieval_service(),
        arxiv_service_factory=lambda: get_arxiv_api_service(),
        get_embedding_config=get_current_embedding_config,
        qa_index_builder=get_paper_qa_index_builder(),
    )
