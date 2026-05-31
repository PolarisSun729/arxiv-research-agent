from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, Iterable

from utils.config import CORE_CONFIG, get_recommendation_clustering_runtime_config

DATA_SOURCE = CORE_CONFIG["arxiv_data_source"]
ARXIV_PROXY_URL = CORE_CONFIG.get("arxiv_proxy_url", "")
SERVICE_LOAD_MODE = str(CORE_CONFIG.get("service_load_mode", "lazy")).strip().lower()

logger = logging.getLogger(__name__)


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


def normalize_service_load_mode(load_mode: str | None = None) -> str:
    mode = str(load_mode or SERVICE_LOAD_MODE or "lazy").strip().lower()
    if mode in {"preload", "eager", "startup", "startup-load"}:
        return "preload"
    return "lazy"


def iter_service_getters() -> list[tuple[str, Callable[[], Any]]]:
    return [
        ("database_service", get_database_service),
        ("oai_database_service", get_oai_database_service),
        ("embedding_service", get_embedding_service),
        ("vector_store_service", get_vector_store_service),
        ("generation_service", get_generation_service),
        ("enhanced_retrieval_service", get_enhanced_retrieval_service),
        ("local_arxiv_service", get_local_arxiv_service),
        ("arxiv_api_service", get_arxiv_api_service),
        ("arxiv_service", get_arxiv_service),
        ("paper_qa_index_builder", get_paper_qa_index_builder),
        ("recommendation_service", get_recommendation_service),
        ("paper_qa_service", get_paper_qa_service),
    ]


def warm_up_services(load_mode: str | None = None) -> list[str]:
    """
    Eagerly construct the backend singletons when startup preloading is enabled.

    The default lazy mode keeps the existing on-demand behavior. When the
    resolved mode is ``preload``, each cached service getter is executed once so
    the heavy imports, model setup, database connections, and vector-store
    clients are initialized before the app starts serving traffic.
    """

    if normalize_service_load_mode(load_mode) != "preload":
        return []

    warmed_services: list[str] = []
    started_at = perf_counter()
    for name, getter in iter_service_getters():
        try:
            getter()
            warmed_services.append(name)
        except Exception:
            logger.exception("Failed to preload backend service: %s", name)
            raise

    logger.info(
        "Backend service preload finished in %.2fs: %s",
        perf_counter() - started_at,
        ", ".join(warmed_services),
    )
    return warmed_services
