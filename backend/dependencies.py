from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterable

from utils.config import CORE_CONFIG, get_recommendation_clustering_runtime_config

if TYPE_CHECKING:
    # 这些类型只服务静态检查和编辑器提示；运行时仍保持懒加载，避免启动阶段实例化重型服务。
    from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
    from services.arxiv.arxiv_search_service import ArxivSearchService
    from services.arxiv.contracts import ArxivSearchBackend
    from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
    from services.llm.generation_service import GenerationService
    from services.memory import MemoryService
    from services.paper_qa.index_job_manager import IndexJobManager
    from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
    from services.paper_qa.paper_qa_service import PaperQAService
    from services.recommendation.recommendation_service import RecommendationService
    from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService
    from services.storage.sqlite import StorageContainer
    from services.storage.sqlite.stores import (
        AgentRuntimeCheckpointStore,
        AgentSessionStore,
        AgentWorkStore,
        InterestVectorStore,
        LangGraphCheckpointStore,
        PaperCatalogStore,
        PaperChatMessageStore,
        PaperChatSessionStore,
        PaperNoteStore,
        PaperProfileEvidenceStore,
        PaperQAIndexStore,
        PaperQATurnStore,
        ProfileBuildJobStore,
        ProfileEventStore,
        ResearchProfileStore,
        UserPreferenceStore,
    )
    from services.storage.vector_store_service import VectorStoreService

DATA_SOURCE = CORE_CONFIG["arxiv_data_source"]
ARXIV_PROXY_URL = CORE_CONFIG.get("arxiv_proxy_url", "")
SERVICE_LOAD_MODE = str(CORE_CONFIG.get("service_load_mode", "lazy")).strip().lower()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestActor:
    """当前 demo 用户命名空间；它提供业务隔离，但不等价于经过认证的身份。"""

    user_id: str


def get_request_actor(user_id: str | None = None) -> RequestActor:
    from services.storage.sqlite.shared import DEFAULT_USER_ID

    normalized = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
    return RequestActor(user_id=normalized)


@lru_cache(maxsize=1)
def get_storage_container() -> StorageContainer:
    from services.storage.sqlite import StorageContainer

    return StorageContainer()


def get_paper_catalog_store() -> PaperCatalogStore:
    return get_storage_container().paper_catalog


def get_user_preference_store() -> UserPreferenceStore:
    return get_storage_container().user_preferences


def get_interest_vector_store() -> InterestVectorStore:
    return get_storage_container().interest_vectors


def get_paper_profile_evidence_store() -> PaperProfileEvidenceStore:
    return get_storage_container().paper_profile_evidence


def get_paper_qa_index_store() -> PaperQAIndexStore:
    return get_storage_container().paper_qa_index


def get_paper_chat_session_store() -> PaperChatSessionStore:
    return get_storage_container().paper_chat_sessions


def get_paper_chat_message_store() -> PaperChatMessageStore:
    return get_storage_container().paper_chat_messages


def get_paper_qa_turn_store() -> PaperQATurnStore:
    return get_storage_container().paper_qa_turns


def get_paper_note_store() -> PaperNoteStore:
    return get_storage_container().paper_notes


def get_profile_event_store() -> ProfileEventStore:
    return get_storage_container().profile_events


def get_profile_build_job_store() -> ProfileBuildJobStore:
    return get_storage_container().profile_build_jobs


def get_research_profile_store() -> ResearchProfileStore:
    return get_storage_container().research_profiles


def get_agent_session_store() -> AgentSessionStore:
    return get_storage_container().agent_sessions


def get_agent_runtime_checkpoint_store() -> AgentRuntimeCheckpointStore:
    return get_storage_container().agent_runtime_checkpoints


def get_agent_work_store() -> AgentWorkStore:
    return get_storage_container().agent_work


def get_langgraph_checkpoint_store() -> LangGraphCheckpointStore:
    return get_storage_container().langgraph_checkpoints


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
def get_memory_service() -> MemoryService:
    from services.memory import MemoryService

    return MemoryService(
        paper_catalog_store=get_paper_catalog_store(),
        user_preference_store=get_user_preference_store(),
        interest_vector_store=get_interest_vector_store(),
        paper_profile_evidence_store=get_paper_profile_evidence_store(),
        paper_chat_session_store=get_paper_chat_session_store(),
        paper_chat_message_store=get_paper_chat_message_store(),
        paper_note_store=get_paper_note_store(),
        profile_event_store=get_profile_event_store(),
        profile_build_job_store=get_profile_build_job_store(),
        research_profile_store=get_research_profile_store(),
        agent_session_store=get_agent_session_store(),
        generation_service=get_generation_service(),
        interest_model_refresher=lambda user_id: get_recommendation_service().generate_user_interest_vector(user_id),
    )


@lru_cache(maxsize=1)
def get_enhanced_retrieval_service() -> EnhancedRetrievalService:
    from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService

    return EnhancedRetrievalService(
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        generation_service=get_generation_service(),
    )


@lru_cache(maxsize=1)
def get_arxiv_api_service() -> ArxivSearchService:
    from services.arxiv.arxiv_search_service import ArxivSearchService

    return ArxivSearchService(proxy_url=ARXIV_PROXY_URL)


@lru_cache(maxsize=1)
def get_arxiv_search_backend() -> ArxivSearchBackend:
    """返回当前配置选中的 arXiv 搜索后端。

    后端选择只允许出现在依赖注入层：业务代码拿到的是统一搜索协议，
    不再感知本地兼容封装或远程 API 的具体类名。
    """
    if DATA_SOURCE == "api":
        return get_arxiv_api_service()
    return get_oai_database_service()


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
        paper_qa_index_store=get_paper_qa_index_store(),
        paper_catalog_store=get_paper_catalog_store(),
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        generation_service=get_generation_service(),
        arxiv_service_factory=lambda: get_arxiv_api_service(),
        oai_db_service=get_oai_database_service(),
        get_embedding_config=get_current_embedding_config,
        loading_service_factory=LoadingService,
        chunking_service_factory=ChunkingService,
    )


@lru_cache(maxsize=1)
def get_index_job_manager() -> IndexJobManager:
    from services.paper_qa.index_job_manager import IndexJobManager

    return IndexJobManager(
        paper_qa_index_store=get_paper_qa_index_store(),
        qa_index_builder=get_paper_qa_index_builder(),
        # 使用服务层状态视图校验 active build/version；lambda 延迟求值，避免组合根初始化时形成循环依赖。
        qa_status_reader=lambda arxiv_id: get_paper_qa_service().get_qa_status(arxiv_id),
        on_job_succeeded=lambda job_id, result: get_agent_work_store().mark_job_ready(
            job_id=job_id,
            validated_result=result,
        ),
        on_job_failed=lambda job_id, code, message: get_agent_work_store().mark_job_failed(
            job_id=job_id,
            error_code=code,
            error_message=message,
        ),
    )


@lru_cache(maxsize=1)
def get_background_work_coordinator():
    """装配通用后台协调器；业务 handler 注册集中在组合根，执行器不识别 QA 工具名。"""
    from agents.arxiv_search_agent.execution.background_jobs import (
        BackgroundJobHandlerRegistry,
        PaperQAIndexBackgroundHandler,
        PersistentBackgroundWorkCoordinator,
    )

    manager = get_index_job_manager()

    def _active_job(arxiv_id: str):
        for job in get_paper_qa_index_store().list_paper_index_jobs(arxiv_id=arxiv_id, limit=20):
            if str(job.get("status") or "") in {"pending", "running", "retrying"}:
                return job
        return None

    handlers = BackgroundJobHandlerRegistry()
    handlers.register(
        "paper_qa_index",
        PaperQAIndexBackgroundHandler(
            status_reader=lambda arxiv_id: get_paper_qa_service().get_qa_status(arxiv_id),
            active_job_reader=_active_job,
            job_submitter=lambda arxiv_id, loading_method: manager.submit_job(arxiv_id, loading_method),
        ),
    )
    return PersistentBackgroundWorkCoordinator(
        store=get_agent_work_store(),
        approval_store=get_storage_container().approval_grants,
        handlers=handlers,
    )


@lru_cache(maxsize=1)
def get_agent_work_continuation_service():
    from agents.arxiv_search_agent.execution.continuations import AgentWorkContinuationService

    coordinator = get_background_work_coordinator()
    return AgentWorkContinuationService(
        store=get_agent_work_store(),
        paper_job_store=get_paper_qa_index_store(),
        handlers=coordinator.handlers,
    )


@lru_cache(maxsize=1)
def get_agent_resume_run_manager():
    from agents.arxiv_search_agent.execution.continuations import AgentResumeRunManager

    return AgentResumeRunManager(
        storage=get_storage_container(),
        background_work_coordinator=get_background_work_coordinator(),
    )


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    from services.recommendation.recommendation_service import RecommendationService

    return RecommendationService(
        paper_catalog_store=get_paper_catalog_store(),
        user_preference_store=get_user_preference_store(),
        interest_vector_store=get_interest_vector_store(),
        paper_profile_evidence_store=get_paper_profile_evidence_store(),
        research_profile_store=get_research_profile_store(),
        profile_event_store=get_profile_event_store(),
        memory_service=get_memory_service(),
        embedding_service=get_embedding_service(),
        vector_store_service=get_vector_store_service(),
        get_embedding_config=get_current_embedding_config,
        get_clustering_config=get_current_recommendation_clustering_config,
        arxiv_service_factory=lambda: get_arxiv_search_backend(),
        oai_db_service=get_oai_database_service(),
    )


@lru_cache(maxsize=1)
def get_paper_qa_service() -> PaperQAService:
    from services.paper_qa.paper_qa_service import PaperQAService

    return PaperQAService(
        paper_qa_index_store=get_paper_qa_index_store(),
        paper_catalog_store=get_paper_catalog_store(),
        paper_chat_session_store=get_paper_chat_session_store(),
        paper_qa_turn_store=get_paper_qa_turn_store(),
        research_profile_store=get_research_profile_store(),
        agent_runtime_checkpoint_store=get_agent_runtime_checkpoint_store(),
        memory_service=get_memory_service(),
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
        ("storage_container", get_storage_container),
        ("oai_database_service", get_oai_database_service),
        ("embedding_service", get_embedding_service),
        ("vector_store_service", get_vector_store_service),
        ("generation_service", get_generation_service),
        ("memory_service", get_memory_service),
        ("enhanced_retrieval_service", get_enhanced_retrieval_service),
        ("arxiv_api_service", get_arxiv_api_service),
        ("arxiv_search_backend", get_arxiv_search_backend),
        ("paper_qa_index_builder", get_paper_qa_index_builder),
        ("index_job_manager", get_index_job_manager),
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
