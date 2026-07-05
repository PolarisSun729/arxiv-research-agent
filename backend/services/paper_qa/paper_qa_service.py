from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from core.errors import AppError, ErrorCode
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.context_lifecycle import ContextLifecycleService
from services.document.chunking_service import ChunkingService
from services.memory import MemoryService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService, RetrievalOptions
from services.llm.generation_service import GenerationService
from services.document.loading_service import LoadingService
from services.paper_qa.answer_generator import AnswerGenerator
from services.paper_qa.context_pack_builder import ContextPackBuilder
from services.paper_qa.evidence_verifier import EvidenceVerifier
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.paper_qa.qa_observation import build_error_qa_observation, build_qa_observation
from services.paper_qa.question_contextualizer import QuestionContextualizer
from services.paper_qa.session_service import PaperQASessionService
from services.storage.sqlite.stores import (
    AgentRuntimeCheckpointStore,
    PaperCatalogStore,
    PaperChatSessionStore,
    PaperQAIndexStore,
    PaperQATurnStore,
    ResearchProfileStore,
)
from services.storage.vector_store_service import VectorStoreService
from utils.config import get_memory_runtime_config
from utils.logging_utils import RequestTrace, info_event

logger = logging.getLogger(__name__)


def _write_qa_trace(trace: RequestTrace, *, reason: str) -> Optional[str]:
    """QA trace 只辅助排查完整输入输出；失败时不能影响问答主链路。"""
    try:
        trace_path = trace.write(reason=reason)
    except Exception as exc:  # pragma: no cover
        logger.warning("paper QA trace write failed: run_id=%s reason=%s error=%s", trace.run_id, reason, exc)
        return None
    if trace_path:
        info_event(logger, "request.trace_written", run_id=trace.run_id, reason=reason, trace_path=trace_path)
    return trace_path

class PaperQAService:
    """论文 QA 主流程编排服务。

    该服务只保留 router、工具入口和 QA 主流程需要的高层入口；会话解析、上下文打包、
    答案生成、证据校验等底层能力由专门组件负责，避免继续把底层 Store 能力堆回主服务。
    """

    def __init__(
        self,
        *,
        paper_qa_index_store: PaperQAIndexStore,
        paper_catalog_store: PaperCatalogStore,
        paper_chat_session_store: PaperChatSessionStore,
        paper_qa_turn_store: PaperQATurnStore,
        research_profile_store: ResearchProfileStore,
        agent_runtime_checkpoint_store: AgentRuntimeCheckpointStore,
        memory_service: MemoryService,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        generation_service: Optional[GenerationService] = None,
        enhanced_retrieval_service: Optional[EnhancedRetrievalService] = None,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        oai_db_service: Optional[ArxivOaiDatabaseService] = None,
        get_embedding_config: Optional[Callable[[], EmbeddingConfig]] = None,
        loading_service_factory: Optional[Callable[[], LoadingService]] = None,
        chunking_service_factory: Optional[Callable[[], ChunkingService]] = None,
        qa_index_builder: Optional[PaperQAIndexBuilder] = None,
    ):
        """初始化单篇论文问答服务，并组装问答、检索、记忆与索引构建依赖。"""
        self.paper_qa_index_store = paper_qa_index_store
        self.paper_catalog_store = paper_catalog_store
        self.paper_chat_session_store = paper_chat_session_store
        self.paper_qa_turn_store = paper_qa_turn_store
        self.research_profile_store = research_profile_store
        self.agent_runtime_checkpoint_store = agent_runtime_checkpoint_store
        self.memory_service = memory_service
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.generation_service = generation_service or GenerationService()
        self.enhanced_retrieval_service = enhanced_retrieval_service or EnhancedRetrievalService(
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
        )
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
        self.oai_db_service = oai_db_service
        self.get_embedding_config = get_embedding_config or self.embedding_service.get_default_embedding_config
        self.loading_service_factory = loading_service_factory or LoadingService
        self.chunking_service_factory = chunking_service_factory or ChunkingService
        self.qa_index_builder = qa_index_builder or PaperQAIndexBuilder(
            paper_qa_index_store=self.paper_qa_index_store,
            paper_catalog_store=self.paper_catalog_store,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
            arxiv_service_factory=self.arxiv_service_factory,
            oai_db_service=self.oai_db_service,
            get_embedding_config=self.get_embedding_config,
            loading_service_factory=self.loading_service_factory,
            chunking_service_factory=self.chunking_service_factory,
        )
        self.memory_runtime_config = get_memory_runtime_config()
        self.session_service = PaperQASessionService(
            paper_chat_session_store=self.paper_chat_session_store,
            paper_qa_turn_store=self.paper_qa_turn_store,
            research_profile_store=self.research_profile_store,
            agent_runtime_checkpoint_store=self.agent_runtime_checkpoint_store,
            memory_service=self.memory_service,
            memory_runtime_config=self.memory_runtime_config,
        )
        self.question_contextualizer = QuestionContextualizer(
            generation_service=self.generation_service,
            session_service=self.session_service,
        )
        self.context_pack_builder = ContextPackBuilder()
        self.answer_generator = AnswerGenerator(generation_service=self.generation_service)
        self.evidence_verifier = EvidenceVerifier()
        self.context_lifecycle_service = ContextLifecycleService(
            agent_runtime_checkpoint_store=self.agent_runtime_checkpoint_store,
        )

    def _build_memory_runtime_debug(self) -> Dict[str, Any]:
        """构造 QA 主流程需要的记忆运行时 debug 快照。"""
        return self.session_service.build_memory_runtime_debug()

    @staticmethod
    def _payload_get(payload: Any, key: str, default: Any = None) -> Any:
        """统一读取 dict 或对象风格 payload，避免主流程散落字段访问分支。"""
        return PaperQASessionService.payload_get(payload, key, default)

    @staticmethod
    def _resolve_user_id(value: Any = None) -> str:
        """统一 QA 主流程的用户 ID 兜底规则。"""
        return PaperQASessionService.resolve_user_id(value)

    def _get_preferred_answer_style(self, payload: Any) -> str:
        """从会话与长期画像中解析回答风格，主流程只消费解析结果。"""
        return self.session_service.get_preferred_answer_style(payload)

    @staticmethod
    def _apply_answer_style_to_question(question: str, preferred_answer_style: str) -> str:
        """把回答风格约束附加到生成问题，同时保持证据约束不被绕过。"""
        return PaperQASessionService.apply_answer_style_to_question(question, preferred_answer_style)

    def _resolve_chat_session(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        """解析或创建当前论文的 QA 会话，主流程不直接操作会话表。"""
        return self.session_service.resolve_chat_session(arxiv_id, payload)

    def persist_completed_turn(
        self,
        *,
        chat_session: Dict[str, Any],
        question: str,
        answer: str,
        source_payload: List[Dict[str, Any]],
        retrieval_debug: Optional[Dict[str, Any]],
        contextualized_question: str,
        question_contextualization: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """持久化一轮完整 QA turn；原子写入细节由 session_service 统一维护。"""
        return self.session_service.persist_completed_turn(
            chat_session=chat_session,
            question=question,
            answer=answer,
            source_payload=source_payload,
            retrieval_debug=retrieval_debug,
            contextualized_question=contextualized_question,
            question_contextualization=question_contextualization,
        )

    def get_qa_status(self, arxiv_id: str) -> Dict[str, Any]:
        """查询指定论文当前是否已完成 QA 索引构建，以及索引摘要信息。"""
        qa_index = self.paper_qa_index_store.get_paper_qa_index(arxiv_id)
        latest_build = self.paper_qa_index_store.get_latest_paper_qa_index_build(arxiv_id)
        building_build = self.paper_qa_index_store.get_latest_paper_qa_index_build(arxiv_id, statuses=["building"])
        last_failed_build = self.paper_qa_index_store.get_latest_paper_qa_index_build(arxiv_id, statuses=["build_failed", "orphaned"])
        cleanup_pending_count = self.paper_qa_index_store.count_paper_qa_index_builds(
            arxiv_id,
            statuses=["cleanup_pending", "orphaned", "build_failed"],
        )
        if qa_index:
            has_active_index = qa_index["status"] == "indexed" and bool(qa_index.get("collection_name"))
            return {
                "arxiv_id": arxiv_id,
                "has_index": has_active_index,
                "status": qa_index["status"],
                "collection_name": qa_index["collection_name"],
                "chunk_count": qa_index["chunk_count"],
                "embedding_model": qa_index["embedding_model"],
                "active_collection_name": qa_index.get("collection_name") if has_active_index else "",
                "active_index_version": qa_index.get("active_index_version"),
                "active_build_id": qa_index.get("active_build_id"),
                "active_chunk_count": qa_index.get("chunk_count") if has_active_index else 0,
                "active_embedding_model": qa_index.get("embedding_model") if has_active_index else "",
                "building_status": building_build.get("status") if building_build else None,
                "latest_build_job": latest_build,
                "last_failed_build": last_failed_build,
                "cleanup_pending_count": cleanup_pending_count,
                "pdf_path": qa_index.get("pdf_path"),
                "chunk_file": qa_index.get("chunk_file"),
                "retrieval_index_file": qa_index.get("retrieval_index_file"),
                "retrieval_index_count": qa_index.get("retrieval_index_count"),
                "retrieval_index_types": qa_index.get("retrieval_index_types"),
                "retrieval_index_version": qa_index.get("retrieval_index_version"),
                "sparse_index_dir": qa_index.get("sparse_index_dir"),
                "sparse_index_manifest_file": qa_index.get("sparse_index_manifest_file"),
                "sparse_index_document_count": qa_index.get("sparse_index_document_count"),
                "sparse_index_token_count": qa_index.get("sparse_index_token_count"),
                "sparse_index_backend": qa_index.get("sparse_index_backend"),
                "sparse_index_schema_version": qa_index.get("sparse_index_schema_version"),
                "sparse_index_source_file": qa_index.get("sparse_index_source_file"),
                "sparse_index_source_hash": qa_index.get("sparse_index_source_hash"),
                "sparse_index_avgdl": qa_index.get("sparse_index_avgdl"),
                "embedding_file": qa_index.get("embedding_file"),
                "loading_method": qa_index.get("loading_method"),
                "chunking_strategy": qa_index.get("chunking_strategy"),
                "current_stage": qa_index.get("current_stage"),
                "failed_stage": qa_index.get("failed_stage"),
                "error_message": qa_index.get("error_message"),
                "artifact_status": qa_index.get("artifact_status"),
                "indexed_at": qa_index.get("indexed_at"),
            }
        return {
            "arxiv_id": arxiv_id,
            "has_index": False,
            "status": "not_indexed",
            "active_collection_name": "",
            "active_index_version": None,
            "active_chunk_count": 0,
            "active_embedding_model": "",
            "building_status": building_build.get("status") if building_build else None,
            "latest_build_job": latest_build,
            "last_failed_build": last_failed_build,
            "cleanup_pending_count": cleanup_pending_count,
        }

    def build_qa_index(self, arxiv_id: str, loading_method: str = "docling", *, run_id: Optional[str] = None) -> Dict[str, Any]:
        # run_id 只透传给索引构建日志，保证 Agent 触发的长任务能和主请求关联。
        return self.qa_index_builder.build_qa_index(arxiv_id, loading_method=loading_method, run_id=run_id)

    def delete_qa_index(self, arxiv_id: str) -> Dict[str, Any]:
        """删除论文 QA 索引时先清理外部 artifact，再把数据库记录标记为不可检索。"""
        qa_index = self.paper_qa_index_store.get_paper_qa_index(arxiv_id)
        if not qa_index:
            return {"status": "not_found", "arxiv_id": arxiv_id}

        cleanup_result = self.qa_index_builder.cleanup_qa_index_artifacts(arxiv_id, qa_index, allow_active=True)
        updated = self.paper_qa_index_store.update_paper_qa_index(
            arxiv_id,
            status="deleted",
            current_stage="delete_qa_index",
            artifact_status="deleted",
            error_message="",
            failed_stage="",
        )
        if not updated:
            raise RuntimeError("QA index artifacts were cleaned, but SQLite failed to mark the index as deleted")
        active_build_id = qa_index.get("active_build_id")
        if active_build_id:
            self.paper_qa_index_store.mark_paper_qa_index_build_deleted(active_build_id)
        return {
            "status": "deleted",
            "arxiv_id": arxiv_id,
            "cleanup": cleanup_result,
        }

    def build_source_payload(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把检索结果整理成统一的来源载荷，供前端展示与会话持久化复用。"""
        # source_id 与资产字段由 ContextPackBuilder 统一分配，避免会话记忆和前端证据不一致。
        return self.context_pack_builder.build_source_payload(search_results)

    def build_qa_context(self, arxiv_id: str, payload: Any, *, run_id: Optional[str] = None):
        qa_index = self.paper_qa_index_store.get_paper_qa_index(arxiv_id)
        if not qa_index or qa_index["status"] != "indexed":
            # 未建索引属于前置检索失败，也要生成观察结构，便于 Agent 直接决定是否重建索引。
            qa_observation = build_error_qa_observation(
                error_code=ErrorCode.QA_INDEX_NOT_FOUND,
                error_stage="build_qa_context",
                error_reason=f"index_status:{(qa_index or {}).get('status', 'missing')}",
            )
            # 未建索引是可预期的业务状态，前端需要用稳定 code 引导用户先构建索引。
            raise AppError(
                ErrorCode.QA_INDEX_NOT_FOUND,
                detail={
                    "arxiv_id": arxiv_id,
                    "stage": "build_qa_context",
                    "index_status": (qa_index or {}).get("status", "missing"),
                    "qa_observation": qa_observation,
                },
                context={"arxiv_id": arxiv_id, "stage": "build_qa_context", "qa_observation": qa_observation},
            )

        question = str(self._payload_get(payload, "question", "") or "").strip()
        user_id = self._resolve_user_id(self._payload_get(payload, "user_id"))
        resolved_run_id = str(run_id or self._payload_get(payload, "run_id", "") or "").strip() or str(uuid4())
        chat_session = self._resolve_chat_session(arxiv_id, payload)
        memory_runtime = self._build_memory_runtime_debug()
        collection_name = qa_index["collection_name"]
        paper = self.paper_catalog_store.get_paper(arxiv_id) or {}
        paper_context = {
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": paper.get("authors", ""),
            "categories": paper.get("categories", ""),
            "published_date": paper.get("published_date", ""),
            "url": paper.get("url", ""),
            # 检索画像用 DB index record 校验缓存是否还匹配当前 active collection。
            "chunk_count": qa_index.get("chunk_count", 0),
            "embedding_model": qa_index.get("embedding_model", ""),
            "chunk_file": qa_index.get("chunk_file", ""),
            "retrieval_index_file": qa_index.get("retrieval_index_file", ""),
            "retrieval_index_count": qa_index.get("retrieval_index_count", 0),
            "retrieval_index_types": qa_index.get("retrieval_index_types", ""),
            "retrieval_index_version": qa_index.get("retrieval_index_version", ""),
            "sparse_index_dir": qa_index.get("sparse_index_dir", ""),
            "sparse_index_manifest_file": qa_index.get("sparse_index_manifest_file", ""),
            "sparse_index_document_count": qa_index.get("sparse_index_document_count", 0),
            "sparse_index_token_count": qa_index.get("sparse_index_token_count", 0),
            "sparse_index_backend": qa_index.get("sparse_index_backend", ""),
            "sparse_index_schema_version": qa_index.get("sparse_index_schema_version", ""),
            "sparse_index_source_file": qa_index.get("sparse_index_source_file", ""),
            "sparse_index_source_hash": qa_index.get("sparse_index_source_hash", ""),
            "sparse_index_avgdl": qa_index.get("sparse_index_avgdl", 0),
            "active_build_id": qa_index.get("active_build_id"),
            "active_index_version": qa_index.get("active_index_version"),
            # run_id 只向下游检索链路传递日志关联键，不参与排序、召回或生成决策。
            "run_id": resolved_run_id,
        }
        session_state = self.session_service.load_conversation_state(
            arxiv_id=arxiv_id,
            payload=payload,
            chat_session=chat_session,
        )
        conversation_context = session_state["conversation_context"]
        recent_conversation_context = session_state.get("recent_conversation_context", conversation_context)
        session_summary = session_state.get("session_summary")
        short_term_debug = session_state["short_term_debug"]

        try:
            question_contextualization = self.question_contextualizer.contextualize(question, paper_context, conversation_context)
        except Exception as exc:
            logger.warning("Question contextualization raised unexpectedly, fallback to original question: %s", exc)
            question_contextualization = {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "memory_reason": "Contextualization raised unexpectedly, so the original question was used.",
                "used_short_term_memory": False,
                "status": "fallback_original",
                "error": str(exc),
            }

        retrieval_question = str(question_contextualization.get("contextualized_question", question) or question).strip() or question
        short_term_debug = self.session_service.update_short_term_debug_with_contextualization(
            short_term_debug,
            question_contextualization,
        )

        try:
            memory_context = self.session_service.build_memory_context(
                conversation_context,
                question_contextualization,
                retrieval_question,
            )
        except Exception as exc:
            logger.warning("Memory context construction failed, skip memory-aware retrieval: %s", exc)
            memory_context = {
                "enabled": False,
                "reason": "Memory context construction failed.",
                "query_keywords": [],
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "candidates": [],
                "fallback_reason": str(exc),
            }

        session_debug = self.session_service.build_session_debug(chat_session)
        try:
            retrieval_result = self.enhanced_retrieval_service.enhanced_retrieve(
                collection_name=collection_name,
                user_query=retrieval_question,
                paper_context=paper_context,
                options=RetrievalOptions(
                    top_k=self._payload_get(payload, "top_k", None) or 15,
                    enable_query_rewrite=self._payload_get(payload, "enable_query_rewrite", None),
                    enable_hyde=self._payload_get(payload, "enable_hyde", None),
                    enable_keyword_search=self._payload_get(payload, "enable_keyword_search", None),
                    enable_table_structured_route=self._payload_get(payload, "enable_table_structured_route", None),
                    enable_llm_rerank=self._payload_get(payload, "enable_llm_rerank", None),
                    enable_context_expansion=self._payload_get(payload, "enable_context_expansion", None),
                    debug=self._payload_get(payload, "debug", None),
                    memory_context=memory_context,
                ),
            )
        except AppError:
            raise
        except Exception as exc:
            # 检索异常时没有 sources，但仍要把失败阶段和建议动作结构化暴露给 Agent。
            qa_observation = build_error_qa_observation(
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                error_stage="enhanced_retrieve",
                error_reason=str(exc),
            )
            # 检索链路异常不能降级成“没有相关 chunk”，否则前端无法区分数据为空和服务故障。
            logger.exception(
                "QA retrieval failed: code=%s arxiv_id=%s collection_name=%s user_id=%s stage=%s",
                ErrorCode.VECTOR_STORE_ERROR,
                arxiv_id,
                collection_name,
                user_id,
                "enhanced_retrieve",
            )
            raise AppError(
                ErrorCode.VECTOR_STORE_ERROR,
                detail={"detail": exc, "qa_observation": qa_observation},
                context={
                    "arxiv_id": arxiv_id,
                    "user_id": user_id,
                    "stage": "enhanced_retrieve",
                    "collection_name": collection_name,
                    "qa_observation": qa_observation,
                },
            ) from exc

        retrieval_debug = retrieval_result.get("debug")
        if retrieval_debug is None:
            retrieval_debug = {}
        elif not isinstance(retrieval_debug, dict):
            retrieval_debug = {"raw_debug": retrieval_debug}
        final_context_results = retrieval_result["chunks"]
        search_results = final_context_results
        if not search_results:
            # 空召回和检索服务异常都需要可观察状态；这里保留 debug 以区分 route 空、rerank 失败或预算层兜底。
            qa_observation = build_error_qa_observation(
                error_code=ErrorCode.VECTOR_STORE_ERROR,
                error_stage="enhanced_retrieve",
                error_reason="empty_retrieval_chunks",
                retrieval_debug=retrieval_debug,
                sources=[],
            )
            raise AppError(
                ErrorCode.VECTOR_STORE_ERROR,
                message="检索服务没有返回可用的论文片段，请检查索引后重试。",
                detail={"arxiv_id": arxiv_id, "stage": "enhanced_retrieve", "collection_name": collection_name, "qa_observation": qa_observation},
                context={"arxiv_id": arxiv_id, "user_id": user_id, "stage": "enhanced_retrieve", "qa_observation": qa_observation},
            )

        info_event(
            logger,
            "qa.retrieval_done",
            run_id=resolved_run_id,
            session_id=chat_session.get("session_id"),
            user_id=user_id,
            arxiv_id=arxiv_id,
            collection_name=collection_name,
            input=question,
            contextualized_question=retrieval_question,
            result_count=len(search_results),
            trace_path=(retrieval_result.get("trace_export") or {}).get("json") if isinstance(retrieval_result.get("trace_export"), dict) else None,
        )
        context_pack = self.context_pack_builder.build(search_results)
        retrieval_debug["original_question"] = question
        retrieval_debug["contextualized_question"] = retrieval_question
        retrieval_debug["question_contextualization"] = question_contextualization
        retrieval_debug["memory_context"] = memory_context
        retrieval_debug["memory_runtime"] = memory_runtime
        retrieval_debug["memory_modules"] = self.session_service.build_memory_modules_debug(
            short_term_debug=short_term_debug,
            session_debug=session_debug,
            memory_context=memory_context,
        )
        # context_pack debug 只记录预算和类型分布，不重复塞入完整 chunk，避免 retrieval_debug 过大。
        retrieval_debug["context_pack"] = {
            "context_budget_debug": context_pack.get("context_budget_debug", {}),
        }
        retrieval_debug["context_lifecycle"] = self.context_lifecycle_service.build_paper_qa_health_debug(
            user_id=user_id,
            session_id=str(chat_session.get("session_id") or ""),
            short_term_debug=short_term_debug,
            session_summary=session_summary,
            degraded={
                "memory_context_fallback": bool(memory_context.get("fallback_reason")) if isinstance(memory_context, dict) else False,
                "question_contextualization_status": question_contextualization.get("status") if isinstance(question_contextualization, dict) else None,
            },
        )

        return qa_index, search_results, {
            "text_context": context_pack["text_context"],
            "image_inputs": context_pack["image_inputs"],
            "asset_metadata": context_pack["asset_metadata"],
            "source_payload": context_pack["source_payload"],
            "context_pack": context_pack,
            "context_budget_debug": context_pack.get("context_budget_debug", {}),
            "generation_question": retrieval_question,
            "original_question": question,
            "question_contextualization": question_contextualization,
            "conversation_context": conversation_context,
            "recent_conversation_context": recent_conversation_context,
            "session_summary": session_summary,
            "memory_context": memory_context,
            "chat_session": chat_session,
            "memory_runtime": memory_runtime,
        }, retrieval_debug

    def _load_user_memory_summary_for_prompt(self, payload: Any) -> Dict[str, Any]:
        """为 PromptContextBuilder 加载用户长期记忆；失败不影响本轮 QA。"""
        user_id = self._resolve_user_id(self._payload_get(payload, "user_id"))
        try:
            summary = self.memory_service.build_user_memory_summary(user_id)
            return summary if isinstance(summary, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load user memory summary for prompt context: user_id=%s error=%s", user_id, exc)
            return {}

    def answer_question(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        question = str(self._payload_get(payload, "question", "") or "").strip()
        run_id = str(self._payload_get(payload, "run_id", "") or "").strip() or str(uuid4())
        started = perf_counter()
        user_id = self._resolve_user_id(self._payload_get(payload, "user_id"))
        trace = RequestTrace(
            run_id=run_id,
            route="paper_qa.answer",
            session_id=str(self._payload_get(payload, "session_id", "") or "") or None,
            user_id=user_id,
            input=question,
        )
        info_event(
            logger,
            "qa.request_start",
            run_id=run_id,
            session_id=trace.session_id,
            user_id=user_id,
            arxiv_id=arxiv_id,
            input=question,
        )
        trace.add_event("qa.request_start", arxiv_id=arxiv_id, user_id=user_id)
        try:
            _, _search_results, qa_context, retrieval_debug = self.build_qa_context(arxiv_id, payload, run_id=run_id)
        except Exception as exc:
            trace.mark_failed()
            error_payload_builder = getattr(exc, "to_payload", None)
            trace.set_output(error_payload_builder() if callable(error_payload_builder) else {"error": str(exc)})
            trace.add_event("qa.request_done", status="error", error_type=type(exc).__name__)
            trace_path = _write_qa_trace(trace, reason="qa_context_error")
            info_event(
                logger,
                "qa.request_done",
                run_id=run_id,
                session_id=trace.session_id,
                user_id=user_id,
                arxiv_id=arxiv_id,
                status="error",
                error_type=type(exc).__name__,
                output=str(exc),
                elapsed_ms=round((perf_counter() - started) * 1000, 1),
                trace_path=trace_path,
            )
            raise
        context_pack = qa_context.get("context_pack") or self.context_pack_builder.build(_search_results)
        source_payload = list(qa_context.get("source_payload") or context_pack.get("source_payload") or [])
        generation_question = str(qa_context.get("generation_question", question) or question).strip() or question
        question_contextualization = qa_context.get("question_contextualization", {}) or {}
        chat_session = qa_context.get("chat_session", {}) or {}
        preferred_answer_style = self._get_preferred_answer_style(payload)
        user_memory_summary = self._load_user_memory_summary_for_prompt(payload)
        styled_generation_question = self._apply_answer_style_to_question(generation_question, preferred_answer_style)
        if isinstance(retrieval_debug, dict) and preferred_answer_style:
            retrieval_debug["preferred_answer_style"] = preferred_answer_style

        logger.debug("Generating answer...")
        try:
            generation_result = self.answer_generator.generate(
                generation_question=styled_generation_question,
                context_pack=context_pack,
                preferred_answer_style=preferred_answer_style,
                style_already_applied=bool(preferred_answer_style),
                original_question=question,
                session_summary=qa_context.get("session_summary"),
                recent_turns=qa_context.get("recent_conversation_context") or qa_context.get("conversation_context") or [],
                user_memory_summary=user_memory_summary,
            )
            answer = generation_result["answer"]
        except Exception as exc:
            # 生成失败发生在检索之后，因此观察结构要保留已有 sources/debug，方便区分“检索弱”和“LLM 失败”。
            qa_observation = build_error_qa_observation(
                error_code=ErrorCode.LLM_GENERATION_FAILED,
                error_stage="paper_qa_final_answer",
                error_reason=str(exc),
                retrieval_debug=retrieval_debug if isinstance(retrieval_debug, dict) else None,
                sources=source_payload,
            )
            # LLM 失败时保留明确错误码，避免把检索结果拼成成功答案误导用户。
            logger.exception(
                "QA generation failed: code=%s arxiv_id=%s session_id=%s stage=%s",
                ErrorCode.LLM_GENERATION_FAILED,
                arxiv_id,
                chat_session.get("session_id"),
                "paper_qa_final_answer",
            )
            trace.mark_failed()
            trace.set_output({"error": str(exc), "qa_observation": qa_observation})
            trace.add_event("qa.request_done", status="error", code=ErrorCode.LLM_GENERATION_FAILED)
            trace_path = _write_qa_trace(trace, reason="generation_error")
            info_event(
                logger,
                "qa.request_done",
                run_id=run_id,
                session_id=chat_session.get("session_id") or trace.session_id,
                user_id=user_id,
                arxiv_id=arxiv_id,
                status="error",
                code=ErrorCode.LLM_GENERATION_FAILED,
                output=str(exc),
                elapsed_ms=round((perf_counter() - started) * 1000, 1),
                trace_path=trace_path,
            )
            raise AppError(
                ErrorCode.LLM_GENERATION_FAILED,
                detail={"detail": exc, "qa_observation": qa_observation},
                context={
                    "arxiv_id": arxiv_id,
                    "session_id": chat_session.get("session_id"),
                    "stage": "paper_qa_final_answer",
                    "qa_observation": qa_observation,
                },
            ) from exc

        verification_result = self.evidence_verifier.verify(
            answer=answer,
            sources=source_payload,
            cited_source_ids=generation_result.get("cited_source_ids", []),
            claims=generation_result.get("claims", []),
            generation_insufficient_evidence=bool(generation_result.get("insufficient_evidence", False)),
        )
        verified_answer = self.evidence_verifier.apply_answer_guardrail(answer, verification_result)
        if isinstance(retrieval_debug, dict):
            # debug 按模块分组，方便回放：检索 -> 上下文打包 -> 生成 -> 证据校验。
            retrieval_debug["generation"] = generation_result.get("generation_debug", {})
            retrieval_debug["verification"] = verification_result
            retrieval_debug["context_lifecycle"] = self.context_lifecycle_service.build_paper_qa_health_debug(
                user_id=self._resolve_user_id(self._payload_get(payload, "user_id")),
                session_id=str(chat_session.get("session_id") or ""),
                short_term_debug=((retrieval_debug.get("memory_modules") or {}).get("short_term_memory") or {}),
                session_summary=qa_context.get("session_summary"),
                prompt_context_debug=(generation_result.get("generation_debug", {}) or {}).get("prompt_context"),
                degraded={
                    "verification_status": verification_result.get("status") if isinstance(verification_result, dict) else None,
                    "generation_insufficient_evidence": bool(generation_result.get("insufficient_evidence", False)),
                },
            )

        qa_observation = build_qa_observation(
            retrieval_debug=retrieval_debug if isinstance(retrieval_debug, dict) else None,
            sources=source_payload,
            verification_result=verification_result,
            generation_result=generation_result,
        )
        if isinstance(retrieval_debug, dict):
            # 顶层 qa_observation 是 Agent 稳定读取入口；debug 内副本用于历史快照和人工排查。
            retrieval_debug["qa_observation"] = qa_observation

        try:
            persisted_turn = self.persist_completed_turn(
                chat_session=chat_session,
                question=question,
                answer=verified_answer,
                source_payload=source_payload,
                retrieval_debug=self.context_lifecycle_service.prepare_debug_snapshot(retrieval_debug) if isinstance(retrieval_debug, dict) else None,
                contextualized_question=generation_question,
                question_contextualization=question_contextualization,
            )
        except AppError as exc:
            trace.mark_failed()
            trace.set_output({"error": exc.to_payload() if hasattr(exc, "to_payload") else str(exc), "qa_observation": qa_observation})
            trace.add_event("qa.request_done", status="error", code=exc.code)
            trace_path = _write_qa_trace(trace, reason="persist_error")
            info_event(
                logger,
                "qa.request_done",
                run_id=run_id,
                session_id=chat_session.get("session_id") or trace.session_id,
                user_id=user_id,
                arxiv_id=arxiv_id,
                status="error",
                code=exc.code,
                output=str(exc.detail or exc.message),
                elapsed_ms=round((perf_counter() - started) * 1000, 1),
                trace_path=trace_path,
            )
            raise AppError(
                exc.code,
                message=exc.message,
                detail={"detail": exc.detail, "qa_observation": qa_observation},
                recoverable=exc.recoverable,
                status_code=exc.status_code,
                context={**exc.context, "qa_observation": qa_observation},
            ) from exc

        result = {
            "status": "success",
            "arxiv_id": arxiv_id,
            "question": question,
            "session_id": chat_session.get("session_id"),
            "chat_session": persisted_turn.get("chat_session", chat_session),
            "turn_id": persisted_turn.get("turn_id"),
            "original_question": question,
            "contextualized_question": generation_question,
            "used_short_term_memory": bool(question_contextualization.get("used_short_term_memory", False)),
            "question_contextualization": question_contextualization,
            "answer": verified_answer,
            "sources": source_payload,
            "image_inputs": qa_context["image_inputs"],
            "asset_metadata": qa_context["asset_metadata"],
            "generation_debug": generation_result.get("generation_debug", {}),
            "verification_debug": verification_result,
            "qa_observation": qa_observation,
            "retrieval_debug": retrieval_debug,
        }
        trace.session_id = str(chat_session.get("session_id") or trace.session_id or "") or None
        trace.set_output(result)
        trace.add_event(
            "qa.request_done",
            status="success",
            source_count=len(source_payload),
            answer_chars=len(str(verified_answer or "")),
        )
        trace_path = _write_qa_trace(trace, reason="auto")
        info_event(
            logger,
            "qa.request_done",
            run_id=run_id,
            session_id=trace.session_id,
            user_id=user_id,
            arxiv_id=arxiv_id,
            status="success",
            source_count=len(source_payload),
            output=verified_answer,
            elapsed_ms=round((perf_counter() - started) * 1000, 1),
            trace_path=trace_path,
        )
        return result
