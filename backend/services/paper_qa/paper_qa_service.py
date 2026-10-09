from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Callable, Dict, Generator, List, Optional
from uuid import uuid4

from core.errors import AppError, ErrorCode
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.document.chunking_service import ChunkingService
from services.evaluation import write_eval_record, build_success_eval_record, build_error_eval_record
from services.llm.call_metrics import LLMCallStats, use_call_stats
from services.memory import MemoryService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.paper_evidence_research import (
    PaperEvidenceResearchRequest,
    PaperEvidenceResearchResult,
    PaperEvidenceResearchError,
    PaperEvidenceResearchService,
    ResearchLimits,
)
from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService
from services.llm.generation_service import GenerationService
from services.document.loading_service import LoadingService
from services.paper_qa.evidence_contract import build_public_source_payload, resolve_evidence_asset_path
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.paper_qa.qa_observation import build_error_qa_observation, build_research_qa_observation
from services.paper_evidence_research.dependencies.production_adapters import PaperRetrievalTargetResolver
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

    该服务只保留 router、工具入口和 QA 主流程需要的高层入口；会话解析由 session_service 负责，
    检索、作答与证据校验统一交给 PaperEvidenceResearchService，避免继续把底层 Store 能力堆回主服务。
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
        research_service: PaperEvidenceResearchService,
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
        self.research_service = research_service
        self._research_target_resolver = PaperRetrievalTargetResolver(
            index_store=paper_qa_index_store, catalog_store=paper_catalog_store,
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

    @staticmethod
    def _build_public_source_payload(
        search_results: List[Dict[str, Any]],
        *,
        arxiv_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """生成前端可持久化的证据契约，并隔离本地文件路径。

        检索和多模态生成仍然需要内部绝对路径，但路径不能进入 API、会话历史或
        Agent 输出；图片只能通过受控的 ``asset_url`` 回放，避免把服务器文件系统
        暴露给浏览器。
        """
        return build_public_source_payload(search_results, arxiv_id=arxiv_id)

    def resolve_evidence_asset_path(self, arxiv_id: str, source_id: str) -> str:
        """根据论文索引反查图片路径，不接受客户端提交的任意文件路径。"""
        return resolve_evidence_asset_path(
            arxiv_id=arxiv_id,
            source_id=source_id,
            paper_qa_index_store=self.paper_qa_index_store,
            enhanced_retrieval_service=self.enhanced_retrieval_service,
        )

    def _prepare_research_context(
        self, arxiv_id: str, payload: Any, request: PaperEvidenceResearchRequest,
    ) -> Dict[str, Any]:
        """会话消歧和索引前置检查共用真实服务接口，避免同步与流式入口发生漂移。"""
        target = self._research_target_resolver(arxiv_id)
        if target is None:
            raise AppError(
                ErrorCode.QA_INDEX_NOT_FOUND,
                context={"arxiv_id": arxiv_id, "stage": "prepare_research"},
            )
        chat_session = self._resolve_chat_session(arxiv_id, payload)
        state = self.session_service.load_conversation_state(
            arxiv_id=arxiv_id, payload=payload, chat_session=chat_session,
        )
        question = request.original_question
        try:
            contextualization = self.question_contextualizer.contextualize(
                question, target.paper_context, state["conversation_context"],
            )
        except Exception:
            # 会话消歧只是增强能力；失败回到原问题，但不能把模型响应或底层异常带到 API。
            logger.warning("Research question contextualization failed: run_id=%s", request.research_run_id, exc_info=True)
            contextualization = {
                "original_question": question, "contextualized_question": question,
                "is_follow_up": False, "referenced_turn_ids": [], "referenced_source_ids": [],
                "used_short_term_memory": False, "status": "fallback_original",
                "memory_reason": "问题消歧不可用，本轮使用原始问题。", "error": "contextualization_failed",
            }
        rewritten = str(contextualization.get("contextualized_question") or question).strip() or question
        state["short_term_debug"] = self.session_service.update_short_term_debug_with_contextualization(
            state["short_term_debug"], contextualization,
        )
        # stateless 标识只满足图的运行隔离，不能写回 chat_session 冒充持久化会话。
        request = request.model_copy(update={
            "original_question": rewritten,
            "session_id": chat_session.get("session_id") or f"stateless-{request.research_run_id}",
            "conversation_snapshot": {
                "recent_turns": state.get("recent_conversation_context") or [],
                "session_summary": state.get("session_summary"),
            },
            "preferred_answer_style": self._get_preferred_answer_style(payload),
        })
        return {
            "request": request, "chat_session": chat_session, "session_state": state,
            "question_contextualization": contextualization, "paper_context": target.paper_context,
        }

    def _finalize_research_answer(
        self, *, arxiv_id: str, question: str, prepared: Dict[str, Any],
        research_result: PaperEvidenceResearchResult, trace_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """统一结果、观察与会话落盘；完成持久化后才允许对外发送 done。"""
        request = prepared["request"]
        summary = research_result.research_summary.model_dump(mode="json")
        sources = self._build_public_source_payload(
            [dict(c.model_dump(mode="json"), chunk_id=c.source_id) for c in research_result.citations],
            arxiv_id=arxiv_id,
        )
        cited_ids = [citation.source_id for citation in research_result.citations]
        observation = build_research_qa_observation(
            outcome=research_result.outcome, research_summary=summary, sources=sources,
        )
        generation_debug = {"status": "completed", "draft_attempt_count": summary["draft_attempt_count"]}
        verification_debug = {
            "status": "passed" if sources else "insufficient_evidence",
            "source_count": len(sources), "cited_source_count": len(cited_ids),
            "supported_claim_count": summary["supported_claim_count"],
            "insufficient_evidence": research_result.outcome == "abstained",
        }
        # API/debug 只返回可展示的摘要。完整候选和逐主张校验留在私有 eval record 中用于重评分。
        retrieval_debug = {
            "engine": "paper_evidence_research", "research_summary": summary, "qa_observation": observation,
            "intent_profile": {"main_intent": next(
                (e["main_intent"] for e in trace_events if e.get("main_intent")), "unknown",
            )},
            "short_term_memory": prepared["session_state"]["short_term_debug"],
            "generation": generation_debug, "verification": verification_debug,
        }
        contextualization = prepared["question_contextualization"]
        persisted = self.persist_completed_turn(
            chat_session=prepared["chat_session"], question=question, answer=research_result.answer,
            source_payload=sources, retrieval_debug=retrieval_debug,
            contextualized_question=request.original_question, question_contextualization=contextualization,
        )
        chat_session = persisted.get("chat_session") or prepared["chat_session"]
        return {
            "status": "success", "arxiv_id": arxiv_id, "question": question,
            "session_id": chat_session.get("session_id"), "chat_session": chat_session,
            "turn_id": persisted.get("turn_id"), "original_question": question,
            "persistence_status": "saved" if persisted.get("turn_id") else "not_saved",
            "contextualized_question": request.original_question,
            "used_short_term_memory": bool(contextualization.get("used_short_term_memory")),
            "question_contextualization": contextualization, "answer": research_result.answer,
            "sources": sources, "cited_source_ids": cited_ids,
            "citation_debug": {"verified": True, "repair_attempt_count": summary["citation_repair_count"]},
            "citation_warning": None, "generation_debug": generation_debug,
            "verification_debug": verification_debug, "retrieval_debug": retrieval_debug,
            "outcome": research_result.outcome, "research_summary": summary,
            "citations": [dict(c.model_dump(mode="json"), citation_id=c.source_id) for c in research_result.citations],
            "qa_observation": observation,
        }

    def answer_question(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        """同步入口消费同一执行流；显式捕获 return，避免 for 吞掉最终结果。"""
        stream = self.answer_question_with_research_stream(arxiv_id, payload)
        while True:
            try:
                next(stream)
            except StopIteration as stop:
                return stop.value

    def answer_question_with_research_stream(
        self, arxiv_id: str, payload: Any,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """逐阶段 yield 进度及唯一 done；异常统一上抛，由 router 发出一次 error。"""
        question = str(self._payload_get(payload, "question", "") or "").strip()
        run_id = str(self._payload_get(payload, "run_id", "") or "").strip() or str(uuid4())
        user_id = self._resolve_user_id(self._payload_get(payload, "user_id"))
        started = perf_counter()
        stats = LLMCallStats()
        events: List[Dict[str, Any]] = []
        request = None
        research_result = None
        research_stream = None
        trace = RequestTrace(
            run_id=run_id, route="paper_qa.research", user_id=user_id, input=question,
            session_id=str(self._payload_get(payload, "session_id", "") or "") or None,
        )
        stage = "prepare_research"
        trace.add_event("qa.request_start", arxiv_id=arxiv_id, user_id=user_id, engine="research")
        info_event(logger, "qa.request_start", run_id=run_id, arxiv_id=arxiv_id, user_id=user_id, engine="research")
        try:
            if not question:
                raise AppError(ErrorCode.REQUEST_VALIDATION_ERROR)
            request = PaperEvidenceResearchRequest(
                arxiv_id=arxiv_id, original_question=question, user_id=user_id,
                session_id=f"stateless-{run_id}", research_run_id=run_id, limits=ResearchLimits(),
            )
            with use_call_stats(stats):
                prepared = self._prepare_research_context(arxiv_id, payload, request)
            request = prepared["request"]
            trace.session_id = prepared["chat_session"].get("session_id")
            stage = "research"
            research_stream = self.research_service.research_stream(
                request, trace_listener=lambda _run_id, trace_events: events.extend(trace_events),
            )
            while True:
                try:
                    # Starlette 可以在不同工作线程恢复同步生成器；每次 next 单独绑定上下文，
                    # 并在 yield 前退出，既不泄漏计数器，也不会跨 Context 重置 token。
                    with use_call_stats(stats):
                        event = next(research_stream)
                except StopIteration as stop:
                    research_result = stop.value
                    break
                yield {"event": event["event"], "data": event}
            if research_result is None:
                raise AppError(ErrorCode.UNKNOWN_ERROR, context={"stage": "research_result_missing"})
            stage = "persist_completed_turn"
            with use_call_stats(stats):
                result = self._finalize_research_answer(
                    arxiv_id=arxiv_id, question=question, prepared=prepared,
                    research_result=research_result, trace_events=events,
                )
            elapsed_ms = (perf_counter() - started) * 1000
            result["usage"] = stats.to_dict()
            write_eval_record(
                record=build_success_eval_record(
                    request=request,
                    result=research_result,
                    trace_events=events,
                    turn_id=result.get("turn_id", ""),
                    latency_ms=elapsed_ms,
                    llm_usage=stats.to_dict(),
                )
            )
            trace.set_output(result)
            trace.add_event("qa.request_done", status="success", outcome=research_result.outcome)
            trace_path = _write_qa_trace(trace, reason="auto")
            info_event(
                logger, "qa.request_done", run_id=run_id, arxiv_id=arxiv_id, status="success",
                outcome=research_result.outcome, elapsed_ms=round(elapsed_ms, 1), trace_path=trace_path,
            )
        except Exception as exc:
            logger.exception("Research QA failed: arxiv_id=%s run_id=%s stage=%s", arxiv_id, run_id, stage)
            code = ErrorCode.DATABASE_WRITE_FAILED if stage == "persist_completed_turn" else ErrorCode.UNKNOWN_ERROR
            if isinstance(exc, AppError):
                code = exc.code
                stage = exc.context.get("stage") or stage
            elif isinstance(exc, PaperEvidenceResearchError):
                stage = exc.stage
                if "retrieval" in exc.code or "target_unavailable" in exc.code:
                    code = ErrorCode.VECTOR_STORE_ERROR
                elif stage in {"draft_generation", "claim_verification", "claim_extraction"}:
                    code = ErrorCode.LLM_GENERATION_FAILED
            # 公开错误仅使用稳定 code/stage；原异常保留在服务器日志，不能绕过 AppError 暴露给 SSE。
            observation = build_error_qa_observation(error_code=code, error_stage=stage, error_reason=code)
            error = AppError(
                code, detail={"stage": stage, "arxiv_id": arxiv_id},
                context={"arxiv_id": arxiv_id, "run_id": run_id, "stage": stage, "qa_observation": observation},
            )
            elapsed_ms = (perf_counter() - started) * 1000
            if request is not None:
                write_eval_record(
                    record=build_error_eval_record(
                        request=request,
                        error={"code": code, "stage": stage, "error_type": type(exc).__name__},
                        trace_events=events,
                        latency_ms=elapsed_ms,
                        llm_usage=stats.to_dict(),
                    )
                )
            trace.mark_failed()
            trace.set_output(error.to_payload())
            trace.add_event("qa.request_done", status="error", code=code)
            trace_path = _write_qa_trace(trace, reason="research_error")
            info_event(
                logger, "qa.request_done", run_id=run_id, arxiv_id=arxiv_id,
                status="error", code=code, elapsed_ms=round(elapsed_ms, 1), trace_path=trace_path,
            )
            raise error from exc
        finally:
            if research_stream is not None:
                research_stream.close()

        yield {"event": "done", "data": result}
        return result
