from __future__ import annotations

import logging
from typing import Any, Callable, Generator

from .contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult
from .errors import PaperEvidenceResearchError
from .graph import ResearchGraphDependencies, build_paper_evidence_research_graph
from .state import PaperEvidenceResearchState

logger = logging.getLogger(__name__)
TraceListener = Callable[[str, list[dict[str, Any]]], Any]


class PaperEvidenceResearchService:
    """有界证据研究入口；进度流和完整审计轨迹使用不同的投影。"""

    def __init__(
        self, *, question_analyzer: Any, decision_policy: Any, retriever: Any,
        draft_generator: Any, claim_extractor: Any, claim_verifier: Any,
        checkpointer: Any = None, trace_sink: Any = None, configuration_provider: Any = None,
    ) -> None:
        dependencies = ResearchGraphDependencies(
            question_analyzer=question_analyzer, decision_policy=decision_policy, retriever=retriever,
            draft_generator=draft_generator, claim_extractor=claim_extractor, claim_verifier=claim_verifier,
            configuration_provider=configuration_provider,
        )
        self._graph = build_paper_evidence_research_graph(dependencies, checkpointer=checkpointer)
        self._trace_sink = trace_sink

    def research(
        self, request: PaperEvidenceResearchRequest, *, trace_listener: TraceListener | None = None,
    ) -> PaperEvidenceResearchResult:
        # 普通 for 会吞掉 StopIteration.value；同步入口明确读取流的返回值，图只执行一次。
        stream = self.research_stream(request, trace_listener=trace_listener)
        while True:
            try:
                next(stream)
            except StopIteration as stop:
                return stop.value

    def research_stream(
        self, request: PaperEvidenceResearchRequest, *, trace_listener: TraceListener | None = None,
    ) -> Generator[dict[str, Any], None, PaperEvidenceResearchResult]:
        """yield 面向调用方的阶段进度；最终研究结果通过生成器 return 交付。"""
        final_state = PaperEvidenceResearchState(request=request)
        emitted = 0
        yield {
            "event": "research_started", "research_run_id": request.research_run_id,
            "arxiv_id": request.arxiv_id, "question": request.original_question,
        }
        try:
            # values 每次给出完整状态，不把某个节点的局部 update 误当成最终状态。
            for raw_state in self._graph.stream(
                final_state, stream_mode="values",
                config={"recursion_limit": 100, "configurable": {"thread_id": request.research_run_id}},
            ):
                final_state = PaperEvidenceResearchState.model_validate(raw_state)
                for event in final_state.trace_events[emitted:]:
                    progress = self._progress_event(event, final_state)
                    if progress is not None:
                        yield progress
                emitted = len(final_state.trace_events)
            if final_state.result is None:
                raise PaperEvidenceResearchError(
                    code="research_result_missing", stage="graph_finalize",
                    message="证据研究图已结束但没有生成研究结果",
                    detail={"research_run_id": request.research_run_id},
                )
        except Exception as exc:
            # 错误轨迹保留已完成阶段，技术失败不能投影成 abstained。
            final_state.trace_events.append({
                "sequence": len(final_state.trace_events) + 1, "event_type": "research_failed",
                "error_type": type(exc).__name__,
            })
            self._emit_trace(request.research_run_id, final_state.trace_events, trace_listener)
            if isinstance(exc, PaperEvidenceResearchError):
                raise
            raise PaperEvidenceResearchError(
                code="paper_evidence_research_failed", stage="graph_invoke",
                message="证据研究执行失败",
                detail={"research_run_id": request.research_run_id, "error_type": type(exc).__name__},
            ) from exc
        self._emit_trace(request.research_run_id, final_state.trace_events, trace_listener)
        return final_state.result

    @staticmethod
    def _progress_event(event: dict[str, Any], state: PaperEvidenceResearchState) -> dict[str, Any] | None:
        kind = event.get("event_type")
        # 普通进度流只给计数和业务终态；完整主张/校验数据只交给运行内监听器与私有轨迹。
        if kind == "retrieval_completed":
            return {"event": kind, "retrieval_count": state.retrieval_count,
                    "new_candidate_count": event.get("new_candidate_count", 0)}
        if kind == "draft_created":
            return {"event": kind, "draft_attempt": event.get("version", state.draft_attempt_count)}
        if kind == "claim_verification_completed":
            return {"event": kind, "draft_attempt": event.get("draft_version"),
                    "supported_count": event.get("supported_claim_count", 0)}
        if kind == "research_completed" and state.result is not None:
            return {"event": kind, "outcome": state.result.outcome,
                    "termination_reason": state.result.research_summary.termination_reason}
        return None

    def _emit_trace(
        self, run_id: str, events: list[dict[str, Any]], trace_listener: TraceListener | None = None,
    ) -> None:
        for sink in (self._trace_sink, trace_listener):
            if sink is None:
                continue
            try:
                # 每次调用显式传监听器，不修改共享服务的 sink，防止并发运行串线。
                sink(run_id, events)
            except Exception:
                logger.warning("研究轨迹写入失败: run_id=%s", run_id, exc_info=True)
