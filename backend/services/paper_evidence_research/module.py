from __future__ import annotations

import logging
from typing import Any, Callable, Generator

from .contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult
from .errors import PaperEvidenceResearchError
from .actions import DraftAnswerAction, SearchPaperAction
from .graph import ResearchGraphDependencies, build_paper_evidence_research_graph
from .state import PaperEvidenceResearchState

logger = logging.getLogger(__name__)
TraceListener = Callable[[str, list[dict[str, Any]]], Any]
# 对调用方有意义、值得报"开始"的节点；validate/project_coverage/finalize 等纯簿记节点不报。
PROGRESS_STAGES = frozenset({"prepare", "decide", "search", "draft", "extract_claims", "verify_claims"})
_INACTIVE_NEED_STATUSES = frozenset({"abandoned", "superseded"})


def _need_numbers(state: PaperEvidenceResearchState) -> dict[str, int]:
    # 进度流按"需求 1、需求 2"称呼需求，编号就是问题分析给出的顺序，内部 need_id 不外露。
    return {need.need_id: number for number, need in enumerate(state.evidence_needs, 1)}


def _action_progress(action: Any, state: PaperEvidenceResearchState) -> dict[str, Any]:
    numbers = _need_numbers(state)
    if isinstance(action, SearchPaperAction):
        need = next((item for item in state.evidence_needs if item.need_id == action.target_need_id), None)
        return {"action": action.action, "need_number": numbers.get(action.target_need_id),
                "need_description": need.description if need else "", "query": action.query}
    if isinstance(action, DraftAnswerAction):
        return {"action": action.action, "revision": state.current_draft is not None,
                "need_numbers": [numbers[need_id] for need_id in action.addressed_need_ids if need_id in numbers]}
    return {"action": action.action}


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
            # values 每次给出完整状态，不把某个节点的局部 update 误当成最终状态；
            # tasks 在节点开始前先到，用来告诉调用方"现在正在做什么"，长耗时的 LLM 节点期间不至于没有动静。
            for mode, chunk in self._graph.stream(
                final_state, stream_mode=["tasks", "values"],
                config={"recursion_limit": 100, "configurable": {"thread_id": request.research_run_id}},
            ):
                if mode == "tasks":
                    started = self._stage_started_event(chunk, final_state)
                    if started is not None:
                        yield started
                    continue
                final_state = PaperEvidenceResearchState.model_validate(chunk)
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
    def _stage_started_event(chunk: Any, state: PaperEvidenceResearchState) -> dict[str, Any] | None:
        # 只认"任务开始"（带 input）；state 是上一个节点结束后的完整状态，本节点的轮次要在它上面 +1。
        if not isinstance(chunk, dict) or "input" not in chunk:
            return None
        stage = str(chunk.get("name") or "")
        if stage not in PROGRESS_STAGES:
            return None
        event: dict[str, Any] = {"event": "stage_started", "stage": stage}
        if stage == "search":
            event["retrieval_round"] = state.retrieval_count + 1
            action = state.pending_action
            need_id = getattr(action, "target_need_id", None)
            need = next((item for item in state.evidence_needs if item.need_id == need_id), None)
            if need is not None:
                event["need_description"] = need.description
                event["need_number"] = _need_numbers(state).get(need.need_id)
        elif stage == "draft":
            event["draft_attempt"] = state.draft_attempt_count + 1
        elif stage in {"extract_claims", "verify_claims"} and state.current_draft is not None:
            event["draft_attempt"] = state.current_draft.version
        return event

    @staticmethod
    def _progress_event(event: dict[str, Any], state: PaperEvidenceResearchState) -> dict[str, Any] | None:
        kind = event.get("event_type")
        # 进度流给"做了什么"：需求、每次决策、检索词、计数与覆盖；完整主张/校验数据和证据原文只交给运行内监听器与私有轨迹。
        if kind == "research_started":
            # 轨迹里的 research_started 由 prepare 节点写入，此时问题已拆成证据需求；流开头的同名事件另有含义，这里换个名字。
            return {"event": "question_analyzed", "need_count": len(state.evidence_needs), "needs": [
                {"number": number, "description": need.description, "importance": need.importance}
                for number, need in enumerate(state.evidence_needs, 1)
            ]}
        if kind == "action_accepted" and state.pending_action is not None:
            return {"event": "action_decided", **_action_progress(state.pending_action, state)}
        if kind == "action_rejected":
            return {"event": kind, "action": event.get("action"), "rejection_code": event.get("rejection_code")}
        if kind == "claims_extracted":
            return {"event": kind, "draft_attempt": event.get("draft_version"), "claim_count": event.get("claim_count", 0),
                    "extractor_source": event.get("extractor_source")}
        if kind == "retrieval_completed":
            return {"event": kind, "retrieval_count": state.retrieval_count,
                    "need_number": _need_numbers(state).get(event.get("target_need_id")), "query": event.get("query"),
                    "new_candidate_count": event.get("new_candidate_count", 0),
                    "duplicate_candidate_count": event.get("duplicate_candidate_count", 0)}
        if kind == "evidence_coverage_projected":
            numbers = _need_numbers(state)
            active = [need for need in state.evidence_needs if need.status not in _INACTIVE_NEED_STATUSES]
            return {"event": "coverage_projected", "draft_attempt": event.get("draft_version"),
                    "need_count": len(active),
                    "satisfied_need_numbers": [numbers[need.need_id] for need in active if need.status == "satisfied"],
                    "unsatisfied_need_numbers": [numbers[need.need_id] for need in active if need.status != "satisfied"]}
        if kind == "draft_created":
            return {"event": kind, "draft_attempt": event.get("version", state.draft_attempt_count)}
        if kind == "claim_verification_completed":
            return {"event": kind, "draft_attempt": event.get("draft_version"),
                    "supported_count": event.get("supported_claim_count", 0),
                    "claim_count": len(event.get("claims") or [])}
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
