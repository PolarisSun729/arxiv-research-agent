from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .schemas import FailureCategory, ObservationResult, PlanRuntime, PlanStep, RecoveryActionType, RecoverySeverity
from .state import AgentState
from .tool_adapters.models import ToolExecutionResult


def _tokenize(text: Any) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", str(text or "").lower()) if len(token) >= 2]


def _tool_result_payload(value: Any) -> Any:
    """兼容 ToolExecutionResult 实例和 checkpoint 后的 dict envelope。"""
    if isinstance(value, ToolExecutionResult):
        return value.model_dump()
    if isinstance(value, Mapping) and {"ok", "tool_name", "adapter_name"}.issubset(set(value.keys())):
        return dict(value)
    return None


def _unwrap_payload(value: Any) -> Any:
    """业务观察只读取工具 data，避免被统一 envelope 外层字段干扰。"""
    tool_payload = _tool_result_payload(value)
    if isinstance(tool_payload, Mapping):
        return tool_payload.get("data")
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return value


def _extract_papers(value: Any) -> List[Dict[str, Any]]:
    value = _unwrap_payload(value)
    if isinstance(value, Mapping):
        papers = value.get("papers")
        if isinstance(papers, list):
            return [dict(item) for item in papers if isinstance(item, Mapping)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _recovery_semantics(
    *,
    failure_category: FailureCategory,
    severity: RecoverySeverity = "warning",
    recoverable: bool = True,
    suggested_recovery_types: Optional[Sequence[RecoveryActionType]] = None,
    evidence: Optional[Mapping[str, Any]] = None,
    retryable: Optional[bool] = None,
    requires_user_input: bool = False,
) -> Dict[str, Any]:
    """集中生成恢复语义，避免 Observer 把分类字段和最终补救动作混在一起。"""
    return {
        "failure_category": failure_category,
        "severity": severity,
        "recoverable": recoverable,
        "suggested_recovery_types": list(suggested_recovery_types or []),
        "evidence": dict(evidence or {}),
        "retryable": retryable,
        "requires_user_input": requires_user_input,
    }


def _tool_recovery_types(suggested_recovery: Any, *, retryable: bool) -> List[RecoveryActionType]:
    """把 ToolError.suggested_recovery 映射到 Replanner 可识别的恢复动作。"""
    normalized = str(suggested_recovery or "").strip()
    if normalized in {"ask_clarification", "request_confirmation", "fallback_answer", "abort_with_error", "retry_step"}:
        return [normalized]  # type: ignore[list-item]
    if retryable:
        return ["retry_step"]
    return ["fallback_answer"]


def _tool_failure_category(error_code: str, suggested_recovery: Any) -> FailureCategory:
    """根据结构化 ToolError 选择恢复分类，避免 Replanner 继续解析错误字符串。"""
    if error_code == "output_validation_error":
        return "tool_invalid_output"
    if error_code in {"input_validation_error", "missing_arxiv_id"}:
        return "ambiguous_user_request"
    if error_code == "preference_target_missing":
        return "preference_target_missing"
    if str(suggested_recovery or "").strip() == "ask_clarification":
        return "ambiguous_user_request"
    return "tool_runtime_error"


class Observer:
    """区分“工具执行成功”和“结果质量达标”，为后续重规划提供统一语义。"""

    def observe(
        self,
        *,
        step: PlanStep,
        resolved_input: Mapping[str, Any],
        raw_output: Any,
        normalized_output: Any,
        runtime: PlanRuntime,
        state: AgentState,
        error: Optional[str] = None,
    ) -> ObservationResult:
        tool_payload = _tool_result_payload(raw_output)
        if isinstance(tool_payload, Mapping) and not bool(tool_payload.get("ok", True)):
            tool_error = tool_payload.get("error") if isinstance(tool_payload.get("error"), Mapping) else None
            error_code = str((tool_error or {}).get("error_code") or "tool_failed")
            suggested_recovery = (tool_error or {}).get("suggested_recovery")
            retryable = bool((tool_error or {}).get("retryable"))
            recoverable = bool((tool_error or {}).get("recoverable", True))
            return ObservationResult(
                status="tool_error",
                reason=error_code,
                confidence=0.0,
                suggested_action=suggested_recovery or "fallback",
                **_recovery_semantics(
                    failure_category=_tool_failure_category(error_code, suggested_recovery),
                    severity="error",
                    suggested_recovery_types=_tool_recovery_types(suggested_recovery, retryable=retryable),
                    evidence={"tool_error": tool_error},
                    retryable=retryable,
                    recoverable=recoverable,
                ),
            )
        if error:
            error_text = str(error or "")
            timeout_like = "timeout" in error_text.lower() or "timed out" in error_text.lower() or "超时" in error_text
            return ObservationResult(
                status="tool_error",
                reason=error,
                confidence=0.0,
                suggested_action="fallback",
                **_recovery_semantics(
                    failure_category="tool_timeout" if timeout_like else "tool_runtime_error",
                    severity="error",
                    suggested_recovery_types=["retry_step", "fallback_answer"] if timeout_like else ["fallback_answer"],
                    evidence={"error": error},
                    retryable=timeout_like,
                ),
            )

        handler = getattr(self, f"_observe_{step.tool_name}", None)
        if callable(handler):
            return handler(resolved_input=resolved_input, raw_output=raw_output, normalized_output=normalized_output, runtime=runtime, state=state)
        return ObservationResult(status="success", reason="no_special_rule", confidence=1.0)

    def _observe_search_arxiv(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del normalized_output, runtime, state
        papers = _extract_papers(raw_output)
        if not papers:
            return ObservationResult(
                status="empty_result",
                reason="search_returned_no_papers",
                confidence=0.1,
                details={"paper_count": 0},
                suggested_action="rewrite_arxiv_query",
                **_recovery_semantics(
                    failure_category="search_empty",
                    suggested_recovery_types=["patch_plan"],
                    evidence={"paper_count": 0},
                    retryable=True,
                ),
            )
        query_tokens = _tokenize((resolved_input.get("search_spec") or {}).get("query") if isinstance(resolved_input.get("search_spec"), Mapping) else None)
        if query_tokens:
            matched_titles = 0
            for paper in papers[:5]:
                title = str(paper.get("title") or "")
                abstract = str(paper.get("summary") or paper.get("abstract") or "")
                if any(token in f"{title} {abstract}".lower() for token in query_tokens):
                    matched_titles += 1
            if matched_titles == 0:
                return ObservationResult(
                    status="low_confidence",
                    reason="search_results_not_related_to_query",
                    confidence=0.25,
                    details={"paper_count": len(papers)},
                    suggested_action="rewrite_arxiv_query",
                    **_recovery_semantics(
                        failure_category="search_low_confidence",
                        suggested_recovery_types=["patch_plan"],
                        evidence={"paper_count": len(papers), "matched_top_titles": matched_titles},
                        retryable=True,
                    ),
                )
        return ObservationResult(status="success", reason="search_results_available", confidence=0.85, details={"paper_count": len(papers)})

    def _observe_validate_arxiv_results(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del runtime, state
        quality = normalized_output if isinstance(normalized_output, Mapping) else raw_output if isinstance(raw_output, Mapping) else {}
        result_count = int(quality.get("result_count") or 0)
        if result_count <= 0:
            return ObservationResult(
                status="empty_result",
                reason="validated_arxiv_results_empty",
                confidence=0.1,
                details={"result_count": result_count},
                suggested_action="rewrite_arxiv_query",
                **_recovery_semantics(
                    failure_category="search_empty",
                    suggested_recovery_types=["patch_plan"],
                    evidence={"result_count": result_count},
                    retryable=True,
                ),
            )
        warnings = list(quality.get("warnings") or [])
        if "tool_failed" in warnings or "no_results" in warnings:
            return ObservationResult(
                status="low_confidence",
                reason="validated_arxiv_results_low_quality",
                confidence=0.35,
                details={"warnings": warnings},
                suggested_action="rewrite_arxiv_query",
                **_recovery_semantics(
                    failure_category="search_low_confidence",
                    suggested_recovery_types=["patch_plan"],
                    evidence={"warnings": warnings},
                    retryable=True,
                ),
            )
        papers = _extract_papers(resolved_input.get("arxiv_results"))
        titles = [str(paper.get("title") or "").strip().lower() for paper in papers if str(paper.get("title") or "").strip()]
        if len(set(titles)) < max(1, len(titles) // 2):
            return ObservationResult(
                status="low_confidence",
                reason="validated_arxiv_results_too_many_duplicates",
                confidence=0.45,
                details={"paper_count": len(papers)},
                suggested_action="rewrite_arxiv_query",
                **_recovery_semantics(
                    failure_category="search_low_confidence",
                    suggested_recovery_types=["patch_plan"],
                    evidence={"paper_count": len(papers), "unique_title_count": len(set(titles))},
                    retryable=True,
                ),
            )
        return ObservationResult(status="success", reason="validated_arxiv_results_ok", confidence=0.85, details={"result_count": result_count})

    def _observe_check_paper_index(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        normalized_output = _unwrap_payload(normalized_output)
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        status = str(payload.get("status") or "").lower()
        has_index = bool(payload.get("has_index"))
        if status in {"available", "indexed"} or has_index:
            return ObservationResult(status="success", reason="paper_index_available", confidence=0.95)
        if status in {"stale", "outdated", "collection_missing", "vector_store_unavailable"}:
            return ObservationResult(
                status="low_confidence",
                reason="paper_index_stale",
                confidence=0.35,
                suggested_action="recover_index",
                **_recovery_semantics(
                    failure_category="paper_index_stale",
                    severity="warning",
                    suggested_recovery_types=["patch_plan", "request_confirmation", "fallback_answer"],
                    evidence={
                        "index_status": status,
                        "has_last_good_index": bool(payload.get("last_good_index") or payload.get("last_good_index_id")),
                    },
                    retryable=False,
                    requires_user_input=not bool(payload.get("last_good_index") or payload.get("last_good_index_id")),
                ),
            )
        if status in {"missing", "not_found"} or not has_index:
            return ObservationResult(
                status="need_confirmation",
                reason="paper_index_missing",
                confidence=0.9,
                suggested_action="request_confirmation",
                **_recovery_semantics(
                    failure_category="paper_index_missing",
                    suggested_recovery_types=["patch_plan", "request_confirmation"],
                    evidence={"index_status": status, "has_index": has_index},
                    retryable=False,
                    requires_user_input=True,
                ),
            )
        if status == "corrupted":
            return ObservationResult(
                status="tool_error",
                reason="paper_index_corrupted",
                confidence=0.1,
                suggested_action="rebuild_index",
                **_recovery_semantics(
                    failure_category="paper_index_corrupted",
                    severity="error",
                    suggested_recovery_types=["patch_plan", "request_confirmation"],
                    evidence={"index_status": status},
                    retryable=False,
                    requires_user_input=True,
                ),
            )
        return ObservationResult(status="partial_success", reason="paper_index_status_unknown", confidence=0.5)

    def _observe_request_confirmation(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, normalized_output, runtime, state
        # request_confirmation 在新的执行模型里只负责准备确认上下文；
        # 真正的暂停点统一放在有副作用 step 的工具调用前，避免提前在“准备步骤”上打断并劫持恢复顺序。
        return ObservationResult(status="success", reason="confirmation_already_available", confidence=1.0)

    def _observe_answer_paper_question(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        normalized_output = _unwrap_payload(normalized_output)
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        answer = str(payload.get("answer") or "").strip()
        sources = payload.get("sources")
        if not answer:
            return ObservationResult(
                status="low_confidence",
                reason="paper_qa_answer_empty",
                confidence=0.2,
                details={"status": payload.get("status"), "error": payload.get("error")},
                suggested_action="answer_with_available_context",
                **_recovery_semantics(
                    failure_category="qa_no_answer",
                    suggested_recovery_types=["retry_step", "ask_clarification", "fallback_answer"],
                    evidence={"status": payload.get("status"), "error": payload.get("error")},
                    retryable=True,
                ),
            )
        if not sources:
            return ObservationResult(
                status="low_confidence",
                reason="paper_qa_answer_without_sources",
                confidence=0.55,
                details={"answer_preview": answer[:120]},
                **_recovery_semantics(
                    failure_category="qa_no_sources",
                    severity="warning",
                    recoverable=True,
                    suggested_recovery_types=["retry_step", "fallback_answer"],
                    evidence={"answer_preview": answer[:120]},
                    retryable=True,
                ),
            )
        return ObservationResult(status="success", reason="paper_qa_answer_available", confidence=0.85, details={"source_count": len(sources) if isinstance(sources, list) else 0})

    def _observe_load_user_profile(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        normalized_output = _unwrap_payload(normalized_output)
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        profile = payload.get("research_profile") if isinstance(payload.get("research_profile"), Mapping) else {}
        summary = payload.get("user_memory_summary")
        if not profile and not summary:
            return ObservationResult(
                status="empty_result",
                reason="user_profile_empty",
                confidence=0.2,
                suggested_action="fallback_to_message_recommendation",
                **_recovery_semantics(
                    failure_category="empty_user_profile",
                    suggested_recovery_types=["patch_plan"],
                    evidence={"has_research_profile": False, "has_user_memory_summary": False},
                    retryable=False,
                ),
            )
        if not profile or not summary:
            return ObservationResult(status="partial_success", reason="user_profile_partial", confidence=0.55)
        return ObservationResult(status="success", reason="user_profile_available", confidence=0.85)

    def _observe_generate_recommendations(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        normalized_output = _unwrap_payload(normalized_output)
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        recommendations = list(payload.get("recommendations") or [])
        if not recommendations:
            return ObservationResult(
                status="empty_result",
                reason="recommendations_empty",
                confidence=0.1,
                suggested_action="fallback",
                **_recovery_semantics(
                    failure_category="insufficient_context",
                    suggested_recovery_types=["fallback_answer"],
                    evidence={"recommendation_count": 0},
                    retryable=False,
                ),
            )
        titles = [str((item or {}).get("title") or "").strip().lower() for item in recommendations if isinstance(item, Mapping)]
        if len(set([title for title in titles if title])) < len([title for title in titles if title]):
            return ObservationResult(
                status="low_confidence",
                reason="recommendations_duplicated",
                confidence=0.35,
                details={"count": len(recommendations)},
                suggested_action="fallback",
                **_recovery_semantics(
                    failure_category="tool_invalid_output",
                    suggested_recovery_types=["fallback_answer"],
                    evidence={"recommendation_count": len(recommendations)},
                    retryable=False,
                ),
            )
        return ObservationResult(status="success", reason="recommendations_available", confidence=0.8, details={"count": len(recommendations)})

    def _observe_update_preference_store(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del raw_output, runtime, state
        normalized_output = _unwrap_payload(normalized_output)
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        if not str(payload.get("arxiv_id") or "").strip():
            return ObservationResult(
                status="need_clarification",
                reason="preference_target_missing",
                confidence=0.2,
                suggested_action="clarify_target",
                **_recovery_semantics(
                    failure_category="preference_target_missing",
                    suggested_recovery_types=["ask_clarification"],
                    evidence={"has_arxiv_id": False},
                    retryable=False,
                    requires_user_input=True,
                ),
            )
        tool_result = payload.get("tool_result") if isinstance(payload.get("tool_result"), Mapping) else {}
        if tool_result and not bool(tool_result.get("ok", False)):
            return ObservationResult(
                status="tool_error",
                reason="preference_store_write_failed",
                confidence=0.1,
                details={"tool_error": tool_result.get("error")},
                suggested_action="fallback",
                **_recovery_semantics(
                    failure_category="preference_write_failed",
                    severity="error",
                    suggested_recovery_types=["fallback_answer"],
                    evidence={"tool_error": tool_result.get("error")},
                    retryable=False,
                ),
            )
        return ObservationResult(status="success", reason="preference_store_updated", confidence=0.9)


__all__ = ["Observer"]
