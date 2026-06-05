from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

from ..schemas import ToolCallRequest
from ..state import AgentState
from ..utils.result_utils import _extract_error_message, _extract_papers_from_tool_result, _result_mapping, _result_ok, _result_text
from ..utils.state_utils import _append_step, _coerce_state, _get_execution_plan_step
from ..utils.text_utils import _normalize_text
from .tool_node import execute_tool

RECOMMENDATION_TOOL_NAME = "recommend_papers"


def _get_recommendation_execution_plan_step_id(state: AgentState) -> Optional[str]:
    step = _get_execution_plan_step(state, step_type="recommendation_generation")
    if step is None:
        return None
    step_id = str(getattr(step, "step_id", "") or "").strip()
    return step_id or None


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_recommendation_topic_hint(message: str, context: Mapping[str, Any]) -> Optional[str]:
    explicit = _normalize_text(context.get("recommendation_topic") or context.get("topic_hint"))
    if explicit:
        return explicit
    normalized_message = _normalize_text(message)
    return normalized_message or None


def build_recommendation_tool_args(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "recommendation":
        next_state.tool_call_request = None
        return _append_step(
            next_state,
            step="build_recommendation_tool_args",
            status="skipped",
            action="构造推荐工具参数",
            inputs={"intent": next_state.intent},
            outputs={"reason": "当前请求不是 recommendation"},
        )

    context = dict(next_state.context or {})
    user_memory_summary = context.get("user_memory_summary") or context.get("memory_summary")
    user_memory_summary_text = _normalize_text(user_memory_summary)
    research_profile = _coerce_mapping(context.get("research_profile"))
    resolved_user_id = str(next_state.user_id or context.get("user_id") or "").strip()
    user_id = resolved_user_id or "default"
    topic_hint = _resolve_recommendation_topic_hint(next_state.message or "", context)
    warnings = list(next_state.warnings or [])
    if not resolved_user_id:
        warnings.append("当前未提供明确 user_id，推荐将使用默认用户画像。")
    if not topic_hint:
        warnings.append("当前未提供明确主题提示，推荐将主要依赖历史偏好与用户画像。")
    if not user_memory_summary_text and not research_profile:
        warnings.append("当前缺少显式用户画像摘要，推荐将优先依赖兴趣向量和历史偏好。")

    tool_args = {
        "user_id": user_id,
        "top_n": _coerce_positive_int(context.get("top_n") or context.get("recommendation_top_n"), 10),
        "max_age_months": _coerce_positive_int(context.get("max_age_months") or context.get("recommendation_max_age_months"), 6),
        "message": _normalize_text(next_state.message),
        "topic_hint": topic_hint,
        "user_memory_summary": user_memory_summary_text,
        "research_profile": research_profile or None,
        "request_context": {
            "has_user_memory_summary": bool(user_memory_summary_text),
            "has_research_profile": bool(research_profile),
            "context_keys": sorted(context.keys()),
        },
    }
    next_state.warnings = warnings
    next_state.tool_name = RECOMMENDATION_TOOL_NAME
    next_state.tool_args = dict(tool_args)
    next_state.tool_call_request = ToolCallRequest(
        tool_name=RECOMMENDATION_TOOL_NAME,
        arguments=dict(tool_args),
        reason="根据用户画像、兴趣向量和历史偏好生成个性化论文推荐",
        expected_result="返回推荐论文列表、推荐说明和个性化信号摘要",
        plan_step_id=_get_recommendation_execution_plan_step_id(next_state),
    )
    debug = dict(next_state.debug or {})
    debug["recommendation_request"] = {
        "user_id": user_id,
        "topic_hint": topic_hint,
        "uses_user_memory_summary": bool(user_memory_summary_text),
        "uses_research_profile": bool(research_profile),
        "top_n": tool_args["top_n"],
        "max_age_months": tool_args["max_age_months"],
    }
    next_state.debug = debug
    return _append_step(
        next_state,
        step="build_recommendation_tool_args",
        status="success",
        action="构造推荐工具参数",
        inputs={"intent": next_state.intent, "context": context},
        outputs={
            "tool_call_request": next_state.tool_call_request.model_dump(),
            "warnings": list(next_state.warnings or []),
        },
    )


def invoke_recommendation_tool(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    next_state = execute_tool(state)
    if next_state.steps and next_state.steps[-1].step == "execute_tool":
        next_state.steps[-1].step = "recommendation_tool_call"
        next_state.steps[-1].action = "调用论文推荐工具"
    return next_state


def adapt_recommendation_tool_result(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    result = dict(next_state.tool_result or {})
    papers = list(_extract_papers_from_tool_result(result))
    data = result.get("data") if isinstance(result.get("data"), Mapping) else {}
    if not papers and isinstance(data, Mapping):
        for key in ("recommended_papers", "recommendations", "items"):
            value = data.get(key)
            if isinstance(value, list):
                papers = list(value)
                break

    next_state.papers = [paper for paper in papers if isinstance(paper, dict)]
    debug = dict(next_state.debug or {})
    recommendation_data = dict(data or {}) if isinstance(data, Mapping) else {}
    debug["recommendation_tool_result"] = result
    debug["recommendation_result"] = {
        "paper_count": len(next_state.papers),
        "personalization_signals": dict(recommendation_data.get("personalization_signals") or {}),
        "interest_profile_mode": recommendation_data.get("interest_profile_mode"),
        "interest_cluster_count": recommendation_data.get("interest_cluster_count"),
        "recall_mode": recommendation_data.get("recall_mode"),
    }
    next_state.debug = debug

    if next_state.tool_observations:
        last_observation = next_state.tool_observations[-1]
        if last_observation.tool_name == RECOMMENDATION_TOOL_NAME:
            last_observation.is_sufficient = bool(next_state.papers)
            last_observation.next_action_hint = None if next_state.papers else "adjust_recommendation_constraints_or_collect_more_preferences"

    if _result_ok(result) and not next_state.papers:
        next_state.warnings = list(next_state.warnings or []) + ["当前推荐结果为空，可以补充主题偏好或先标记几篇喜欢的论文。"]

    if _result_ok(result):
        return _append_step(
            next_state,
            step="adapt_recommendation_tool_result",
            status="success",
            action="适配推荐工具返回结果",
            inputs={"tool_name": next_state.tool_name},
            outputs={
                "paper_count": len(next_state.papers),
                "summary": _result_text(result, "summary"),
                "trace": _result_mapping(result, "trace") or {},
                "personalization_signals": dict(recommendation_data.get("personalization_signals") or {}),
            },
        )

    return _append_step(
        next_state,
        step="adapt_recommendation_tool_result",
        status="failed",
        action="适配推荐工具返回结果",
        inputs={"tool_name": next_state.tool_name},
        outputs={"paper_count": len(next_state.papers), "error": _result_mapping(result, "error")},
        error=_extract_error_message(result),
    )


__all__ = [
    "build_recommendation_tool_args",
    "invoke_recommendation_tool",
    "adapt_recommendation_tool_result",
]
