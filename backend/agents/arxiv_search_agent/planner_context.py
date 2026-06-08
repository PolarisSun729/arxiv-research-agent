"""Planner 输入上下文构造器。

这个模块只负责把分散在 state/context/runtime/tool contract 中的信息整理成稳定输入层；
它不生成计划，也不执行工具，避免 planner 决策继续依赖零散字段读取。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .schemas import Goal, PlannerContext, PlannerToolContext, ToolSpec
from .state import AgentState
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


def build_planner_context(
    *,
    goal: Goal,
    state: AgentState,
    tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY,
) -> PlannerContext:
    """构造本轮 planner 唯一可信的输入快照。

    构造失败应由 planner 入口兜底到固定模板；这里保持纯整理逻辑，让错误边界集中在
    build_executable_plan_for_goal，避免执行链路因为局部上下文字段异常直接中断。
    """

    context = state.context if isinstance(state.context, Mapping) else {}
    runtime_outputs = _extract_runtime_outputs(state)
    intermediate_results = _build_intermediate_results(state, context, runtime_outputs)
    available_tools = [_tool_context_from_spec(tool) for tool in tool_registry.list_tools()]

    return PlannerContext(
        raw_user_request=_normalize_text(state.message),
        normalized_goal=goal,
        goal_type=_normalize_text(goal.goal_type),
        intent=_normalize_text(goal.intent or state.intent),
        intent_confidence=_normalize_confidence(state.llm_confidence),
        selected_paper=_normalize_mapping(context.get("selected_paper")),
        last_papers=_normalize_paper_list(context.get("last_papers") or context.get("papers")),
        paper_qa_result=_normalize_mapping(state.paper_qa_result or context.get("paper_qa_result")),
        pending_action=_normalize_mapping(state.pending_action or context.get("pending_action")),
        user_memory_summary=context.get("user_memory_summary", context.get("memory_summary")),
        research_profile=context.get("research_profile"),
        available_tools=available_tools,
        available_tool_names=[tool.tool_name for tool in available_tools],
        context_refs=_build_context_refs(state, context, runtime_outputs),
        context_field_summary=_compact_mapping(context),
        session_state={
            "user_id": state.user_id,
            "session_id": state.session_id,
            "intent_source": state.intent_source,
            "fallback_reason": state.fallback_reason,
            "has_search_spec": state.search_spec is not None,
            "paper_count": len(list(state.papers or [])),
            "warning_count": len(list(state.warnings or [])),
        },
        intermediate_results=intermediate_results,
        reusable_outputs=_build_reusable_outputs(intermediate_results),
        high_risk_tools=[
            tool.tool_name
            for tool in available_tools
            if tool.requires_confirmation or tool.side_effect_level in {"persistent_write", "external_call"}
        ],
    )


def planner_context_debug(planner_context: PlannerContext) -> Dict[str, Any]:
    """生成适合 debug 暴露的轻量摘要。

    原始上下文可能包含较大的论文列表或工具输出；debug 只暴露字段形态和关键引用，
    既能说明 planner 看到了什么，也避免把大对象重复塞进响应。
    """

    return {
        "raw_user_request": planner_context.raw_user_request,
        "goal": planner_context.normalized_goal.model_dump() if planner_context.normalized_goal else None,
        "goal_type": planner_context.goal_type,
        "intent": planner_context.intent,
        "intent_confidence": planner_context.intent_confidence,
        "context_refs": list(planner_context.context_refs or []),
        "context_field_summary": dict(planner_context.context_field_summary or {}),
        "used_context_fields": _used_context_fields(planner_context),
        "selected_paper": _paper_debug_summary(planner_context.selected_paper),
        "last_papers_count": len(list(planner_context.last_papers or [])),
        "paper_qa_result_status": (
            planner_context.paper_qa_result or {}
        ).get("status") if isinstance(planner_context.paper_qa_result, Mapping) else None,
        "pending_action_type": (
            planner_context.pending_action or {}
        ).get("type") if isinstance(planner_context.pending_action, Mapping) else None,
        "session_state": dict(planner_context.session_state or {}),
        "intermediate_result_keys": sorted((planner_context.intermediate_results or {}).keys()),
        "reusable_output_keys": sorted((planner_context.reusable_outputs or {}).keys()),
        "available_tool_names": list(planner_context.available_tool_names or []),
        "high_risk_tools": list(planner_context.high_risk_tools or []),
        "available_tools": [
            {
                "tool_name": tool.tool_name,
                "capability_tags": list(tool.capability_tags or []),
                "side_effect_level": tool.side_effect_level,
                "requires_confirmation": tool.requires_confirmation,
                "failure_modes": list(tool.failure_modes or []),
                "recovery_policy": dict(tool.recovery_policy or {}),
                "confirmation_policy": dict(tool.confirmation_policy or {}),
            }
            for tool in list(planner_context.available_tools or [])
        ],
    }


def _tool_context_from_spec(tool: ToolSpec) -> PlannerToolContext:
    return PlannerToolContext(
        tool_name=tool.tool_name,
        description=tool.description,
        capability_tags=list(tool.capability_tags or []),
        side_effect_level=tool.side_effect_level,
        requires_confirmation=bool(tool.requires_confirmation),
        can_retry=bool(tool.can_retry),
        failure_modes=list(tool.failure_modes or []),
        recovery_policy=dict(tool.recovery_policy or {}),
        confirmation_policy=dict(tool.confirmation_policy or {}),
        input_schema=dict(tool.input_schema or {}),
        output_schema=dict(tool.output_schema or {}),
    )


def _extract_runtime_outputs(state: AgentState) -> Dict[str, Any]:
    runtime = state.plan_runtime
    if runtime is None and state.runtime_state is not None:
        outputs = getattr(state.runtime_state, "outputs", None)
        return dict(outputs or {}) if isinstance(outputs, Mapping) else {}
    outputs = getattr(runtime, "outputs", None)
    return dict(outputs or {}) if isinstance(outputs, Mapping) else {}


def _build_intermediate_results(
    state: AgentState,
    context: Mapping[str, Any],
    runtime_outputs: Mapping[str, Any],
) -> Dict[str, Any]:
    results: Dict[str, Any] = dict(runtime_outputs or {})
    # planner 只需要知道哪些结果已经存在；这里保留引用摘要，后续 builder 再决定是否复用。
    for key, value in {
        "search_spec": state.search_spec.model_dump() if state.search_spec is not None else None,
        "papers": list(state.papers or []),
        "tool_result": state.tool_result,
        "paper_qa_result": state.paper_qa_result or context.get("paper_qa_result"),
        "preference_action_result": state.preference_action_result,
        "selected_paper": context.get("selected_paper"),
        "last_papers": context.get("last_papers") or context.get("papers"),
    }.items():
        if value not in (None, "", [], {}):
            results.setdefault(key, value)
    return results


def _build_reusable_outputs(intermediate_results: Mapping[str, Any]) -> Dict[str, Any]:
    reusable: Dict[str, Any] = {}
    for key in ("search_spec", "papers", "selected_paper", "last_papers", "paper_ref", "paper_qa_result"):
        value = intermediate_results.get(key)
        if value not in (None, "", [], {}):
            reusable[key] = value
    return reusable


def _build_context_refs(
    state: AgentState,
    context: Mapping[str, Any],
    runtime_outputs: Mapping[str, Any],
) -> List[str]:
    refs: List[str] = []
    for key in ("selected_paper", "last_papers", "user_memory_summary", "memory_summary", "research_profile", "pending_action", "paper_qa_result"):
        value = context.get(key)
        if value not in (None, "", [], {}):
            refs.append(key)
    if isinstance(state.pending_action, Mapping):
        refs.append("state.pending_action")
    if isinstance(state.paper_qa_result, Mapping):
        refs.append("state.paper_qa_result")
    if runtime_outputs:
        refs.append("runtime.outputs")
    refs.extend(str(key) for key in context.keys())
    return _dedupe_strings(refs)


def _used_context_fields(planner_context: PlannerContext) -> List[str]:
    fields: List[str] = []
    if planner_context.selected_paper:
        fields.append("selected_paper")
    if planner_context.last_papers:
        fields.append("last_papers")
    if planner_context.paper_qa_result:
        fields.append("paper_qa_result")
    if planner_context.pending_action:
        fields.append("pending_action")
    if planner_context.user_memory_summary not in (None, "", [], {}):
        fields.append("user_memory_summary")
    if planner_context.research_profile not in (None, "", [], {}):
        fields.append("research_profile")
    if planner_context.intermediate_results:
        fields.append("intermediate_results")
    return fields


def _normalize_mapping(value: Any) -> Optional[Dict[str, Any]]:
    return dict(value) if isinstance(value, Mapping) else None


def _normalize_paper_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    papers: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            papers.append(dict(item))
    return papers


def _paper_debug_summary(paper: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(paper, Mapping):
        return None
    return {
        "arxiv_id": paper.get("arxiv_id") or paper.get("paper_id"),
        "title": paper.get("title"),
    }


def _compact_mapping(mapping: Mapping[str, Any], *, limit: int = 16) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for index, (key, value) in enumerate(mapping.items()):
        if index >= limit:
            compact["_truncated"] = True
            break
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[str(key)] = value
        elif isinstance(value, Mapping):
            compact[str(key)] = {"type": "object", "keys": list(value.keys())[:8]}
        elif isinstance(value, list):
            compact[str(key)] = {"type": "list", "count": len(value)}
        else:
            compact[str(key)] = {"type": type(value).__name__}
    return compact


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_strings(items: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for item in items:
        text = _normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


__all__ = ["build_planner_context", "planner_context_debug"]
