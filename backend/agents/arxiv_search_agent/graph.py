from __future__ import annotations

from typing import Any, Mapping, Optional

from langgraph.graph import END, START, StateGraph

from .nodes import (
    apply_preference_action,
    build_search_tool_args,
    check_search_result,
    classify_pending_action_confirmation,
    handle_paper_reading_request,
    handle_pending_action_confirmation,
    invoke_search_tool,
    parse_search_request,
    personalized_rank_and_annotate_papers,
    relax_search_for_retry,
    synthesize_response,
)
from .state import AgentState


def route_after_parse(state: Any) -> str:
    current_state = _coerce_state(state)
    pending_action = current_state.context.get("pending_action") if isinstance(current_state.context, dict) else None
    if isinstance(pending_action, dict) and str(pending_action.get("type") or "").strip() == "parse_then_qa" and str(
        pending_action.get("status") or ""
    ).strip() == "waiting_confirmation":
        return "classify_pending_action_confirmation"

    intent = str(current_state.intent or "").strip()
    if intent in {
        "arxiv_search",
        "paper_detail",
        "paper_summary",
        "paper_qa",
        "recommendation",
        "preference_action",
        "reading_list_action",
        "unclear",
        "unsupported",
    }:
        return intent
    return "unsupported"


def route_after_pending_confirmation(state: Any) -> str:
    current_state = _coerce_state(state)
    decision = str((current_state.debug or {}).get("pending_action_decision") or "").strip().lower()
    if decision == "confirm":
        return "handle_pending_action_confirmation"
    return "synthesize_response"


def route_after_check(state: Any) -> str:
    """搜索结果检查后的路由：空结果且未超重试次数则放宽后重试。"""
    current_state = _coerce_state(state)
    papers = list(current_state.papers or [])
    retry_count = int(current_state.search_retry_count or 0)
    tool_result = current_state.tool_result
    tool_ok = bool(tool_result and isinstance(tool_result, dict) and tool_result.get("ok"))

    # 只有搜索成功但结果为空，且未超过最大重试次数时才触发 fallback
    if tool_ok and not papers and retry_count < 3:
        return "relax_search_for_retry"
    return "personalized_rank_and_annotate_papers"


def build_arxiv_search_graph(generation_service: Optional[Any] = None) -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("parse_search_request", lambda state: parse_search_request(state, generation_service=generation_service))
    graph.add_node("build_search_tool_args", build_search_tool_args)
    graph.add_node("invoke_search_tool", invoke_search_tool)
    graph.add_node("check_search_result", check_search_result)
    graph.add_node("relax_search_for_retry", relax_search_for_retry)
    graph.add_node("personalized_rank_and_annotate_papers", personalized_rank_and_annotate_papers)
    graph.add_node("apply_preference_action", apply_preference_action)
    graph.add_node(
        "classify_pending_action_confirmation",
        lambda state: classify_pending_action_confirmation(state, generation_service=generation_service),
    )
    graph.add_node("handle_pending_action_confirmation", handle_pending_action_confirmation)
    graph.add_node("handle_paper_reading_request", handle_paper_reading_request)
    graph.add_node("synthesize_response", synthesize_response)

    graph.add_edge(START, "parse_search_request")
    graph.add_conditional_edges(
        "parse_search_request",
        route_after_parse,
        {
            "arxiv_search": "build_search_tool_args",
            "paper_detail": "handle_paper_reading_request",
            "paper_summary": "handle_paper_reading_request",
            "paper_qa": "handle_paper_reading_request",
            "recommendation": "synthesize_response",
            "preference_action": "apply_preference_action",
            "classify_pending_action_confirmation": "classify_pending_action_confirmation",
            "reading_list_action": "synthesize_response",
            "unclear": "synthesize_response",
            "unsupported": "synthesize_response",
        },
    )
    graph.add_conditional_edges(
        "classify_pending_action_confirmation",
        route_after_pending_confirmation,
        {
            "handle_pending_action_confirmation": "handle_pending_action_confirmation",
            "synthesize_response": "synthesize_response",
        },
    )
    graph.add_edge("apply_preference_action", "synthesize_response")
    graph.add_edge("handle_pending_action_confirmation", "synthesize_response")
    graph.add_edge("handle_paper_reading_request", "synthesize_response")
    graph.add_edge("build_search_tool_args", "invoke_search_tool")
    graph.add_edge("invoke_search_tool", "check_search_result")
    graph.add_conditional_edges(
        "check_search_result",
        route_after_check,
        {
            "relax_search_for_retry": "relax_search_for_retry",
            "personalized_rank_and_annotate_papers": "personalized_rank_and_annotate_papers",
        },
    )
    graph.add_edge("relax_search_for_retry", "build_search_tool_args")
    graph.add_edge("personalized_rank_and_annotate_papers", "synthesize_response")
    graph.add_edge("synthesize_response", END)

    return graph.compile()


def _coerce_state(state: Any) -> AgentState:
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)


__all__ = ["build_arxiv_search_graph", "route_after_parse", "START", "END"]
