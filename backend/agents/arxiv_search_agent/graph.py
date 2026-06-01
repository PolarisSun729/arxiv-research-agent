from __future__ import annotations

from typing import Any, Mapping, Optional

from langgraph.graph import END, START, StateGraph

from .nodes import (
    build_search_tool_args,
    check_search_result,
    apply_preference_action,
    invoke_search_tool,
    parse_search_request,
    personalized_rank_and_annotate_papers,
    synthesize_response,
)
from .state import AgentState


def route_after_parse(state: Any) -> str:
    current_state = _coerce_state(state)
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


def build_arxiv_search_graph(generation_service: Optional[Any] = None) -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("parse_search_request", lambda state: parse_search_request(state, generation_service=generation_service))
    graph.add_node("build_search_tool_args", build_search_tool_args)
    graph.add_node("invoke_search_tool", invoke_search_tool)
    graph.add_node("check_search_result", check_search_result)
    graph.add_node("personalized_rank_and_annotate_papers", personalized_rank_and_annotate_papers)
    graph.add_node("apply_preference_action", apply_preference_action)
    graph.add_node("synthesize_response", synthesize_response)

    graph.add_edge(START, "parse_search_request")
    graph.add_conditional_edges(
        "parse_search_request",
        route_after_parse,
        {
            "arxiv_search": "build_search_tool_args",
            "paper_detail": "synthesize_response",
            "paper_summary": "synthesize_response",
            "paper_qa": "synthesize_response",
            "recommendation": "synthesize_response",
            # 偏好动作先经过一个轻量占位节点，后续接入真正的写入逻辑时不需要改路由入口。
            "preference_action": "apply_preference_action",
            "reading_list_action": "synthesize_response",
            "unclear": "synthesize_response",
            "unsupported": "synthesize_response",
        },
    )
    graph.add_edge("apply_preference_action", "synthesize_response")
    graph.add_edge("build_search_tool_args", "invoke_search_tool")
    graph.add_edge("invoke_search_tool", "check_search_result")
    graph.add_edge("check_search_result", "personalized_rank_and_annotate_papers")
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
