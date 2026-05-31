from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Type

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from .nodes import (
    build_search_tool_args,
    check_search_result,
    invoke_search_tool,
    parse_search_request,
    personalized_rank_and_annotate_papers,
    synthesize_response,
)
from .state import AgentState

try:  # pragma: no cover - optional dependency in this workspace
    from langgraph.graph import END, START, StateGraph as LangGraphStateGraph  # type: ignore

    HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - fallback for current environment
    HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"

    class LangGraphStateGraph:  # type: ignore[override]
        def __init__(self, state_type: Type[Any]):
            self.state_type = state_type
            self._nodes: Dict[str, Callable[[Any], Any]] = {}
            self._edges: Dict[str, str] = {}
            self._conditional_edges: Dict[str, tuple[Callable[[Any], str], Dict[str, str], Optional[str]]] = {}

        def add_node(self, name: str, func: Callable[[Any], Any]) -> "LangGraphStateGraph":
            self._nodes[name] = func
            return self

        def add_edge(self, source: str, target: str) -> "LangGraphStateGraph":
            self._edges[source] = target
            return self

        def add_conditional_edges(
            self,
            source: str,
            condition: Callable[[Any], str],
            mapping: Dict[str, str],
        ) -> "LangGraphStateGraph":
            self._conditional_edges[source] = (condition, mapping, mapping.get("__default__"))
            return self

        def compile(self) -> "_CompiledArxivSearchGraph":
            return _CompiledArxivSearchGraph(
                nodes=dict(self._nodes),
                edges=dict(self._edges),
                conditional_edges=dict(self._conditional_edges),
            )


class _CompiledArxivSearchGraph:
    def __init__(
        self,
        *,
        nodes: Dict[str, Callable[[Any], Any]],
        edges: Dict[str, str],
        conditional_edges: Dict[str, tuple[Callable[[Any], str], Dict[str, str], Optional[str]]],
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges

    def invoke(self, state: Any, config: Optional[Dict[str, Any]] = None) -> AgentState:
        current_state = _coerce_state(state)
        current_node = self._edges.get(START, "parse_search_request")

        while current_node != END:
            node = self._nodes.get(current_node)
            if node is None:
                break

            result = node(current_state)
            current_state = _coerce_state(result)

            if current_node in self._conditional_edges:
                condition, mapping, default_target = self._conditional_edges[current_node]
                branch = condition(current_state)
                current_node = mapping.get(branch, default_target or END)
            else:
                current_node = self._edges.get(current_node, END)

        return current_state


def route_after_parse(state: Any) -> str:
    current_state = _coerce_state(state)
    intent = str(current_state.intent or "").strip()
    if intent in {"arxiv_search", "unclear", "unsupported"}:
        return intent
    return "unsupported"


def build_arxiv_search_graph(generation_service: Optional[Any] = None) -> Any:
    graph = LangGraphStateGraph(AgentState)

    graph.add_node("parse_search_request", lambda state: parse_search_request(state, generation_service=generation_service))
    graph.add_node("build_search_tool_args", build_search_tool_args)
    graph.add_node("invoke_search_tool", invoke_search_tool)
    graph.add_node("check_search_result", check_search_result)
    graph.add_node("personalized_rank_and_annotate_papers", personalized_rank_and_annotate_papers)
    graph.add_node("synthesize_response", synthesize_response)

    graph.add_edge(START, "parse_search_request")
    graph.add_conditional_edges(
        "parse_search_request",
        route_after_parse,
        {
            "arxiv_search": "build_search_tool_args",
            "unclear": "synthesize_response",
            "unsupported": "synthesize_response",
        },
    )
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
