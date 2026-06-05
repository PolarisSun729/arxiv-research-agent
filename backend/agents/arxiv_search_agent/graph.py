from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from langgraph.graph import END, START, StateGraph

from .node import parse_search_request
from .plan_executor import run_agent_turn
from .schemas import AgentStep, AgentTurnResult
from .state import AgentState

logger = logging.getLogger(__name__)

_ARXIV_GRAPH_NODE_NAMES = (
    "parse_search_request",
    "run_agent_turn",
)


def _coerce_state(state: Any) -> AgentState:
    """把 LangGraph 运行时可能传入的 dict/model 统一转换成 AgentState。"""
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)


def _state_from_turn_result(source_state: AgentState, result: AgentTurnResult) -> AgentState:
    """把统一执行入口的结果回写到 AgentState，供 service/stream 复用响应适配器。"""
    next_state = source_state.model_copy(deep=True)
    next_state.goal = result.plan.goal if result.plan is not None else source_state.goal
    next_state.execution_plan = result.plan
    next_state.plan_runtime = result.runtime
    next_state.answer = result.final_answer
    next_state.pending_action = result.pending_confirmation

    next_state.debug = dict(next_state.debug or {})
    next_state.debug["agent_turn"] = {
        "status": result.status,
        "error": result.error,
        "output_keys": sorted(result.outputs.keys()),
        "trace_events": [trace.event for trace in list(result.trace or [])],
    }

    if result.status == "waiting_confirmation":
        next_state.paper_qa_result = {
            "status": "waiting_confirmation",
            "pending_confirmation": result.pending_confirmation,
        }

    if "preference_action_result" in result.outputs:
        preference_result = result.outputs.get("preference_action_result")
        next_state.preference_action_result = dict(preference_result) if isinstance(preference_result, Mapping) else {"value": preference_result}

    if "ranked_papers" in result.outputs and isinstance(result.outputs.get("ranked_papers"), list):
        next_state.papers = [dict(item) for item in result.outputs["ranked_papers"] if isinstance(item, Mapping)]

    # 这里记录的是新 runtime 的单节点摘要；详细步骤以 result.trace / plan_runtime 为准。
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="run_agent_turn",
            status="success" if result.status in {"success", "waiting_confirmation", "need_clarification", "fallback"} else "failed",
            action="execute_executable_plan",
            inputs={"intent": next_state.intent, "message": next_state.message},
            outputs={"status": result.status, "output_keys": sorted(result.outputs.keys())},
            error=result.error,
        )
    ]
    return next_state


def run_agent_turn_node(state: Any) -> AgentState:
    """LangGraph 主流程节点：直接执行新的 plan runtime。"""
    current_state = _coerce_state(state)
    result = run_agent_turn(current_state)
    return _state_from_turn_result(current_state, result)


def route_after_parse(state: Any) -> str:
    """兼容旧导出名；Step 6 主图不再使用条件路由。"""
    del state
    return "run_agent_turn"


def build_arxiv_search_graph(generation_service: Optional[Any] = None) -> Any:
    """构建 Step 6 主图：parse_search_request -> run_agent_turn -> END。"""
    graph = StateGraph(AgentState)

    graph.add_node("parse_search_request", lambda state: parse_search_request(state, generation_service=generation_service))
    graph.add_node("run_agent_turn", run_agent_turn_node)

    graph.add_edge(START, "parse_search_request")
    graph.add_edge("parse_search_request", "run_agent_turn")
    graph.add_edge("run_agent_turn", END)

    return graph.compile()


def export_arxiv_search_graph_mermaid(generation_service: Optional[Any] = None) -> dict[str, Any]:
    """导出当前主流程图结构，优先使用 LangGraph 原生 Mermaid。"""
    compiled_graph = build_arxiv_search_graph(generation_service=generation_service)
    drawable_graph = None
    mermaid = ""
    render_source = "langgraph"

    try:
        drawable_graph = compiled_graph.get_graph()
        mermaid = drawable_graph.draw_mermaid()
    except Exception as exc:  # pragma: no cover - 绘图依赖不可用时走兜底
        render_source = "fallback"
        logger.warning("arxiv_agent graph mermaid export fallback: error=%s", exc)
        mermaid = _build_fallback_mermaid()

    return {
        "graph_name": "arxiv_search_agent",
        "render_source": render_source,
        "node_names": list(_ARXIV_GRAPH_NODE_NAMES),
        "mermaid": mermaid,
        "supports_png": bool(drawable_graph and hasattr(drawable_graph, "draw_mermaid_png")),
    }


def _build_fallback_mermaid() -> str:
    """生成与 Step 6 主流程一致的最小 Mermaid 兜底图。"""
    return "\n".join(
        [
            "graph TD;",
            "    START([START]) --> parse_search_request;",
            "    parse_search_request --> run_agent_turn;",
            "    run_agent_turn --> END([END]);",
        ]
    )


__all__ = [
    "build_arxiv_search_graph",
    "export_arxiv_search_graph_mermaid",
    "route_after_parse",
    "START",
    "END",
]
