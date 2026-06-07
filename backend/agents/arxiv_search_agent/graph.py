from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .node import parse_search_request
from .plan_executor import run_agent_turn, run_agent_turn_in_graph
from .schemas import AgentStep, AgentTurnResult
from .state import AgentState

logger = logging.getLogger(__name__)

# 默认内存 checkpointer 只创建一次，保证同一进程内同一个 thread_id 的中断状态可恢复。
# 这里故意把“默认实例”提升到模块级，避免每次请求临时 new 一个内存存储导致 session 无法续跑。
# 后续生产环境如果要接 Redis/数据库等持久化实现，只需要从 service 或依赖注入层传入新的 checkpointer。
DEFAULT_GRAPH_CHECKPOINTER = InMemorySaver()

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
    confirmation_payload = result.pending_confirmation.model_dump() if result.pending_confirmation is not None else None
    next_state.pending_action = _build_compatible_pending_action(result)

    next_state.debug = dict(next_state.debug or {})
    next_state.debug["agent_turn"] = {
        "status": result.status,
        "error": result.error,
        "output_keys": sorted(result.outputs.keys()),
        "trace_events": [trace.event for trace in list(result.trace or [])],
    }
    if confirmation_payload is not None:
        next_state.debug["pending_confirmation"] = confirmation_payload

    if result.status == "waiting_confirmation":
        next_state.paper_qa_result = {
            "status": "waiting_confirmation",
            "pending_confirmation": confirmation_payload,
        }

    if "preference_action_result" in result.outputs:
        preference_result = result.outputs.get("preference_action_result")
        next_state.preference_action_result = dict(preference_result) if isinstance(preference_result, Mapping) else {"value": preference_result}

    if "paper_qa_result" in result.outputs:
        # answer_paper_question 的输出来自 PaperQAService 真实 RAG 链路，sources/retrieval_debug 必须原样带给前端。
        paper_qa_result = result.outputs.get("paper_qa_result")
        next_state.paper_qa_result = _build_paper_qa_result(next_state, paper_qa_result)
        if next_state.paper_qa_result.get("answer"):
            next_state.answer = str(next_state.paper_qa_result.get("answer") or "")

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


def _build_paper_qa_result(state: AgentState, payload: Any) -> dict[str, Any]:
    """把真实 PaperQA 工具输出整理成前端沿用的 paper_qa_result。

    这里只做字段适配，不补造 chunk 或证据；retrieval_debug/sources 均以 PaperQAService 返回为准。
    """
    data = dict(payload) if isinstance(payload, Mapping) else {"value": payload}
    context = state.context if isinstance(state.context, Mapping) else {}
    selected_paper = context.get("selected_paper") if isinstance(context.get("selected_paper"), Mapping) else {}
    answer = str(data.get("answer") or "").strip()
    status = str(data.get("status") or ("success" if answer else "failed")).strip() or "failed"
    return {
        "status": status,
        "arxiv_id": data.get("arxiv_id") or context.get("arxiv_id") or selected_paper.get("arxiv_id"),
        "title": data.get("title") or selected_paper.get("title"),
        "question": data.get("question") or state.message,
        "answer": answer,
        "sources": data.get("sources", []),
        "retrieval_debug": data.get("retrieval_debug"),
        "error": data.get("error"),
        "tool_result": data.get("tool_result"),
    }


def _build_compatible_pending_action(result: AgentTurnResult) -> Optional[dict[str, Any]]:
    """把新的确认请求结构映射成前端沿用的 pending_action 外显字段。

    这里继续提供一个轻量 dict 作为确认卡片的数据源；
    恢复执行的真源是 LangGraph checkpointer 中的 interrupt 现场，而不是这个展示镜像。
    """
    confirmation = result.pending_confirmation
    if confirmation is None:
        return None
    payload = confirmation.model_dump()
    target_paper = dict(confirmation.target_paper or {})
    arguments_summary = dict(confirmation.arguments_summary or {})
    return {
        "type": "tool_approval",
        "status": "waiting_confirmation",
        "decision": None,
        "step_id": confirmation.step_id,
        "tool_name": confirmation.tool_name,
        "action_type": confirmation.action_type,
        "side_effect_level": confirmation.side_effect_level,
        "reason": confirmation.reason,
        "title": target_paper.get("title") or confirmation.title,
        "title_text": confirmation.title,
        "description": confirmation.description,
        "arxiv_id": target_paper.get("arxiv_id"),
        "original_question": confirmation.original_question,
        "target_paper": target_paper or None,
        "allowed_decisions": [item.code for item in list(confirmation.allowed_decisions or [])],
        "allow_argument_edit": confirmation.allow_argument_edit,
        "allow_reject": confirmation.allow_reject,
        "allow_note": confirmation.allow_note,
        "arguments_summary": arguments_summary,
        "confirmation_request": payload,
        "thread_id": confirmation.thread_id,
        "session_id": confirmation.session_id,
        "plan_id": confirmation.plan_id,
        "trace_id": confirmation.trace_id,
        "qa_question": arguments_summary.get("qa_question") or arguments_summary.get("question"),
    }


def run_agent_turn_node(state: Any) -> AgentState:
    """LangGraph 主流程节点：直接执行新的 plan runtime。"""
    current_state = _coerce_state(state)
    result = run_agent_turn_in_graph(current_state)
    return _state_from_turn_result(current_state, result)


def route_after_parse(state: Any) -> str:
    """兼容旧导出名；Step 6 主图不再使用条件路由。"""
    del state
    return "run_agent_turn"


def build_arxiv_search_graph(
    generation_service: Optional[Any] = None,
    *,
    checkpointer: Optional[Any] = None,
) -> Any:
    """构建 Step 6 主图：parse_search_request -> run_agent_turn -> END。

    默认使用进程级共享的内存 checkpointer，先满足本地开发和测试场景下的 interrupt/resume。
    外部显式传入 checkpointer 时，以外部实现为准，给后续替换成持久化 checkpoint 预留入口。
    """
    graph = StateGraph(AgentState)

    graph.add_node("parse_search_request", lambda state: parse_search_request(state, generation_service=generation_service))
    graph.add_node("run_agent_turn", run_agent_turn_node)

    graph.add_edge(START, "parse_search_request")
    graph.add_edge("parse_search_request", "run_agent_turn")
    graph.add_edge("run_agent_turn", END)

    compiled_checkpointer = checkpointer if checkpointer is not None else DEFAULT_GRAPH_CHECKPOINTER
    return graph.compile(checkpointer=compiled_checkpointer)


def export_arxiv_search_graph_mermaid(
    generation_service: Optional[Any] = None,
    *,
    checkpointer: Optional[Any] = None,
) -> dict[str, Any]:
    """导出当前主流程图结构，优先使用 LangGraph 原生 Mermaid。"""
    compiled_graph = build_arxiv_search_graph(
        generation_service=generation_service,
        checkpointer=checkpointer,
    )
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
    "DEFAULT_GRAPH_CHECKPOINTER",
    "build_arxiv_search_graph",
    "export_arxiv_search_graph_mermaid",
    "route_after_parse",
    "START",
    "END",
]
