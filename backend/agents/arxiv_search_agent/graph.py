from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from langgraph.graph import END, START, StateGraph

from .node import parse_search_request
from .planner import GoalBuilder, build_executable_plan_for_goal, build_plan_runtime
from .plan_executor import PlanExecutor
from .runtime_checkpoint import build_agent_checkpointer
from .schemas import AgentRuntimeState, AgentStep, AgentTurnResult, ExecutablePlan, PlanRuntime, StepExecutionResult
from .state import AgentState
from .tool_registry import PLANNER_TOOL_REGISTRY

logger = logging.getLogger(__name__)

# 默认 checkpointer 走项目数据库持久化；只有通过 AGENT_RUNTIME_CHECKPOINT_BACKEND=memory 显式切换时，
# 才会退回进程内开发模式，避免生产路径把 resume 真源绑死在单进程内存里。
DEFAULT_GRAPH_CHECKPOINTER = build_agent_checkpointer()

_ARXIV_GRAPH_NODE_NAMES = (
    "parse_search_request",
    "build_goal",
    "build_plan",
    "select_next_step",
    "execute_step",
    "observe_step",
    "route_after_observation",
    "replan",
    "finalize",
    "error_finalize",
)


def _coerce_state(state: Any) -> AgentState:
    """把 LangGraph 运行时可能传入的 dict/model 统一转换成 AgentState。"""
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)


def _executor() -> PlanExecutor:
    """统一创建执行器，图节点只关心编排，不直接触碰工具注册细节。"""
    return PlanExecutor(tool_registry=PLANNER_TOOL_REGISTRY)


def build_goal_node(state: Any) -> AgentState:
    """只负责把 parse 后的 intent/context 转成结构化 Goal。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    goal = GoalBuilder.from_state(next_state)
    next_state.goal = goal
    next_state.debug = dict(next_state.debug or {})
    next_state.debug["goal"] = goal.model_dump()
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="build_goal",
            status="success",
            action="根据解析结果构建本轮目标",
            inputs={"intent": next_state.intent, "message": next_state.message},
            outputs={"goal_type": goal.goal_type, "risk_level": goal.risk_level},
            error=None,
        )
    ]
    return next_state


def build_plan_node(state: Any) -> AgentState:
    """只负责生成和校验 ExecutablePlan，并初始化 PlanRuntime。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    goal = next_state.goal or GoalBuilder.from_state(next_state)
    # 规划策略仍复用现有 rule/template/tool-aware 能力，但调用位置已经变成图上的 planning 节点。
    goal, plan, planning_debug = build_executable_plan_for_goal(goal, next_state, tool_registry=PLANNER_TOOL_REGISTRY)
    runtime = build_plan_runtime(next_state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
    runtime.outputs = {}
    runtime.trace = []
    runtime.retry_counts = {}
    runtime.replan_counts = {}
    runtime.step_replan_counts = {}

    next_state.goal = goal
    next_state.execution_plan = plan
    next_state.plan_runtime = runtime
    next_state.debug = dict(next_state.debug or {})
    next_state.debug["planner"] = planning_debug
    # 计划节点只建立现场，不执行任何工具；后续节点通过 runtime_state 继续推进。
    next_state.runtime_state = _runtime_state_from_runtime(next_state, runtime)
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="build_plan",
            status="success",
            action="生成并校验可执行计划",
            inputs={"goal_type": goal.goal_type},
            outputs={"plan_id": plan.plan_id, "step_count": len(plan.steps or [])},
            error=None,
        )
    ]
    return next_state


def select_next_step_node(state: Any) -> AgentState:
    """只负责选择下一可执行 step，不调用工具。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    runtime = _ensure_runtime(next_state)
    step = _executor().select_next_step(runtime, next_state)
    next_state.plan_runtime = runtime
    next_state.runtime_state = _runtime_state_from_runtime(next_state, runtime)
    next_state.debug = dict(next_state.debug or {})
    next_state.debug["agent_route"] = {
        "phase": "select_next_step",
        "current_step_id": step.step_id if step else None,
        "current_tool_name": step.tool_name if step else None,
    }
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="select_next_step",
            status="success",
            action="选择下一可执行计划步骤",
            inputs={"pending_statuses": dict(runtime.step_status or {})},
            outputs={"step_id": step.step_id if step else None, "tool_name": step.tool_name if step else None},
            error=None,
        )
    ]
    return next_state


def execute_step_node(state: Any) -> AgentState:
    """只负责执行当前 step 对应工具。

    这里关闭 executor 的 auto_replan，让低质量结果留给图上的 replan 节点处理；
    这样 Mermaid 和流式事件都能看到 execute -> observe -> replan 的真实路径。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    runtime = _ensure_runtime(next_state)
    result = _executor().execute_current_step_tool(runtime, next_state, allow_interrupt=True)
    next_state.plan_runtime = runtime
    next_state.runtime_state = _runtime_state_from_runtime(next_state, runtime)
    _apply_step_result(next_state, result)
    return next_state


def observe_step_node(state: Any) -> AgentState:
    """只负责把最近 observation 投影到结构化 runtime/debug，供条件边路由。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    runtime = _ensure_runtime(next_state)
    result = _executor().observe_current_step(runtime, next_state)
    observation = dict(runtime.last_observation or {}) if isinstance(runtime.last_observation, Mapping) else None
    next_state.debug = dict(next_state.debug or {})
    next_state.debug["last_observation"] = observation
    next_state.runtime_state = _runtime_state_from_runtime(next_state, runtime)
    _apply_step_result(next_state, result)
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="observe_step",
            status="success" if observation else "skipped",
            action="整理最近一次工具结果观察",
            inputs={"current_step_id": runtime.current_step_id},
            outputs={"observation_status": observation.get("status") if observation else None, "needs_replan": bool(runtime.needs_replan)},
            error=None,
        )
    ]
    return next_state


def route_after_observation_node(state: Any) -> AgentState:
    """只负责记录 observation 后的路由决策，不执行工具或改写计划。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    decision = route_after_observation(next_state)
    runtime = _ensure_runtime(next_state, allow_missing=True)
    next_state.debug = dict(next_state.debug or {})
    # 将条件边决策写入状态，方便流式事件和调试面板看到图层为什么继续、重规划或收束。
    next_state.debug["agent_route"] = {
        "phase": "route_after_observation",
        "decision": decision,
        "current_step_id": runtime.current_step_id if runtime else None,
        "needs_replan": bool(runtime.needs_replan) if runtime else False,
        "pending_confirmation": bool(runtime.pending_confirmation) if runtime else False,
        "turn_status": runtime.turn_status if runtime else None,
        "error": runtime.error if runtime else "missing_runtime",
    }
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="route_after_observation",
            status="success" if decision != "error" else "failed",
            action="根据 observation 决定下一条图路径",
            inputs={
                "current_step_id": runtime.current_step_id if runtime else None,
                "last_observation": runtime.last_observation if runtime else None,
            },
            outputs={"decision": decision},
            error=runtime.error if runtime and decision == "error" else None,
        )
    ]
    return next_state


def replan_node(state: Any) -> AgentState:
    """只负责根据 observation 执行重规划或兜底。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    runtime = _ensure_runtime(next_state)
    turn_result = _executor().replan_after_observation(runtime, next_state)
    next_state.plan_runtime = runtime
    next_state.execution_plan = runtime.plan
    next_state.runtime_state = _runtime_state_from_runtime(next_state, runtime)
    if turn_result is not None:
        _apply_turn_result(next_state, turn_result)
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="replan",
            status="success" if not runtime.error else "failed",
            action="根据 observation 更新计划或生成兜底结果",
            inputs={"current_step_id": runtime.current_step_id, "last_observation": runtime.last_observation},
            outputs={"needs_replan": bool(runtime.needs_replan), "turn_status": runtime.turn_status, "error": runtime.error},
            error=runtime.error,
        )
    ]
    return next_state


def finalize_node(state: Any) -> AgentState:
    """只负责把 runtime 汇总成对外响应所需的 AgentState 字段。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    runtime = _ensure_runtime(next_state)
    turn_result = _executor().finalize_runtime(runtime)
    _apply_turn_result(next_state, turn_result)
    next_state.runtime_state = _runtime_state_from_runtime(next_state, runtime)
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="finalize",
            status="success" if turn_result.status in {"success", "waiting_confirmation", "need_clarification", "fallback"} else "failed",
            action="汇总 runtime 并生成最终响应状态",
            inputs={"turn_status": turn_result.status},
            outputs={"output_keys": sorted(turn_result.outputs.keys()), "pending_confirmation": bool(turn_result.pending_confirmation)},
            error=turn_result.error,
        )
    ]
    return next_state


def error_finalize_node(state: Any) -> AgentState:
    """只负责异常终态归一，避免执行错误继续落到普通 finalize 路径。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    runtime = _ensure_runtime(next_state, allow_missing=True)
    if runtime is not None:
        runtime.turn_status = "failed"
        runtime.error = runtime.error or "agent_graph_error"
        next_state.plan_runtime = runtime
        next_state.runtime_state = _runtime_state_from_runtime(next_state, runtime)
    next_state.intent = next_state.intent or "unsupported"
    next_state.answer = next_state.answer or "Agent 执行过程中发生错误，请稍后重试。"
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="error_finalize",
            status="failed",
            action="归一化 Agent 图执行错误",
            inputs={},
            outputs={"error": runtime.error if runtime else "missing_runtime"},
            error=runtime.error if runtime else "missing_runtime",
        )
    ]
    return next_state


def route_after_selection(state: Any) -> str:
    """根据选步结果决定执行 step 还是直接 finalize。"""
    current_state = _coerce_state(state)
    runtime = current_state.plan_runtime
    if runtime is None:
        return "error"
    if runtime.error:
        return "error"
    if not runtime.current_step_id:
        return "finalize"
    return "execute"


def route_after_execution(state: Any) -> str:
    """执行节点后决定进入观察、重新选步、确认收束或错误收束。"""
    current_state = _coerce_state(state)
    runtime = current_state.plan_runtime
    last_step_result = dict((current_state.debug or {}).get("last_step_result") or {})
    if runtime is None:
        return "error"
    if runtime.pending_confirmation:
        return "finalize"
    if runtime.error:
        return "error"
    if last_step_result.get("next_action") == "continue" and not runtime.last_step_output:
        # approve resume 后当前 step 会回到 pending，需要重新走选步再执行，不能进入 observe。
        return "select_next_step"
    return "observe"


def route_after_observation(state: Any) -> str:
    """根据结构化 runtime 决定继续、重规划、等待确认、失败或结束。"""
    current_state = _coerce_state(state)
    runtime = current_state.plan_runtime
    if runtime is None:
        return "error"
    if runtime.pending_confirmation:
        return "finalize"
    if runtime.error and not runtime.final_answer:
        return "error"
    if runtime.needs_replan:
        return "replan"
    if runtime.turn_status in {"failed", "fallback", "need_clarification", "waiting_confirmation"}:
        return "finalize"
    return "select_next_step"


def route_after_replan(state: Any) -> str:
    """重规划后根据 runtime 状态继续执行或收束。"""
    current_state = _coerce_state(state)
    runtime = current_state.plan_runtime
    if runtime is None:
        return "error"
    if runtime.pending_confirmation:
        return "finalize"
    if runtime.error and runtime.turn_status not in {"fallback", "success"}:
        return "error"
    if runtime.turn_status in {"fallback", "failed", "need_clarification", "waiting_confirmation"}:
        return "finalize"
    return "select_next_step"


def build_arxiv_search_graph(
    generation_service: Optional[Any] = None,
    *,
    checkpointer: Optional[Any] = None,
) -> Any:
    """构建显式 Agent 执行环。

    主图不再把 planner/executor/observer/replanner 全部藏进 run_agent_turn；
    每个节点只承担一个职责，循环由条件边表达，便于中断、恢复和可观测调试。
    """
    graph = StateGraph(AgentState)

    graph.add_node("parse_search_request", lambda state: parse_search_request(state, generation_service=generation_service))
    graph.add_node("build_goal", build_goal_node)
    graph.add_node("build_plan", build_plan_node)
    graph.add_node("select_next_step", select_next_step_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("observe_step", observe_step_node)
    graph.add_node("route_after_observation", route_after_observation_node)
    graph.add_node("replan", replan_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("error_finalize", error_finalize_node)

    graph.add_edge(START, "parse_search_request")
    graph.add_edge("parse_search_request", "build_goal")
    graph.add_edge("build_goal", "build_plan")
    graph.add_edge("build_plan", "select_next_step")
    graph.add_conditional_edges(
        "select_next_step",
        route_after_selection,
        {
            "execute": "execute_step",
            "finalize": "finalize",
            "error": "error_finalize",
        },
    )
    graph.add_conditional_edges(
        "execute_step",
        route_after_execution,
        {
            "observe": "observe_step",
            "select_next_step": "select_next_step",
            "finalize": "finalize",
            "error": "error_finalize",
        },
    )
    graph.add_edge("observe_step", "route_after_observation")
    graph.add_conditional_edges(
        "route_after_observation",
        route_after_observation,
        {
            "select_next_step": "select_next_step",
            "replan": "replan",
            "finalize": "finalize",
            "error": "error_finalize",
        },
    )
    graph.add_conditional_edges(
        "replan",
        route_after_replan,
        {
            "select_next_step": "select_next_step",
            "finalize": "finalize",
            "error": "error_finalize",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("error_finalize", END)

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


def _ensure_runtime(state: AgentState, *, allow_missing: bool = False) -> Optional[PlanRuntime] | PlanRuntime:
    """读取当前 PlanRuntime；缺失时只在错误归一节点允许返回 None。"""
    if state.plan_runtime is not None:
        return state.plan_runtime
    if allow_missing:
        return None
    raise ValueError("missing_plan_runtime")


def _apply_step_result(state: AgentState, result: StepExecutionResult) -> None:
    """把单步执行结果写回 AgentState，供流式事件和调试面板消费。"""
    state.debug = dict(state.debug or {})
    state.debug["last_step_result"] = result.model_dump(mode="json")
    display_status = _display_step_status_from_plan_status(result.step_status, next_action=result.next_action)
    state.steps = list(state.steps or []) + [
        AgentStep(
            step="execute_step",
            status=display_status,
            action="执行当前计划步骤",
            inputs={"step_id": result.step_id},
            outputs={
                "next_action": result.next_action,
                "plan_step_status": result.step_status,
                "output_key": result.output_key,
                "has_observation": bool(result.observation),
                "pending_confirmation": bool(result.pending_confirmation),
            },
            error=result.error,
        )
    ]
    if result.turn_result is not None and isinstance(result.turn_result, AgentTurnResult):
        _apply_turn_result(state, result.turn_result)


def _display_step_status_from_plan_status(status: Optional[str], *, next_action: Optional[str]) -> str:
    """把执行计划内部状态转换成 AgentStep 展示状态。

    PlanRuntime 需要保留 running / waiting_confirmation 这类中间态，供 observe/replan
    继续接管；AgentStep 只描述图节点本身是否完成，不能直接写入这些内部状态。
    """
    if status == "failed" or next_action == "fail":
        return "failed"
    if status == "skipped" or next_action == "skip":
        return "skipped"
    return "success"


# 产出可展示论文的工具，按优先级排列：个性化重排结果优先于原始检索结果。
# 出站 papers 投影必须和 synthesize_arxiv_response 读取的来源一致，否则会出现
# “回答说检索到 N 篇，但 papers 为空”的不一致。
_PAPER_PRODUCING_TOOLS = ("personalize_paper_results", "search_arxiv")


def _papers_from_value(value: Any) -> list[Dict[str, Any]]:
    """从任意工具输出形态中提取论文列表。

    兼容 personalize_paper_results 的 ranked_papers、search_arxiv 的 papers，
    以及后端 tool_result.data 包裹体，避免因输出包装层级不同而漏取。
    """
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    for papers_key in ("ranked_papers", "papers"):
        sequence = value.get(papers_key)
        if isinstance(sequence, list):
            papers = [dict(item) for item in sequence if isinstance(item, Mapping)]
            if papers:
                return papers
    data = value.get("data")
    if isinstance(data, Mapping):
        nested = _papers_from_value(data)
        if nested:
            return nested
    tool_result = value.get("tool_result")
    if isinstance(tool_result, Mapping):
        nested = _papers_from_value(tool_result.get("data") if isinstance(tool_result.get("data"), Mapping) else tool_result)
        if nested:
            return nested
    return []


def _papers_output_keys_by_tool(plan: Optional[ExecutablePlan]) -> list[str]:
    """按工具优先级解析出本轮真实承载论文的 output_key。

    LLM planner 只锁定 tool_name，output_key 可任意命名；这里像 input_bindings 一样
    通过 tool_name -> output_key 做确定性解析，使 papers 投影与回答口径一致，
    不再依赖 output_key 字面量是否恰好叫 arxiv_results/ranked_papers。
    """
    if plan is None:
        return []
    ordered_keys: list[str] = []
    for tool_name in _PAPER_PRODUCING_TOOLS:
        for step in list(plan.steps or []):
            if step.tool_name == tool_name and step.output_key and step.output_key not in ordered_keys:
                ordered_keys.append(step.output_key)
    return ordered_keys


def _extract_arxiv_papers_from_outputs(
    outputs: Mapping[str, Any],
    *,
    plan: Optional[ExecutablePlan] = None,
) -> list[Dict[str, Any]]:
    """从执行输出中提取可展示论文列表。

    解析优先级：
    1. 按 plan 中 papers 工具（personalize/search）的真实 output_key 取值，与回答口径对齐；
    2. 退回固定 output_key（ranked_papers/arxiv_results），兼容规则型 planner；
    3. 最后对全部 outputs 做一次兜底扫描，避免 LLM 任意命名导致“有回答无卡片”。
    """
    for output_key in _papers_output_keys_by_tool(plan):
        papers = _papers_from_value(outputs.get(output_key))
        if papers:
            return papers

    for fixed_key in ("ranked_papers", "arxiv_results"):
        papers = _papers_from_value(outputs.get(fixed_key))
        if papers:
            return papers

    for value in outputs.values():
        papers = _papers_from_value(value)
        if papers:
            return papers
    return []


def _warn_if_answer_claims_unprojected_papers(state: AgentState, result: AgentTurnResult) -> None:
    """当回答声称检索到论文、但 papers 投影为空时，记录关键诊断日志。

    这是“回答正常但 Paper 展示区为空”不一致的最后一道哨兵：正常情况下不应触发；
    若触发，日志会带上 output_keys 与 result_count，便于定位是哪种工具输出形态没被识别。
    """
    quality = result.outputs.get("arxiv_result_quality") if isinstance(result.outputs, Mapping) else None
    result_count = int(quality.get("result_count") or 0) if isinstance(quality, Mapping) else 0
    if result_count <= 0 and not state.papers:
        return
    if state.papers:
        return
    logger.warning(
        "arxiv_agent papers projection empty while answer implies results: "
        "intent=%s result_count=%s output_keys=%s plan_tools=%s",
        state.intent,
        result_count,
        sorted(dict(result.outputs or {}).keys()),
        [step.tool_name for step in list((result.plan.steps if result.plan else []) or [])],
    )


def _apply_turn_result(state: AgentState, result: AgentTurnResult) -> None:
    """把统一执行结果回写到 AgentState，供 service/stream 复用响应适配器。"""
    state.goal = result.plan.goal if result.plan is not None else state.goal
    state.execution_plan = result.plan
    state.plan_runtime = result.runtime
    state.answer = result.final_answer or state.answer
    previous_pending_action = dict(state.pending_action or {}) if isinstance(state.pending_action, Mapping) else None
    confirmation_payload = result.pending_confirmation.model_dump() if result.pending_confirmation is not None else None
    if confirmation_payload is not None:
        state.pending_action = _build_pending_action_mirror(result)
    elif previous_pending_action and previous_pending_action.get("confirmation_consumed") is True:
        # 确认已被消费时，最终响应仍要保留 approved/rejected 结果给前端和日志层，
        # 但它不会再被当成新的待确认卡片，因为 display 层只展示 waiting_confirmation。
        state.pending_action = previous_pending_action
    else:
        state.pending_action = None
    state.debug = dict(state.debug or {})
    state.debug["agent_turn"] = {
        "status": result.status,
        "error": result.error,
        "output_keys": sorted(result.outputs.keys()),
        "trace_events": [trace.event for trace in list(result.trace or [])],
    }
    if confirmation_payload is not None:
        state.debug["pending_confirmation"] = confirmation_payload
    else:
        state.debug.pop("pending_confirmation", None)
    if result.status == "waiting_confirmation":
        state.paper_qa_result = {"status": "waiting_confirmation", "pending_confirmation": confirmation_payload}
    elif previous_pending_action and previous_pending_action.get("confirmation_consumed") is True:
        # 确认已消费后，要同步撤掉业务快照里的 waiting_confirmation 残留，
        # 避免后续持久化上下文或前端状态机继续把旧确认当成待处理任务。
        existing_paper_qa_result = dict(state.paper_qa_result or {}) if isinstance(state.paper_qa_result, Mapping) else {}
        if str(existing_paper_qa_result.get("status") or "").strip() == "waiting_confirmation":
            state.paper_qa_result = {
                **existing_paper_qa_result,
                "status": "cancelled" if previous_pending_action.get("decision") == "reject" else "ready",
                "pending_confirmation": None,
                "confirmation_consumed": True,
                "confirmation_decision": previous_pending_action.get("decision"),
            }
    if "preference_action_result" in result.outputs:
        preference_result = result.outputs.get("preference_action_result")
        state.preference_action_result = dict(preference_result) if isinstance(preference_result, Mapping) else {"value": preference_result}
    if "paper_qa_result" in result.outputs:
        # answer_paper_question 的输出来自 PaperQAService 真实 RAG 链路，sources/retrieval_debug 必须原样带给前端。
        paper_qa_result = result.outputs.get("paper_qa_result")
        state.paper_qa_result = _build_paper_qa_result(state, paper_qa_result)
        if state.paper_qa_result.get("answer"):
            state.answer = str(state.paper_qa_result.get("answer") or "")
    arxiv_papers = _extract_arxiv_papers_from_outputs(result.outputs, plan=result.plan or state.execution_plan)
    if arxiv_papers:
        state.papers = arxiv_papers
    else:
        _warn_if_answer_claims_unprojected_papers(state, result)
    runtime = state.plan_runtime
    state.runtime_state = _runtime_state_from_runtime(state, runtime) if runtime is not None else state.runtime_state


def _extract_resolved_paper_from_runtime(state: AgentState) -> Dict[str, Any]:
    """从执行现场取出 resolve_paper 的结果，作为本轮 QA 真实目标论文。"""
    runtime = state.plan_runtime
    outputs = runtime.outputs if runtime is not None and isinstance(runtime.outputs, Mapping) else {}
    paper_ref = outputs.get("paper_ref") if isinstance(outputs, Mapping) else None
    if not isinstance(paper_ref, Mapping):
        return {}
    nested_paper = paper_ref.get("paper")
    if isinstance(nested_paper, Mapping):
        # Target Resolver 会同时返回顶层 arxiv_id/title 和完整 paper；
        # 嵌套 paper 字段通常更完整，但顶层字段代表最终解析结果，保留其优先级。
        return {
            **dict(nested_paper),
            **{key: value for key, value in dict(paper_ref).items() if key in {"arxiv_id", "title"} and value not in (None, "", [], {})},
        }
    return dict(paper_ref)


def _build_paper_qa_result(state: AgentState, payload: Any) -> Dict[str, Any]:
    """把真实 PaperQA 工具输出整理成前端沿用的 paper_qa_result。"""
    data = dict(payload) if isinstance(payload, Mapping) else {"value": payload}
    context = state.context if isinstance(state.context, Mapping) else {}
    selected_paper = context.get("selected_paper") if isinstance(context.get("selected_paper"), Mapping) else {}
    resolved_paper = _extract_resolved_paper_from_runtime(state)
    answer = str(data.get("answer") or "").strip()
    status = str(data.get("status") or ("success" if answer else "failed")).strip() or "failed"
    return {
        "status": status,
        # arxiv_id/title 优先取本轮 resolve_paper 的结果，避免“第二篇”被旧 selected_paper 覆盖。
        "arxiv_id": data.get("arxiv_id") or resolved_paper.get("arxiv_id") or context.get("arxiv_id") or selected_paper.get("arxiv_id"),
        "title": data.get("title") or resolved_paper.get("title") or selected_paper.get("title"),
        "question": data.get("question") or state.message,
        "answer": answer,
        "sources": data.get("sources", []),
        "retrieval_debug": data.get("retrieval_debug"),
        "qa_observation": data.get("qa_observation"),
        "error": data.get("error"),
        "tool_result": data.get("tool_result"),
    }


def _build_pending_action_mirror(result: AgentTurnResult) -> Optional[Dict[str, Any]]:
    """把结构化确认请求投影成旧前端仍读取的 pending_action 镜像。

    pending_action 只承担展示兼容职责，真实可恢复状态仍以
    pending_confirmation 和 LangGraph resume/checkpoint 为准。
    """
    confirmation = result.pending_confirmation
    if confirmation is None:
        return None
    payload = confirmation.model_dump()
    target_paper = dict(confirmation.target_paper or {})
    arguments_summary = dict(confirmation.arguments_summary or {})
    return {
        "type": confirmation.request_type,
        "request_type": confirmation.request_type,
        "status": "waiting_confirmation",
        "decision": None,
        "pending_action_id": confirmation.pending_action_id,
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
        "original_message": confirmation.original_message,
        "target_paper": target_paper or None,
        "candidates": list(confirmation.candidates or []),
        "recommended_candidate": dict(confirmation.recommended_candidate or {}) if confirmation.recommended_candidate else None,
        "default_candidate_id": confirmation.default_candidate_id,
        "reference_hint": dict(confirmation.reference_hint or {}),
        "target_resolution": dict(confirmation.target_resolution or {}),
        "confirmation_fields": dict(confirmation.confirmation_fields or {}),
        "created_at": confirmation.created_at,
        "expires_at": confirmation.expires_at,
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


def _runtime_state_from_runtime(state: AgentState, runtime: PlanRuntime) -> AgentRuntimeState:
    """从 PlanRuntime 投影出可序列化执行现场。"""
    return AgentRuntimeState(
        request_state=_json_safe(runtime.state or {}),
        goal=runtime.goal,
        plan=runtime.plan,
        current_step_id=runtime.current_step_id,
        current_step_index=runtime.current_step_index,
        step_status=dict(runtime.step_status or {}),
        outputs=_json_safe(runtime.outputs or {}),
        last_observation=_json_safe(runtime.last_observation) if runtime.last_observation else None,
        last_step_output=_json_safe(runtime.last_step_output) if runtime.last_step_output else None,
        trace=list(runtime.trace or []),
        retry_counts=dict(runtime.retry_counts or {}),
        replan_counts=dict(runtime.replan_counts or {}),
        step_replan_counts=dict(runtime.step_replan_counts or {}),
        approved_step_ids=list(runtime.approved_step_ids or []),
        pending_confirmation=runtime.pending_confirmation,
        needs_replan=bool(runtime.needs_replan),
        is_finished=bool(runtime.turn_status),
        failure_reason=runtime.error,
        recovery_strategy=_json_safe(runtime.recovery_strategy) if runtime.recovery_strategy else None,
        turn_status=runtime.turn_status,
        final_answer=runtime.final_answer or state.answer,
    )


def _json_safe(value: Any) -> Any:
    """把 runtime_state 投影限制在 JSON 友好数据内，避免 checkpoint 写入复杂实例。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in dict(value or {}).items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return repr(value)


def _build_fallback_mermaid() -> str:
    """生成与显式执行环一致的最小 Mermaid 兜底图。"""
    return "\n".join(
        [
            "graph TD;",
            "    START([START]) --> parse_search_request;",
            "    parse_search_request --> build_goal;",
            "    build_goal --> build_plan;",
            "    build_plan --> select_next_step;",
            "    select_next_step --> execute_step;",
            "    execute_step --> observe_step;",
            "    execute_step --> select_next_step;",
            "    execute_step --> finalize;",
            "    execute_step --> error_finalize;",
            "    observe_step --> select_next_step;",
            "    observe_step --> replan;",
            "    replan --> select_next_step;",
            "    observe_step --> finalize;",
            "    finalize --> END([END]);",
            "    error_finalize --> END;",
        ]
    )


__all__ = [
    "DEFAULT_GRAPH_CHECKPOINTER",
    "build_arxiv_search_graph",
    "export_arxiv_search_graph_mermaid",
    "route_after_observation",
    "route_after_replan",
    "route_after_execution",
    "route_after_selection",
    "START",
    "END",
]
