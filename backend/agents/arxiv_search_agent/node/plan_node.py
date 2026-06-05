"""Planner 节点入口。

这里不再承载巨型 if-else 规划逻辑，只负责：
1. 调用新的 planner 入口生成 goal 和 ExecutablePlan；
2. 构造 PlanRuntime；
3. 回写调试信息与节点级步骤轨迹。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Union

from ..planner import build_executable_plan, build_plan_runtime
from ..state import AgentState
from ..utils.state_utils import _append_step, _coerce_state, _compact_search_spec, _refresh_execution_plan_runtime, _update_execution_plan_step


def _build_goal_and_plan(state: AgentState):
    """临时 facade，后续可在调用方全部迁移后删除。"""
    goal, execution_plan, planning_debug = build_executable_plan(state)
    turn_status = "waiting_confirmation" if any(step.status == "waiting_confirmation" for step in list(execution_plan.steps or [])) else "success"
    plan_runtime = build_plan_runtime(state, goal=goal, plan=execution_plan, turn_status=turn_status)
    return goal, execution_plan, plan_runtime, planning_debug


def plan_task(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """基于新 planner 架构生成 goal、ExecutablePlan 和 PlanRuntime。"""
    current_state = _coerce_state(state)

    try:
        goal, execution_plan, plan_runtime, planning_debug = _build_goal_and_plan(current_state)
        next_state = current_state.model_copy(deep=True)
        next_state.goal = goal
        next_state.execution_plan = execution_plan
        next_state.plan_runtime = plan_runtime
        next_state = _refresh_execution_plan_runtime(next_state)

        # 请求规范化是 planner 自己可完成的输入整理动作，构造完计划后即可视为完成。
        if goal.goal_type == "arxiv_search":
            next_state = _update_execution_plan_step(next_state, step_id="normalize_request", status="success")

        next_state.debug = dict(next_state.debug or {})
        next_state.debug["plan_task"] = {
            **planning_debug,
            "plan_runtime": next_state.plan_runtime.model_dump() if next_state.plan_runtime else None,
        }
        return _append_step(
            next_state,
            step="plan_task",
            status="success",
            action="generate_goal_and_executable_plan",
            inputs={
                "intent": current_state.intent,
                "message": current_state.message,
                "search_spec": _compact_search_spec(current_state.search_spec),
                "context_keys": sorted((current_state.context or {}).keys()) if isinstance(current_state.context, dict) else [],
            },
            outputs={
                "goal_type": goal.goal_type,
                "goal_id": goal.goal_id,
                "execution_plan_steps": len(execution_plan.steps or []),
                "entry_step_ids": list(execution_plan.entry_step_ids or []),
                "final_step_ids": list(execution_plan.final_step_ids or []),
            },
        )
    except Exception as exc:
        next_state = current_state.model_copy(deep=True)
        fallback_goal, fallback_plan, planning_debug = build_executable_plan(
            current_state.model_copy(update={"intent": "unsupported"})
        )
        next_state.goal = fallback_goal
        next_state.execution_plan = fallback_plan
        next_state.plan_runtime = build_plan_runtime(next_state, goal=fallback_goal, plan=fallback_plan, turn_status="fallback")
        next_state = _refresh_execution_plan_runtime(next_state)
        next_state.debug = dict(next_state.debug or {})
        next_state.debug["plan_task"] = {
            **planning_debug,
            "error": str(exc),
            "source": "plan_task_fallback",
        }
        return _append_step(
            next_state,
            step="plan_task",
            status="failed",
            action="generate_goal_and_executable_plan",
            inputs={"intent": current_state.intent, "message": current_state.message},
            outputs={"goal_type": fallback_goal.goal_type, "execution_plan_steps": len(fallback_plan.steps or [])},
            error=str(exc),
        )
