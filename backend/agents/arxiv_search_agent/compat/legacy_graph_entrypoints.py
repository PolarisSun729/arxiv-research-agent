from __future__ import annotations

from typing import Any

from ..state import AgentState


def route_after_parse(state: Any) -> str:
    """旧图路由兼容入口：parse 后固定进入 build_goal，不属于当前主图执行链。

    保留原因是方便排查外部旧导入；新代码应直接使用
    build_arxiv_search_graph() 生成的显式节点链。该入口计划在旧调用清零后删除。
    """
    del state
    return "build_goal"


def run_agent_turn_node(state: Any) -> AgentState:
    """旧单节点执行兼容入口：把整轮执行折回 run_agent_turn_in_graph。

    当前主图已经拆成 build_goal/build_plan/select/execute/observe/replan/finalize
    等显式节点；这里仅服务迁移期的旧导入，不参与正式 Agent 执行路径。
    """
    from ..graph import _apply_turn_result, _coerce_state
    from ..plan_executor import run_agent_turn_in_graph

    current_state = _coerce_state(state)
    result = run_agent_turn_in_graph(current_state)
    next_state = current_state.model_copy(deep=True)
    _apply_turn_result(next_state, result)
    return next_state


__all__ = ["route_after_parse", "run_agent_turn_node"]
