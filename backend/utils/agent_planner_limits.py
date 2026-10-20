"""Agent 规划预算与恢复上限：配置项、环境变量和默认值的唯一一处定义（设计见 notes/agent-redesign.md D4-3 / D4-4 / D4-5）。

utils/config.py 按这张表读环境变量；planner、replanner、LLM 草稿生成器和测试 stub 在拿不到 config 时
也回退到这里的默认值，不再各自写一份。模块只依赖标准库，config 和 agent 包都能安全导入。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


# 配置项 -> (环境变量, 默认值)
AGENT_PLANNER_LIMITS: Dict[str, Tuple[str, int]] = {
    # 规划预算（D4-3 I-6）：总步数不单独配，按「每目标步数 × 目标数」算（planner.plan_step_budget）。
    "llm_plan_max_steps_per_goal": ("LLM_PLAN_MAX_STEPS_PER_GOAL", 6),
    "llm_plan_max_search_calls": ("LLM_PLAN_MAX_SEARCH_CALLS", 3),
    "llm_plan_max_index_per_paper": ("LLM_PLAN_MAX_INDEX_PER_PAPER", 1),
    # 只定义、暂未使用：D4-4 修复回路接入时读取（计划校验不过时最多让 LLM 修几次）。
    "llm_plan_max_repair_rounds": ("LLM_PLAN_MAX_REPAIR_ROUNDS", 2),
    # 只定义、暂未使用：D4-5 重规划接入时读取（每轮最多交给规划 LLM 重规划几次）。
    "llm_replan_max_rounds": ("LLM_REPLAN_MAX_ROUNDS", 2),
    # 恢复次数上限（D4-3 I-10），Replanner 读取：同一工具同一失败类别 / 同一步 / 每轮总次数。
    "recovery_max_per_reason": ("AGENT_RECOVERY_MAX_PER_REASON", 2),
    "recovery_max_per_step": ("AGENT_RECOVERY_MAX_PER_STEP", 2),
    "recovery_max_per_turn": ("AGENT_RECOVERY_MAX_PER_TURN", 5),
}

AGENT_PLANNER_LIMIT_DEFAULTS: Dict[str, int] = {key: default for key, (_, default) in AGENT_PLANNER_LIMITS.items()}


def planner_limit(config: Mapping[str, Any], key: str) -> int:
    """从 config 取一项上限；缺失、非法或小于 1 时回退默认值。"""
    default = AGENT_PLANNER_LIMIT_DEFAULTS[key]
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


__all__ = ["AGENT_PLANNER_LIMITS", "AGENT_PLANNER_LIMIT_DEFAULTS", "planner_limit"]
