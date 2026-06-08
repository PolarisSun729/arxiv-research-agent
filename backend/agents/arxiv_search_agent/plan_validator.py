"""ExecutablePlan 合法性校验器。"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .schemas import ExecutablePlan, PlanStep
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


class PlanValidationError(ValueError):
    """用于聚合返回 plan 校验失败原因。"""


class PlanValidator:
    """校验 planner 产出的计划是否满足工具和拓扑约束。"""

    def __init__(self) -> None:
        self._registry = PLANNER_TOOL_REGISTRY

    def validate(self, plan: ExecutablePlan, tool_registry: Optional[ToolRegistry] = None) -> None:
        registry = tool_registry or self._registry
        steps = list(plan.steps or [])
        if not steps:
            raise PlanValidationError("Plan must contain at least one step")

        steps_by_id: Dict[str, PlanStep] = {}
        output_keys: Dict[str, str] = {}
        adjacency: Dict[str, List[str]] = {}
        indegree: Dict[str, int] = {}

        for step in steps:
            if step.step_id in steps_by_id:
                raise PlanValidationError(f"Duplicate step_id: {step.step_id}")
            steps_by_id[step.step_id] = step
            adjacency[step.step_id] = []
            indegree[step.step_id] = 0

            # planner 不能引用未注册工具，否则 executor 无法保证契约。
            try:
                registry.validate_tool_name(step.tool_name)
            except ValueError as exc:
                raise PlanValidationError(str(exc)) from exc
            tool_spec = registry.get(step.tool_name)
            if tool_spec is None:
                raise PlanValidationError(f"Unknown tool: {step.tool_name}")
            contract = registry.get_contract(step.tool_name)
            if contract is None:
                raise PlanValidationError(f"Unknown tool contract: {step.tool_name}")

            # step.tool 只是 planner/debug 的 contract 投影；校验阶段必须确认它仍来自当前 registry，
            # 避免恢复旧 checkpoint 或手写 plan 时把过期工具契约带进 executor。
            if getattr(step.tool, "contract_source", None) != contract.contract_source:
                raise PlanValidationError(
                    f"Step {step.step_id} tool contract source does not match registry contract source"
                )
            if getattr(step.tool, "adapter", None) != contract.to_tool_spec().adapter:
                raise PlanValidationError(
                    f"Step {step.step_id} tool adapter does not match registry contract adapter"
                )

            if step.side_effect_level != tool_spec.side_effect_level:
                raise PlanValidationError(
                    f"Step {step.step_id} side_effect_level={step.side_effect_level} "
                    f"does not match tool {step.tool_name} side_effect_level={tool_spec.side_effect_level}"
                )

            if tool_spec.requires_confirmation and step.confirmation_policy is None:
                raise PlanValidationError(f"Step {step.step_id} uses confirmation-required tool {step.tool_name} without confirmation_policy")

            if step.output_key:
                if step.output_key in output_keys:
                    raise PlanValidationError(
                        f"Duplicate output_key {step.output_key}: used by {output_keys[step.output_key]} and {step.step_id}"
                    )
                output_keys[step.output_key] = step.step_id

        for step in steps:
            for dependency_step_id in list(step.depends_on or []):
                if dependency_step_id not in steps_by_id:
                    raise PlanValidationError(f"Step {step.step_id} depends on missing step {dependency_step_id}")
                adjacency[dependency_step_id].append(step.step_id)
                indegree[step.step_id] += 1

        if self._has_cycle(adjacency):
            raise PlanValidationError("Plan contains circular dependencies")

        for final_step_id in list(plan.final_step_ids or []):
            if final_step_id not in steps_by_id:
                raise PlanValidationError(f"Unknown final_step_id: {final_step_id}")

        for entry_step_id in list(plan.entry_step_ids or []):
            if entry_step_id not in steps_by_id:
                raise PlanValidationError(f"Unknown entry_step_id: {entry_step_id}")
            if indegree.get(entry_step_id, 0) != 0:
                raise PlanValidationError(f"Entry step {entry_step_id} still has dependencies")

        if not list(plan.entry_step_ids or []):
            raise PlanValidationError("Plan must have at least one entry step")

        goal_type = str(getattr(plan.goal, "goal_type", "") or "").strip()
        if goal_type == "unclear":
            forbidden_tools = {
                "search_arxiv",
                "generate_recommendations",
                "answer_paper_question",
                "update_preference_store",
                "parse_and_index_paper",
            }
            used_forbidden_tools = [step.tool_name for step in steps if step.tool_name in forbidden_tools]
            if used_forbidden_tools:
                raise PlanValidationError(f"Clarification plan contains forbidden tools: {used_forbidden_tools}")

        if goal_type == "unsupported":
            unsupported_tools = [step.tool_name for step in steps if step.tool_name != "generate_fallback_response"]
            if unsupported_tools:
                raise PlanValidationError(f"Unsupported plan must only use generate_fallback_response, got: {unsupported_tools}")

    def _has_cycle(self, adjacency: Dict[str, List[str]]) -> bool:
        visited: Set[str] = set()
        stack: Set[str] = set()

        def dfs(node: str) -> bool:
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for child in adjacency.get(node, []):
                if dfs(child):
                    return True
            stack.remove(node)
            return False

        return any(dfs(node) for node in adjacency.keys() if node not in visited)


__all__ = ["PlanValidationError", "PlanValidator"]
