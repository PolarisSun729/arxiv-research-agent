"""ExecutablePlan 合法性校验器。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .schemas import ExecutablePlan, PlanStep, StepInputBinding
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
            if step.side_effect_level == "persistent_write" and not self._has_confirmation_policy(step):
                raise PlanValidationError(f"Step {step.step_id} uses persistent_write tool {step.tool_name} without confirmation_policy")
            if step.side_effect_level == "external_call" and not self._has_risk_strategy(step, tool_spec):
                raise PlanValidationError(f"Step {step.step_id} uses external_call tool {step.tool_name} without retry, failure, recovery, or confirmation policy")
            self._validate_required_inputs(step, tool_spec.input_schema)

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
        unreachable_steps = self._unreachable_steps(adjacency, set(plan.entry_step_ids or []), set(steps_by_id.keys()))
        if unreachable_steps:
            raise PlanValidationError(f"Plan contains unreachable steps: {sorted(unreachable_steps)}")

        goal_type = str(getattr(plan.goal, "goal_type", "") or "").strip()
        if not self._has_user_facing_terminal_step(steps):
            raise PlanValidationError("Plan must end with answer, fallback, or clarification step")
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
        if goal_type == "paper_qa" and not self._is_clarification_only_plan(steps):
            # 目标论文缺失时 planner 可以安全降级为澄清计划；只有真正进入 Paper QA 业务链路时才强制 resolve/check/answer。
            self._validate_paper_qa_plan(steps_by_id, steps)
        if goal_type == "preference_action":
            self._validate_preference_plan(steps_by_id, steps)

    def _validate_required_inputs(self, step: PlanStep, input_schema: Dict[str, Any]) -> None:
        if not input_schema:
            return
        bindings = list(step.input_bindings or [])
        if not bindings:
            raise PlanValidationError(f"Step {step.step_id} missing required input bindings")
        for binding in bindings:
            self._validate_binding_source(step, binding)

    def _validate_binding_source(self, step: PlanStep, binding: StepInputBinding) -> None:
        if not str(binding.input_key or "").strip():
            raise PlanValidationError(f"Step {step.step_id} has input binding with empty input_key")
        if binding.source_type == "step_output" and not str(binding.step_id or "").strip():
            raise PlanValidationError(f"Step {step.step_id} input {binding.input_key} references step_output without step_id")
        if binding.source_type in {"state", "context", "goal"} and binding.required and not str(binding.source_key or "").strip():
            raise PlanValidationError(f"Step {step.step_id} input {binding.input_key} missing source_key for {binding.source_type}")
        if binding.source_type == "literal" and binding.required and binding.value is None:
            raise PlanValidationError(f"Step {step.step_id} input {binding.input_key} uses empty required literal")

    def _has_confirmation_policy(self, step: PlanStep) -> bool:
        return bool(step.confirmation_policy and step.confirmation_policy.requires_confirmation)

    def _has_risk_strategy(self, step: PlanStep, tool_spec: Any) -> bool:
        return bool(
            self._has_confirmation_policy(step)
            or step.retry_policy is not None
            or step.failure_policy is not None
            or bool(getattr(tool_spec, "recovery_policy", None) or {})
            or bool(getattr(tool_spec, "confirmation_policy", None) or {})
        )

    def _unreachable_steps(self, adjacency: Dict[str, List[str]], entry_step_ids: Set[str], all_step_ids: Set[str]) -> Set[str]:
        visited: Set[str] = set()
        stack = list(entry_step_ids)
        while stack:
            step_id = stack.pop()
            if step_id in visited:
                continue
            visited.add(step_id)
            stack.extend(adjacency.get(step_id, []))
        return all_step_ids - visited

    def _has_user_facing_terminal_step(self, steps: List[PlanStep]) -> bool:
        terminal_tools = {
            "synthesize_arxiv_response",
            "answer_paper_question",
            "explain_recommendations",
            "synthesize_preference_response",
            "generate_clarification",
            "generate_fallback_response",
        }
        for step in steps:
            tags = set(step.tool.capability_tags or [])
            if step.tool_name in terminal_tools or tags.intersection({"answer", "fallback", "clarify"}):
                return True
        return False

    def _is_clarification_only_plan(self, steps: List[PlanStep]) -> bool:
        tool_names = {step.tool_name for step in steps}
        return bool(tool_names) and tool_names.issubset({"analyze_ambiguity", "generate_clarification"}) and "generate_clarification" in tool_names

    def _validate_paper_qa_plan(self, steps_by_id: Dict[str, PlanStep], steps: List[PlanStep]) -> None:
        tool_by_name = {step.tool_name: step for step in steps}
        for tool_name in ("resolve_paper", "check_paper_index", "answer_paper_question"):
            if tool_name not in tool_by_name:
                raise PlanValidationError(f"paper_qa plan missing required tool {tool_name}")
        answer_step = tool_by_name["answer_paper_question"]
        if not self._step_depends_on_tool(answer_step, steps_by_id, "resolve_paper") or not self._step_depends_on_tool(answer_step, steps_by_id, "check_paper_index"):
            raise PlanValidationError("paper_qa answer_paper_question must depend on resolve_paper and check_paper_index")
        self._assert_tool_precedes("resolve_paper", "check_paper_index", steps_by_id, steps)
        self._assert_tool_precedes("check_paper_index", "answer_paper_question", steps_by_id, steps)
        if not self._step_has_binding_from_step(answer_step, steps_by_id, "paper_ref", "resolve_paper"):
            raise PlanValidationError("paper_qa answer_paper_question must bind paper_ref from resolve_paper")

    def _validate_preference_plan(self, steps_by_id: Dict[str, PlanStep], steps: List[PlanStep]) -> None:
        tool_by_name = {step.tool_name: step for step in steps}
        if "update_preference_store" not in tool_by_name:
            return
        for tool_name in ("resolve_preference_target", "verify_preference_update", "synthesize_preference_response"):
            if tool_name not in tool_by_name:
                raise PlanValidationError(f"preference_action plan missing required tool {tool_name}")
        update_step = tool_by_name["update_preference_store"]
        if not self._step_depends_on_tool(update_step, steps_by_id, "resolve_preference_target"):
            raise PlanValidationError("preference_action update_preference_store must depend on resolve_preference_target")
        if not self._step_has_binding_from_step(update_step, steps_by_id, "paper_reference", "resolve_preference_target") and not self._step_has_literal_target(update_step, "paper_reference"):
            raise PlanValidationError("preference_action update_preference_store requires explicit paper_reference target")
        self._assert_tool_precedes("resolve_preference_target", "update_preference_store", steps_by_id, steps)
        self._assert_tool_precedes("update_preference_store", "verify_preference_update", steps_by_id, steps)
        self._assert_tool_precedes("verify_preference_update", "synthesize_preference_response", steps_by_id, steps)

    def _assert_tool_precedes(self, previous_tool: str, next_tool: str, steps_by_id: Dict[str, PlanStep], steps: List[PlanStep]) -> None:
        position = {step.step_id: index for index, step in enumerate(steps)}
        previous_steps = [step for step in steps if step.tool_name == previous_tool]
        next_steps = [step for step in steps if step.tool_name == next_tool]
        if not previous_steps or not next_steps:
            return
        if position[previous_steps[0].step_id] > position[next_steps[0].step_id]:
            raise PlanValidationError(f"Tool order invalid: {previous_tool} must precede {next_tool}")

    def _step_depends_on_tool(self, step: PlanStep, steps_by_id: Dict[str, PlanStep], source_tool_name: str) -> bool:
        for dependency in list(step.depends_on or []):
            dependency_step = steps_by_id.get(dependency)
            if dependency_step and dependency_step.tool_name == source_tool_name:
                return True
        return False

    def _step_has_binding_from_step(self, step: PlanStep, steps_by_id: Dict[str, PlanStep], input_key: str, source_tool_name: str) -> bool:
        for binding in list(step.input_bindings or []):
            if binding.input_key != input_key or binding.source_type != "step_output":
                continue
            if not binding.step_id:
                continue
            dependency_step = steps_by_id.get(binding.step_id)
            if dependency_step and dependency_step.tool_name == source_tool_name:
                return True
        return False

    def _step_has_literal_target(self, step: PlanStep, input_key: str) -> bool:
        for binding in list(step.input_bindings or []):
            if binding.input_key != input_key or binding.source_type != "literal":
                continue
            value = binding.value
            if isinstance(value, dict) and any(str(value.get(key) or "").strip() for key in ("arxiv_id", "paper_id", "title")):
                return True
        return False

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
