from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .plan_validator import PlanValidator
from .schemas import ExecutablePlan, Goal, ObservationResult, PlanRuntime, PlanStep, StepInputBinding, StepPolicy
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


@dataclass(frozen=True)
class ReplanRule:
    rule_name: str
    tool_name: str
    observation_status: str


@dataclass
class ReplanDecision:
    updated_plan: Optional[ExecutablePlan] = None
    updated_runtime: Optional[PlanRuntime] = None
    fallback: bool = False
    fallback_reason: Optional[str] = None


def _binding(
    input_key: str,
    *,
    source_type: str,
    source_key: Optional[str] = None,
    step_id: Optional[str] = None,
    value: Any = None,
    required: bool = True,
) -> StepInputBinding:
    return StepInputBinding(
        input_key=input_key,
        source_type=source_type,  # type: ignore[arg-type]
        source_key=source_key,
        step_id=step_id,
        value=value,
        required=required,
    )


class Replanner:
    """第一版只做规则型重规划，避免执行器在低质量场景下直接退化为失败或静默继续。"""

    MAX_REASON_REPLANS = 2
    MAX_STEP_REPLANS = 2
    MAX_PLAN_REPLANS = 5

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry
        self.validator = PlanValidator()

    def replan(
        self,
        *,
        goal: Goal,
        current_plan: ExecutablePlan,
        runtime: PlanRuntime,
        failed_or_low_quality_step: PlanStep,
        observation_result: ObservationResult,
    ) -> ReplanDecision:
        reason_key = f"{failed_or_low_quality_step.tool_name}:{observation_result.status}"
        current_reason_count = int(runtime.replan_counts.get(reason_key, 0) or 0)
        current_step_count = int(runtime.step_replan_counts.get(failed_or_low_quality_step.step_id, 0) or 0)
        total_replans = sum(int(value or 0) for value in runtime.replan_counts.values())
        # 重规划是质量补救，不是无限重试机制；按 reason、step、plan 三层同时设上限。
        if current_reason_count >= self.MAX_REASON_REPLANS or current_step_count >= self.MAX_STEP_REPLANS or total_replans >= self.MAX_PLAN_REPLANS:
            return ReplanDecision(fallback=True, fallback_reason=f"replan_limit_exceeded:{reason_key}")

        plan_copy = current_plan.model_copy(deep=True)
        runtime_copy = runtime.model_copy(deep=True)
        plan_copy.steps = list(plan_copy.steps or [])

        decision = self._apply_rule(
            goal=goal,
            plan=plan_copy,
            runtime=runtime_copy,
            step=failed_or_low_quality_step,
            observation=observation_result,
        )
        if decision is None:
            return ReplanDecision(fallback=True, fallback_reason=f"no_replan_rule:{reason_key}")

        runtime_copy.replan_counts[reason_key] = current_reason_count + 1
        runtime_copy.step_replan_counts[failed_or_low_quality_step.step_id] = current_step_count + 1
        runtime_copy.trace.append(
            self._trace_for_replan(
                step_id=failed_or_low_quality_step.step_id,
                observation=observation_result,
                rule_name=decision["rule_name"],
                reason_key=reason_key,
            )
        )

        if decision.get("fallback"):
            return ReplanDecision(fallback=True, fallback_reason=str(decision.get("fallback_reason") or reason_key), updated_runtime=runtime_copy)

        self._refresh_plan_topology(plan_copy)
        self.validator.validate(plan_copy, self.tool_registry)
        runtime_copy.plan = plan_copy
        runtime_copy.step_status = {
            step.step_id: runtime_copy.step_status.get(step.step_id, step.status)
            for step in list(plan_copy.steps or [])
        }
        return ReplanDecision(updated_plan=plan_copy, updated_runtime=runtime_copy)

    def _apply_rule(
        self,
        *,
        goal: Goal,
        plan: ExecutablePlan,
        runtime: PlanRuntime,
        step: PlanStep,
        observation: ObservationResult,
    ) -> Optional[Dict[str, Any]]:
        if step.tool_name == "search_arxiv" and observation.status == "empty_result":
            self._rewrite_search_chain(plan, runtime, trigger_step_id=step.step_id)
            return {"rule_name": "rule_arxiv_empty_result"}
        if step.tool_name == "validate_arxiv_results" and observation.status == "low_confidence":
            self._rewrite_search_chain(plan, runtime, trigger_step_id=step.step_id)
            return {"rule_name": "rule_arxiv_low_confidence"}
        if step.tool_name == "check_paper_index" and observation.status == "need_confirmation":
            self._inject_index_confirmation_chain(plan, runtime, trigger_step_id=step.step_id)
            return {"rule_name": "rule_missing_paper_index"}
        if step.tool_name == "load_user_profile" and observation.status == "empty_result":
            self._downgrade_recommendation_chain(plan, runtime, trigger_step_id=step.step_id)
            return {"rule_name": "rule_empty_user_profile"}
        return None

    def _build_step(
        self,
        *,
        step_id: str,
        action_type: str,
        tool_name: str,
        output_key: Optional[str],
        input_bindings: Optional[List[StepInputBinding]] = None,
        depends_on: Optional[List[str]] = None,
        retry_policy: Optional[StepPolicy] = None,
        confirmation_policy: Optional[StepPolicy] = None,
        side_effect_level: Optional[str] = None,
    ) -> PlanStep:
        normalized_tool_name = self.tool_registry.validate_tool_name(tool_name)
        tool = self.tool_registry.get(normalized_tool_name)
        if tool is None:
            raise ValueError(f"Unknown planner tool: {tool_name}")
        if confirmation_policy is None and tool.requires_confirmation:
            confirmation_policy = StepPolicy(
                policy_type="confirmation",
                mode="explicit_user_confirmation_required",
                requires_confirmation=True,
            )
        return PlanStep(
            step_id=step_id,
            action_type=action_type,
            tool_name=normalized_tool_name,
            tool=tool,
            input_bindings=list(input_bindings or []),
            output_key=output_key,
            depends_on=list(depends_on or []),
            retry_policy=retry_policy,
            confirmation_policy=confirmation_policy,
            side_effect_level=side_effect_level or tool.side_effect_level,  # type: ignore[arg-type]
            status="pending",
        )

    def _make_unique_step_id(self, plan: ExecutablePlan, base_step_id: str) -> str:
        existing_ids = {step.step_id for step in list(plan.steps or [])}
        candidate = base_step_id
        suffix = 1
        while candidate in existing_ids:
            candidate = f"{base_step_id}__replan_{suffix}"
            suffix += 1
        return candidate

    def _make_unique_output_key(self, plan: ExecutablePlan, base_output_key: str) -> str:
        existing_output_keys = {step.output_key for step in list(plan.steps or []) if step.output_key}
        candidate = base_output_key
        suffix = 1
        while candidate in existing_output_keys:
            candidate = f"{base_output_key}__replan_{suffix}"
            suffix += 1
        return candidate

    def _insert_steps_after(self, plan: ExecutablePlan, anchor_step_id: str, new_steps: Sequence[PlanStep]) -> None:
        steps = list(plan.steps or [])
        anchor_index = next((index for index, item in enumerate(steps) if item.step_id == anchor_step_id), len(steps) - 1)
        plan.steps = steps[: anchor_index + 1] + list(new_steps) + steps[anchor_index + 1 :]

    def _repoint_pending_bindings(
        self,
        plan: ExecutablePlan,
        runtime: PlanRuntime,
        *,
        old_step_ids: Sequence[str],
        new_step_id: str,
        input_keys: Optional[Sequence[str]] = None,
        exclude_step_ids: Optional[Sequence[str]] = None,
    ) -> None:
        old_step_id_set = {str(step_id) for step_id in old_step_ids}
        input_key_set = {str(input_key) for input_key in list(input_keys or [])}
        exclude_step_id_set = {str(step_id) for step_id in list(exclude_step_ids or [])}
        for plan_step in list(plan.steps or []):
            if plan_step.step_id in exclude_step_id_set:
                continue
            if runtime.step_status.get(plan_step.step_id) not in {None, "pending"}:
                continue
            updated_bindings: List[StepInputBinding] = []
            for binding in list(plan_step.input_bindings or []):
                if binding.source_type == "step_output" and str(binding.step_id or "") in old_step_id_set:
                    if input_key_set and binding.input_key not in input_key_set:
                        updated_bindings.append(binding)
                        continue
                    updated_bindings.append(binding.model_copy(update={"step_id": new_step_id}))
                else:
                    updated_bindings.append(binding)
            updated_depends_on = [new_step_id if dependency in old_step_id_set else dependency for dependency in list(plan_step.depends_on or [])]
            deduped_depends_on = list(dict.fromkeys(updated_depends_on))
            plan_step.input_bindings = updated_bindings
            plan_step.depends_on = deduped_depends_on

    def _rewrite_search_chain(self, plan: ExecutablePlan, runtime: PlanRuntime, *, trigger_step_id: str) -> None:
        rewrite_step_id = self._make_unique_step_id(plan, "rewrite_arxiv_query")
        search_step_id = self._make_unique_step_id(plan, "search_arxiv")
        validate_step_id = self._make_unique_step_id(plan, "validate_arxiv_results")
        rewrite_output_key = self._make_unique_output_key(plan, "rewritten_search_spec")
        search_output_key = self._make_unique_output_key(plan, "arxiv_results_retry")
        validate_output_key = self._make_unique_output_key(plan, "arxiv_result_quality_retry")

        rewrite_step = self._build_step(
            step_id=rewrite_step_id,
            action_type="rewrite",
            tool_name="rewrite_arxiv_query",
            output_key=rewrite_output_key,
            depends_on=[trigger_step_id],
            input_bindings=[_binding("search_spec", source_type="step_output", step_id="build_arxiv_search_spec", required=False)],
        )
        search_step = self._build_step(
            step_id=search_step_id,
            action_type="search",
            tool_name="search_arxiv",
            output_key=search_output_key,
            depends_on=[rewrite_step_id],
            retry_policy=StepPolicy(policy_type="retry", mode="allow_search_relaxation", max_attempts=2),
            input_bindings=[_binding("search_spec", source_type="step_output", step_id=rewrite_step_id)],
        )
        validate_step = self._build_step(
            step_id=validate_step_id,
            action_type="validate",
            tool_name="validate_arxiv_results",
            output_key=validate_output_key,
            depends_on=[search_step_id],
            input_bindings=[_binding("arxiv_results", source_type="step_output", step_id=search_step_id)],
        )
        self._insert_steps_after(plan, trigger_step_id, [rewrite_step, search_step, validate_step])
        inserted_step_ids = [rewrite_step_id, search_step_id, validate_step_id]
        self._repoint_pending_bindings(plan, runtime, old_step_ids=["search_arxiv"], new_step_id=search_step_id, input_keys=["arxiv_results"], exclude_step_ids=inserted_step_ids)
        self._repoint_pending_bindings(plan, runtime, old_step_ids=["validate_arxiv_results"], new_step_id=validate_step_id, input_keys=["arxiv_result_quality"], exclude_step_ids=inserted_step_ids)

    def _inject_index_confirmation_chain(self, plan: ExecutablePlan, runtime: PlanRuntime, *, trigger_step_id: str) -> None:
        request_step_id = self._make_unique_step_id(plan, "request_confirmation")
        parse_step_id = self._make_unique_step_id(plan, "parse_and_index_paper")
        request_output_key = self._make_unique_output_key(plan, "confirmation_status")
        parse_output_key = self._make_unique_output_key(plan, "index_build_result")

        request_step = self._build_step(
            step_id=request_step_id,
            action_type="clarify",
            tool_name="request_confirmation",
            output_key=request_output_key,
            depends_on=[trigger_step_id],
            input_bindings=[
                _binding(
                    "pending_action",
                    source_type="literal",
                    value={
                        "type": "tool_approval",
                        "status": "waiting_confirmation",
                        "step_id": parse_step_id,
                        "tool_name": "parse_and_index_paper",
                        "action_label": "解析并索引论文",
                        "description": "目标论文还没有 QA 索引，需要先确认是否解析 PDF 并创建全文检索索引。",
                    },
                ),
            ],
        )
        parse_step = self._build_step(
            step_id=parse_step_id,
            action_type="index",
            tool_name="parse_and_index_paper",
            output_key=parse_output_key,
            depends_on=[request_step_id],
            confirmation_policy=StepPolicy(policy_type="confirmation", mode="explicit_user_confirmation_required", requires_confirmation=True),
            input_bindings=[_binding("paper_reference", source_type="step_output", step_id="resolve_paper")],
        )
        self._insert_steps_after(plan, trigger_step_id, [request_step, parse_step])

        for plan_step in list(plan.steps or []):
            if plan_step.step_id in {request_step_id, parse_step_id}:
                continue
            if runtime.step_status.get(plan_step.step_id) not in {None, "pending"}:
                continue
            if "check_paper_index" in list(plan_step.depends_on or []):
                updated_depends_on = []
                for dependency in list(plan_step.depends_on or []):
                    if dependency == "check_paper_index":
                        updated_depends_on.extend([dependency, parse_step_id])
                    else:
                        updated_depends_on.append(dependency)
                plan_step.depends_on = list(dict.fromkeys(updated_depends_on))

    def _downgrade_recommendation_chain(self, plan: ExecutablePlan, runtime: PlanRuntime, *, trigger_step_id: str) -> None:
        candidate_step = next((step for step in list(plan.steps or []) if step.tool_name == "load_candidate_papers"), None)
        if candidate_step is not None:
            candidate_step.depends_on = []
            candidate_step.input_bindings = [
                _binding("recommendation_profile", source_type="literal", value={"message": None, "user_memory_summary": None, "research_profile": {}, "request_context": {}}),
            ]
        generate_step = next((step for step in list(plan.steps or []) if step.tool_name == "generate_recommendations"), None)
        if generate_step is not None:
            updated_bindings: List[StepInputBinding] = []
            replaced_profile = False
            for binding in list(generate_step.input_bindings or []):
                if binding.input_key == "recommendation_profile":
                    updated_bindings.append(
                        _binding(
                            "recommendation_profile",
                            source_type="literal",
                            value={"message": None, "user_memory_summary": None, "research_profile": {}, "request_context": {}},
                        )
                    )
                    replaced_profile = True
                else:
                    updated_bindings.append(binding)
            if not replaced_profile:
                updated_bindings.append(
                    _binding(
                        "recommendation_profile",
                        source_type="literal",
                        value={"message": None, "user_memory_summary": None, "research_profile": {}, "request_context": {}},
                    )
                )
            generate_step.input_bindings = updated_bindings
            generate_step.depends_on = [dependency for dependency in list(generate_step.depends_on or []) if dependency != trigger_step_id]
        runtime.step_status[trigger_step_id] = "skipped"

    def _refresh_plan_topology(self, plan: ExecutablePlan) -> None:
        step_ids = [step.step_id for step in list(plan.steps or [])]
        depended_ids = {dependency for step in list(plan.steps or []) for dependency in list(step.depends_on or [])}
        plan.entry_step_ids = [step.step_id for step in list(plan.steps or []) if not step.depends_on]
        plan.final_step_ids = [step_id for step_id in step_ids if step_id not in depended_ids]

    def _trace_for_replan(self, *, step_id: str, observation: ObservationResult, rule_name: str, reason_key: str):
        from .schemas import ExecutionTrace

        return ExecutionTrace(
            step_id=step_id,
            event="plan_replanned",
            status="running",
            detail={
                "rule_name": rule_name,
                "observation_status": observation.status,
                "observation_reason": observation.reason,
                "reason_key": reason_key,
            },
        )


__all__ = ["Replanner", "ReplanRule", "ReplanDecision"]
