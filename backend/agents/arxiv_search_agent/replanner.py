from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .failure_classifier import FailureClassifier
from .plan_patcher import PlanPatcher
from .recovery_chooser import RecoveryChooser
from .recovery_diagnosis import LLMRecoveryDiagnoser
from .recovery_policy import DEFAULT_RECOVERY_POLICY_REGISTRY, RecoveryPolicyRegistry
from .recovery_safety import RecoverySafetyGuard
from .schemas import ExecutablePlan, Goal, ObservationResult, PlanRuntime, PlanStep
from .state import AgentState
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
    recovery_debug: Dict[str, Any] = field(default_factory=dict)


class Replanner:
    """恢复策略编排器：只串联分类、候选生成、动作选择和 plan patch。"""

    MAX_REASON_REPLANS = 2
    MAX_STEP_REPLANS = 2
    MAX_PLAN_REPLANS = 5

    def __init__(
        self,
        tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY,
        recovery_policy_registry: RecoveryPolicyRegistry = DEFAULT_RECOVERY_POLICY_REGISTRY,
        failure_classifier: Optional[FailureClassifier] = None,
        recovery_chooser: Optional[RecoveryChooser] = None,
        recovery_diagnoser: Optional[LLMRecoveryDiagnoser] = None,
        safety_guard: Optional[RecoverySafetyGuard] = None,
        plan_patcher: Optional[PlanPatcher] = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.recovery_policy_registry = recovery_policy_registry
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.recovery_chooser = recovery_chooser or RecoveryChooser(tool_registry=tool_registry)
        self.recovery_diagnoser = recovery_diagnoser or LLMRecoveryDiagnoser()
        self.safety_guard = safety_guard or RecoverySafetyGuard(tool_registry=tool_registry)
        self.plan_patcher = plan_patcher or PlanPatcher(tool_registry=tool_registry)

    def replan(
        self,
        *,
        goal: Goal,
        current_plan: ExecutablePlan,
        runtime: PlanRuntime,
        failed_or_low_quality_step: PlanStep,
        observation_result: ObservationResult,
        state: Optional[AgentState] = None,
    ) -> ReplanDecision:
        classified_observation = self.failure_classifier.classify(
            step=failed_or_low_quality_step,
            observation=observation_result,
        )
        reason_key = f"{failed_or_low_quality_step.tool_name}:{classified_observation.status}"
        current_reason_count = int(runtime.replan_counts.get(reason_key, 0) or 0)
        current_step_count = int(runtime.step_replan_counts.get(failed_or_low_quality_step.step_id, 0) or 0)
        total_replans = sum(int(value or 0) for value in runtime.replan_counts.values())
        # 重规划是质量补救，不是无限重试机制；先做全局限流，再进入策略选择。
        if current_reason_count >= self.MAX_REASON_REPLANS or current_step_count >= self.MAX_STEP_REPLANS or total_replans >= self.MAX_PLAN_REPLANS:
            return ReplanDecision(fallback=True, fallback_reason=f"replan_limit_exceeded:{reason_key}")

        recovery_candidates = self.recovery_policy_registry.get_candidates(
            step=failed_or_low_quality_step,
            observation=classified_observation,
            runtime=runtime,
            state=state,
        )
        llm_diagnosis = self.recovery_diagnoser.diagnose(
            goal=goal,
            current_plan=current_plan,
            runtime=runtime,
            failed_step=failed_or_low_quality_step,
            observation=classified_observation,
            recovery_candidates=recovery_candidates,
            state=state,
        )
        recovery_choice = self.recovery_chooser.choose(
            goal=goal,
            current_plan=current_plan,
            runtime=runtime,
            failed_step=failed_or_low_quality_step,
            observation=classified_observation,
            recovery_candidates=recovery_candidates,
            llm_diagnosis=llm_diagnosis,
        )
        safety_check = self.safety_guard.check(action=recovery_choice.action, failed_step=failed_or_low_quality_step, runtime=runtime)
        selected_action = safety_check.fallback_action or recovery_choice.action
        patch_result = self.plan_patcher.apply(
            goal=goal,
            current_plan=current_plan,
            runtime=runtime,
            failed_step=failed_or_low_quality_step,
            observation=classified_observation,
            action=selected_action,
            reason_key=reason_key,
            current_reason_count=current_reason_count,
            current_step_count=current_step_count,
            recovery_candidates=recovery_candidates,
            rejected_candidates=recovery_choice.rejected_candidates,
            selection_reason=recovery_choice.selection_reason,
            scored_candidates=recovery_choice.scored_candidates,
            llm_diagnosis=llm_diagnosis.model_dump() if llm_diagnosis is not None else None,
            safety_check_result=safety_check.model_dump(),
        )
        recovery_debug = {
            "failure_category": classified_observation.failure_category,
            "generated_candidate_count": len(recovery_candidates),
            "llm_diagnosis": llm_diagnosis.model_dump() if llm_diagnosis is not None else None,
            "selected_recovery_action": selected_action.model_dump(),
            "safety_check_result": safety_check.model_dump(),
            "selection_reason": recovery_choice.selection_reason,
            "rejected_candidates": list(recovery_choice.rejected_candidates or []),
            "patch_result": dict(patch_result.patch_result or {}),
        }
        if patch_result.fallback:
            return ReplanDecision(
                fallback=True,
                fallback_reason=patch_result.fallback_reason or recovery_choice.action.fallback_reason or reason_key,
                updated_runtime=patch_result.updated_runtime,
                recovery_debug=recovery_debug,
            )
        if patch_result.updated_plan is not None and patch_result.updated_runtime is not None:
            return ReplanDecision(updated_plan=patch_result.updated_plan, updated_runtime=patch_result.updated_runtime, recovery_debug=recovery_debug)
        return ReplanDecision(fallback=True, fallback_reason=f"recovery_patch_failed:{reason_key}", updated_runtime=patch_result.updated_runtime, recovery_debug=recovery_debug)


__all__ = ["Replanner", "ReplanRule", "ReplanDecision"]
