from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .schemas import ExecutablePlan, Goal, LLMRecoveryDiagnosis, ObservationResult, PlanRuntime, PlanStep, RecoveryAction, RecoveryCandidate
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


@dataclass
class RecoveryChoice:
    action: RecoveryAction
    selection_reason: str
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    scored_candidates: List[Dict[str, Any]] = field(default_factory=list)


class RecoveryChooser:
    """规则型恢复动作选择器，不调用 LLM，也不直接修改 plan。"""

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry

    def choose(
        self,
        *,
        goal: Goal,
        current_plan: ExecutablePlan,
        runtime: PlanRuntime,
        failed_step: PlanStep,
        observation: ObservationResult,
        recovery_candidates: Sequence[RecoveryCandidate],
        llm_diagnosis: Optional[LLMRecoveryDiagnosis] = None,
    ) -> RecoveryChoice:
        del current_plan
        rejected: List[Dict[str, Any]] = []
        accepted: List[Dict[str, Any]] = []
        fallback_candidates: List[RecoveryCandidate] = []

        for candidate in list(recovery_candidates or []):
            if candidate.action_type == "fallback_answer":
                fallback_candidates.append(candidate)
                continue
            reject_reason = self._reject_reason(candidate=candidate, goal=goal, runtime=runtime, failed_step=failed_step)
            if reject_reason:
                rejected.append({"candidate_id": candidate.candidate_id, "reason": reject_reason, "candidate": candidate.model_dump()})
                continue
            score_detail = self._score(candidate=candidate, goal=goal, runtime=runtime, failed_step=failed_step, llm_diagnosis=llm_diagnosis)
            accepted.append({"candidate": candidate, **score_detail})

        if accepted:
            accepted.sort(key=lambda item: (-float(item["score"]), str(item["candidate"].candidate_id)))
            selected = accepted[0]["candidate"]
            return RecoveryChoice(
                action=self._action_from_candidate(selected, debug_reason=str(accepted[0]["debug_reason"])),
                selection_reason=str(accepted[0]["debug_reason"]),
                rejected_candidates=rejected,
                scored_candidates=[self._serializable_score(item) for item in accepted],
            )

        if fallback_candidates:
            selected_fallback = sorted(fallback_candidates, key=lambda item: (-item.priority, -item.confidence, item.candidate_id))[0]
            return RecoveryChoice(
                action=self._action_from_candidate(
                    selected_fallback,
                    debug_reason="没有可用的计划修复候选，选择显式 fallback 候选。",
                ),
                selection_reason="fallback_candidate_selected",
                rejected_candidates=rejected,
                scored_candidates=[],
            )

        fallback_reason = f"no_available_recovery_candidate:{failed_step.tool_name}:{observation.status}"
        return RecoveryChoice(
            action=RecoveryAction(
                action_type="fallback_answer",
                target_step_id=failed_step.step_id,
                fallback_reason=fallback_reason,
                debug_reason="所有候选均不可用或策略未生成候选，降级为 fallback。",
            ),
            selection_reason=fallback_reason,
            rejected_candidates=rejected,
            scored_candidates=[],
        )

    def _reject_reason(self, *, candidate: RecoveryCandidate, goal: Goal, runtime: PlanRuntime, failed_step: PlanStep) -> Optional[str]:
        del goal
        current_step_attempts = int(runtime.step_replan_counts.get(candidate.target_step_id or failed_step.step_id, 0) or 0)
        if candidate.max_attempts is not None and current_step_attempts >= int(candidate.max_attempts):
            return f"retry_limit_exceeded:{current_step_attempts}/{candidate.max_attempts}"

        side_effects = self._required_tool_side_effects(candidate)
        if candidate.action_type == "retry_step" and "persistent_write" in side_effects:
            return "persistent_write_not_allowed_for_silent_retry"

        if candidate.risk_level == "high":
            confirmation_capable = bool(candidate.requires_confirmation and self._has_confirmation_capable_tool(candidate))
            if not confirmation_capable:
                return "high_risk_candidate_requires_confirmation_capability"
        return None

    def _score(self, *, candidate: RecoveryCandidate, goal: Goal, runtime: PlanRuntime, failed_step: PlanStep, llm_diagnosis: Optional[LLMRecoveryDiagnosis] = None) -> Dict[str, Any]:
        attempts = int(runtime.step_replan_counts.get(candidate.target_step_id or failed_step.step_id, 0) or 0)
        side_effects = self._required_tool_side_effects(candidate)
        risk_penalty = {"low": 0.0, "medium": 15.0, "high": 35.0}.get(candidate.risk_level, 20.0)
        retry_penalty = attempts * 20.0
        confirmation_bonus = 8.0 if candidate.requires_confirmation and self._has_confirmation_capable_tool(candidate) else 0.0
        clarification_bonus = 15.0 if attempts >= 1 and candidate.action_type == "ask_clarification" else 0.0
        side_effect_penalty = 10.0 if "external_call" in side_effects else 0.0
        goal_risk_penalty = 8.0 if str(goal.goal_type or "") == "preference_action" and candidate.risk_level != "low" else 0.0
        llm_rank_bonus = self._llm_rank_bonus(candidate=candidate, llm_diagnosis=llm_diagnosis)
        score = (
            float(candidate.priority) * 10.0
            + float(candidate.confidence) * 100.0
            + confirmation_bonus
            + clarification_bonus
            + llm_rank_bonus
            - retry_penalty
            - risk_penalty
            - side_effect_penalty
            - goal_risk_penalty
        )
        return {
            "score": score,
            "debug_reason": (
                f"priority={candidate.priority}, confidence={candidate.confidence}, attempts={attempts}, "
                f"risk={candidate.risk_level}, side_effects={sorted(side_effects)}, llm_rank_bonus={llm_rank_bonus}"
            ),
        }

    def _llm_rank_bonus(self, *, candidate: RecoveryCandidate, llm_diagnosis: Optional[LLMRecoveryDiagnosis]) -> float:
        if llm_diagnosis is None or not llm_diagnosis.ranked_candidate_ids:
            return 0.0
        try:
            rank = list(llm_diagnosis.ranked_candidate_ids).index(candidate.candidate_id)
        except ValueError:
            return 0.0
        # LLM 只能在规则得分内做轻量排序辅助，不能压过安全和 retry 限制。
        return max(0.0, 12.0 - float(rank) * 4.0) * max(0.0, min(float(llm_diagnosis.confidence or 0.0), 1.0))

    def _required_tool_side_effects(self, candidate: RecoveryCandidate) -> set[str]:
        side_effects: set[str] = set()
        for tool_name in list(candidate.required_tools or []):
            tool = self.tool_registry.get(str(tool_name))
            if tool is not None:
                side_effects.add(str(tool.side_effect_level or "none"))
        return side_effects

    def _has_confirmation_capable_tool(self, candidate: RecoveryCandidate) -> bool:
        for tool_name in list(candidate.required_tools or []):
            tool = self.tool_registry.get(str(tool_name))
            if bool(getattr(tool, "requires_confirmation", False)):
                return True
        return False

    def _action_from_candidate(self, candidate: RecoveryCandidate, *, debug_reason: str) -> RecoveryAction:
        return RecoveryAction(
            action_type=candidate.action_type,
            action_semantic=candidate.action_semantic or self._semantic_for_candidate(candidate),
            target_step_id=candidate.target_step_id,
            selected_candidate_id=candidate.candidate_id,
            patch_strategy=candidate.patch_strategy,
            patch_payload={
                "failure_category": candidate.failure_category,
                "required_tools": list(candidate.required_tools or []),
                "risk_level": candidate.risk_level,
                "expected_effect": candidate.expected_effect,
                "max_attempts": candidate.max_attempts,
                "retry_strategy": dict(candidate.strategy_payload or {}),
                "policy_source": candidate.policy_source,
                "tool_recovery_policy": dict(candidate.tool_recovery_policy or {}),
            },
            requires_confirmation=candidate.requires_confirmation,
            fallback_reason=candidate.fallback_if_failed,
            debug_reason=debug_reason or candidate.reason,
        )

    def _semantic_for_candidate(self, candidate: RecoveryCandidate) -> str:
        # 对外暴露更细的恢复动作语义；旧 action_type 继续作为执行兼容字段。
        if candidate.action_type == "retry_step":
            return "retry_same_step" if not candidate.patch_strategy else "patch_current_step_inputs"
        if candidate.action_type == "ask_clarification":
            return "ask_clarification"
        if candidate.action_type == "fallback_answer":
            return "fallback_answer"
        if candidate.action_type == "abort_with_error":
            return "terminate_failed"
        if candidate.action_type == "skip_step":
            return "skip_step"
        strategy = str(candidate.patch_strategy or "")
        if strategy in {"rewrite_search_chain", "inject_index_confirmation_chain"}:
            return "append_step_after_current"
        if strategy == "retry_step_with_adjusted_arguments":
            return "patch_current_step_inputs"
        if strategy in {"downgrade_recommendation_chain", "use_last_good_index"}:
            return "replace_remaining_plan"
        if strategy == "clarification_chain":
            return "ask_clarification"
        return "patch_current_step_inputs"

    def _serializable_score(self, item: Dict[str, Any]) -> Dict[str, Any]:
        candidate = item["candidate"]
        return {
            "candidate_id": candidate.candidate_id,
            "score": item["score"],
            "debug_reason": item["debug_reason"],
        }


__all__ = ["RecoveryChoice", "RecoveryChooser"]
