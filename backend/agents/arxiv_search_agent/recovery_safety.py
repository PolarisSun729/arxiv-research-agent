from __future__ import annotations

from typing import List

from .schemas import PlanRuntime, PlanStep, RecoveryAction, RecoverySafetyCheckResult
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


class RecoverySafetyGuard:
    """恢复动作安全闸门，防止 policy 或 LLM 建议绕过副作用和确认边界。"""

    SAFE_PATCH_STRATEGIES = {
        "rewrite_search_chain",
        "inject_index_confirmation_chain",
        "downgrade_recommendation_chain",
        "clarification_chain",
        "fallback_answer",
        "retry_step_with_adjusted_arguments",
        "use_last_good_index",
    }

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry

    def check(self, *, action: RecoveryAction, failed_step: PlanStep, runtime: PlanRuntime) -> RecoverySafetyCheckResult:
        reasons: List[str] = []
        patch_strategy = str(action.patch_strategy or "")
        required_tools = [str(item) for item in list(action.patch_payload.get("required_tools") or [])]

        if patch_strategy and patch_strategy not in self.SAFE_PATCH_STRATEGIES:
            reasons.append(f"dangerous_patch_strategy:{patch_strategy}")

        max_attempts = action.patch_payload.get("max_attempts")
        current_attempts = int(runtime.step_replan_counts.get(action.target_step_id or failed_step.step_id, 0) or 0)
        if max_attempts is not None and current_attempts >= int(max_attempts):
            reasons.append(f"retry_limit_exceeded:{current_attempts}/{max_attempts}")

        unknown_tools = [tool_name for tool_name in required_tools if self.tool_registry.get(tool_name) is None]
        if unknown_tools:
            reasons.append(f"unknown_required_tools:{','.join(unknown_tools)}")

        side_effects = set()
        confirmation_capable = False
        for tool_name in required_tools:
            tool = self.tool_registry.get(tool_name)
            if tool is None:
                continue
            side_effects.add(str(tool.side_effect_level or "none"))
            confirmation_capable = confirmation_capable or bool(getattr(tool, "requires_confirmation", False))
        side_effects.add(str(failed_step.side_effect_level or "none"))

        if action.action_type == "retry_step" and "persistent_write" in side_effects:
            reasons.append("persistent_write_retry_blocked")
        if str(action.patch_payload.get("risk_level") or "") == "high" and not (action.requires_confirmation and confirmation_capable):
            reasons.append("high_risk_action_without_confirmation")
        if "persistent_write" in side_effects and not action.requires_confirmation and action.action_type != "fallback_answer":
            reasons.append("persistent_write_requires_confirmation_or_explicit_target")

        if reasons:
            fallback_action = RecoveryAction(
                action_type="fallback_answer",
                action_semantic="fallback_answer",
                target_step_id=failed_step.step_id,
                selected_candidate_id=action.selected_candidate_id,
                patch_strategy="fallback_answer",
                patch_payload={"safety_rejected": True, "risk_level": "low", "required_tools": []},
                fallback_reason="recovery_safety_rejected:" + "|".join(reasons),
                debug_reason="SafetyGuard 拒绝恢复动作，改走 fallback。",
            )
            return RecoverySafetyCheckResult(allowed=False, reasons=reasons, fallback_action=fallback_action, checked_action=action.model_dump())
        return RecoverySafetyCheckResult(allowed=True, reasons=[], checked_action=action.model_dump())


__all__ = ["RecoverySafetyGuard"]
