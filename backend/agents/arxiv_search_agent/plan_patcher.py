from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .plan_validator import PlanValidator
from .schemas import (
    ExecutablePlan,
    ExecutionTrace,
    Goal,
    ObservationResult,
    PlanRuntime,
    PlanStep,
    RecoveryAction,
    RecoveryCandidate,
    StepInputBinding,
    StepPolicy,
)
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


@dataclass
class PlanPatchResult:
    updated_plan: Optional[ExecutablePlan] = None
    updated_runtime: Optional[PlanRuntime] = None
    fallback: bool = False
    fallback_reason: Optional[str] = None
    patch_result: Dict[str, Any] = field(default_factory=dict)


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


class PlanPatcher:
    """把 RecoveryAction 落地为计划修改或 fallback，并统一负责 patch 后校验。"""

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry
        self.validator = PlanValidator()
        self._patch_dispatch = {
            "rewrite_search_chain": self.patch_search_rewrite_chain,
            "inject_index_confirmation_chain": self.patch_index_confirmation_chain,
            "downgrade_recommendation_chain": self.patch_recommendation_downgrade_chain,
            "clarification_chain": self.patch_clarification_chain,
            "fallback_answer": self.patch_fallback_answer,
            "retry_step_with_adjusted_arguments": self.retry_step_with_adjusted_arguments,
            "use_last_good_index": self.patch_use_last_good_index,
        }

    def apply(
        self,
        *,
        goal: Goal,
        current_plan: ExecutablePlan,
        runtime: PlanRuntime,
        failed_step: PlanStep,
        observation: ObservationResult,
        action: RecoveryAction,
        reason_key: str,
        current_reason_count: int,
        current_step_count: int,
        recovery_candidates: Sequence[RecoveryCandidate],
        rejected_candidates: Sequence[Dict[str, Any]],
        selection_reason: str,
        scored_candidates: Sequence[Dict[str, Any]],
        llm_diagnosis: Optional[Dict[str, Any]] = None,
        safety_check_result: Optional[Dict[str, Any]] = None,
    ) -> PlanPatchResult:
        del goal
        plan_copy = current_plan.model_copy(deep=True)
        runtime_copy = runtime.model_copy(deep=True)
        plan_copy.steps = list(plan_copy.steps or [])
        runtime_copy.replan_counts[reason_key] = current_reason_count + 1
        runtime_copy.step_replan_counts[failed_step.step_id] = current_step_count + 1

        patch_strategy = str(action.patch_strategy or "")
        patch_func = self._patch_dispatch.get(patch_strategy)
        if action.action_type == "fallback_answer":
            patch_func = self.patch_fallback_answer
        if patch_func is None:
            fallback_reason = f"unsupported_patch_strategy:{patch_strategy or action.action_type}"
            return self._fallback_result(
                runtime_copy=runtime_copy,
                failed_step=failed_step,
                observation=observation,
                action=action,
                reason_key=reason_key,
                recovery_candidates=recovery_candidates,
                rejected_candidates=rejected_candidates,
                selection_reason=selection_reason,
                scored_candidates=scored_candidates,
                llm_diagnosis=llm_diagnosis,
                safety_check_result=safety_check_result,
                fallback_reason=fallback_reason,
            )

        patch_result = patch_func(plan_copy=plan_copy, runtime_copy=runtime_copy, failed_step=failed_step, action=action)
        if bool(patch_result.get("fallback")):
            fallback_reason = str(patch_result.get("fallback_reason") or action.fallback_reason or reason_key)
            return self._fallback_result(
                runtime_copy=runtime_copy,
                failed_step=failed_step,
                observation=observation,
                action=action,
                reason_key=reason_key,
                recovery_candidates=recovery_candidates,
                rejected_candidates=rejected_candidates,
                selection_reason=selection_reason,
                scored_candidates=scored_candidates,
                llm_diagnosis=llm_diagnosis,
                safety_check_result=safety_check_result,
                fallback_reason=fallback_reason,
                patch_result=patch_result,
            )

        self._refresh_plan_topology(plan_copy)
        self.validator.validate(plan_copy, self.tool_registry)
        runtime_copy.plan = plan_copy
        runtime_copy.step_status = {
            step.step_id: runtime_copy.step_status.get(step.step_id, step.status)
            for step in list(plan_copy.steps or [])
        }
        self._append_recovery_trace(
            runtime_copy,
            step_id=failed_step.step_id,
            observation=observation,
            action=action,
            reason_key=reason_key,
            recovery_candidates=recovery_candidates,
            rejected_candidates=rejected_candidates,
            selection_reason=selection_reason,
            scored_candidates=scored_candidates,
            llm_diagnosis=llm_diagnosis,
            safety_check_result=safety_check_result,
            patch_result=patch_result,
        )
        return PlanPatchResult(updated_plan=plan_copy, updated_runtime=runtime_copy, patch_result=patch_result)

    def patch_search_rewrite_chain(self, *, plan_copy: ExecutablePlan, runtime_copy: PlanRuntime, failed_step: PlanStep, action: RecoveryAction) -> Dict[str, Any]:
        rewrite_step_id = self._make_unique_step_id(plan_copy, "rewrite_arxiv_query")
        search_step_id = self._make_unique_step_id(plan_copy, "search_arxiv")
        validate_step_id = self._make_unique_step_id(plan_copy, "validate_arxiv_results")
        rewrite_step = self._build_step(
            step_id=rewrite_step_id,
            action_type="rewrite",
            tool_name="rewrite_arxiv_query",
            output_key=self._make_unique_output_key(plan_copy, "rewritten_search_spec"),
            depends_on=[failed_step.step_id],
            input_bindings=[_binding("search_spec", source_type="step_output", step_id="build_arxiv_search_spec", required=False)],
        )
        search_step = self._build_step(
            step_id=search_step_id,
            action_type="search",
            tool_name="search_arxiv",
            output_key=self._make_unique_output_key(plan_copy, "arxiv_results_retry"),
            depends_on=[rewrite_step_id],
            retry_policy=StepPolicy(policy_type="retry", mode="allow_search_relaxation", max_attempts=2),
            input_bindings=[_binding("search_spec", source_type="step_output", step_id=rewrite_step_id)],
        )
        validate_step = self._build_step(
            step_id=validate_step_id,
            action_type="validate",
            tool_name="validate_arxiv_results",
            output_key=self._make_unique_output_key(plan_copy, "arxiv_result_quality_retry"),
            depends_on=[search_step_id],
            input_bindings=[_binding("arxiv_results", source_type="step_output", step_id=search_step_id)],
        )
        inserted_step_ids = [rewrite_step_id, search_step_id, validate_step_id]
        self._insert_steps_after(plan_copy, failed_step.step_id, [rewrite_step, search_step, validate_step])
        self._repoint_pending_bindings(
            plan_copy,
            runtime_copy,
            old_step_ids=["search_arxiv"],
            new_step_id=search_step_id,
            input_keys=["arxiv_results"],
            exclude_step_ids=inserted_step_ids,
        )
        self._repoint_pending_bindings(
            plan_copy,
            runtime_copy,
            old_step_ids=["validate_arxiv_results"],
            new_step_id=validate_step_id,
            input_keys=["arxiv_result_quality"],
            exclude_step_ids=inserted_step_ids,
        )
        return {
            "ok": True,
            "rule_name": "rule_arxiv_low_confidence" if action.patch_payload.get("failure_category") == "search_low_confidence" else "rule_arxiv_empty_result",
            "inserted_step_ids": inserted_step_ids,
            "patch_strategy": "rewrite_search_chain",
        }

    def patch_index_confirmation_chain(self, *, plan_copy: ExecutablePlan, runtime_copy: PlanRuntime, failed_step: PlanStep, action: RecoveryAction) -> Dict[str, Any]:
        del action
        request_step_id = self._make_unique_step_id(plan_copy, "request_confirmation")
        parse_step_id = self._make_unique_step_id(plan_copy, "parse_and_index_paper")
        request_step = self._build_step(
            step_id=request_step_id,
            action_type="clarify",
            tool_name="request_confirmation",
            output_key=self._make_unique_output_key(plan_copy, "confirmation_status"),
            depends_on=[failed_step.step_id],
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
                )
            ],
        )
        parse_step = self._build_step(
            step_id=parse_step_id,
            action_type="index",
            tool_name="parse_and_index_paper",
            output_key=self._make_unique_output_key(plan_copy, "index_build_result"),
            depends_on=[request_step_id],
            confirmation_policy=StepPolicy(policy_type="confirmation", mode="explicit_user_confirmation_required", requires_confirmation=True),
            input_bindings=[_binding("paper_reference", source_type="step_output", step_id="resolve_paper")],
        )
        self._insert_steps_after(plan_copy, failed_step.step_id, [request_step, parse_step])
        return {
            "ok": True,
            "rule_name": "rule_missing_paper_index",
            "inserted_step_ids": [request_step_id, parse_step_id],
            "patch_strategy": "inject_index_confirmation_chain",
        }

    def patch_recommendation_downgrade_chain(self, *, plan_copy: ExecutablePlan, runtime_copy: PlanRuntime, failed_step: PlanStep, action: RecoveryAction) -> Dict[str, Any]:
        del action
        empty_profile = {"message": None, "user_memory_summary": None, "research_profile": {}, "request_context": {}}
        for plan_step in list(plan_copy.steps or []):
            if plan_step.tool_name not in {"load_candidate_papers", "generate_recommendations"}:
                continue
            plan_step.depends_on = [dependency for dependency in list(plan_step.depends_on or []) if dependency != failed_step.step_id]
            plan_step.input_bindings = [
                binding for binding in list(plan_step.input_bindings or []) if binding.input_key != "recommendation_profile"
            ]
            plan_step.input_bindings.append(_binding("recommendation_profile", source_type="literal", value=empty_profile))
        runtime_copy.step_status[failed_step.step_id] = "skipped"
        return {"ok": True, "rule_name": "rule_empty_user_profile", "patch_strategy": "downgrade_recommendation_chain"}

    def patch_clarification_chain(self, *, plan_copy: ExecutablePlan, runtime_copy: PlanRuntime, failed_step: PlanStep, action: RecoveryAction) -> Dict[str, Any]:
        del plan_copy, runtime_copy, failed_step
        return {"ok": False, "fallback": True, "fallback_reason": action.fallback_reason or "clarification_required", "patch_strategy": "clarification_chain"}

    def patch_fallback_answer(self, *, plan_copy: ExecutablePlan, runtime_copy: PlanRuntime, failed_step: PlanStep, action: RecoveryAction) -> Dict[str, Any]:
        del plan_copy, runtime_copy, failed_step
        return {"ok": False, "fallback": True, "fallback_reason": action.fallback_reason or "recovery_fallback", "patch_strategy": "fallback_answer"}

    def retry_step_with_adjusted_arguments(self, *, plan_copy: ExecutablePlan, runtime_copy: PlanRuntime, failed_step: PlanStep, action: RecoveryAction) -> Dict[str, Any]:
        retry_strategy = dict(action.patch_payload.get("retry_strategy") or {})
        if failed_step.tool_name == "answer_paper_question":
            return self._insert_qa_retry_step(
                plan_copy=plan_copy,
                runtime_copy=runtime_copy,
                failed_step=failed_step,
                retry_strategy=retry_strategy,
            )
        target_step = next((step for step in list(plan_copy.steps or []) if step.step_id == failed_step.step_id), None)
        if target_step is None:
            return {"ok": False, "fallback": True, "fallback_reason": f"retry_target_missing:{failed_step.step_id}"}
        # 非 QA 工具先沿用原步骤有限重试；具体参数调整由工具自身或后续策略层承载。
        target_step.status = "pending"
        runtime_copy.step_status[failed_step.step_id] = "pending"
        return {"ok": True, "rule_name": "rule_retry_step", "patch_strategy": "retry_step_with_adjusted_arguments"}

    def patch_use_last_good_index(self, *, plan_copy: ExecutablePlan, runtime_copy: PlanRuntime, failed_step: PlanStep, action: RecoveryAction) -> Dict[str, Any]:
        del action
        target_step = next((step for step in list(plan_copy.steps or []) if step.step_id == failed_step.step_id), None)
        if target_step is None:
            return {"ok": False, "fallback": True, "fallback_reason": f"last_good_index_target_missing:{failed_step.step_id}"}
        output_key = target_step.output_key or "paper_index_status"
        # last good index 是低风险恢复：不重建、不删除，仅把索引检查输出标记为可复用，供后续 QA 步骤继续执行。
        runtime_copy.outputs[output_key] = {
            "status": "indexed",
            "recovery_mode": "use_last_good_index",
            "source_step_id": failed_step.step_id,
            "last_good_index": True,
        }
        runtime_copy.step_status[failed_step.step_id] = "success"
        for plan_step in list(plan_copy.steps or []):
            if plan_step.tool_name != "answer_paper_question" or plan_step.step_id == failed_step.step_id:
                continue
            existing_keys = {binding.input_key for binding in list(plan_step.input_bindings or [])}
            if "index_strategy" not in existing_keys:
                # 与 PlanExecutor 的结构化输入契约保持一致，避免恢复策略变成只存在于 trace 里的标记。
                plan_step.input_bindings.append(
                    _binding("index_strategy", source_type="literal", value={"mode": "use_last_good_index"}, required=False)
                )
        return {
            "ok": True,
            "rule_name": "rule_use_last_good_index",
            "patch_strategy": "use_last_good_index",
            "reused_output_key": output_key,
        }

    def _insert_qa_retry_step(
        self,
        *,
        plan_copy: ExecutablePlan,
        runtime_copy: PlanRuntime,
        failed_step: PlanStep,
        retry_strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        retry_step_id = self._make_unique_step_id(plan_copy, "answer_paper_question")
        retry_step = self._build_step(
            step_id=retry_step_id,
            action_type="answer",
            tool_name="answer_paper_question",
            output_key=self._make_unique_output_key(plan_copy, "paper_qa_result_retry"),
            depends_on=[dependency for dependency in list(failed_step.depends_on or [])],
            input_bindings=[
                binding.model_copy(deep=True)
                for binding in list(failed_step.input_bindings or [])
                if binding.input_key != "qa_recovery_strategy"
            ],
        )
        # QA recovery 的策略必须以结构化 payload 进入工具层，避免在执行器里散落临时字段。
        retry_step.input_bindings.append(
            _binding("qa_recovery_strategy", source_type="literal", value=dict(retry_strategy or {}), required=False)
        )
        self._insert_steps_after(plan_copy, failed_step.step_id, [retry_step])
        self._repoint_pending_bindings(
            plan_copy,
            runtime_copy,
            old_step_ids=[failed_step.step_id],
            new_step_id=retry_step_id,
            input_keys=["paper_qa_result"],
            exclude_step_ids=[retry_step_id],
        )
        runtime_copy.step_status[failed_step.step_id] = "low_confidence"
        runtime_copy.step_status[retry_step_id] = "pending"
        return {
            "ok": True,
            "rule_name": "rule_qa_recovery_retry",
            "patch_strategy": "retry_step_with_adjusted_arguments",
            "inserted_step_ids": [retry_step_id],
            "retry_strategy": dict(retry_strategy or {}),
        }

    def _fallback_result(
        self,
        *,
        runtime_copy: PlanRuntime,
        failed_step: PlanStep,
        observation: ObservationResult,
        action: RecoveryAction,
        reason_key: str,
        recovery_candidates: Sequence[RecoveryCandidate],
        rejected_candidates: Sequence[Dict[str, Any]],
        selection_reason: str,
        scored_candidates: Sequence[Dict[str, Any]],
        llm_diagnosis: Optional[Dict[str, Any]],
        safety_check_result: Optional[Dict[str, Any]],
        fallback_reason: str,
        patch_result: Optional[Dict[str, Any]] = None,
    ) -> PlanPatchResult:
        payload = dict(patch_result or {"ok": False})
        payload.setdefault("fallback", True)
        payload.setdefault("fallback_reason", fallback_reason)
        self._append_recovery_trace(
            runtime_copy,
            step_id=failed_step.step_id,
            observation=observation,
            action=action,
            reason_key=reason_key,
            recovery_candidates=recovery_candidates,
            rejected_candidates=rejected_candidates,
            selection_reason=selection_reason,
            scored_candidates=scored_candidates,
            llm_diagnosis=llm_diagnosis,
            safety_check_result=safety_check_result,
            patch_result=payload,
        )
        return PlanPatchResult(updated_runtime=runtime_copy, fallback=True, fallback_reason=fallback_reason, patch_result=payload)

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
            confirmation_policy = StepPolicy(policy_type="confirmation", mode="explicit_user_confirmation_required", requires_confirmation=True)
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
                    else:
                        updated_bindings.append(binding.model_copy(update={"step_id": new_step_id}))
                else:
                    updated_bindings.append(binding)
            plan_step.input_bindings = updated_bindings
            plan_step.depends_on = list(dict.fromkeys([new_step_id if item in old_step_id_set else item for item in list(plan_step.depends_on or [])]))

    def _refresh_plan_topology(self, plan: ExecutablePlan) -> None:
        step_ids = [step.step_id for step in list(plan.steps or [])]
        depended_ids = {dependency for step in list(plan.steps or []) for dependency in list(step.depends_on or [])}
        plan.entry_step_ids = [step.step_id for step in list(plan.steps or []) if not step.depends_on]
        plan.final_step_ids = [step_id for step_id in step_ids if step_id not in depended_ids]

    def _append_recovery_trace(
        self,
        runtime: PlanRuntime,
        *,
        step_id: str,
        observation: ObservationResult,
        action: RecoveryAction,
        reason_key: str,
        recovery_candidates: Sequence[RecoveryCandidate],
        rejected_candidates: Sequence[Dict[str, Any]],
        selection_reason: str,
        scored_candidates: Sequence[Dict[str, Any]],
        llm_diagnosis: Optional[Dict[str, Any]],
        safety_check_result: Optional[Dict[str, Any]],
        patch_result: Dict[str, Any],
    ) -> None:
        runtime.trace.append(
            ExecutionTrace(
                step_id=step_id,
                event="plan_replanned",
                status="running",
                detail={
                    "rule_name": patch_result.get("rule_name"),
                    "failure_category": observation.failure_category,
                    "observation_status": observation.status,
                    "observation_reason": observation.reason,
                    "reason_key": reason_key,
                    "retry_count": runtime.step_replan_counts.get(step_id, 0),
                    "risk_level": action.patch_payload.get("risk_level"),
                    "recovery_candidates": [candidate.model_dump() for candidate in list(recovery_candidates or [])],
                    "generated_candidates": [candidate.model_dump() for candidate in list(recovery_candidates or [])],
                    "llm_diagnosis": dict(llm_diagnosis or {}),
                    "selected_recovery_candidate_id": action.selected_candidate_id,
                    "selected_recovery_action": action.model_dump(),
                    "selected_action": action.model_dump(),
                    "safety_check_result": dict(safety_check_result or {}),
                    "rejected_candidates": list(rejected_candidates or []),
                    "scored_candidates": list(scored_candidates or []),
                    "selection_reason": selection_reason,
                    "patch_result": dict(patch_result or {}),
                    "plan_patch_result": dict(patch_result or {}),
                    "fallback_used": bool(patch_result.get("fallback")),
                    "fallback_reason": patch_result.get("fallback_reason"),
                    "final_recovery_status": "fallback" if bool(patch_result.get("fallback")) else "patched",
                },
            )
        )


__all__ = ["PlanPatchResult", "PlanPatcher"]
