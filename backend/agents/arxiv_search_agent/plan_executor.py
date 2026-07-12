


from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from langgraph.types import interrupt
from services.storage.sqlite.stores import ApprovalGrantStore

from . import tool_registry as agent_tool_registry
from .execution.bindings import (
    InputBindingResolver,
    can_reuse_side_effect_output,
    evaluate_condition,
    existing_step_output,
    step_output,
)
from .execution.tool_execution import ToolExecutionService
from .execution.state_projector import RuntimeStateProjector
from .execution.execution_guard import ExecutionGuard
from .execution.target_selection import TargetSelectionCoordinator
from .execution.recovery import ExecutionRecoveryCoordinator
from .execution.side_effects import SideEffectInvocationService
from .observer import Observer
from .planner import build_executable_plan, build_plan_runtime
from .replanner import Replanner
from .response_assembler import assemble_final_answer
from .schemas import (
    AgentTurnResult,
    AgentRuntimeState,
    ExecutablePlan,
    ExecutionTrace,
    Goal,
    ObservationResult,
    PlanRuntime,
    PlanStep,
    StepExecutionResult,
    StepCondition,
)
from .state import AgentState
from .tool_adapters.models import ToolError, ToolExecutionResult
from .tool_result_projector import project_tool_result
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry

logger = logging.getLogger(__name__)
invoke_backend_tool = agent_tool_registry.invoke_backend_tool


from .execution.traces import (
    _build_execution_path_summary,
    _build_existing_index_skip_output,
    _build_paper_qa_quality_trace,
    _compact_step_output_for_trace,
    _compact_tool_execution_for_trace,
    _extract_arxiv_id_from_paper_payload,
    _json_safe,
    _is_paper_target_resolution_step,
    _log_plan_done,
    _make_tool_error,
    _model_to_plain,
    _record_step_output,
    _safe_compact,
    _should_preserve_non_success_observation_output,
    _state_run_id,
    _tool_error_to_text,
    _utcnow,
)
class PlanExecutor:
    """按计划拓扑、输入绑定和策略约束执行 ExecutablePlan。"""

    def __init__(
        self,
        tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY,
        approval_store: Optional[ApprovalGrantStore] = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.execution_guard = ExecutionGuard(approval_store)
        self.side_effects = SideEffectInvocationService(approval_store)
        self.target_selection = TargetSelectionCoordinator()
        self.input_bindings = InputBindingResolver()
        self.tool_execution = ToolExecutionService(
            registry=tool_registry,
            backend_invoker=lambda tool_name, **kwargs: invoke_backend_tool(tool_name, **kwargs),
        )
        self.state_projector = RuntimeStateProjector(execution_path_builder=_build_execution_path_summary)
        self.observer = Observer()
        self.replanner = Replanner(tool_registry=tool_registry)
        self.recovery = ExecutionRecoveryCoordinator(replanner=self.replanner)

    def execute(self, plan: ExecutablePlan, state: AgentState) -> AgentTurnResult:
        runtime = build_plan_runtime(state, goal=plan.goal, plan=plan, turn_status="success")
        runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
        runtime.outputs = {}
        runtime.trace = []
        runtime.retry_counts = {}
        runtime.replan_counts = {}
        runtime.step_replan_counts = {}
        return self._execute_runtime(runtime, state, allow_interrupt=False)

    def execute_runtime(self, runtime: PlanRuntime, state: AgentState) -> AgentTurnResult:
        """兼容旧入口：在已有 runtime 基础上连续执行到终态或确认暂停。"""
        return self._execute_runtime(runtime, state, allow_interrupt=False)

    def finalize_runtime(self, runtime: PlanRuntime) -> AgentTurnResult:
        """把当前 runtime 收束成 AgentTurnResult，供图上的 finalize 节点调用。"""
        self._finalize_blocked_pending_steps(runtime)
        return self._build_turn_result(runtime)

    def select_next_step(self, runtime: PlanRuntime, state: AgentState) -> Optional[PlanStep]:
        """选择下一个可执行 step。

        这个方法是给 LangGraph 节点准备的显式边界：选择逻辑只判断依赖、条件和前置约束，
        不触发任何工具调用，便于上层把“路由到哪个节点”与“真正执行工具”分开。
        """
        step = self._find_executable_step(runtime, state)
        self._sync_runtime_state(state, runtime, current_step=step)
        return step

    def execute_next_step(
        self,
        runtime: PlanRuntime,
        state: AgentState,
        *,
        allow_interrupt: bool = False,
        auto_replan: bool = True,
    ) -> StepExecutionResult:
        """只执行一个待执行 step，并返回结构化结果。

        旧执行器的大循环会继续调用这个方法完成兼容路径；未来 LangGraph 节点可以直接消费
        StepExecutionResult.runtime_patch，把单步输出、observation、确认态和恢复建议写入 checkpoint。
        """
        step = self.select_next_step(runtime, state)
        if step is None:
            self._finalize_blocked_pending_steps(runtime)
            turn_result = self._build_turn_result(runtime)
            self._sync_runtime_state(state, runtime, current_step=None)
            return StepExecutionResult(
                step_id=None,
                step_status=None,
                next_action="finish" if turn_result.status not in {"failed", "fallback"} else "fail",
                error=turn_result.error,
                interaction=turn_result.interaction,
                runtime_patch=self._build_runtime_patch(runtime, current_step=None),
                turn_result=turn_result,
            )

        previous_outputs = dict(runtime.outputs or {})
        previous_trace_len = len(runtime.trace or [])
        previous_error = runtime.error
        turn_result = self._execute_step(step, runtime, state, allow_interrupt=allow_interrupt, auto_replan=auto_replan)
        step_result = self._build_step_execution_result(
            step=step,
            runtime=runtime,
            previous_outputs=previous_outputs,
            previous_trace_len=previous_trace_len,
            previous_error=previous_error,
            turn_result=turn_result,
        )
        self._sync_runtime_state(state, runtime, current_step=step)
        return step_result

    def execute_current_step_tool(self, runtime: PlanRuntime, state: AgentState, *, allow_interrupt: bool = False) -> StepExecutionResult:
        """只执行当前 step 的工具调用，不做 observation / replan。

        这是显式 LangGraph `execute_step` 节点的入口。工具输出会写入 runtime.last_step_output，
        后续 `observe_step` 节点再独立判断结果质量，避免观察和重规划继续藏在执行节点里。
        """
        step = self._get_current_step(runtime)
        if step is None:
            return self.execute_next_step(runtime, state, allow_interrupt=allow_interrupt, auto_replan=False)

        runtime.step_status[step.step_id] = "running"
        runtime.last_step_output = None
        started_at = _utcnow()
        resolved_input, missing_input = self._resolve_input_bindings(step, runtime, state)
        if missing_input:
            runtime.step_status[step.step_id] = "failed"
            runtime.error = f"missing_input:{step.step_id}:{missing_input}"

            runtime.recovery_strategy = {"type": "ask_clarification", "reason": "missing_required_input", "input_key": missing_input}
            self._append_trace(
                runtime,
                step,
                event="step_failed",
                status="failed",
                detail={"failure_reason": "missing_input", "missing_input": missing_input, "resolved_input": _safe_compact(resolved_input), "started_at": started_at, "finished_at": _utcnow()},
            )
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(step=step, runtime=runtime, next_action="fail", error=runtime.error)

        if can_reuse_side_effect_output(runtime, step):
            # resume 或局部 replan 可能重新走到同一副作用 step；已有输出时复用结果，避免重复外部调用/持久写入。
            reused_output = existing_step_output(runtime, step)
            runtime.step_status[step.step_id] = "success"
            runtime.needs_replan = False
            runtime.last_step_output = {
                "step_id": step.step_id,
                "reused_output": True,
                "tool_contract": _json_safe(self.tool_registry.describe_contract(step.tool_name)),
                "resolved_input": _json_safe(resolved_input),
                "normalized_output": _json_safe(reused_output),
                "started_at": started_at,
                "finished_at": _utcnow(),
            }
            self._append_trace(
                runtime,
                step,
                event="step_reused_output",
                status="success",
                detail={
                    "tool_name": step.tool_name,
                    "side_effect_level": step.side_effect_level,
                    "output_key": step.output_key,
                    "reason": "side_effect_output_already_available",
                },
            )
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(step=step, runtime=runtime, next_action="continue", output=reused_output)

        guard_decision = None
        if step.confirmation_policy and step.confirmation_policy.requires_confirmation:
            guard_decision = self.execution_guard.evaluate(
                step=step,
                runtime=runtime,
                state=state,
                arguments=resolved_input,
                reason="explicit_user_confirmation_required",
            )
            if guard_decision.action == "wait_for_interaction":
                runtime.interaction = guard_decision.interaction
                runtime.step_status[step.step_id] = "waiting_interaction"
                runtime.turn_status = "waiting_interaction"
                self._append_trace(
                    runtime,
                    step,
                    event="interaction_requested",
                    status="waiting_interaction",
                    detail={
                        "interaction_id": guard_decision.interaction.interaction_id,
                        "kind": guard_decision.interaction.kind,
                        "arguments_fingerprint": guard_decision.arguments_fingerprint,
                    },
                )
                self._sync_runtime_state(state, runtime, current_step=step)
                if allow_interrupt:
                    # LangGraph checkpoint 保存 interaction 后中断；业务 resume 事务成功后会从节点开头重入。
                    resume_payload = interrupt(guard_decision.interaction.model_dump(mode="json"))
                    resume_decision = str((resume_payload or {}).get("decision") or "").strip().lower() if isinstance(resume_payload, Mapping) else ""
                    if resume_decision in {"reject", "cancel"}:
                        runtime.interaction = None
                        runtime.step_status[step.step_id] = "skipped"
                        runtime.turn_status = "failed"
                        runtime.error = f"interaction_rejected:{step.step_id}"
                        self._append_trace(
                            runtime,
                            step,
                            event="interaction_cancelled",
                            status="skipped",
                            detail={"interaction_id": guard_decision.interaction.interaction_id},
                        )
                        return self._step_result_from_runtime(
                            step=step,
                            runtime=runtime,
                            next_action="fail",
                            error=runtime.error,
                        )

                return self._step_result_from_runtime(
                    step=step,
                    runtime=runtime,
                    next_action="wait_for_interaction",
                    interaction=runtime.interaction,
                )
            runtime.interaction = None
            runtime.turn_status = None

        invocation_id: Optional[str] = None
        if guard_decision is not None and guard_decision.grant_id:
            invocation_id = self.side_effects.prepare(
                grant_id=guard_decision.grant_id,
                plan_id=str(runtime.plan.plan_id if runtime.plan else ""),
                step_id=step.step_id,
                tool_name=step.tool_name,
                arguments_fingerprint=guard_decision.arguments_fingerprint,
            )

        self._append_trace(
            runtime,
            step,
            event="step_started",
            status="running",
            detail={
                "tool_name": step.tool_name,
                "tool_contract": self.tool_registry.describe_contract(step.tool_name),
                "resolved_input": _safe_compact(resolved_input),
                "started_at": started_at,
            },
        )
        raw_output: Any = None
        normalized_output: Any = None
        attempts = 0
        last_error: Optional[str] = None
        # 已消费授权的副作用调用禁止通用重试；未知结果必须进入 indeterminate。
        max_attempts = 1 if invocation_id else max(int(getattr(step.retry_policy, "max_attempts", 0) or 0), 1)
        while attempts < max_attempts:
            attempts += 1
            runtime.retry_counts[step.step_id] = attempts - 1
            try:
                raw_output = self._invoke_step_tool(step, resolved_input, state, runtime)
            except Exception as exc:  # pragma: no cover - 这里兜住未知工具实现异常
                if invocation_id:
                    self.side_effects.mark_indeterminate(invocation_id, exc)
                last_error = str(exc)
                if attempts < max_attempts and bool(step.tool.can_retry or step.retry_policy):
                    continue
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"tool_execution_failed:{step.step_id}:{last_error}"
                runtime.recovery_strategy = {"type": "retry_or_abort", "reason": "tool_execution_failed", "error": last_error}
                self._append_trace(
                    runtime,
                    step,
                    event="step_failed",
                    status="failed",
                    detail={"failure_reason": "tool_execution_failed", "error": last_error, "resolved_input": _safe_compact(resolved_input), "started_at": started_at, "finished_at": _utcnow()},
                )
                self._sync_runtime_state(state, runtime, current_step=step)
                return self._step_result_from_runtime(step=step, runtime=runtime, next_action="fail", error=runtime.error)
            if isinstance(raw_output, ToolExecutionResult) and not raw_output.ok:
                tool_error = raw_output.error or _make_tool_error(error_code="tool_failed", message="工具执行失败")
                last_error = _tool_error_to_text(tool_error)
                if attempts < max_attempts and (tool_error.retryable or bool(step.tool.can_retry or step.retry_policy)):
                    continue
                # 结构化工具错误仍交给 Observer/Replanner 判断，执行器只保存 envelope 和调度下一节点。
                normalized_output = None
                break
            normalized_output = project_tool_result(step, raw_output) if isinstance(raw_output, ToolExecutionResult) else raw_output
            break

        if step.output_key and step.output_key in runtime.outputs:
            runtime.step_status[step.step_id] = "failed"
            runtime.error = f"duplicate_output_key:{step.output_key}"
            runtime.recovery_strategy = {"type": "abort_with_error", "reason": "duplicate_output_key", "output_key": step.output_key}
            self._append_trace(runtime, step, event="step_failed", status="failed", detail={"failure_reason": "duplicate_output_key", "output_key": step.output_key})
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(step=step, runtime=runtime, next_action="fail", error=runtime.error)

        tool_failed = isinstance(raw_output, ToolExecutionResult) and not raw_output.ok
        if invocation_id:
            self.side_effects.mark_terminal(
                invocation_id,
                succeeded=not tool_failed,
                result_summary=_compact_tool_execution_for_trace(step, raw_output),
                error_code=raw_output.error.error_code if tool_failed and raw_output.error else "",
            )
        candidate_outputs = dict(runtime.outputs, **({step.output_key: normalized_output} if step.output_key and not tool_failed else {}))
        if not tool_failed and any(not evaluate_condition(condition, state, runtime.model_copy(update={"outputs": candidate_outputs})) for condition in list(step.postconditions or [])):
            runtime.step_status[step.step_id] = "failed"
            runtime.error = f"postcondition_failed:{step.step_id}"
            runtime.recovery_strategy = {"type": "abort_with_error", "reason": "postcondition_failed"}
            self._append_trace(runtime, step, event="step_failed", status="failed", detail={"failure_reason": "postcondition_failed", "raw_output": _safe_compact(raw_output)})
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(step=step, runtime=runtime, next_action="fail", error=runtime.error)

        runtime.last_step_output = {
            "step_id": step.step_id,
            "tool_contract": _json_safe(self.tool_registry.describe_contract(step.tool_name)),
            "tool_execution": _json_safe(raw_output) if isinstance(raw_output, ToolExecutionResult) else None,
            "resolved_input": _json_safe(resolved_input),
            "raw_output": _json_safe(_model_to_plain(raw_output.data) if isinstance(raw_output, ToolExecutionResult) else raw_output),
            "normalized_output": _json_safe(normalized_output),
            "started_at": started_at,
            "finished_at": _utcnow(),
        }
        self._sync_runtime_state(state, runtime, current_step=step)
        return self._step_result_from_runtime(step=step, runtime=runtime, next_action="continue", output=normalized_output)

    def observe_current_step(self, runtime: PlanRuntime, state: AgentState, *, allow_interrupt: bool = True) -> StepExecutionResult:
        """只观察当前 step 的工具输出质量，不执行工具、不重规划。"""
        step = self._get_current_step(runtime)
        output_payload = runtime.last_step_output if isinstance(runtime.last_step_output, Mapping) else None
        if step is None or not output_payload:
            runtime.step_status[str(runtime.current_step_id or "unknown")] = "failed"
            runtime.error = runtime.error or "observation_missing_step_output"
            runtime.recovery_strategy = {"type": "abort_with_error", "reason": "observation_missing_step_output"}
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(step=step, runtime=runtime, next_action="fail", error=runtime.error)

        raw_output = output_payload.get("tool_execution") or output_payload.get("raw_output")
        normalized_output = output_payload.get("normalized_output")
        resolved_input = output_payload.get("resolved_input") if isinstance(output_payload.get("resolved_input"), Mapping) else {}
        observation = self.observer.observe(
            step=step,
            resolved_input=resolved_input,
            raw_output=raw_output,
            normalized_output=normalized_output,

            runtime=runtime,
            state=state,
        )
        runtime.last_observation = observation.model_dump()
        self._append_trace(
            runtime,
            step,
            event="step_observed",
            status="running",
            detail={
                "observation_status": observation.status,
                "observation_reason": observation.reason,
                "failure_category": observation.failure_category,
                "severity": observation.severity,
                "recoverable": observation.recoverable,
                "retryable": observation.retryable,
                "requires_user_input": observation.requires_user_input,
                "suggested_recovery_types": list(observation.suggested_recovery_types or []),
                "evidence": _safe_compact(observation.evidence),
                "suggested_action": observation.suggested_action,
                "confidence": observation.confidence,
            },
        )

        if observation.status == "need_confirmation" and _is_paper_target_resolution_step(step):
            payload = _model_to_plain(normalized_output)
            payload = payload if isinstance(payload, Mapping) else {}
            if step.output_key:
                _record_step_output(runtime, step, dict(payload))
            interaction = self.target_selection.build_interaction(step=step, runtime=runtime, payload=payload)
            runtime.interaction = interaction
            runtime.step_status[step.step_id] = "waiting_interaction"
            runtime.turn_status = "waiting_interaction"
            self._append_trace(
                runtime,
                step,
                event="interaction_requested",
                status="waiting_interaction",
                detail={"interaction_id": interaction.interaction_id, "kind": interaction.kind},
            )
            if allow_interrupt:
                resume_payload = interrupt(interaction.model_dump(mode="json"))
                decision = str((resume_payload or {}).get("decision") or "").strip().lower() if isinstance(resume_payload, Mapping) else ""
                if decision == "select" and isinstance((resume_payload or {}).get("selected_candidate"), Mapping):
                    confirmed_output = self.target_selection.materialize(
                        selected_candidate=resume_payload["selected_candidate"],
                        payload=payload,
                    )
                    if step.output_key:
                        _record_step_output(runtime, step, confirmed_output)
                    runtime.interaction = None
                    runtime.step_status[step.step_id] = "success"
                    runtime.turn_status = None
                    runtime.needs_replan = False
                    runtime.last_observation = {
                        "status": "success",
                        "reason": "paper_target_user_selected",
                        "confidence": 1.0,
                    }
                    self._append_trace(
                        runtime,
                        step,
                        event="interaction_resolved",
                        status="success",
                        detail={"interaction_id": interaction.interaction_id, "kind": interaction.kind},
                    )
                    self._sync_runtime_state(state, runtime, current_step=step)
                    return self._step_result_from_runtime(
                        step=step,
                        runtime=runtime,
                        next_action="continue",
                        output=confirmed_output,
                        observation=runtime.last_observation,
                    )
                runtime.interaction = None
                runtime.step_status[step.step_id] = "skipped"
                runtime.turn_status = "failed"
                runtime.error = f"interaction_cancelled:{step.step_id}"
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(
                step=step,
                runtime=runtime,
                next_action="wait_for_interaction" if runtime.interaction else "fail",
                observation=runtime.last_observation,
                interaction=runtime.interaction,
                error=runtime.error,
            )

        if observation.status not in {"success", "partial_success"}:
            if _should_preserve_non_success_observation_output(step, normalized_output):
                _record_step_output(runtime, step, normalized_output)
            runtime.needs_replan = True
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(step=step, runtime=runtime, next_action="replan", observation=runtime.last_observation)


        if step.output_key:
            _record_step_output(runtime, step, normalized_output)
            runtime.final_answer = assemble_final_answer(runtime)

        runtime.step_status[step.step_id] = "success"
        runtime.needs_replan = False
        self.recovery.append_paper_qa_quality_trace(runtime, step, normalized_output)
        self._append_trace(
            runtime,
            step,
            event="step_succeeded",
            status="success",
            detail={
                "tool_name": step.tool_name,
                "tool_contract": self.tool_registry.describe_contract(step.tool_name),
                "tool_execution": _compact_tool_execution_for_trace(step, raw_output),
                "resolved_input": _safe_compact(resolved_input),
                "raw_output": _compact_step_output_for_trace(step, _model_to_plain(raw_output.data) if isinstance(raw_output, ToolExecutionResult) else raw_output),
                "normalized_output": _compact_step_output_for_trace(step, normalized_output),
                "started_at": output_payload.get("started_at"),

                "finished_at": output_payload.get("finished_at") or _utcnow(),
            },
        )
        self._sync_runtime_state(state, runtime, current_step=step)
        return self._step_result_from_runtime(step=step, runtime=runtime, next_action="continue", output=normalized_output, observation=runtime.last_observation)

    def _execute_runtime(self, runtime: PlanRuntime, state: AgentState, *, allow_interrupt: bool) -> AgentTurnResult:
        plan = runtime.plan
        if plan is None:
            runtime.error = "missing_plan"
            runtime.turn_status = "failed"
            return AgentTurnResult(status="failed", error=runtime.error, plan=plan, outputs=dict(runtime.outputs), trace=list(runtime.trace), runtime=runtime)

        while True:
            step_result = self.execute_next_step(runtime, state, allow_interrupt=allow_interrupt, auto_replan=True)
            if step_result.turn_result is not None:
                return step_result.turn_result
            if step_result.next_action not in {"continue", "replan"}:
                break

        return self._build_turn_result(runtime)

    def _find_executable_step(self, runtime: PlanRuntime, state: AgentState) -> Optional[PlanStep]:
        plan = runtime.plan
        if plan is None:
            return None
        for step in list(plan.steps or []):
            status = runtime.step_status.get(step.step_id, "pending")
            if status != "pending":
                continue
            dependency_statuses = [runtime.step_status.get(step_id, "pending") for step_id in list(step.depends_on or [])]
            if any(status not in {"success", "skipped"} for status in dependency_statuses):
                continue
            if not evaluate_condition(step.condition, state, runtime):
                runtime.step_status[step.step_id] = "skipped"
                self._append_trace(runtime, step, event="condition_skipped", status="skipped", detail={"condition": step.condition.model_dump() if step.condition else None})
                continue
            if any(not evaluate_condition(condition, state, runtime) for condition in list(step.preconditions or [])):
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"precondition_failed:{step.step_id}"
                self._append_trace(runtime, step, event="precondition_failed", status="failed", detail={"preconditions": [condition.model_dump() for condition in list(step.preconditions or [])]})
                continue
            return step
        return None

    def _execute_step(self, step: PlanStep, runtime: PlanRuntime, state: AgentState, *, allow_interrupt: bool, auto_replan: bool = True) -> Optional[AgentTurnResult]:
        """兼容入口：旧连续执行循环的单步执行。

        重构后这里不再内联工具执行/观察/重规划，而是委托给同一套显式 step 执行内核：
        execute_current_step_tool -> observe_current_step -> replan_after_observation。
        这样新旧路径共享唯一的工具调用、确认、副作用复用与观察重规划逻辑，避免双份实现漂移。

        返回语义保持与旧实现一致：返回 AgentTurnResult 表示本轮已收口（等待确认/失败/兜底），
        返回 None 表示本步已处理完（成功、低质量但未触发兜底、或 auto_replan=False 时暂停），
        由调用方继续推进后续步骤。
        """
        # 显式 step 执行内核以 runtime.current_step_id 定位当前步；兼容入口已经选中 step，先对齐定位。
        runtime.current_step_id = step.step_id

        exec_result = self.execute_current_step_tool(runtime, state, allow_interrupt=allow_interrupt)
        if exec_result.turn_result is not None:
            # 等待确认/拒绝收口等已经产出终态 turn。
            return exec_result.turn_result
        if exec_result.next_action == "wait_for_confirmation":
            return self._build_turn_result(runtime)
        if exec_result.next_action == "fail":
            # 输入缺失、工具执行异常、duplicate_output_key、postcondition 失败：与旧实现一致返回 None，
            # 由上层循环在没有可执行 step 后统一收口为 failed/fallback。
            return None
        if exec_result.next_action != "continue":
            return None

        # continue 之后，仅当当前 step 仍处于 running 才需要观察：
        # reused 副作用输出、确认桥接、目标解析 resume 等已在执行内核内置为非 running 终态，
        # 与旧 _execute_step “复用输出直接完成、不再观察” 的语义保持一致。
        if runtime.step_status.get(step.step_id) != "running":
            return None

        observe_result = self.observe_current_step(runtime, state, allow_interrupt=allow_interrupt)
        if observe_result.turn_result is not None:
            return observe_result.turn_result
        if observe_result.next_action == "wait_for_confirmation":
            return self._build_turn_result(runtime)
        if observe_result.next_action != "replan":
            # success / fail 等：本步已收尾，交回上层推进。
            return None

        # 低质量观察 -> 需要重规划。
        if not auto_replan:
            # 新 LangGraph 执行环把重规划显式暴露成图节点，这里只记录观察结果并暂停本步。
            return None
        return self.replan_after_observation(runtime, state)

    def replan_after_observation(self, runtime: PlanRuntime, state: AgentState) -> Optional[AgentTurnResult]:
        """根据 runtime 中最近一次 observation 显式执行重规划。

        这是 LangGraph `replan` 节点使用的入口：execute_step 只负责运行工具并留下观察结果，
        是否 patch plan、fallback 或失败由这个节点单独决定，避免重规划继续藏在执行黑盒里。
        """
        step = self._get_current_step(runtime)
        observation_payload = runtime.last_observation if isinstance(runtime.last_observation, Mapping) else None
        if step is None or not observation_payload:
            runtime.error = runtime.error or "replan_missing_observation"
            runtime.needs_replan = False
            runtime.recovery_strategy = {"type": "abort_with_error", "reason": "replan_missing_observation"}
            return self._build_turn_result(runtime)

        observation = ObservationResult.model_validate(dict(observation_payload))
        step_output = runtime.last_step_output if isinstance(runtime.last_step_output, Mapping) else {}
        # 低质量 observation 不会写入 step_succeeded trace，replan 必须优先消费 execute_step 留下的真实工具输出。
        raw_output = step_output.get("tool_execution") or step_output.get("raw_output")
        normalized_output = step_output.get("normalized_output")
        if raw_output is None:
            raw_output = self._extract_trace_value(runtime, step.step_id, "step_succeeded", "raw_output")
        if normalized_output is None:
            normalized_output = self._extract_trace_value(runtime, step.step_id, "step_succeeded", "normalized_output")
        return self.recovery.handle_observation_replan(
            step=step,
            runtime=runtime,
            state=state,
            observation=observation,
            raw_output=raw_output,
            normalized_output=normalized_output,
        )

    def _resolve_input_bindings(self, step: PlanStep, runtime: PlanRuntime, state: AgentState) -> Tuple[Dict[str, Any], Optional[str]]:
        return self.input_bindings.resolve(step, runtime, state)

    def _invoke_step_tool(self, step: PlanStep, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime) -> Any:
        return self.tool_execution.execute(step=step, arguments=resolved_input, state=state)

    def _append_trace(
        self,
        runtime: PlanRuntime,
        step: PlanStep,
        *,
        event: str,
        status: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """统一委托 trace 写入，执行器本身不再构造领域事件模型。"""
        self.recovery.append_trace(runtime, step, event=event, status=status, detail=detail)

    def _build_step_execution_result(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        previous_outputs: Mapping[str, Any],
        previous_trace_len: int,
        previous_error: Optional[str],
        turn_result: Optional[AgentTurnResult],
    ) -> StepExecutionResult:
        """把一次 step 执行后的 runtime 变化归一成单步结果。

        执行器内部仍会维护 runtime，但 LangGraph 节点不应该反向解析 runtime diff；
        这里把本步 output、observation、确认态和下一步动作显式算出来，作为后续节点边界。
        """
        status = runtime.step_status.get(step.step_id)
        output = runtime.outputs.get(step.output_key) if step.output_key else None
        if step.output_key and step.output_key not in runtime.outputs and step.output_key in previous_outputs:
            output = previous_outputs.get(step.output_key)
        new_traces = list(runtime.trace or [])[previous_trace_len:]
        observation = self._extract_observation_from_traces(new_traces) or runtime.last_observation
        next_action = self._infer_next_action(
            runtime=runtime,
            status=status,
            observation=observation,
            previous_error=previous_error,
            turn_result=turn_result,
        )
        return StepExecutionResult(
            step_id=step.step_id,
            step_status=status,
            output_key=step.output_key,
            output=output,
            error=runtime.error if runtime.error != previous_error else None,
            observation=observation,
            next_action=next_action,
            interaction=runtime.interaction,
            runtime_patch=self._build_runtime_patch(runtime, current_step=step),
            turn_result=turn_result,
        )

    def _step_result_from_runtime(
        self,
        *,
        step: Optional[PlanStep],
        runtime: PlanRuntime,
        next_action: str,
        output: Any = None,
        observation: Optional[Mapping[str, Any]] = None,
        interaction: Optional[Any] = None,
        error: Optional[str] = None,
        turn_result: Optional[AgentTurnResult] = None,
    ) -> StepExecutionResult:
        """把显式图节点中的 runtime 当前状态包装成 StepExecutionResult。"""
        return StepExecutionResult(
            step_id=step.step_id if step else runtime.current_step_id,
            step_status=runtime.step_status.get(step.step_id) if step else None,
            output_key=step.output_key if step else None,
            output=output,
            error=error,
            observation=dict(observation or {}) if observation else None,
            next_action=next_action,  # type: ignore[arg-type]
            interaction=interaction,
            runtime_patch=self._build_runtime_patch(runtime, current_step=step),
            turn_result=turn_result,
        )

    def _extract_observation_from_traces(self, traces: Sequence[ExecutionTrace]) -> Optional[Dict[str, Any]]:
        """从本步新增 trace 中提取最近 observation 摘要，避免调用方扫描完整 trace。"""
        for trace in reversed(list(traces or [])):
            if trace.event != "step_observed":
                continue
            detail = dict(trace.detail or {})
            return {
                "status": detail.get("observation_status"),
                "observation_signal": detail.get("observation_signal"),
                "reason": detail.get("observation_reason"),
                "confidence": detail.get("confidence"),
                "failure_category": detail.get("failure_category"),
                "severity": detail.get("severity"),
                "recoverable": detail.get("recoverable"),
                "retryable": detail.get("retryable"),
                "requires_user_input": detail.get("requires_user_input"),
                "suggested_recovery_types": list(detail.get("suggested_recovery_types") or []),
                "suggested_action": detail.get("suggested_action"),
                "evidence": dict(detail.get("evidence") or {}),
            }
        return None

    def _get_current_step(self, runtime: PlanRuntime) -> Optional[PlanStep]:
        """从 runtime.current_step_id 找回当前 step，供显式 observe/replan/finalize 节点使用。"""
        current_step_id = str(runtime.current_step_id or "").strip()
        if not current_step_id or runtime.plan is None:
            return None
        for step in list(runtime.plan.steps or []):
            if step.step_id == current_step_id:
                return step

        return None

    def _extract_trace_value(self, runtime: PlanRuntime, step_id: str, event: str, detail_key: str) -> Any:
        """从 trace 中提取本步最近一次紧凑记录，仅作为 replan debug 输入。"""
        for trace in reversed(list(runtime.trace or [])):
            if trace.step_id == step_id and trace.event == event:

                return dict(trace.detail or {}).get(detail_key)
        return None

    def _infer_next_action(
        self,
        *,
        runtime: PlanRuntime,
        status: Optional[str],
        observation: Optional[Mapping[str, Any]],
        previous_error: Optional[str],
        turn_result: Optional[AgentTurnResult],
    ) -> str:
        """把 runtime 状态归一成上层可路由的下一步动作。"""
        if turn_result is not None:
            if turn_result.status == "waiting_interaction":
                return "wait_for_interaction"

            if turn_result.status in {"failed", "fallback"}:
                return "fail"
            return "finish"
        if runtime.interaction is not None:
            return "wait_for_interaction"
        if status == "failed" or (runtime.error and runtime.error != previous_error):
            return "fail"
        if runtime.needs_replan or (observation and observation.get("status") not in {None, "success", "partial_success"}):
            return "replan"
        return "continue"

    def _build_runtime_patch(self, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> Dict[str, Any]:
        return self.state_projector.build_patch(runtime, current_step=current_step)

    def _build_agent_runtime_state(self, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> AgentRuntimeState:
        return self.state_projector.build_runtime_state(runtime, current_step=current_step)

    def _sync_runtime_state(self, state: AgentState, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> None:
        self.state_projector.apply(state, runtime, current_step=current_step)

    def _finalize_blocked_pending_steps(self, runtime: PlanRuntime) -> None:
        plan = runtime.plan
        if plan is None:
            return
        for step in list(plan.steps or []):
            if runtime.step_status.get(step.step_id) == "pending":
                runtime.step_status[step.step_id] = "skipped"
                self._append_trace(runtime, step, event="dependency_blocked", status="skipped", detail={"depends_on": list(step.depends_on or [])})

    def _build_turn_result(self, runtime: PlanRuntime) -> AgentTurnResult:
        self._sync_plan_step_status(runtime)
        goal_type = str((runtime.goal.goal_type if runtime.goal else "") or "").strip()
        if runtime.interaction:
            runtime.turn_status = "waiting_interaction"
            return AgentTurnResult(
                status="waiting_interaction",
                final_answer=runtime.final_answer,
                plan=runtime.plan,
                outputs=dict(runtime.outputs),
                trace=list(runtime.trace),
                interaction=runtime.interaction,
                error=runtime.error,
                runtime=runtime,
            )
        if any(status == "failed" for status in list(runtime.step_status.values())) and not runtime.final_answer:
            runtime.turn_status = "failed"
            return AgentTurnResult(status="failed", final_answer=runtime.final_answer, plan=runtime.plan, outputs=dict(runtime.outputs), trace=list(runtime.trace), error=runtime.error, runtime=runtime)
        if goal_type == "unclear":
            runtime.turn_status = "need_clarification"
            return AgentTurnResult(status="need_clarification", final_answer=runtime.final_answer, plan=runtime.plan, outputs=dict(runtime.outputs), trace=list(runtime.trace), error=runtime.error, runtime=runtime)
        if goal_type == "unsupported":
            runtime.turn_status = "fallback"
            return AgentTurnResult(status="fallback", final_answer=runtime.final_answer, plan=runtime.plan, outputs=dict(runtime.outputs), trace=list(runtime.trace), error=runtime.error, runtime=runtime)
        runtime.turn_status = "success"
        return AgentTurnResult(status="success", final_answer=runtime.final_answer, plan=runtime.plan, outputs=dict(runtime.outputs), trace=list(runtime.trace), error=runtime.error, runtime=runtime)

    def _sync_plan_step_status(self, runtime: PlanRuntime) -> None:
        self.state_projector.sync_plan_step_status(runtime)


def run_agent_turn(state: AgentState, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> AgentTurnResult:
    """统一 Agent 单轮入口：通过 planner 门面拿到已校验计划后执行。

    planner 门面优先使用 LLM/规则型 Tool-Aware 路径；legacy 模板只在主 planner 不可用时兜底，
    执行器这里只消费可信 ExecutablePlan，避免运行期再关心草稿来源。
    """
    goal, plan, planning_debug = build_executable_plan(state, tool_registry=tool_registry)
    state.goal = goal
    state.execution_plan = plan
    state.debug = dict(state.debug or {})
    state.debug["planner"] = planning_debug
    _log_plan_done(state, goal, plan, planning_debug)
    result = PlanExecutor(tool_registry=tool_registry).execute(plan, state)
    if result.runtime is not None:
        state.plan_runtime = result.runtime
    return result


def run_agent_turn_in_graph(state: AgentState, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> AgentTurnResult:
    """图内执行入口：和普通入口共享 planner 门面，但允许 interrupt/resume 确认链路。"""
    goal, plan, planning_debug = build_executable_plan(state, tool_registry=tool_registry)
    state.goal = goal
    state.execution_plan = plan
    state.debug = dict(state.debug or {})
    state.debug["planner"] = planning_debug
    _log_plan_done(state, goal, plan, planning_debug)
    runtime = build_plan_runtime(state, goal=plan.goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
    runtime.outputs = {}
    runtime.trace = []
    runtime.retry_counts = {}
    runtime.replan_counts = {}
    runtime.step_replan_counts = {}
    result = PlanExecutor(tool_registry=tool_registry)._execute_runtime(runtime, state, allow_interrupt=True)
    if result.runtime is not None:
        state.plan_runtime = result.runtime
    return result


def execute_executable_plan(state: AgentState, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> AgentTurnResult:
    """便捷入口：从 AgentState 先构造计划，再立刻执行。"""
    return run_agent_turn(state, tool_registry=tool_registry)


__all__ = ["PlanExecutor", "run_agent_turn", "run_agent_turn_in_graph", "execute_executable_plan"]
