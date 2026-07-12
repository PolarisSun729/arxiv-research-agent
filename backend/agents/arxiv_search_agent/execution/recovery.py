from __future__ import annotations

from typing import Any, Dict, Optional

from ..fallbacks import build_fallback_record
from ..replanner import Replanner
from ..response_assembler import record_recovery_fallback
from ..schemas import AgentTurnResult, ExecutionTrace, ObservationResult, PlanRuntime, PlanStep
from ..state import AgentState
from .traces import (
    _build_paper_qa_quality_trace,
    _latest_paper_qa_repair_trace,
    _record_step_output,
    _safe_compact,
)


class ExecutionRecoveryCoordinator:
    """负责 observation 后的 replan、fallback 和业务质量决策记录。"""

    def __init__(self, *, replanner: Replanner) -> None:
        self.replanner = replanner

    def handle_observation_replan(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        observation: ObservationResult,
        raw_output: Any,
        normalized_output: Any,
    ) -> Optional[AgentTurnResult]:
        """质量问题优先走重规划，而不是把“工具成功但结果差”误判成完成。"""
        plan = runtime.plan
        goal = runtime.goal
        if plan is None or goal is None:
            runtime.step_status[step.step_id] = "failed"
            runtime.error = f"observation_failed_without_plan:{step.step_id}"
            return None

        replan_decision = self.replanner.replan(
            goal=goal,
            current_plan=plan,
            runtime=runtime,
            failed_or_low_quality_step=step,
            observation_result=observation,
            state=state,
        )
        if replan_decision.updated_plan is not None and replan_decision.updated_runtime is not None:
            runtime.plan = replan_decision.updated_plan
            runtime.replan_counts = dict(replan_decision.updated_runtime.replan_counts or {})
            runtime.step_replan_counts = dict(replan_decision.updated_runtime.step_replan_counts or {})
            runtime.trace = list(replan_decision.updated_runtime.trace or [])
            runtime.step_status = dict(replan_decision.updated_runtime.step_status or runtime.step_status)
            runtime.interaction = replan_decision.updated_runtime.interaction
            runtime.needs_replan = False
            runtime.recovery_strategy = {"type": "patch_plan", "reason": observation.reason or observation.status}
            # 低质量 final_answer 只作为触发重规划的观察对象，不能提前污染最终答案。
            if step.output_key and step.output_key not in runtime.outputs and step.output_key != "final_answer":
                _record_step_output(runtime, step, normalized_output)
            runtime.step_status[step.step_id] = "success"
            self.append_trace(
                runtime,
                step,
                event="step_replanned",
                status="success",
                detail={
                    "observation_status": observation.status,
                    "observation_signal": observation.observation_signal,
                    "observation_reason": observation.reason,
                    "failure_category": observation.failure_category,
                    "raw_output": _safe_compact(raw_output),
                },
            )
            return None

        if replan_decision.updated_runtime is not None:
            runtime.trace = list(replan_decision.updated_runtime.trace or runtime.trace)
            runtime.replan_counts = dict(replan_decision.updated_runtime.replan_counts or runtime.replan_counts)
            runtime.step_replan_counts = dict(replan_decision.updated_runtime.step_replan_counts or runtime.step_replan_counts)
        runtime.step_status[step.step_id] = "failed"
        fallback_reason = str(replan_decision.fallback_reason or observation.reason or observation.status)
        existing_fallback_record = dict(replan_decision.fallback_record or {})
        fallback_record = dict(existing_fallback_record)
        # replan 阶段可能已经给出一个 fallback_record，但最终 turn 是在 executor
        # 收口为 fallback_answer。这里统一再归一化一次，确保最终 trace/输出表达的是
        # “本轮最终如何结束”，而不是保留上一层的临时 reason。
        if not fallback_record or fallback_record.get("code") == fallback_reason:
            fallback_record = build_fallback_record(
                fallback_reason,
                stage="executor",
                source="PlanExecutor._handle_observation_replan",
                detail={"step_id": step.step_id, "tool_name": step.tool_name},
            )
        runtime.error = f"fallback:{step.step_id}:{fallback_reason}"
        runtime.needs_replan = False
        runtime.recovery_strategy = {"type": "fallback_answer", "reason": fallback_reason, "fallback_record": fallback_record}
        record_recovery_fallback(runtime, fallback_reason=fallback_reason, fallback_record=fallback_record)
        self.append_trace(
            runtime,
            step,
            event="replan_fallback",
            status="failed",
            detail={
                "observation_status": observation.status,
                "observation_reason": observation.reason,
                "failure_category": observation.failure_category,
                "fallback_reason": fallback_reason,
                "fallback_record": fallback_record,
                "paper_qa_final_decision": self._build_paper_qa_fallback_decision(runtime, step, observation, fallback_reason),
            },
        )
        runtime.turn_status = "fallback"
        return AgentTurnResult(
            status="fallback",
            final_answer=runtime.final_answer,
            plan=runtime.plan,
            outputs=dict(runtime.outputs),
            trace=list(runtime.trace),
            error=runtime.error,
            runtime=runtime,
        )

    def append_trace(self, runtime: PlanRuntime, step: PlanStep, *, event: str, status: str, detail: Optional[Dict[str, Any]] = None) -> None:
        runtime.trace.append(
            ExecutionTrace(
                step_id=step.step_id,
                event=event,
                status=status,  # type: ignore[arg-type]
                detail=detail or {},
            )
        )

    def append_paper_qa_quality_trace(self, runtime: PlanRuntime, step: PlanStep, normalized_output: Any) -> None:
        """质量门完成时单独记录 Paper QA 决策闭环，便于验收脚本无需解析通用 trace。"""
        quality_trace = _build_paper_qa_quality_trace(runtime, step, normalized_output)
        if not quality_trace:
            return
        self.append_trace(
            runtime,
            step,
            event="paper_qa_quality_decision",
            status="success",
            detail={"paper_qa_quality_trace": quality_trace},
        )

    def _build_paper_qa_fallback_decision(
        self,
        runtime: PlanRuntime,
        step: PlanStep,
        observation: ObservationResult,
        fallback_reason: str,
    ) -> Dict[str, Any]:
        """把 Paper QA 兜底原因显式化，避免证据不足被展示成普通工具失败。"""
        if step.tool_name not in {"answer_paper_question", "assess_paper_qa_quality"}:
            return {}
        repair_trace = _latest_paper_qa_repair_trace(runtime)
        return {
            "final_decision": "fallback",
            "final_decision_reason": fallback_reason,
            "observation_status": observation.status,
            "observation_reason": observation.reason,
            "failure_category": observation.failure_category,
            "selected_repair_actions": list(repair_trace.get("selected_repair_actions") or []),
            "repair_attempted": bool(repair_trace),
            "max_repair_limit_triggered": fallback_reason.startswith("replan_limit_exceeded:"),
        }

