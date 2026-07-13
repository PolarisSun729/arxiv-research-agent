from typing import Any, Callable, Dict, Optional

from ..schemas import AgentRuntimeState, PlanRuntime, PlanStep
from ..state import AgentState


def _json_safe(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"))
        except TypeError:
            return _json_safe(model_dump())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


class RuntimeStateProjector:
    """集中维护运行态投影，防止执行组件分别修改多个 AgentState 副本。"""

    def __init__(self, *, execution_path_builder: Callable[..., Dict[str, Any]]) -> None:
        self.execution_path_builder = execution_path_builder

    def build_runtime_state(self, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> AgentRuntimeState:
        step_index: Optional[int] = None
        if current_step is not None and runtime.plan is not None:
            step_index = next(
                (index for index, step in enumerate(runtime.plan.steps or []) if step.step_id == current_step.step_id),
                None,
            )
        statuses = list((runtime.step_status or {}).values())
        is_finished = bool(runtime.turn_status) or bool(
            statuses and all(
                status in {
                    "success", "failed", "skipped", "waiting_confirmation",
                    "waiting_interaction", "waiting_background_job",
                }
                for status in statuses
            )
        )
        return AgentRuntimeState(
            request_state=_json_safe(runtime.state or {}),
            goal=runtime.goal,
            plan=runtime.plan,
            current_step_id=current_step.step_id if current_step else None,
            current_step_index=step_index,
            step_status=dict(runtime.step_status or {}),
            outputs=_json_safe(runtime.outputs or {}),
            last_observation=_json_safe(runtime.last_observation) if runtime.last_observation else None,
            last_step_output=_json_safe(runtime.last_step_output) if runtime.last_step_output else None,
            trace=list(runtime.trace or []),
            retry_counts=dict(runtime.retry_counts or {}),
            replan_counts=dict(runtime.replan_counts or {}),
            step_replan_counts=dict(runtime.step_replan_counts or {}),
            interaction=runtime.interaction,
            needs_replan=bool(runtime.needs_replan),
            is_finished=is_finished,
            failure_reason=runtime.error,
            recovery_strategy=_json_safe(runtime.recovery_strategy) if runtime.recovery_strategy else None,
            turn_status=runtime.turn_status,
            final_answer=runtime.final_answer,
        )

    def build_patch(self, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> Dict[str, Any]:
        runtime_state = self.build_runtime_state(runtime, current_step=current_step)
        return {
            "runtime_state": _json_safe(runtime_state),
            "plan_runtime": _json_safe(runtime),
            "current_step_id": runtime_state.current_step_id,
            "step_status": dict(runtime.step_status or {}),
            "outputs": _json_safe(runtime.outputs or {}),
            "last_observation": _json_safe(runtime.last_observation),
            "last_step_output": _json_safe(runtime.last_step_output),
            "interaction": _json_safe(runtime.interaction) if runtime.interaction else None,
            "needs_replan": bool(runtime.needs_replan),
            "is_finished": bool(runtime_state.is_finished),
            "failure_reason": runtime.error,
            "recovery_strategy": _json_safe(runtime.recovery_strategy or {}),
            "execution_path": self.execution_path_builder(runtime, current_step_id=runtime_state.current_step_id),
        }

    def apply(self, state: AgentState, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> None:
        runtime.current_step_id = current_step.step_id if current_step else None
        runtime.current_step_index = None
        if current_step is not None and runtime.plan is not None:
            runtime.current_step_index = next(
                (index for index, step in enumerate(runtime.plan.steps or []) if step.step_id == current_step.step_id),
                None,
            )
        state.plan_runtime = runtime
        state.runtime_state = self.build_runtime_state(runtime, current_step=current_step)
        state.debug = dict(state.debug or {})
        state.debug["execution_path"] = self.execution_path_builder(
            runtime,
            current_step_id=current_step.step_id if current_step else runtime.current_step_id,
        )

    @staticmethod
    def sync_plan_step_status(runtime: PlanRuntime) -> None:
        if runtime.plan is None:
            return
        runtime.plan.steps = [
            step.model_copy(update={"status": runtime.step_status.get(step.step_id, step.status)})
            for step in list(runtime.plan.steps or [])
        ]
