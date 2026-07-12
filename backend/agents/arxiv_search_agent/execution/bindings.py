from typing import Any, Dict, Mapping, Optional, Tuple

from ..schemas import ExecutablePlan, Goal, PlanRuntime, PlanStep, StepCondition
from ..state import AgentState


def _path_value(value: Any, path: Optional[str]) -> Any:
    current = value
    for part in [item for item in str(path or "").split(".") if item]:
        if current is None:
            return None
        current = current.get(part) if isinstance(current, Mapping) else getattr(current, part, None)
    return current


def _state_mapping(state: AgentState) -> Dict[str, Any]:
    return state.model_dump() if hasattr(state, "model_dump") else dict(state)


def _state_value(state: AgentState, key: Optional[str]) -> Any:
    if key in {"user_request", "original_question", "question"}:
        return state.message
    if key and hasattr(state, key):
        return getattr(state, key)
    mapping = _state_mapping(state)
    if key in mapping:
        return mapping.get(key)
    context = state.context if isinstance(state.context, Mapping) else {}
    return context.get(key) if key else None


def _step_output(runtime: PlanRuntime, plan: ExecutablePlan, step_id: Optional[str]) -> Any:
    for plan_step in list(plan.steps or []):
        if plan_step.step_id == str(step_id or "").strip() and plan_step.output_key:
            return runtime.outputs.get(plan_step.output_key)
    return None


def step_output(runtime: PlanRuntime, plan: ExecutablePlan, step_id: Optional[str]) -> Any:
    """按计划步骤读取其声明的输出，供其他执行组件复用同一解析规则。"""

    return _step_output(runtime, plan, step_id)


def existing_step_output(runtime: PlanRuntime, step: PlanStep) -> Any:
    return runtime.outputs.get(step.output_key) if step.output_key else None


def can_reuse_side_effect_output(runtime: PlanRuntime, step: PlanStep) -> bool:
    """副作用已有结果时必须复用，避免恢复或重试再次触发外部操作。"""

    return bool(
        step.side_effect_level in {"persistent_write", "external_call"}
        and step.output_key
        and runtime.outputs.get(step.output_key) not in (None, "", [], {})
    )


def evaluate_condition(condition: Optional[StepCondition], state: AgentState, runtime: PlanRuntime) -> bool:
    if condition is None or condition.condition_type == "always":
        return True
    if condition.condition_type in {"field_exists", "field_equals"}:
        path = condition.field_path or ""
        root, _, remaining = path.partition(".")
        roots = {
            "state": _state_mapping(state),
            "goal": runtime.goal.model_dump() if runtime.goal else {},
            "outputs": runtime.outputs,
            "context": state.context if isinstance(state.context, Mapping) else {},
        }
        value = _path_value(roots.get(root, _state_mapping(state)), remaining if root in roots else path)
    elif condition.condition_type == "step_output_exists":
        value = _step_output(runtime, runtime.plan or ExecutablePlan(goal=Goal(), plan_id="unknown"), condition.step_id)
    elif condition.condition_type == "step_status_is":
        value = runtime.step_status.get(str(condition.step_id or "").strip())
    else:
        value = True
    result = (
        value not in (None, "", [], {})
        if condition.condition_type in {"field_exists", "step_output_exists"}
        else value == condition.expected_value
        if condition.condition_type in {"field_equals", "step_status_is"}
        else True
    )
    return not result if condition.negate else result


class InputBindingResolver:
    """解析计划声明的输入来源，返回参数或首个缺失字段，不修改共享状态。"""

    def resolve(self, step: PlanStep, runtime: PlanRuntime, state: AgentState) -> Tuple[Dict[str, Any], Optional[str]]:
        resolved: Dict[str, Any] = {}
        for binding in list(step.input_bindings or []):
            if binding.source_type == "state":
                value = _state_value(state, binding.source_key or binding.input_key)
            elif binding.source_type == "context":
                value = (state.context if isinstance(state.context, Mapping) else {}).get(binding.source_key or binding.input_key)
            elif binding.source_type == "goal":
                value = _path_value(runtime.goal.model_dump() if runtime.goal else {}, binding.source_key or binding.input_key)
            elif binding.source_type == "search_spec":
                value = (
                    state.search_spec
                    if state.search_spec is not None and not binding.source_key
                    else getattr(state.search_spec, binding.source_key, None)
                    if state.search_spec is not None
                    else None
                )
            elif binding.source_type == "step_output":
                value = _step_output(runtime, runtime.plan or ExecutablePlan(goal=Goal(), plan_id="unknown"), binding.step_id)
            else:
                value = binding.value if binding.source_type == "literal" else None
            if value in (None, "", [], {}) and binding.required:
                return resolved, binding.input_key
            if value not in (None, "", [], {}) or binding.required:
                resolved[binding.input_key] = value
        return resolved, None
