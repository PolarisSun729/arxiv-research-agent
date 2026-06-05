"""AgentState 压缩、执行轨迹追加和计划运行态维护工具。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..schemas import AgentStep, ArxivSearchSpec, ExecutablePlan, ExecutionTrace, PlanRuntime, PlanStep
from ..state import AgentState


_PLAN_ACTIVE_STATUSES = {"pending", "running"}
_PLAN_TERMINAL_STATUSES = {"success", "failed", "skipped", "waiting_confirmation"}
_STATUS_ALIAS = {
    "in_progress": "running",
    "completed": "success",
}
_LEGACY_STEP_ID_ALIAS = {
    "goal_interpretation": "normalize_request",
    "search_execution": "search_arxiv",
    "result_validation": "validate_search_result",
    "personalization": "personalize_results",
    "response_synthesis": "synthesize_search_answer",
    "paper_resolution": "resolve_paper",
    "qa_index_check": "check_qa_index",
    "confirmation_gate": "confirm_parse_if_needed",
    "paper_response": "answer_paper_request",
    "preference_update": "mutate_preference",
    "memory_sync": "sync_memory",
    "profile_loading": "load_profile",
    "recommendation_generation": "recommend_papers",
    "recommendation_explanation": "answer_recommendation",
    "action_resolution": "resolve_reading_list_action",
    "state_update": "apply_reading_list_action",
    "ambiguity_analysis": "identify_missing_information",
    "clarification_response": "ask_for_clarification",
    "capability_check": "check_capability_boundary",
    "fallback_response": "answer_unsupported_request",
}


def _normalize_plan_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return _STATUS_ALIAS.get(normalized, normalized or "pending")


def _resolve_step_selector(step_type: Optional[str]) -> Optional[str]:
    normalized = str(step_type or "").strip()
    if not normalized:
        return None
    return _LEGACY_STEP_ID_ALIAS.get(normalized, normalized)


def _get_plan_steps(state: AgentState) -> List[PlanStep]:
    plan = state.execution_plan
    if not isinstance(plan, ExecutablePlan):
        return []
    return list(plan.steps or [])


def _replace_plan_steps(state: AgentState, steps: List[PlanStep]) -> AgentState:
    next_state = state.model_copy(deep=True)
    if not isinstance(next_state.execution_plan, ExecutablePlan):
        return next_state
    next_state.execution_plan = next_state.execution_plan.model_copy(update={"steps": steps})
    return next_state


def _ensure_plan_runtime(state: AgentState) -> PlanRuntime:
    if isinstance(state.plan_runtime, PlanRuntime):
        return state.plan_runtime
    return PlanRuntime(
        state={},
        goal=state.goal,
        plan=state.execution_plan,
        pending_confirmation=state.pending_action if isinstance(state.pending_action, Mapping) else None,
    )


def _compact_search_spec(spec: Optional[ArxivSearchSpec]) -> Dict[str, Any]:
    """把搜索规格压缩成适合 debug 和轨迹输出的轻量字典。"""
    if spec is None:
        return {}
    payload = {
        "intent": spec.intent,
        "query": spec.query,
        "title_query": spec.title_query,
        "abstract_query": spec.abstract_query,
        "categories": list(spec.categories or []),
        "submitted_days_ago": spec.submitted_days_ago,
        "max_results": spec.max_results,
        "sort_by": spec.sort_by,
        "sort_order": spec.sort_order,
        "field_operator": spec.field_operator,
        "category_operator": spec.category_operator,
        "reasoning_summary": spec.reasoning_summary,
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _compact_paper_summaries(papers: Sequence[Mapping[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    """只保留少量论文摘要字段，避免把完整论文对象写进轨迹。"""
    summaries: List[Dict[str, Any]] = []
    for paper in list(papers or [])[: max(0, limit)]:
        if not isinstance(paper, Mapping):
            continue
        summary: Dict[str, Any] = {}
        for key in ("arxiv_id", "title", "published", "primary_category", "score", "rank"):
            value = paper.get(key)
            if value not in (None, ""):
                summary[key] = value
        if summary:
            summaries.append(summary)
    return summaries


def _append_step(
    state: AgentState,
    *,
    step: str,
    status: str,
    action: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> AgentState:
    """统一追加节点级执行轨迹，保证前端和日志消费格式稳定。"""
    next_state = state.model_copy(deep=True)
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step=step,
            status=status,
            action=action,
            inputs=inputs or {},
            outputs=outputs or {},
            error=error,
        )
    ]
    return next_state


def _update_execution_plan_step(
    state: AgentState,
    *,
    step_id: Optional[str] = None,
    step_type: Optional[str] = None,
    status: Optional[str] = None,
) -> AgentState:
    """按 step_id 或 action_type 更新计划步骤状态，并同步回写 plan_runtime。"""
    normalized_status = _normalize_plan_status(status)
    if not status:
        return state.model_copy(deep=True)

    next_state = state.model_copy(deep=True)
    updated_steps: List[PlanStep] = []
    matched = False
    resolved_step_selector = _resolve_step_selector(step_type)
    for plan_step in _get_plan_steps(next_state):
        current_step_id = str(plan_step.step_id or "").strip()
        current_action_type = str(plan_step.action_type or "").strip()
        is_target = bool(step_id and current_step_id == str(step_id).strip())
        if not is_target and resolved_step_selector:
            is_target = current_action_type == resolved_step_selector or current_step_id == resolved_step_selector
        if is_target:
            matched = True
            updated_steps.append(plan_step.model_copy(update={"status": normalized_status}))
        else:
            updated_steps.append(plan_step)

    if matched:
        next_state = _replace_plan_steps(next_state, updated_steps)
    return _refresh_execution_plan_runtime(next_state)


def _get_execution_plan_step(
    state: AgentState,
    *,
    step_id: Optional[str] = None,
    step_type: Optional[str] = None,
) -> Optional[PlanStep]:
    """按 step_id 或 action_type 获取计划中的某一步。"""
    resolved_step_selector = _resolve_step_selector(step_type)
    for plan_step in _get_plan_steps(state):
        current_step_id = str(plan_step.step_id or "").strip()
        current_action_type = str(plan_step.action_type or "").strip()
        if step_id and current_step_id == str(step_id).strip():
            return plan_step
        if resolved_step_selector and (current_action_type == resolved_step_selector or current_step_id == resolved_step_selector):
            return plan_step
    return None


def _get_next_incomplete_plan_step(state: AgentState) -> Optional[PlanStep]:
    """返回第一个未进入终态的计划步骤。"""
    for plan_step in _get_plan_steps(state):
        if _normalize_plan_status(plan_step.status) not in {"success", "skipped"}:
            return plan_step
    return None


def _get_blocking_dependency_ids(state: AgentState, step: PlanStep) -> List[str]:
    """返回阻塞当前步骤执行的依赖步骤 ID。"""
    plan_steps_by_id = {str(plan_step.step_id or "").strip(): plan_step for plan_step in _get_plan_steps(state)}
    blocked_by: List[str] = []
    for dependency_step_id in list(step.depends_on or []):
        normalized_dependency_step_id = str(dependency_step_id or "").strip()
        if not normalized_dependency_step_id:
            continue
        dependency_step = plan_steps_by_id.get(normalized_dependency_step_id)
        dependency_status = _normalize_plan_status(getattr(dependency_step, "status", "")) if dependency_step else "missing"
        if dependency_status != "success":
            blocked_by.append(normalized_dependency_step_id)
    return blocked_by


def _can_execute_plan_step(state: AgentState, step: PlanStep) -> bool:
    """判断当前步骤是否满足执行条件。"""
    if _normalize_plan_status(step.status) != "pending":
        return False
    return not _get_blocking_dependency_ids(state, step)


def _get_next_executable_plan_step(state: AgentState) -> Optional[PlanStep]:
    """返回当前可执行的下一步。"""
    for plan_step in _get_plan_steps(state):
        if _normalize_plan_status(plan_step.status) == "running":
            return plan_step
    for plan_step in _get_plan_steps(state):
        if _can_execute_plan_step(state, plan_step):
            return plan_step
    return None


def _refresh_execution_plan_runtime(state: AgentState) -> AgentState:
    """根据 execution_plan 回写轻量运行态摘要和结构化 PlanRuntime。"""
    next_state = state.model_copy(deep=True)
    plan_steps = _get_plan_steps(next_state)
    debug = dict(next_state.debug or {})
    runtime = _ensure_plan_runtime(next_state)
    runtime = runtime.model_copy(
        update={
            "goal": next_state.goal,
            "plan": next_state.execution_plan,
            "pending_confirmation": next_state.pending_action if isinstance(next_state.pending_action, Mapping) else runtime.pending_confirmation,
            "final_answer": next_state.answer,
        }
    )

    if not plan_steps:
        next_state.plan_runtime = runtime.model_copy(update={"step_status": {}, "state": {}, "trace": list(runtime.trace or [])})
        debug["execution_plan_runtime"] = {
            "step_count": 0,
            "current_step_id": None,
            "current_action_type": None,
            "next_executable_step_id": None,
            "next_executable_action_type": None,
            "status_counts": {},
            "steps": [],
        }
        next_state.debug = debug
        return next_state

    status_counts: Dict[str, int] = {}
    runtime_steps: List[Dict[str, Any]] = []
    completed_step_ids: List[str] = []
    failed_step_ids: List[str] = []
    skipped_step_ids: List[str] = []
    step_status: Dict[str, str] = {}
    plan_step_observations = dict(debug.get("plan_step_observations", {}))

    current_step = _get_next_executable_plan_step(next_state) or _get_next_incomplete_plan_step(next_state)
    next_executable_step = _get_next_executable_plan_step(next_state)

    for plan_step in plan_steps:
        step_id = str(plan_step.step_id or "").strip()
        action_type = str(plan_step.action_type or "").strip()
        status = _normalize_plan_status(plan_step.status)
        blocked_by = _get_blocking_dependency_ids(next_state, plan_step)
        can_execute = _can_execute_plan_step(next_state, plan_step)

        step_status[step_id] = status
        status_counts[status] = int(status_counts.get(status, 0) or 0) + 1
        if status == "success" and step_id:
            completed_step_ids.append(step_id)
        elif status == "failed" and step_id:
            failed_step_ids.append(step_id)
        elif status == "skipped" and step_id:
            skipped_step_ids.append(step_id)

        runtime_step = {
            "step_id": step_id,
            "action_type": action_type,
            "tool_name": plan_step.tool_name,
            "status": status,
            "output_key": plan_step.output_key,
            "depends_on": list(plan_step.depends_on or []),
            "blocked_by": blocked_by,
            "can_execute": can_execute,
            "is_current": bool(current_step and step_id == str(current_step.step_id or "").strip()),
        }
        last_observation = plan_step_observations.get(step_id)
        if isinstance(last_observation, Mapping) and last_observation:
            runtime_step["last_tool_observation"] = dict(last_observation)
        runtime_steps.append(runtime_step)

    runtime_state = {
        "step_count": len(plan_steps),
        "current_step_id": str(getattr(current_step, "step_id", None) or "").strip() or None,
        "current_action_type": str(getattr(current_step, "action_type", None) or "").strip() or None,
        "current_step_status": _normalize_plan_status(getattr(current_step, "status", None)),
        "next_executable_step_id": str(getattr(next_executable_step, "step_id", None) or "").strip() or None,
        "next_executable_action_type": str(getattr(next_executable_step, "action_type", None) or "").strip() or None,
        "status_counts": status_counts,
        "completed_step_ids": completed_step_ids,
        "failed_step_ids": failed_step_ids,
        "skipped_step_ids": skipped_step_ids,
        "steps": runtime_steps,
    }
    trace = list(runtime.trace or []) + [
        ExecutionTrace(
            step_id=str(getattr(current_step, "step_id", None) or "plan_runtime"),
            event="runtime_refresh",
            status=_normalize_plan_status(getattr(current_step, "status", None)) if current_step else None,
            detail={"step_count": len(plan_steps), "current_step_id": runtime_state["current_step_id"]},
        )
    ]
    next_state.plan_runtime = runtime.model_copy(update={"state": runtime_state, "step_status": step_status, "trace": trace})
    debug["execution_plan_runtime"] = runtime_state
    next_state.debug = debug
    return next_state


def _compact_tool_args(tool_args: Mapping[str, Any]) -> Dict[str, Any]:
    """筛选并压缩工具参数，生成适合展示和追踪的参数快照。"""
    payload: Dict[str, Any] = {}
    for key in (
        "query",
        "title_query",
        "abstract_query",
        "author_query",
        "categories",
        "comment_query",
        "journal_ref_query",
        "report_number_query",
        "id_list",
        "field_operator",
        "category_operator",
        "submitted_days_ago",
        "max_results",
        "start",
        "sort_by",
        "sort_order",
    ):
        value = tool_args.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    return payload


def _coerce_state(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """把节点收到的 state 统一转换成 AgentState 副本。"""
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    model_dump = getattr(state, "model_dump", None)
    if callable(model_dump):
        return AgentState.model_validate(model_dump())
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)
