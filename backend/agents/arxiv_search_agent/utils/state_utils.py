"""AgentState 压缩、追加步骤和状态统一化工具。

这个模块主要负责把较重的状态对象整理成更适合记录、调试和跨节点传递的形式，
避免每个 node 都各自实现一套类似的状态拼装逻辑。

职责重点包括：
1. 把复杂对象压缩成轻量快照；
2. 统一追加执行步骤轨迹；
3. 把外部传入的 state 规范化为 AgentState。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, List, Union

from ..schemas import AgentStep, ArxivSearchSpec, ExecutionPlanStep
from ..state import AgentState


_PLAN_ACTIVE_STATUSES = {"pending", "in_progress"}
_PLAN_TERMINAL_STATUSES = {"completed", "failed", "skipped"}


def _compact_search_spec(spec: Optional[ArxivSearchSpec]) -> Dict[str, Any]:
    """把搜索规格对象压缩成适合日志、调试和 step 输出的轻量字典。

    这里不会保留完整模型对象，而是提取出最有解释价值的字段，
    便于写入 debug、step outputs 或前端展示区域。
    """
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
    """从论文列表中抽取少量摘要字段，避免在轨迹里保存完整论文数据。

    完整论文对象通常字段很多、体积较大，直接写入状态轨迹会让调试信息过重。
    因此这里只挑选标题、arXiv ID、发布时间和简单排序字段作为摘要。
    """
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
    """向状态轨迹中追加一个新的执行步骤，并返回新的 state 副本。

    该函数统一维护步骤追加方式，确保所有节点记录的 step 结构一致，
    这样前端、日志系统和调试工具都可以稳定消费这些轨迹数据。
    """
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
    """按 step_id 或 step_type 更新 execution_plan 中某一步的状态。

    这个辅助函数只负责稳定地回写计划状态，不引入新的业务判断：
    - 如果目标步骤不存在，则保持原状态不变；
    - 如果没有传入 status，也保持原状态；
    - 返回新的 state 副本，方便节点链式调用。
    """
    next_state = state.model_copy(deep=True)
    if not next_state.execution_plan or not status:
        return next_state

    updated_plan: List[ExecutionPlanStep] = []
    matched = False
    for plan_step in list(next_state.execution_plan or []):
        current_step_id = str(getattr(plan_step, "step_id", "") or "").strip()
        current_step_type = str(getattr(plan_step, "step_type", "") or "").strip()
        is_target = False
        if step_id and current_step_id == str(step_id).strip():
            is_target = True
        elif step_type and current_step_type == str(step_type).strip():
            is_target = True

        if is_target:
            matched = True
            updated_plan.append(plan_step.model_copy(update={"status": status}))
        else:
            updated_plan.append(plan_step)

    if matched:
        next_state.execution_plan = updated_plan
    return _refresh_execution_plan_runtime(next_state)


def _get_execution_plan_step(
    state: AgentState,
    *,
    step_id: Optional[str] = None,
    step_type: Optional[str] = None,
) -> Optional[ExecutionPlanStep]:
    """按 step_id 或 step_type 获取 execution_plan 中的某一步。"""
    for plan_step in list(state.execution_plan or []):
        current_step_id = str(getattr(plan_step, "step_id", "") or "").strip()
        current_step_type = str(getattr(plan_step, "step_type", "") or "").strip()
        if step_id and current_step_id == str(step_id).strip():
            return plan_step
        if step_type and current_step_type == str(step_type).strip():
            return plan_step
    return None


def _get_next_incomplete_plan_step(state: AgentState) -> Optional[ExecutionPlanStep]:
    """返回 execution_plan 中第一个尚未完成的步骤。"""
    for plan_step in list(state.execution_plan or []):
        status = str(getattr(plan_step, "status", "") or "").strip()
        if status not in {"completed", "skipped"}:
            return plan_step
    return None


def _get_blocking_dependency_ids(state: AgentState, step: ExecutionPlanStep) -> List[str]:
    """返回阻塞当前步骤执行的依赖步骤 ID 列表。"""
    plan_steps_by_id = {
        str(getattr(plan_step, "step_id", "") or "").strip(): plan_step
        for plan_step in list(state.execution_plan or [])
        if str(getattr(plan_step, "step_id", "") or "").strip()
    }

    blocked_by: List[str] = []
    for dependency_step_id in list(getattr(step, "depends_on", []) or []):
        normalized_dependency_step_id = str(dependency_step_id or "").strip()
        if not normalized_dependency_step_id:
            continue
        dependency_step = plan_steps_by_id.get(normalized_dependency_step_id)
        dependency_status = str(getattr(dependency_step, "status", "") or "").strip() if dependency_step is not None else "missing"
        if dependency_status != "completed":
            blocked_by.append(normalized_dependency_step_id)
    return blocked_by


def _can_execute_plan_step(state: AgentState, step: ExecutionPlanStep) -> bool:
    """判断当前步骤是否满足执行条件。"""
    status = str(getattr(step, "status", "") or "").strip()
    if status != "pending":
        return False
    return not _get_blocking_dependency_ids(state, step)


def _get_next_executable_plan_step(state: AgentState) -> Optional[ExecutionPlanStep]:
    """返回 execution_plan 中当前可执行的下一步。"""
    for plan_step in list(state.execution_plan or []):
        if str(getattr(plan_step, "status", "") or "").strip() == "in_progress":
            return plan_step

    for plan_step in list(state.execution_plan or []):
        if _can_execute_plan_step(state, plan_step):
            return plan_step
    return None


def _refresh_execution_plan_runtime(state: AgentState) -> AgentState:
    """根据 execution_plan 当前状态回写一份轻量运行态摘要到 debug。"""
    next_state = state.model_copy(deep=True)
    plan_steps = list(next_state.execution_plan or [])
    debug = dict(next_state.debug or {})

    if not plan_steps:
        debug["execution_plan_runtime"] = {
            "step_count": 0,
            "current_step_id": None,
            "current_step_type": None,
            "next_executable_step_id": None,
            "next_executable_step_type": None,
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
    plan_step_observations = dict(debug.get("plan_step_observations", {}))

    current_step = _get_next_executable_plan_step(next_state)
    if current_step is None:
        current_step = _get_next_incomplete_plan_step(next_state)

    for plan_step in plan_steps:
        step_id = str(getattr(plan_step, "step_id", "") or "").strip()
        step_type = str(getattr(plan_step, "step_type", "") or "").strip()
        status = str(getattr(plan_step, "status", "") or "").strip() or "pending"
        blocked_by = _get_blocking_dependency_ids(next_state, plan_step)
        can_execute = _can_execute_plan_step(next_state, plan_step)

        status_counts[status] = int(status_counts.get(status, 0) or 0) + 1
        if status == "completed" and step_id:
            completed_step_ids.append(step_id)
        elif status == "failed" and step_id:
            failed_step_ids.append(step_id)
        elif status == "skipped" and step_id:
            skipped_step_ids.append(step_id)

        runtime_step = {
            "step_id": step_id,
            "step_type": step_type,
            "status": status,
            "depends_on": list(getattr(plan_step, "depends_on", []) or []),
            "blocked_by": blocked_by,
            "can_execute": can_execute,
            "is_current": bool(current_step is not None and step_id and step_id == str(getattr(current_step, "step_id", "") or "").strip()),
        }
        last_observation = plan_step_observations.get(step_id)
        if isinstance(last_observation, Mapping) and last_observation:
            runtime_step["last_tool_observation"] = dict(last_observation)
        runtime_steps.append(runtime_step)

    debug["execution_plan_runtime"] = {
        "step_count": len(plan_steps),
        "current_step_id": str(getattr(current_step, "step_id", None) or "").strip() or None,
        "current_step_type": str(getattr(current_step, "step_type", None) or "").strip() or None,
        "current_step_status": str(getattr(current_step, "status", None) or "").strip() or None,
        "next_executable_step_id": str(getattr(_get_next_executable_plan_step(next_state), "step_id", None) or "").strip() or None,
        "next_executable_step_type": str(getattr(_get_next_executable_plan_step(next_state), "step_type", None) or "").strip() or None,
        "status_counts": status_counts,
        "completed_step_ids": completed_step_ids,
        "failed_step_ids": failed_step_ids,
        "skipped_step_ids": skipped_step_ids,
        "steps": runtime_steps,
    }
    next_state.debug = debug
    return next_state


def _compact_tool_args(tool_args: Mapping[str, Any]) -> Dict[str, Any]:
    """筛选并压缩工具参数，生成适合展示和追踪的参数快照。

    只保留与搜索执行和问题定位最相关的字段，避免把过多冗余参数写入 trace。
    """
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
    """把节点收到的 state 统一转换成 AgentState 副本。

    这样各个节点就不需要关心调用方传进来的是完整模型实例还是普通字典，
    只需要面对统一的 AgentState 接口即可。
    """
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    model_dump = getattr(state, "model_dump", None)
    if callable(model_dump):
        return AgentState.model_validate(model_dump())
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)
