from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.logging_utils import info_event

from ..schemas import ExecutablePlan, Goal, PlanRuntime, PlanStep
from ..state import AgentState
from ..tool_adapters.models import ToolError, ToolExecutionResult

logger = logging.getLogger(__name__)


def _state_run_id(state: AgentState) -> Optional[str]:
    """从 AgentState 内部上下文取日志关联 ID；缺失时回退 session_id，保证 INFO 可串联。"""
    context = state.context if isinstance(state.context, Mapping) else {}
    debug = state.debug if isinstance(state.debug, Mapping) else {}
    value = context.get("run_id") or debug.get("run_id") or state.session_id
    text = str(value or "").strip()
    return text or None


def _tool_result_count(result: ToolExecutionResult) -> Optional[int]:
    data = result.data
    if isinstance(data, Mapping):
        for key in ("papers", "sources", "chunks", "results", "items"):
            value = data.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return len(value)
    return None


def _log_plan_done(state: AgentState, goal: Goal, plan: ExecutablePlan, planning_debug: Mapping[str, Any]) -> None:
    planner_summary = dict(planning_debug.get("planner_summary") or {}) if isinstance(planning_debug, Mapping) else {}
    info_event(
        logger,
        "arxiv_agent.plan_done",
        run_id=_state_run_id(state),
        session_id=state.session_id,
        goal_type=getattr(goal, "goal_type", None),
        step_count=len(list(getattr(plan, "steps", []) or [])),
        requested_path=planner_summary.get("requested_path"),
        selected_path=planner_summary.get("selected_path"),
        final_path=planner_summary.get("final_path"),
        llm_attempted=planner_summary.get("llm_draft_attempted"),
        llm_valid=planner_summary.get("llm_draft_valid"),
        fallback=planner_summary.get("fallback_used"),
        fallback_reason=planner_summary.get("fallback_reason"),
    )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_paper_target_resolution_step(step: PlanStep) -> bool:
    return step.tool_name in {"resolve_paper", "resolve_preference_target"}


def _safe_compact(value: Any, *, limit: int = 1200) -> Any:
    """Trace 里只保留轻量快照，避免把大对象原样塞进运行轨迹。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        compact: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 12:
                compact["__truncated__"] = True
                break
            compact[str(key)] = _safe_compact(item, limit=limit)
        return compact
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value[:5]) if hasattr(value, "__getitem__") else list(value)
        compact_items = [_safe_compact(item, limit=limit) for item in items[:5]]
        if len(items) > 5:
            compact_items.append({"__truncated__": True})
        return compact_items
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _json_safe(value: Any) -> Any:
    """把跨节点运行现场裁剪成 JSON 友好的数据。

    PlanRuntime 内部仍可能短暂持有 Pydantic 对象或工具返回的复杂对象；写入 runtime_state /
    runtime_patch 时必须先降级成普通 dict/list/标量，避免 checkpoint 依赖不可序列化实例。
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in dict(value or {}).items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in list(value or [])]
    return repr(value)



def _record_step_output(runtime: PlanRuntime, step: PlanStep, normalized_output: Any) -> None:
    """写入步骤输出；QA 修复重试要同步 canonical 结果，避免最终响应继续读旧答案。"""
    if not step.output_key:
        return
    runtime.outputs[step.output_key] = normalized_output
    if step.tool_name == "answer_paper_question" and step.output_key != "paper_qa_result":
        runtime.outputs["paper_qa_result"] = normalized_output


def _extract_arxiv_id_from_paper_payload(payload: Mapping[str, Any]) -> str:
    """从论文工具输入中提取最终 arXiv ID；确认兜底只能基于明确目标，避免误跳过用户确认。"""
    candidate_values: List[Any] = [payload.get("arxiv_id")]
    for key in ("paper_reference", "paper_ref", "target_paper", "paper"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidate_values.append(value.get("arxiv_id"))
    for value in candidate_values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _build_existing_index_skip_output(arxiv_id: str, check_result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """把索引状态检查结果归一成 parse_and_index_paper 的成功输出；未知或失败时不改变原确认流程。"""
    if not bool(check_result.get("ok", False)):
        return None
    data = check_result.get("data")
    status_data = dict(data or {}) if isinstance(data, Mapping) else {}
    status = str(status_data.get("status") or "").strip().lower()
    has_index = bool(status_data.get("has_index"))
    if not has_index and status not in {"available", "indexed", "already_indexed", "ready"}:
        return None
    output = {
        **status_data,
        "status": "indexed",
        "has_index": True,
        "arxiv_id": arxiv_id,
        "skipped_rebuild": True,
        "skip_reason": "paper_qa_index_already_available",
        "tool_result": dict(check_result),
    }
    return output


def _should_preserve_non_success_observation_output(step: PlanStep, normalized_output: Any) -> bool:
    """保留“目标解析未完成”类输出，便于 fallback/debug 说明为何不能继续。

    resolve_paper / resolve_preference_target 可能已经生成 reference_hint 和候选列表，
    但 Observer 会因为未得到唯一 final target 而拦截。此类输出不是业务成功结果，
    不能把 step 标成 success；但必须进入 runtime.outputs，供前端和测试读取解析原因。
    """
    if step.tool_name not in {"resolve_paper", "resolve_preference_target"}:
        return False
    if not step.output_key:
        return False
    return normalized_output not in (None, "", [], {})



_PAPER_QA_OBSERVATION_TRACE_KEYS = (
    "outcome",
    "termination_reason",
    "research_summary",
    "schema_version",
    "retrieval_quality",
    "retrieval_quality_reason",
    "answer_quality",
    "answer_quality_reason",
    "answer_insufficient_evidence",
    "missing_evidence_type",
    "weak_source_reason",
    "rerank_failed_reason",
    "degraded_stages",
    "recommended_repair_actions",
    "source_count",
    "error_code",
    "error_stage",
    "observation_reason",
)


def _compact_paper_qa_observation(value: Any) -> Dict[str, Any]:
    """质量闭环 trace 只保留 Agent 决策字段，避免把完整 retrieval_debug 塞回响应。"""
    if not isinstance(value, Mapping):
        return {}
    return {
        key: value.get(key)
        for key in _PAPER_QA_OBSERVATION_TRACE_KEYS
        if value.get(key) not in (None, "", [], {})
    }


def _paper_qa_source_fingerprints(value: Any) -> List[str]:
    """用来源指纹做修复前后对比，不记录 chunk 正文或 prompt。"""
    if not isinstance(value, list):
        return []
    fingerprints: List[str] = []
    for item in value[:8]:
        if not isinstance(item, Mapping):
            continue
        source_id = (
            item.get("chunk_id")
            or item.get("source_id")
            or item.get("id")
            or item.get("chunk_index")
            or item.get("section")
        )
        if source_id is not None:
            fingerprints.append(str(source_id))
    return fingerprints


def _answer_output_for_quality_step(runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
    """从质量门输入绑定反查 answer 输出，供 debug 比较 observation 和 sources。"""
    source_step_id = next(
        (
            str(binding.step_id)
            for binding in list(step.input_bindings or [])
            if binding.input_key == "paper_qa_result" and binding.source_type == "step_output" and binding.step_id
        ),
        "",
    )
    if not source_step_id or runtime.plan is None:
        return {}
    source_step = next((item for item in list(runtime.plan.steps or []) if item.step_id == source_step_id), None)
    if source_step is None or not source_step.output_key:
        return {}
    source_output = runtime.outputs.get(source_step.output_key)
    return dict(source_output or {}) if isinstance(source_output, Mapping) else {}


def _latest_paper_qa_repair_trace(runtime: PlanRuntime) -> Dict[str, Any]:
    """读取最近一次 Paper QA 修复计划摘要，修复后质量门用它生成 before/after 对比。"""
    for trace in reversed(list(runtime.trace or [])):
        if trace.event != "plan_replanned":
            continue
        detail = dict(trace.detail or {})
        repair_trace = detail.get("paper_qa_repair_trace")
        if isinstance(repair_trace, Mapping):
            return dict(repair_trace)
    return {}


def _build_paper_qa_quality_trace(runtime: PlanRuntime, step: PlanStep, quality_output: Any) -> Dict[str, Any]:
    """生成 Paper QA 质量门的固定 debug 结构，供验收直接读取最终决策和修复效果。"""
    if step.tool_name != "assess_paper_qa_quality" or not isinstance(quality_output, Mapping):
        return {}
    answer_output = _answer_output_for_quality_step(runtime, step)
    current_observation = _compact_paper_qa_observation(
        quality_output.get("qa_observation") or answer_output.get("qa_observation")
    )
    current_sources = _paper_qa_source_fingerprints(answer_output.get("sources"))
    repair_trace = _latest_paper_qa_repair_trace(runtime)
    original_observation = dict(repair_trace.get("original_observation") or {})
    before_sources = list(repair_trace.get("before_sources") or [])
    retrieval_before = original_observation.get("retrieval_quality")
    retrieval_after = current_observation.get("retrieval_quality")
    insufficient_before = original_observation.get("answer_insufficient_evidence")
    insufficient_after = current_observation.get("answer_insufficient_evidence") or quality_output.get("answer_insufficient_evidence")
    decision = str(quality_output.get("decision") or "unknown")
    reason = str(quality_output.get("reason") or current_observation.get("observation_reason") or "paper_qa_quality_decision")
    return {
        "original_observation": original_observation,

        "current_observation": current_observation,
        "decision_fields": {
            "decision": decision,
            "status": quality_output.get("status"),
            "reason": reason,
            "retrieval_quality": retrieval_after,
            "answer_insufficient_evidence": insufficient_after,
            "rerank_failed_reason": current_observation.get("rerank_failed_reason") or quality_output.get("rerank_failed_reason"),
            "recommended_repair_actions": list(quality_output.get("repair_actions") or []),
        },
        "selected_repair_actions": list(repair_trace.get("selected_repair_actions") or []),
        "selected_repair_action": repair_trace.get("selected_repair_action"),
        "retrieval_quality_before": retrieval_before,
        "retrieval_quality_after": retrieval_after,
        "answer_insufficient_evidence_before": insufficient_before,
        "answer_insufficient_evidence_after": insufficient_after,
        "sources_before": before_sources,
        "sources_after": current_sources,
        "sources_changed": bool(before_sources or current_sources) and before_sources != current_sources,
        "repair_attempted": bool(repair_trace),
        "final_decision": decision,
        "final_decision_reason": reason,
        "max_repair_limit_triggered": False,
    }


def _compact_step_output_for_trace(step: PlanStep, value: Any) -> Any:
    """Paper QA 输出可能包含长 debug/source；执行 trace 只保留可回放的轻量摘要。"""
    if step.tool_name == "answer_paper_question" and isinstance(value, Mapping):
        retrieval_debug = value.get("retrieval_debug")
        return {
            "status": value.get("status"),
            "answer_preview": str(value.get("answer") or "")[:160],
            "source_count": len(value.get("sources") or []) if isinstance(value.get("sources"), list) else 0,
            "source_fingerprints": _paper_qa_source_fingerprints(value.get("sources")),
            "retrieval_debug_keys": sorted(str(key) for key in retrieval_debug.keys()) if isinstance(retrieval_debug, Mapping) else [],
            "qa_observation": _compact_paper_qa_observation(value.get("qa_observation")),
            "error": value.get("error"),
        }
    if step.tool_name == "assess_paper_qa_quality" and isinstance(value, Mapping):
        return {
            "status": value.get("status"),
            "decision": value.get("decision"),
            "reason": value.get("reason"),
            "repair_required": value.get("repair_required"),
            "repair_optional": value.get("repair_optional"),
            "repair_actions": list(value.get("repair_actions") or []),
            "retrieval_quality": value.get("retrieval_quality"),
            "answer_quality": value.get("answer_quality"),
            "answer_insufficient_evidence": value.get("answer_insufficient_evidence"),
            "rerank_failed_reason": value.get("rerank_failed_reason"),
            "qa_observation": _compact_paper_qa_observation(value.get("qa_observation")),
        }
    return _safe_compact(value)


def _compact_tool_execution_for_trace(step: PlanStep, raw_output: Any) -> Any:
    """工具 envelope 也可能携带完整 data；Paper QA trace 只记录执行状态。"""
    if not isinstance(raw_output, ToolExecutionResult):
        return None
    if step.tool_name in {"answer_paper_question", "assess_paper_qa_quality"}:
        metadata = raw_output.metadata if isinstance(raw_output.metadata, Mapping) else {}
        return {
            "ok": raw_output.ok,
            "tool_name": raw_output.tool_name,
            "adapter_name": raw_output.adapter_name,
            "summary": metadata.get("summary"),
            "error": raw_output.error.model_dump() if raw_output.error is not None else None,
        }
    return _safe_compact(raw_output.model_dump())


def _make_tool_error(
    *,
    error_code: str,
    message: str,
    detail: Optional[Mapping[str, Any]] = None,
    recoverable: bool = True,
    retryable: bool = False,
    suggested_recovery: Optional[str] = None,
    failed_stage: Optional[str] = None,
    raw_exception_type: Optional[str] = None,
    safe_debug: Optional[Mapping[str, Any]] = None,
) -> ToolError:
    return ToolError(
        error_code=error_code,
        message=message,
        detail=dict(detail or {}),
        recoverable=recoverable,
        retryable=retryable,
        suggested_recovery=suggested_recovery,
        failed_stage=failed_stage,
        raw_exception_type=raw_exception_type,
        safe_debug=dict(safe_debug or {}),
    )


def _tool_error_to_text(error: ToolError) -> str:
    return f"{error.error_code}:{error.message}"


def _model_to_plain(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return value


def _compact_validation_errors(errors: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """压缩 Pydantic errors()，保留字段路径和原因，避免日志被完整输入对象淹没。"""
    compact: List[Dict[str, Any]] = []
    for item in list(errors or []):
        detail = dict(item or {})
        loc = detail.get("loc")
        if isinstance(loc, (list, tuple)):
            detail["loc"] = ".".join(str(part) for part in loc)
        if "input" in detail:
            detail["input"] = _safe_compact(_model_to_plain(detail.get("input")), limit=400)
        if isinstance(detail.get("ctx"), Mapping):
            detail["ctx"] = _safe_compact(dict(detail.get("ctx") or {}), limit=400)
        compact.append(detail)
    return compact


_EXECUTION_PATH_TURN_STATUS_TO_FINAL = {
    "waiting_interaction": "waiting_interaction",
    "failed": "failed",
    "fallback": "failed",
    "need_clarification": "need_clarification",
    "success": "success",
}


def _summarize_step_path(runtime: "PlanRuntime", step_id: str) -> Dict[str, Any]:
    """汇总单个 step 的执行路径事实，作为只读 debug 视图。

    只读 runtime 已有的 trace / 计数 / 状态，不改变任何执行语义；目的是让“这一步到底走了哪条
    路径”一眼可见，避免后续拆分时还要靠人肉串 trace 判断 confirmation/resume/retry/replan/复用。
    """
    events = [trace.event for trace in list(runtime.trace or []) if trace.step_id == step_id]
    event_set = set(events)
    return {
        "step_id": step_id,
        "tool_name": next(
            (
                str(trace.detail.get("tool_name"))
                for trace in list(runtime.trace or [])
                if trace.step_id == step_id and trace.detail.get("tool_name")
            ),
            None,
        ),
        "status": runtime.step_status.get(step_id),
        "interaction_requested": "interaction_requested" in event_set,
        "interaction_resolved": "interaction_resolved" in event_set,
        "interaction_cancelled": "interaction_cancelled" in event_set,
        "retried": int(runtime.retry_counts.get(step_id, 0) or 0) > 0,
        "retry_count": int(runtime.retry_counts.get(step_id, 0) or 0),
        "replanned": "step_replanned" in event_set,
        "reused_side_effect_output": "step_reused_output" in event_set,
        "events": events,
    }


def _build_execution_path_summary(runtime: "PlanRuntime", *, current_step_id: Optional[str]) -> Dict[str, Any]:
    """生成一份固定结构的“本轮执行路径”只读摘要，供 debug 视图和验收脚本直接读取。

    它不替代细粒度 trace，而是把任务最关心的几类状态链路结论收敛到一处：当前 step/tool、是否经过
    确认、是否来自 checkpoint resume、是否 retry、是否 replan、是否复用副作用结果，以及本轮最终状态。
    所有字段都从 runtime 现有事实派生，因此是纯附加、可序列化、不影响执行行为的观测信息。
    """
    plan = runtime.plan
    step_ids = [step.step_id for step in list(plan.steps or [])] if plan is not None else list(runtime.step_status or {})

    step_paths = [_summarize_step_path(runtime, step_id) for step_id in step_ids]
    replanned = any(item["replanned"] for item in step_paths) or bool(runtime.replan_counts)
    final_status = _EXECUTION_PATH_TURN_STATUS_TO_FINAL.get(str(runtime.turn_status or "")) if runtime.turn_status else None
    if final_status is None:
        if runtime.interaction is not None:
            final_status = "waiting_interaction"
        elif runtime.error:
            final_status = "failed"
        elif replanned and not runtime.error:
            final_status = "replanned"
        else:
            final_status = "in_progress"
    elif final_status == "success" and replanned:
        # 本轮虽然最终成功，但确实经历过 replan；显式区分出来，便于排查“成功但绕过弯路”的链路。
        final_status = "replanned_success"
    return {
        "plan_id": str(getattr(plan, "plan_id", "") or "") or None,
        "current_step_id": current_step_id,
        "current_tool_name": next(
            (item["tool_name"] for item in step_paths if item["step_id"] == current_step_id),
            None,
        ),
        "interaction_requested": any(item["interaction_requested"] for item in step_paths),
        "interaction_resolved": any(item["interaction_resolved"] for item in step_paths),
        "retry_occurred": any(item["retried"] for item in step_paths),
        "replan_occurred": replanned,
        "reused_side_effect_output": any(item["reused_side_effect_output"] for item in step_paths),
        "pending_interaction": runtime.interaction is not None,
        "needs_replan": bool(runtime.needs_replan),
        "failure_reason": runtime.error,
        "final_status": final_status,
        "retry_counts": dict(runtime.retry_counts or {}),
        "replan_counts": dict(runtime.replan_counts or {}),
        "step_replan_counts": dict(runtime.step_replan_counts or {}),
        "steps": step_paths,
    }


