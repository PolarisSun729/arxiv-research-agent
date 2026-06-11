from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from langgraph.types import interrupt
from pydantic import ValidationError
from utils.config import get_agent_runtime_checkpoint_config

from . import tool_registry as agent_tool_registry
from .fallbacks import build_fallback_record
from .observer import Observer
from .planner import build_executable_plan, build_plan_runtime
from .replanner import Replanner
from .response_assembler import assemble_final_answer, record_confirmation_rejection, record_recovery_fallback
from .schemas import (
    AgentTurnResult,
    AgentRuntimeState,
    ConfirmationDecisionOption,
    ConfirmationRequest,
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


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_ttl_seconds() -> int:
    """读取 checkpoint TTL 作为 pending action 展示过期时间；真正校验仍由 checkpoint manager 执行。"""
    try:
        config = get_agent_runtime_checkpoint_config()
        return max(int(config.get("ttl_seconds") or 0), 60)
    except Exception:
        return 24 * 60 * 60


def _confirmation_expires_at() -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=_checkpoint_ttl_seconds())).isoformat()


def _is_paper_target_resolution_step(step: PlanStep) -> bool:
    return step.tool_name in {"resolve_paper", "resolve_preference_target"}


def _candidate_identity_values(candidate: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("candidate_id", "paper_id", "paperId", "stable_id", "id", "arxiv_id", "arxivId"):
        value = str(candidate.get(key) or "").strip()
        if value and value not in values:
            values.append(value)
    if not values:
        title = str(candidate.get("title") or "").strip()
        if title:
            values.append(title)
    return values


def _candidate_display_id(candidate: Mapping[str, Any]) -> Optional[str]:
    values = _candidate_identity_values(candidate)
    return values[0] if values else None


def _normalize_confirmation_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """裁剪候选论文给前端展示，保留恢复所需稳定身份和来源信息。"""
    payload = dict(candidate or {})
    candidate_id = _candidate_display_id(payload)
    if candidate_id:
        payload["candidate_id"] = candidate_id
    authors = payload.get("authors")
    if isinstance(authors, list):
        visible_authors = [str(item).strip() for item in authors if str(item or "").strip()]
        payload["authors_summary"] = ", ".join(visible_authors[:3])
    elif authors not in (None, "", [], {}):
        payload["authors_summary"] = str(authors).strip()
    payload.setdefault("source_label", payload.get("list_name") or payload.get("source") or payload.get("source_type"))
    return payload


def _confirmation_candidate_sources(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """确认日志只记录候选身份和来源，避免把完整论文摘要写入运行日志。"""
    sources: List[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:8]:
        if not isinstance(candidate, Mapping):
            continue
        sources.append(
            {
                "candidate_id": _candidate_display_id(candidate),
                "arxiv_id": candidate.get("arxiv_id") or candidate.get("arxivId"),
                "source": candidate.get("source_label") or candidate.get("list_name") or candidate.get("source") or candidate.get("source_type"),
                "source_key": candidate.get("source_key"),
                "rank": candidate.get("rank"),
            }
        )
    return sources


def _resume_edited_arguments(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    edited = payload.get("edited_arguments")
    if isinstance(edited, Mapping):
        return dict(edited)
    return {}


def _match_confirmed_candidate(candidates: Sequence[Mapping[str, Any]], edited_arguments: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """只在 pending confirmation 保存的候选内匹配，禁止 resume 时重新解析用户自然语言。"""
    requested_values = [
        str(edited_arguments.get(key) or "").strip()
        for key in (
            "confirmed_paper_id",
            "paper_id",
            "candidate_id",
            "confirmed_arxiv_id",
            "arxiv_id",
            "confirmed_title",
            "title",
        )
    ]
    requested_values = [value for value in requested_values if value]
    if not requested_values:
        return None
    for candidate in list(candidates or []):
        if not isinstance(candidate, Mapping):
            continue
        identities = _candidate_identity_values(candidate)
        if any(value in identities for value in requested_values):
            return dict(candidate)
    return None


def _build_confirmed_paper_target_output(
    *,
    step: PlanStep,
    confirmation_request: ConfirmationRequest,
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    """把用户确认的候选物化成 resolver 的成功输出，供下游工具直接按 paper_id/arxiv_id 执行。"""
    selected = dict(candidate or {})
    reference_hint = dict(confirmation_request.reference_hint or {})
    original_resolution = dict(confirmation_request.target_resolution or {})
    resolution_hint = original_resolution.get("reference_hint") if isinstance(original_resolution.get("reference_hint"), Mapping) else {}
    target_resolution = {
        **original_resolution,
        "status": "resolved",
        "target": selected,
        "recommended_candidate": selected,
        "confirmed_candidate": selected,
        "confidence": max(float(original_resolution.get("confidence") or 0.0), 0.99),
        "resolution_reason": "user_confirmed_target",
        "requires_confirmation": False,
        "user_confirmed": True,
    }
    base = {
        "status": "resolved",
        "reference_type": reference_hint.get("reference_type") or resolution_hint.get("reference_type") or "unknown",
        "value": reference_hint.get("value"),
        "confidence": target_resolution["confidence"],
        "hint_confidence": reference_hint.get("confidence", 0.0),
        "source": reference_hint.get("source"),
        "requires_context": bool(reference_hint.get("requires_context")),
        "reason": "user_confirmed_target",
        "resolution_reason": "user_confirmed_target",
        "final_target_resolved": True,
        "reference_hint": reference_hint,
        "target": selected,
        "paper": selected,
        "arxiv_id": selected.get("arxiv_id") or selected.get("arxivId"),
        "title": selected.get("title"),
        "matched_by": "user_confirmed_target",
        "target_resolution": target_resolution,
        "candidates": list(confirmation_request.candidates or []),
        "recommended_candidate": selected,
        "requires_confirmation": False,
        "risk_level": original_resolution.get("risk_level") or confirmation_request.arguments_summary.get("risk_level"),
        "action_type": original_resolution.get("action_type") or confirmation_request.action_type,
        "confirmed_by_user": True,
        "confirmed_paper_id": selected.get("paper_id") or selected.get("candidate_id") or selected.get("id"),
        "confirmed_arxiv_id": selected.get("arxiv_id") or selected.get("arxivId"),
    }
    alias_payload = dict(base)
    if step.tool_name == "resolve_paper":
        base["paper_ref"] = dict(alias_payload)
    base["paper_reference"] = dict(alias_payload)
    return base


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


def _get_path_value(value: Any, path: Optional[str]) -> Any:
    if not path:
        return value
    current = value
    for part in [segment for segment in str(path).split(".") if segment]:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(part)
            continue
        current = getattr(current, part, None)
    return current


def _coerce_state_mapping(state: AgentState) -> Dict[str, Any]:
    return state.model_dump() if hasattr(state, "model_dump") else dict(state)


def _extract_state_value(state: AgentState, source_key: Optional[str]) -> Any:
    if not source_key:
        return None
    if hasattr(state, source_key):
        return getattr(state, source_key)
    state_mapping = _coerce_state_mapping(state)
    if source_key in state_mapping:
        return state_mapping.get(source_key)
    context = state.context if isinstance(state.context, Mapping) else {}
    return context.get(source_key)


def _extract_search_spec_value(state: AgentState, source_key: Optional[str]) -> Any:
    if state.search_spec is None:
        return None
    if not source_key:
        return state.search_spec
    return getattr(state.search_spec, source_key, None)


def _extract_step_output(runtime: PlanRuntime, plan: ExecutablePlan, step_id: Optional[str]) -> Any:
    normalized_step_id = str(step_id or "").strip()
    if not normalized_step_id:
        return None
    for plan_step in list(plan.steps or []):
        if plan_step.step_id == normalized_step_id and plan_step.output_key:
            return runtime.outputs.get(plan_step.output_key)
    return None


def _existing_step_output(runtime: PlanRuntime, step: PlanStep) -> Any:
    if not step.output_key:
        return None
    return runtime.outputs.get(step.output_key)


def _can_reuse_side_effect_output(runtime: PlanRuntime, step: PlanStep) -> bool:
    """副作用工具不能因为 resume/retry 被重复执行；已有输出优先视为本轮可复用结果。"""
    if step.side_effect_level not in {"persistent_write", "external_call"}:
        return False
    if not step.output_key or step.output_key not in runtime.outputs:
        return False
    existing_output = runtime.outputs.get(step.output_key)
    if existing_output in (None, "", [], {}):
        return False
    return True


def _condition_value(condition: StepCondition, state: AgentState, runtime: PlanRuntime) -> Any:
    if condition.condition_type in {"field_exists", "field_equals"}:
        field_path = condition.field_path or ""
        root_name, _, remaining = field_path.partition(".")
        if root_name == "state":
            return _get_path_value(_coerce_state_mapping(state), remaining)
        if root_name == "goal":
            return _get_path_value(runtime.goal.model_dump() if runtime.goal else {}, remaining)
        if root_name == "outputs":
            return _get_path_value(runtime.outputs, remaining)
        if root_name == "context":
            return _get_path_value(state.context if isinstance(state.context, Mapping) else {}, remaining)
        return _get_path_value(_coerce_state_mapping(state), field_path)
    if condition.condition_type == "step_output_exists":
        return _extract_step_output(runtime, runtime.plan or ExecutablePlan(goal=Goal(), plan_id="unknown"), condition.step_id)
    if condition.condition_type == "step_status_is":
        return runtime.step_status.get(str(condition.step_id or "").strip())
    return True


def _evaluate_condition(condition: Optional[StepCondition], state: AgentState, runtime: PlanRuntime) -> bool:
    if condition is None or condition.condition_type == "always":
        return True
    value = _condition_value(condition, state, runtime)
    if condition.condition_type in {"field_exists", "step_output_exists"}:
        result = value not in (None, "", [], {})
    elif condition.condition_type in {"field_equals", "step_status_is"}:
        result = value == condition.expected_value
    else:
        result = True
    return not result if condition.negate else result


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


def _compact_confirmation_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """只保留确认展示所需的轻量参数摘要，避免把大对象写入确认 payload。"""
    summary: Dict[str, Any] = {}
    for key, value in dict(arguments or {}).items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, Mapping):
            nested_summary: Dict[str, Any] = {}
            for nested_key in ("arxiv_id", "title", "question", "qa_question", "loading_method", "status", "type"):
                nested_value = value.get(nested_key)
                if nested_value not in (None, "", [], {}):
                    nested_summary[nested_key] = nested_value
            if nested_summary:
                summary[key] = nested_summary
            continue
        if isinstance(value, list):
            summary[key] = value[:3]
            continue
        if isinstance(value, (str, int, float, bool)):
            text_value = str(value)
            summary[key] = text_value[:240] if isinstance(value, str) else value
    return summary


class PlanExecutor:
    """按计划拓扑、输入绑定和策略约束执行 ExecutablePlan。"""

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry
        self.observer = Observer()
        self.replanner = Replanner(tool_registry=tool_registry)

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
                pending_confirmation=turn_result.pending_confirmation,
                runtime_patch=self._build_runtime_patch(runtime, current_step=None),
                turn_result=turn_result,
            )

        previous_outputs = dict(runtime.outputs or {})
        previous_trace_len = len(runtime.trace or [])
        previous_error = runtime.error
        previous_pending = runtime.pending_confirmation
        turn_result = self._execute_step(step, runtime, state, allow_interrupt=allow_interrupt, auto_replan=auto_replan)
        step_result = self._build_step_execution_result(
            step=step,
            runtime=runtime,
            previous_outputs=previous_outputs,
            previous_trace_len=previous_trace_len,
            previous_error=previous_error,
            previous_pending=previous_pending,
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

        if (
            allow_interrupt
            and runtime.step_status.get(step.step_id) == "waiting_confirmation"
            and runtime.pending_confirmation is not None
            and runtime.pending_confirmation.step_id == step.step_id
        ):
            # LangGraph resume 会从节点函数开头重入，而不是直接跳到上一轮 interrupt 的下一行。
            # 先消费同一确认请求的 resume payload，并写入 runtime/context 批准态，避免重新经过确认判断时二次弹窗。
            decision = self._resume_pending_confirmation(step=step, runtime=runtime, state=state)
            if decision != "approve":
                turn_result = self._handle_confirmation_rejection(
                    step=step,
                    runtime=runtime,
                    state=state,
                    confirmation_request=runtime.pending_confirmation,
                )
                self._sync_runtime_state(state, runtime, current_step=step)
                return self._step_result_from_runtime(
                    step=step,
                    runtime=runtime,
                    next_action="fail",
                    pending_confirmation=runtime.pending_confirmation,
                    turn_result=turn_result,
                )
            if _is_paper_target_resolution_step(step) and runtime.step_status.get(step.step_id) == "success":
                self._sync_runtime_state(state, runtime, current_step=step)
                return self._step_result_from_runtime(
                    step=step,
                    runtime=runtime,
                    next_action="continue",
                    output=runtime.outputs.get(step.output_key) if step.output_key else None,
                    observation=runtime.last_observation,
                )

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

        if _can_reuse_side_effect_output(runtime, step):
            # resume 或局部 replan 可能重新走到同一副作用 step；已有输出时复用结果，避免重复外部调用/持久写入。
            reused_output = _existing_step_output(runtime, step)
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

        if self._needs_confirmation(step, state):
            confirmation_request = self._build_confirmation_request(
                step=step,
                runtime=runtime,
                state=state,
                resolved_input=resolved_input,
                reason="explicit_user_confirmation_required",
                pending_action=resolved_input.get("pending_action") if isinstance(resolved_input.get("pending_action"), Mapping) else None,
            )
            turn_result = self._handle_confirmation_gate(
                step=step,
                runtime=runtime,
                state=state,
                confirmation_request=confirmation_request,
                resolved_input=resolved_input,
                started_at=started_at,
                allow_interrupt=allow_interrupt,
            )
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(
                step=step,
                runtime=runtime,
                next_action="wait_for_confirmation" if runtime.pending_confirmation else "continue",
                pending_confirmation=runtime.pending_confirmation,
                turn_result=turn_result,
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
        max_attempts = max(int(getattr(step.retry_policy, "max_attempts", 0) or 0), 1)
        while attempts < max_attempts:
            attempts += 1
            runtime.retry_counts[step.step_id] = attempts - 1
            try:
                raw_output = self._invoke_step_tool(step, resolved_input, state, runtime)
            except Exception as exc:  # pragma: no cover - 这里兜住未知工具实现异常
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
        candidate_outputs = dict(runtime.outputs, **({step.output_key: normalized_output} if step.output_key and not tool_failed else {}))
        if not tool_failed and any(not _evaluate_condition(condition, state, runtime.model_copy(update={"outputs": candidate_outputs})) for condition in list(step.postconditions or [])):
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

    def observe_current_step(self, runtime: PlanRuntime, state: AgentState) -> StepExecutionResult:
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
            turn_result = self._handle_paper_target_confirmation_gate(
                step=step,
                runtime=runtime,
                state=state,
                observation=observation,
                normalized_output=normalized_output,
                started_at=str(output_payload.get("started_at") or _utcnow()),
                allow_interrupt=True,
            )
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(
                step=step,
                runtime=runtime,
                next_action="wait_for_confirmation" if runtime.pending_confirmation else "continue",
                observation=runtime.last_observation,
                pending_confirmation=runtime.pending_confirmation,
                turn_result=turn_result,
            )

        if observation.status == "need_confirmation" and step.tool_name == "request_confirmation":
            pending_action = raw_output.get("pending_action") if isinstance(raw_output, Mapping) else None
            confirmation_request = self._build_confirmation_request(
                step=step,
                runtime=runtime,
                state=state,
                resolved_input=resolved_input,
                reason=observation.reason,
                pending_action=pending_action if isinstance(pending_action, Mapping) else None,
            )
            # 观察节点发现确认需求后也走正式 interrupt/resume，避免只生成展示态而没有可恢复现场。
            turn_result = self._handle_confirmation_gate(
                step=step,
                runtime=runtime,
                state=state,
                confirmation_request=confirmation_request,
                resolved_input=resolved_input,
                started_at=str(output_payload.get("started_at") or _utcnow()),
                allow_interrupt=True,
            )
            self._sync_runtime_state(state, runtime, current_step=step)
            return self._step_result_from_runtime(
                step=step,
                runtime=runtime,
                next_action="wait_for_confirmation" if runtime.pending_confirmation else "continue",
                observation=runtime.last_observation,
                pending_confirmation=runtime.pending_confirmation,
                turn_result=turn_result,
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
        self._append_paper_qa_quality_trace(runtime, step, normalized_output)
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
            if not _evaluate_condition(step.condition, state, runtime):
                runtime.step_status[step.step_id] = "skipped"
                self._append_trace(runtime, step, event="condition_skipped", status="skipped", detail={"condition": step.condition.model_dump() if step.condition else None})
                continue
            if any(not _evaluate_condition(condition, state, runtime) for condition in list(step.preconditions or [])):
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"precondition_failed:{step.step_id}"
                self._append_trace(runtime, step, event="precondition_failed", status="failed", detail={"preconditions": [condition.model_dump() for condition in list(step.preconditions or [])]})
                continue
            return step
        return None

    def _execute_step(self, step: PlanStep, runtime: PlanRuntime, state: AgentState, *, allow_interrupt: bool, auto_replan: bool = True) -> Optional[AgentTurnResult]:
        runtime.step_status[step.step_id] = "running"
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
            return None

        if _can_reuse_side_effect_output(runtime, step):
            # 兼容执行循环同样必须遵守副作用幂等边界，避免旧入口绕过显式节点的复用保护。
            reused_output = _existing_step_output(runtime, step)
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
            return None

        if self._needs_confirmation(step, state):
            confirmation_request = self._build_confirmation_request(
                step=step,
                runtime=runtime,
                state=state,
                resolved_input=resolved_input,
                reason="explicit_user_confirmation_required",
                pending_action=resolved_input.get("pending_action") if isinstance(resolved_input.get("pending_action"), Mapping) else None,
            )
            return self._handle_confirmation_gate(
                step=step,
                runtime=runtime,
                state=state,
                confirmation_request=confirmation_request,
                resolved_input=resolved_input,
                started_at=started_at,
                allow_interrupt=allow_interrupt,
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

        attempts = 0
        last_error: Optional[str] = None
        max_attempts = max(int(getattr(step.retry_policy, "max_attempts", 0) or 0), 1)
        while attempts < max_attempts:
            attempts += 1
            runtime.retry_counts[step.step_id] = attempts - 1
            try:
                raw_output = self._invoke_step_tool(step, resolved_input, state, runtime)
            except Exception as exc:  # pragma: no cover - 这里兜住未知工具实现异常
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
                return None

            if isinstance(raw_output, ToolExecutionResult) and not raw_output.ok:
                tool_error = raw_output.error or _make_tool_error(error_code="tool_failed", message="工具执行失败")
                last_error = _tool_error_to_text(tool_error)
                if attempts < max_attempts and (tool_error.retryable or bool(step.tool.can_retry or step.retry_policy)):
                    continue
                # 结构化工具错误仍交给 Observer/Replanner 判断，执行器只保存 envelope 和调度下一节点。
                normalized_output = None
            else:
                normalized_output = project_tool_result(step, raw_output) if isinstance(raw_output, ToolExecutionResult) else raw_output

            if step.output_key and step.output_key in runtime.outputs:
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"duplicate_output_key:{step.output_key}"
                runtime.recovery_strategy = {"type": "abort_with_error", "reason": "duplicate_output_key", "output_key": step.output_key}
                self._append_trace(runtime, step, event="step_failed", status="failed", detail={"failure_reason": "duplicate_output_key", "output_key": step.output_key})
                return None

            tool_failed = isinstance(raw_output, ToolExecutionResult) and not raw_output.ok
            if not tool_failed and any(not _evaluate_condition(condition, state, runtime.model_copy(update={"outputs": dict(runtime.outputs, **({step.output_key: normalized_output} if step.output_key else {}))})) for condition in list(step.postconditions or [])):
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"postcondition_failed:{step.step_id}"
                runtime.recovery_strategy = {"type": "abort_with_error", "reason": "postcondition_failed"}
                self._append_trace(runtime, step, event="step_failed", status="failed", detail={"failure_reason": "postcondition_failed", "raw_output": _safe_compact(raw_output)})
                return None

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
                    "observation_signal": observation.observation_signal,
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
                return self._handle_paper_target_confirmation_gate(
                    step=step,
                    runtime=runtime,
                    state=state,
                    observation=observation,
                    normalized_output=normalized_output,
                    started_at=started_at,
                    allow_interrupt=allow_interrupt,
                )

            if observation.status == "need_confirmation" and step.tool_name == "request_confirmation":
                runtime.step_status[step.step_id] = "waiting_confirmation"
                pending_action = raw_output.get("pending_action") if isinstance(raw_output, Mapping) else None
                confirmation_request = self._build_confirmation_request(
                    step=step,
                    runtime=runtime,
                    state=state,
                    resolved_input=resolved_input,
                    reason=observation.reason,
                    pending_action=pending_action if isinstance(pending_action, Mapping) else None,
                )
                runtime.pending_confirmation = confirmation_request
                logger.info(
                    "arxiv_agent confirmation requested: step_id=%s tool_name=%s side_effect_level=%s allowed_decisions=%s",
                    confirmation_request.step_id,
                    confirmation_request.tool_name,
                    confirmation_request.side_effect_level,
                    [item.code for item in list(confirmation_request.allowed_decisions or [])],
                )
                return self._build_turn_result(runtime)

            if observation.status not in {"success", "partial_success"}:
                # observation 代表“工具执行后质量不足”，先写入 runtime，方便重规划节点或调试视图复盘触发原因。
                if _should_preserve_non_success_observation_output(step, normalized_output):
                    _record_step_output(runtime, step, normalized_output)
                runtime.needs_replan = True
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
                if not auto_replan:
                    # 新 LangGraph 执行环需要把重规划显式暴露成图节点，这里只记录观察结果并暂停本步。
                    return None
                replan_result = self._handle_observation_replan(
                    step=step,
                    runtime=runtime,
                    state=state,
                    observation=observation,
                    raw_output=raw_output,
                    normalized_output=normalized_output,
                )
                if replan_result is not None:
                    return replan_result
                return None

            if step.output_key:
                _record_step_output(runtime, step, normalized_output)
                runtime.final_answer = assemble_final_answer(runtime)

            runtime.step_status[step.step_id] = "success"
            runtime.needs_replan = False
            self._append_paper_qa_quality_trace(runtime, step, normalized_output)
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
                    "started_at": started_at,
                    "finished_at": _utcnow(),
                },
            )
            return None

        runtime.step_status[step.step_id] = "failed"
        runtime.error = f"tool_execution_failed:{step.step_id}:{last_error or 'unknown_error'}"
        runtime.recovery_strategy = {"type": "abort_with_error", "reason": "tool_execution_failed", "error": last_error or "unknown_error"}
        return None

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
        return self._handle_observation_replan(
            step=step,
            runtime=runtime,
            state=state,
            observation=observation,
            raw_output=raw_output,
            normalized_output=normalized_output,
        )

    def _resolve_input_bindings(self, step: PlanStep, runtime: PlanRuntime, state: AgentState) -> Tuple[Dict[str, Any], Optional[str]]:
        resolved: Dict[str, Any] = {}
        for binding in list(step.input_bindings or []):
            value: Any = None
            if binding.source_type == "state":
                value = _extract_state_value(state, binding.source_key or binding.input_key)
            elif binding.source_type == "context":
                context = state.context if isinstance(state.context, Mapping) else {}
                value = context.get(binding.source_key or binding.input_key)
            elif binding.source_type == "goal":
                value = _get_path_value(runtime.goal.model_dump() if runtime.goal else {}, binding.source_key or binding.input_key)
            elif binding.source_type == "search_spec":
                value = _extract_search_spec_value(state, binding.source_key)
            elif binding.source_type == "step_output":
                value = _extract_step_output(runtime, runtime.plan or ExecutablePlan(goal=Goal(), plan_id="unknown"), binding.step_id)
            elif binding.source_type == "literal":
                value = binding.value
            if value in (None, "", [], {}) and binding.required:
                return resolved, binding.input_key
            if value not in (None, "", [], {}) or binding.required:
                resolved[binding.input_key] = value
        return resolved, None

    def _invoke_step_tool(self, step: PlanStep, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime) -> Any:
        return self._invoke_named_tool(step.tool_name, resolved_input, state, runtime, step)

    def _invoke_named_tool(self, tool_name: str, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Any:
        contract = self.tool_registry.get_contract(tool_name)
        if contract is None or contract.adapter is None:
            raise ValueError(f"Unsupported tool contract adapter: {tool_name}")
        # 测试和兼容调用会 monkeypatch plan_executor.invoke_backend_tool；同步给 contract 层后，
        # adapter 与 executor 仍共用同一个后端工具入口，不需要恢复旧的函数映射表。
        agent_tool_registry.invoke_backend_tool = invoke_backend_tool
        if hasattr(contract.adapter, "invoke_backend_tool"):
            contract.adapter.invoke_backend_tool = invoke_backend_tool
        tool_input = self._validate_tool_input(contract, resolved_input, state)
        if isinstance(tool_input, ToolExecutionResult):
            logger.info(
                "arxiv_agent tool input validation failed: step_id=%s tool_name=%s error_code=%s",
                step.step_id,
                tool_name,
                tool_input.error.error_code if tool_input.error else None,
            )
            return tool_input
        started = perf_counter()
        logger.info(
            "arxiv_agent tool execution started: step_id=%s tool_name=%s backend_tool=%s adapter=%s input=%s",
            step.step_id,
            tool_name,
            getattr(contract, "backend_tool_name", None),
            contract.adapter.__class__.__name__,
            _safe_compact(tool_input.model_dump() if hasattr(tool_input, "model_dump") else tool_input),
        )
        try:
            result = contract.adapter.execute(tool_input)
        except Exception:
            logger.exception(
                "arxiv_agent tool execution raised: step_id=%s tool_name=%s backend_tool=%s elapsed_ms=%.1f",
                step.step_id,
                tool_name,
                getattr(contract, "backend_tool_name", None),
                (perf_counter() - started) * 1000,
            )
            raise
        if not isinstance(result, ToolExecutionResult):
            logger.error(
                "arxiv_agent tool adapter contract violation: step_id=%s tool_name=%s returned_type=%s elapsed_ms=%.1f",
                step.step_id,
                tool_name,
                type(result).__name__,
                (perf_counter() - started) * 1000,
            )
            return ToolExecutionResult(
                ok=False,
                data=None,
                error=_make_tool_error(
                    error_code="adapter_contract_violation",
                    message="ToolAdapter 未返回 ToolExecutionResult",
                    detail={"returned_type": type(result).__name__},
                    recoverable=False,
                    failed_stage="adapter_return",
                ),
                adapter_name=contract.adapter.__class__.__name__,
                tool_name=tool_name,
            )
        if not result.ok:
            logger.info(
                "arxiv_agent tool execution failed: step_id=%s tool_name=%s backend_tool=%s error_code=%s message=%s elapsed_ms=%.1f",
                step.step_id,
                tool_name,
                getattr(contract, "backend_tool_name", None),
                result.error.error_code if result.error else None,
                result.error.message if result.error else None,
                (perf_counter() - started) * 1000,
            )
            return result
        validated_result = self._validate_tool_output(contract, result)
        logger.info(
            "arxiv_agent tool execution finished: step_id=%s tool_name=%s backend_tool=%s ok=%s elapsed_ms=%.1f",
            step.step_id,
            tool_name,
            getattr(contract, "backend_tool_name", None),
            getattr(validated_result, "ok", None),
            (perf_counter() - started) * 1000,
        )
        return validated_result

    def _validate_tool_input(self, contract: Any, resolved_input: Dict[str, Any], state: AgentState) -> Any:
        raw_input = self._augment_tool_input(contract.tool_name, resolved_input, state)
        input_model = contract.input_model
        if input_model is None:
            return raw_input
        try:
            return input_model.model_validate(raw_input)
        except ValidationError as exc:
            return ToolExecutionResult(
                ok=False,
                data=None,
                error=_make_tool_error(
                    error_code="input_validation_error",
                    message="工具输入未通过 Pydantic 模型校验",
                    detail={"errors": exc.errors(), "input_keys": sorted(raw_input.keys())},
                    recoverable=True,
                    retryable=False,
                    suggested_recovery="ask_clarification",
                    failed_stage="input_validation",
                    raw_exception_type=exc.__class__.__name__,
                ),
                adapter_name=contract.adapter.__class__.__name__ if contract.adapter is not None else "",
                tool_name=contract.tool_name,
            )

    def _validate_tool_output(self, contract: Any, result: ToolExecutionResult) -> ToolExecutionResult:
        output_model = contract.output_model
        if output_model is None or result.data is None:
            return result
        try:
            output = result.data if isinstance(result.data, output_model) else output_model.model_validate(_model_to_plain(result.data))
            return result.model_copy(update={"data": output})
        except ValidationError as exc:
            return result.model_copy(
                update={
                    "ok": False,
                    "data": None,
                    "error": _make_tool_error(
                        error_code="output_validation_error",
                        message="工具输出未通过 Pydantic 模型校验",
                        detail={"errors": exc.errors()},
                        recoverable=False,
                        retryable=False,
                        failed_stage="output_validation",
                        raw_exception_type=exc.__class__.__name__,
                    ),
                }
            )

    def _augment_tool_input(self, tool_name: str, resolved_input: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        payload = dict(resolved_input or {})
        # adapter 不直接读 state；这些上下文由 executor 在模型校验前显式注入。
        if tool_name in {"resolve_paper", "resolve_preference_target"}:
            payload.setdefault("context", dict(state.context or {}) if isinstance(state.context, Mapping) else {})
        if tool_name in {"load_user_profile", "load_candidate_papers"}:
            payload.setdefault("context", dict(state.context or {}) if isinstance(state.context, Mapping) else {})
        if tool_name in {"load_user_profile", "generate_recommendations", "update_preference_store"}:
            payload.setdefault("user_id", state.user_id)
        if tool_name in {"load_user_profile", "generate_recommendations", "answer_paper_question"}:
            payload.setdefault("message", state.message)
        if tool_name == "analyze_ambiguity":
            # 结构化澄清要根据现有上下文判断“缺的到底是什么”，而不是回退成只看 message 的占位逻辑。
            payload.setdefault("context", dict(state.context or {}) if isinstance(state.context, Mapping) else {})
            payload.setdefault("user_id", state.user_id)
            payload.setdefault("search_spec", _model_to_plain(state.search_spec) if state.search_spec is not None else {})
            payload.setdefault("pending_action", dict(state.pending_action or {}) if isinstance(state.pending_action, Mapping) else {})
            payload.setdefault(
                "goal",
                {
                    "intent": str(state.intent or "").strip() or "unclear",
                    "goal_type": str(state.intent or "").strip() or "unclear",
                },
            )
        if tool_name == "request_confirmation":
            payload.setdefault("pending_state", dict(state.pending_action or {}) if isinstance(state.pending_action, Mapping) else {})
        return payload

    def _handle_observation_replan(
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
            runtime.pending_confirmation = replan_decision.updated_runtime.pending_confirmation
            runtime.needs_replan = False
            runtime.recovery_strategy = {"type": "patch_plan", "reason": observation.reason or observation.status}
            # 低质量 final_answer 只作为触发重规划的观察对象，不能提前污染最终答案。
            if step.output_key and step.output_key not in runtime.outputs and step.output_key != "final_answer":
                _record_step_output(runtime, step, normalized_output)
            runtime.step_status[step.step_id] = "success"
            self._append_trace(
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
        self._append_trace(
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

    def _append_trace(self, runtime: PlanRuntime, step: PlanStep, *, event: str, status: str, detail: Optional[Dict[str, Any]] = None) -> None:
        runtime.trace.append(
            ExecutionTrace(
                step_id=step.step_id,
                event=event,
                status=status,  # type: ignore[arg-type]
                detail=detail or {},
            )
        )

    def _append_paper_qa_quality_trace(self, runtime: PlanRuntime, step: PlanStep, normalized_output: Any) -> None:
        """质量门完成时单独记录 Paper QA 决策闭环，便于验收脚本无需解析通用 trace。"""
        quality_trace = _build_paper_qa_quality_trace(runtime, step, normalized_output)
        if not quality_trace:
            return
        self._append_trace(
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

    def _build_step_execution_result(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        previous_outputs: Mapping[str, Any],
        previous_trace_len: int,
        previous_error: Optional[str],
        previous_pending: Optional[ConfirmationRequest],
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
            previous_pending=previous_pending,
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
            pending_confirmation=runtime.pending_confirmation,
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
        pending_confirmation: Optional[ConfirmationRequest] = None,
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
            pending_confirmation=pending_confirmation,
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
        previous_pending: Optional[ConfirmationRequest],
        turn_result: Optional[AgentTurnResult],
    ) -> str:
        """把 runtime 状态归一成上层可路由的下一步动作。"""
        if turn_result is not None:
            if turn_result.status == "waiting_confirmation":
                return "wait_for_confirmation"
            if turn_result.status in {"failed", "fallback"}:
                return "fail"
            return "finish"
        if runtime.pending_confirmation and runtime.pending_confirmation != previous_pending:
            return "wait_for_confirmation"
        if status == "failed" or (runtime.error and runtime.error != previous_error):
            return "fail"
        if runtime.needs_replan or (observation and observation.get("status") not in {None, "success", "partial_success"}):
            return "replan"
        return "continue"

    def _build_runtime_patch(self, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> Dict[str, Any]:
        """生成可序列化 runtime patch，供图节点写回 AgentState 或 checkpoint。"""
        runtime_state = self._build_agent_runtime_state(runtime, current_step=current_step)
        return {
            "runtime_state": _json_safe(runtime_state),
            "plan_runtime": _json_safe(runtime),
            "current_step_id": runtime_state.current_step_id,
            "step_status": dict(runtime.step_status or {}),
            "outputs": _json_safe(runtime.outputs or {}),
            "last_observation": _json_safe(runtime.last_observation),
            "last_step_output": _json_safe(runtime.last_step_output),
            "pending_confirmation": _json_safe(runtime.pending_confirmation) if runtime.pending_confirmation else None,
            "needs_replan": bool(runtime.needs_replan),
            "is_finished": bool(runtime_state.is_finished),
            "failure_reason": runtime.error,
            "recovery_strategy": _json_safe(runtime.recovery_strategy or {}),
        }

    def _build_agent_runtime_state(self, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> AgentRuntimeState:
        """从内部 PlanRuntime 投影出跨节点可恢复的一等执行现场。"""
        step_index: Optional[int] = None
        if current_step is not None and runtime.plan is not None:
            for index, step in enumerate(list(runtime.plan.steps or [])):
                if step.step_id == current_step.step_id:
                    step_index = index
                    break
        status_values = list((runtime.step_status or {}).values())
        is_finished = bool(runtime.turn_status) or bool(status_values and all(status in {"success", "failed", "skipped", "waiting_confirmation"} for status in status_values))
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
            approved_step_ids=list(runtime.approved_step_ids or []),
            pending_confirmation=runtime.pending_confirmation,
            needs_replan=bool(runtime.needs_replan),
            is_finished=is_finished,
            failure_reason=runtime.error,
            recovery_strategy=_json_safe(runtime.recovery_strategy) if runtime.recovery_strategy else None,
            turn_status=runtime.turn_status,
            final_answer=runtime.final_answer,
        )

    def _sync_runtime_state(self, state: AgentState, runtime: PlanRuntime, *, current_step: Optional[PlanStep]) -> None:
        """把一等执行现场同步回 AgentState。

        AgentState 仍是图里流转的总状态；runtime_state 是其中专门描述执行现场的边界，
        这样调试视图和后续 LangGraph 节点不用再从 debug/pending_action/tool_outputs 里拼现场。
        """
        runtime.current_step_id = current_step.step_id if current_step else None
        runtime.current_step_index = None
        if current_step is not None and runtime.plan is not None:
            for index, step in enumerate(list(runtime.plan.steps or [])):
                if step.step_id == current_step.step_id:
                    runtime.current_step_index = index
                    break
        state.plan_runtime = runtime
        state.runtime_state = self._build_agent_runtime_state(runtime, current_step=current_step)

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
        if runtime.pending_confirmation:
            runtime.turn_status = "waiting_confirmation"
            return AgentTurnResult(
                status="waiting_confirmation",
                final_answer=runtime.final_answer,
                plan=runtime.plan,
                outputs=dict(runtime.outputs),
                trace=list(runtime.trace),
                pending_confirmation=dict(runtime.pending_confirmation),
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
        """PlanRuntime 是执行时真源，返回结果前把状态回写到 plan，便于上层直接消费。"""
        plan = runtime.plan
        if plan is None:
            return
        plan.steps = [
            step.model_copy(update={"status": runtime.step_status.get(step.step_id, step.status)})
            for step in list(plan.steps or [])
        ]

    def _needs_confirmation(self, step: PlanStep, state: AgentState) -> bool:
        policy = step.confirmation_policy
        if not policy or not policy.requires_confirmation:
            return False
        context = state.context if isinstance(state.context, Mapping) else {}
        runtime_approved_step_ids = []
        if state.plan_runtime is not None:
            runtime_approved_step_ids.extend(list(state.plan_runtime.approved_step_ids or []))
        if state.runtime_state is not None:
            runtime_approved_step_ids.extend(list(state.runtime_state.approved_step_ids or []))
        if step.step_id in runtime_approved_step_ids:
            logger.info(
                "arxiv_agent confirmation bypassed: step_id=%s tool_name=%s source=runtime_approved_step_ids",
                step.step_id,
                step.tool_name,
            )
            return False
        approved_step_ids = context.get("approved_step_ids")
        if isinstance(approved_step_ids, list) and step.step_id in approved_step_ids:
            logger.info(
                "arxiv_agent confirmation bypassed: step_id=%s tool_name=%s source=context_approved_step_ids",
                step.step_id,
                step.tool_name,
            )
            return False
        pending_action = state.pending_action if isinstance(state.pending_action, Mapping) else {}
        approval_decision = str(pending_action.get("decision") or "").strip().lower()
        if approval_decision == "approve" and pending_action.get("step_id") == step.step_id:
            logger.info(
                "arxiv_agent confirmation bypassed: step_id=%s tool_name=%s source=pending_action_decision",
                step.step_id,
                step.tool_name,
            )
            return False
        if pending_action.get("status") == "approved" and pending_action.get("step_id") == step.step_id:
            logger.info(
                "arxiv_agent confirmation bypassed: step_id=%s tool_name=%s source=pending_action_status",
                step.step_id,
                step.tool_name,
            )
            return False
        logger.info(
            "arxiv_agent confirmation required: step_id=%s tool_name=%s runtime_approved=%s context_approved=%s pending_status=%s",
            step.step_id,
            step.tool_name,
            runtime_approved_step_ids,
            approved_step_ids if isinstance(approved_step_ids, list) else [],
            pending_action.get("status"),
        )
        return True

    def _handle_paper_target_confirmation_gate(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        observation: ObservationResult,
        normalized_output: Any,
        started_at: str,
        allow_interrupt: bool,
    ) -> Optional[AgentTurnResult]:
        """把目标解析候选转成可恢复确认请求，而不是让 resolver 的候选结果继续流入业务工具。"""
        payload = _model_to_plain(normalized_output)
        payload = payload if isinstance(payload, Mapping) else {}
        if step.output_key:
            # 候选解析结果先进入 outputs，等待确认时前端/debug 能看到候选来源；确认后会被 resolved 输出覆盖。
            _record_step_output(runtime, step, dict(payload))
        confirmation_request = self._build_paper_target_confirmation_request(
            step=step,
            runtime=runtime,
            state=state,
            payload=payload,
            observation=observation,
        )
        return self._handle_confirmation_gate(
            step=step,
            runtime=runtime,
            state=state,
            confirmation_request=confirmation_request,
            resolved_input=payload,
            started_at=started_at,
            allow_interrupt=allow_interrupt,
        )

    def _build_paper_target_confirmation_request(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        payload: Mapping[str, Any],
        observation: ObservationResult,
    ) -> ConfirmationRequest:
        """生成“确认目标论文”请求；它只确认候选选择，不批准后续收藏/不喜欢等副作用动作。"""
        target_resolution = dict(payload.get("target_resolution") or {})
        raw_candidates = payload.get("candidates") or target_resolution.get("candidates") or []
        candidates = [
            _normalize_confirmation_candidate(candidate)
            for candidate in list(raw_candidates or [])
            if isinstance(candidate, Mapping)
        ]
        recommended_source = payload.get("recommended_candidate") or target_resolution.get("recommended_candidate")
        recommended_candidate = (
            _normalize_confirmation_candidate(recommended_source)
            if isinstance(recommended_source, Mapping)
            else (candidates[0] if candidates else None)
        )
        if recommended_candidate:
            recommended_id = _candidate_display_id(recommended_candidate)
            if recommended_id and not any(recommended_id in _candidate_identity_values(candidate) for candidate in candidates):
                candidates.insert(0, dict(recommended_candidate))
        default_candidate_id = _candidate_display_id(recommended_candidate or {}) if recommended_candidate else None
        reference_hint = dict(payload.get("reference_hint") or target_resolution.get("reference_hint") or {})
        created_at = _utcnow()
        pending_action_id = ":".join(
            item
            for item in [
                str(state.session_id or "session"),
                str(runtime.plan.plan_id if runtime.plan else "plan"),
                step.step_id,
                str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            ]
            if item
        )
        action_type = str(payload.get("action_type") or target_resolution.get("action_type") or step.action_type or "").strip()
        risk_level = str(payload.get("risk_level") or target_resolution.get("risk_level") or "low").strip()
        return ConfirmationRequest(
            request_type="paper_target_confirmation",
            pending_action_id=pending_action_id,
            step_id=step.step_id,
            tool_name=step.tool_name,
            action_type=action_type or step.action_type,
            side_effect_level=str(step.side_effect_level or "none"),
            reason=str(payload.get("resolution_reason") or payload.get("reason") or observation.reason or "paper_target_requires_confirmation").strip(),
            title="确认目标论文",
            description="系统根据当前会话状态找到了可能的论文目标，请确认要继续操作哪一篇。",
            arguments_summary={
                "reference_hint": reference_hint,
                "resolution_reason": payload.get("resolution_reason") or payload.get("reason") or observation.reason,
                "candidate_count": len(candidates),
                "default_candidate_id": default_candidate_id,
                "risk_level": risk_level,
                "action_type": action_type,
            },
            original_question=state.message,
            original_message=state.message,
            target_paper=dict(recommended_candidate or {}) or None,
            candidates=candidates,
            recommended_candidate=dict(recommended_candidate or {}) or None,
            default_candidate_id=default_candidate_id,
            reference_hint=reference_hint,
            target_resolution=target_resolution,
            confirmation_fields={
                "payload_location": "resume.edited_arguments",
                "required": ["pending_action_id", "confirmed_paper_id"],
                "optional": ["confirmed_arxiv_id", "note"],
            },
            created_at=created_at,
            expires_at=_confirmation_expires_at(),
            allowed_decisions=[
                ConfirmationDecisionOption(code="approve", label="确认", description="使用选中的论文继续执行原动作"),
                ConfirmationDecisionOption(code="reject", label="取消", description="取消当前论文动作"),
            ],
            allow_argument_edit=True,
            allow_reject=True,
            allow_note=True,
            trace_id=str(runtime.goal.goal_id if runtime.goal else "") or None,
            plan_id=str(runtime.plan.plan_id if runtime.plan else "") or None,
            session_id=state.session_id,
            thread_id=state.session_id,
        )

    def _materialize_paper_target_confirmation(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        confirmation_request: ConfirmationRequest,
        resume_payload: Any,
    ) -> bool:
        """把用户确认的候选写成 resolver 成功输出；失败时不做默认兜底，避免误选论文。"""
        edited_arguments = _resume_edited_arguments(resume_payload)
        submitted_pending_id = str(edited_arguments.get("pending_action_id") or "").strip()
        if confirmation_request.pending_action_id and not submitted_pending_id:
            self._append_trace(
                runtime,
                step,
                event="paper_target_confirmation_invalid",
                status="failed",
                detail={"reason": "pending_action_id_missing"},
            )
            return False
        if confirmation_request.pending_action_id and submitted_pending_id and submitted_pending_id != confirmation_request.pending_action_id:
            self._append_trace(
                runtime,
                step,
                event="paper_target_confirmation_invalid",
                status="failed",
                detail={"reason": "pending_action_id_mismatch", "submitted_pending_action_id": submitted_pending_id},
            )
            return False
        candidate = _match_confirmed_candidate(confirmation_request.candidates, edited_arguments)
        if candidate is None:
            self._append_trace(
                runtime,
                step,
                event="paper_target_confirmation_invalid",
                status="failed",
                detail={"reason": "confirmed_candidate_not_found", "edited_argument_keys": sorted(edited_arguments.keys())},
            )
            return False

        confirmed_output = _build_confirmed_paper_target_output(
            step=step,
            confirmation_request=confirmation_request,
            candidate=candidate,
        )
        if step.output_key:
            _record_step_output(runtime, step, confirmed_output)
        runtime.last_step_output = {
            "step_id": step.step_id,
            "tool_contract": _json_safe(self.tool_registry.describe_contract(step.tool_name)),
            "tool_execution": None,
            "resolved_input": {"confirmed_by_user": True, "pending_action_id": confirmation_request.pending_action_id},
            "raw_output": _json_safe(confirmed_output),
            "normalized_output": _json_safe(confirmed_output),
            "started_at": _utcnow(),
            "finished_at": _utcnow(),
        }
        runtime.last_observation = {
            "status": "success",
            "reason": "paper_target_user_confirmed",
            "confidence": 0.99,
            "evidence": {
                "confirmed_paper_id": confirmed_output.get("confirmed_paper_id"),
                "confirmed_arxiv_id": confirmed_output.get("confirmed_arxiv_id"),
                "source": candidate.get("source_label") or candidate.get("source"),
            },
        }
        state.context = dict(state.context or {})
        approved_step_ids = list(state.context.get("approved_step_ids") or [])
        if step.step_id not in approved_step_ids:
            approved_step_ids.append(step.step_id)
        state.context["approved_step_ids"] = approved_step_ids
        runtime_approved_step_ids = list(runtime.approved_step_ids or [])
        if step.step_id not in runtime_approved_step_ids:
            runtime_approved_step_ids.append(step.step_id)
        runtime.approved_step_ids = runtime_approved_step_ids
        runtime.step_status[step.step_id] = "success"
        runtime.pending_confirmation = None
        runtime.needs_replan = False
        runtime.recovery_strategy = None
        if state.runtime_state is not None:
            state.runtime_state.pending_confirmation = None
            state.runtime_state.approved_step_ids = list(runtime.approved_step_ids or [])
        state.pending_action = {
            "type": "paper_target_confirmation",
            "status": "approved",
            "decision": "approve",
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "pending_action_id": confirmation_request.pending_action_id,
            "confirmed_paper_id": confirmed_output.get("confirmed_paper_id"),
            "confirmed_arxiv_id": confirmed_output.get("confirmed_arxiv_id"),
        }
        self._append_trace(
            runtime,
            step,
            event="confirmation_approved",
            status="success",
            detail={
                "tool_name": step.tool_name,
                "request_type": confirmation_request.request_type,
                "pending_action_id": confirmation_request.pending_action_id,
                "confirmed_paper_id": confirmed_output.get("confirmed_paper_id"),
                "confirmed_arxiv_id": confirmed_output.get("confirmed_arxiv_id"),
                "confirmed_source": candidate.get("source_label") or candidate.get("source") or candidate.get("source_type"),
                "resumed_at": _utcnow(),
            },
        )
        self._append_trace(
            runtime,
            step,
            event="step_succeeded",
            status="success",
            detail={
                "tool_name": step.tool_name,
                "tool_contract": self.tool_registry.describe_contract(step.tool_name),
                "resolved_input": {"confirmed_by_user": True},
                "raw_output": _compact_step_output_for_trace(step, confirmed_output),
                "normalized_output": _compact_step_output_for_trace(step, confirmed_output),
                "started_at": runtime.last_step_output.get("started_at"),
                "finished_at": runtime.last_step_output.get("finished_at"),
            },
        )
        logger.info(
            "arxiv_agent paper target confirmed: step_id=%s tool_name=%s pending_action_id=%s paper_id=%s arxiv_id=%s source=%s",
            step.step_id,
            step.tool_name,
            confirmation_request.pending_action_id,
            confirmed_output.get("confirmed_paper_id"),
            confirmed_output.get("confirmed_arxiv_id"),
            candidate.get("source_label") or candidate.get("source") or candidate.get("source_type"),
        )
        return True

    def _build_confirmation_request(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        resolved_input: Optional[Mapping[str, Any]],
        reason: Optional[str],
        pending_action: Optional[Mapping[str, Any]],
    ) -> ConfirmationRequest:
        """构建统一的确认请求结构。

        这里把执行器内部的 step/runtime/state 摘要化成轻量 payload，
        供 interrupt、前端展示和兼容字段映射共用，避免后续各层自己拼装一套近似结构。
        """
        pending_action_payload = dict(pending_action or {})
        target_paper: Dict[str, Any] = {}
        for source in (
            pending_action_payload.get("target_paper"),
            pending_action_payload,
            resolved_input.get("paper_reference") if isinstance(resolved_input, Mapping) and isinstance(resolved_input.get("paper_reference"), Mapping) else None,
            resolved_input.get("paper_ref") if isinstance(resolved_input, Mapping) and isinstance(resolved_input.get("paper_ref"), Mapping) else None,
            _extract_step_output(runtime, runtime.plan or ExecutablePlan(goal=Goal(), plan_id="unknown"), "resolve_paper"),
            (state.context or {}).get("selected_paper") if isinstance(state.context, Mapping) else None,
        ):
            if isinstance(source, Mapping):
                for key in ("arxiv_id", "title"):
                    value = source.get(key)
                    if value not in (None, "", [], {}):
                        target_paper[key] = value
                if target_paper:
                    break
        original_question = str(
            pending_action_payload.get("original_question")
            or pending_action_payload.get("qa_question")
            or state.message
            or ""
        ).strip() or None
        action_label = str(pending_action_payload.get("action_label") or "执行该工具").strip()
        title = str(
            pending_action_payload.get("title_text")
            or pending_action_payload.get("title")
            or f"确认是否继续 {action_label}"
        ).strip()
        description = str(
            pending_action_payload.get("description")
            or pending_action_payload.get("answer")
            or pending_action_payload.get("message")
            or "该步骤会触发需要用户审批的工具操作，请先确认是否继续。"
        ).strip()
        arguments_source: Mapping[str, Any]
        if isinstance(resolved_input, Mapping) and resolved_input:
            arguments_source = resolved_input
        elif pending_action_payload:
            arguments_source = pending_action_payload
        else:
            arguments_source = {}
        confirmation_request = ConfirmationRequest(
            step_id=step.step_id,
            tool_name=step.tool_name,
            action_type=step.action_type,
            side_effect_level=str(step.side_effect_level or ""),
            reason=str(reason or pending_action_payload.get("reason") or "waiting_confirmation").strip() or None,
            title=title or None,
            description=description or None,
            arguments_summary=_compact_confirmation_arguments(arguments_source),
            original_question=original_question,
            target_paper=target_paper or None,
            allowed_decisions=[
                ConfirmationDecisionOption(code="approve", label="批准", description="继续执行当前工具操作"),
                ConfirmationDecisionOption(code="reject", label="拒绝", description="取消当前工具操作"),
            ],
            allow_argument_edit=False,
            allow_reject=True,
            allow_note=True,
            trace_id=str(runtime.goal.goal_id if runtime.goal else "") or None,
            plan_id=str(runtime.plan.plan_id if runtime.plan else "") or None,
            session_id=state.session_id,
            thread_id=state.session_id,
        )
        return confirmation_request

    def _handle_confirmation_gate(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        confirmation_request: ConfirmationRequest,
        resolved_input: Mapping[str, Any],
        started_at: str,
        allow_interrupt: bool,
    ) -> Optional[AgentTurnResult]:
        """在副作用工具执行前统一处理确认暂停与恢复。

        图内运行时优先使用 LangGraph interrupt/resume。
        图外直接调用执行器的场景暂时保留旧的 waiting_confirmation 返回，避免非图单测和工具化调用被强制改写。
        """
        runtime.step_status[step.step_id] = "waiting_confirmation"
        runtime.pending_confirmation = confirmation_request
        runtime.recovery_strategy = {"type": "request_confirmation", "reason": confirmation_request.reason or "waiting_confirmation"}
        candidate_sources = _confirmation_candidate_sources(confirmation_request.candidates or [])
        self._append_trace(
            runtime,
            step,
            event="confirmation_requested",
            status="waiting_confirmation",
            detail={
                "tool_name": step.tool_name,
                "request_type": confirmation_request.request_type,
                "pending_action_id": confirmation_request.pending_action_id,
                "side_effect_level": step.side_effect_level,
                "allowed_decisions": [item.code for item in list(confirmation_request.allowed_decisions or [])],
                "candidate_count": len(confirmation_request.candidates or []),
                "candidate_sources": candidate_sources,
                "default_candidate_id": confirmation_request.default_candidate_id,
                "reference_hint": _safe_compact(confirmation_request.reference_hint),
                "target_resolution_status": (confirmation_request.target_resolution or {}).get("status") if isinstance(confirmation_request.target_resolution, Mapping) else None,
                "started_at": started_at,
            },
        )
        logger.info(
            "arxiv_agent confirmation requested: step_id=%s tool_name=%s request_type=%s pending_action_id=%s side_effect_level=%s allowed_decisions=%s candidate_count=%s default_candidate_id=%s candidate_sources=%s reference_hint=%s original_message=%r",
            confirmation_request.step_id,
            confirmation_request.tool_name,
            confirmation_request.request_type,
            confirmation_request.pending_action_id,
            confirmation_request.side_effect_level,
            [item.code for item in list(confirmation_request.allowed_decisions or [])],
            len(confirmation_request.candidates or []),
            confirmation_request.default_candidate_id,
            candidate_sources,
            _safe_compact(confirmation_request.reference_hint),
            confirmation_request.original_message or confirmation_request.original_question,
        )

        if not allow_interrupt:
            return self._build_turn_result(runtime)

        resume_payload = interrupt(confirmation_request.model_dump())
        decision = self._normalize_confirmation_resume_payload(resume_payload)
        logger.info(
            "arxiv_agent confirmation resume received: step_id=%s tool_name=%s decision=%s raw_type=%s",
            step.step_id,
            step.tool_name,
            decision,
            type(resume_payload).__name__,
        )
        if decision == "approve":
            if confirmation_request.request_type == "paper_target_confirmation":
                if self._materialize_paper_target_confirmation(
                    step=step,
                    runtime=runtime,
                    state=state,
                    confirmation_request=confirmation_request,
                    resume_payload=resume_payload,
                ):
                    return None
                return self._handle_confirmation_rejection(
                    step=step,
                    runtime=runtime,
                    state=state,
                    confirmation_request=confirmation_request,
                )
            state.context = dict(state.context or {})
            # 恢复后把当前步骤标为已批准，避免同一工具在续跑时再次触发 interrupt。
            approved_step_ids = list(state.context.get("approved_step_ids") or [])
            if step.step_id not in approved_step_ids:
                approved_step_ids.append(step.step_id)
            state.context["approved_step_ids"] = approved_step_ids
            # 批准状态属于执行现场，必须和 step_status 一起进入 PlanRuntime；
            # 只写 context/pending_action 容易在图恢复后的下一节点合并中丢失，导致同一工具二次确认。
            runtime_approved_step_ids = list(runtime.approved_step_ids or [])
            if step.step_id not in runtime_approved_step_ids:
                runtime_approved_step_ids.append(step.step_id)
            runtime.approved_step_ids = runtime_approved_step_ids
            state.pending_action = {
                "type": "tool_approval",
                "status": "approved",
                "decision": "approve",
                "step_id": step.step_id,
                "tool_name": step.tool_name,
            }
            runtime.step_status[step.step_id] = "pending"
            runtime.pending_confirmation = None
            runtime.recovery_strategy = None
            self._append_trace(
                runtime,
                step,
                event="confirmation_approved",
                status="pending",
                detail={"tool_name": step.tool_name, "resumed_at": _utcnow()},
            )
            logger.info("arxiv_agent confirmation approved: step_id=%s tool_name=%s", step.step_id, step.tool_name)
            return None

        return self._handle_confirmation_rejection(
            step=step,
            runtime=runtime,
            state=state,
            confirmation_request=confirmation_request,
        )

    def _resume_pending_confirmation(self, *, step: PlanStep, runtime: PlanRuntime, state: AgentState) -> str:
        """消费已暂停确认请求的 resume payload，并同步写入执行现场。

        这个方法服务于显式 LangGraph 多节点执行：确认暂停后再次进入 execute_step 时，
        节点会从函数开头重跑，所以必须在 `_needs_confirmation()` 之前把批准态落到
        PlanRuntime / AgentRuntimeState / context 三处真源和兼容镜像。
        """
        confirmation_request = runtime.pending_confirmation
        resume_payload = interrupt(confirmation_request.model_dump() if confirmation_request is not None else {})
        decision = self._normalize_confirmation_resume_payload(resume_payload)
        logger.info(
            "arxiv_agent confirmation resume received: step_id=%s tool_name=%s decision=%s raw_type=%s source=pending_confirmation",
            step.step_id,
            step.tool_name,
            decision,
            type(resume_payload).__name__,
        )
        if decision != "approve":
            return decision
        if confirmation_request is not None and confirmation_request.request_type == "paper_target_confirmation":
            if self._materialize_paper_target_confirmation(
                step=step,
                runtime=runtime,
                state=state,
                confirmation_request=confirmation_request,
                resume_payload=resume_payload,
            ):
                return decision
            return "reject"

        state.context = dict(state.context or {})
        approved_step_ids = list(state.context.get("approved_step_ids") or [])
        if step.step_id not in approved_step_ids:
            approved_step_ids.append(step.step_id)
        state.context["approved_step_ids"] = approved_step_ids

        runtime_approved_step_ids = list(runtime.approved_step_ids or [])
        if step.step_id not in runtime_approved_step_ids:
            runtime_approved_step_ids.append(step.step_id)
        runtime.approved_step_ids = runtime_approved_step_ids
        if state.runtime_state is not None:
            runtime_state_approved = list(state.runtime_state.approved_step_ids or [])
            if step.step_id not in runtime_state_approved:
                runtime_state_approved.append(step.step_id)
            state.runtime_state.approved_step_ids = runtime_state_approved
            state.runtime_state.pending_confirmation = None

        state.pending_action = {
            "type": "tool_approval",
            "status": "approved",
            "decision": "approve",
            "step_id": step.step_id,
            "tool_name": step.tool_name,
        }
        runtime.step_status[step.step_id] = "pending"
        runtime.pending_confirmation = None
        runtime.recovery_strategy = None
        self._append_trace(
            runtime,
            step,
            event="confirmation_approved",
            status="pending",
            detail={"tool_name": step.tool_name, "resumed_at": _utcnow(), "source": "pending_confirmation_resume"},
        )
        logger.info("arxiv_agent confirmation approved: step_id=%s tool_name=%s source=pending_confirmation", step.step_id, step.tool_name)
        return decision

    def _normalize_confirmation_resume_payload(self, payload: Any) -> str:
        """把 resume 输入归一成 approve / reject。

        第一版只接受稳定枚举；遇到未知值时按 reject 处理，避免误执行副作用工具。
        """
        if isinstance(payload, Mapping):
            raw_decision = str(payload.get("decision") or "").strip().lower()
        else:
            raw_decision = str(payload or "").strip().lower()
        if raw_decision == "approve":
            return "approve"
        return "reject"

    def _handle_confirmation_rejection(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        confirmation_request: ConfirmationRequest,
    ) -> AgentTurnResult:
        """处理用户拒绝确认后的取消语义。"""
        runtime.pending_confirmation = None
        runtime.step_status[step.step_id] = "skipped"
        runtime.recovery_strategy = {"type": "skip_step", "reason": "confirmation_rejected"}
        state.pending_action = {
            "type": confirmation_request.request_type,
            "status": "cancelled",
            "decision": "reject",
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "pending_action_id": confirmation_request.pending_action_id,
        }
        self._append_trace(
            runtime,
            step,
            event="confirmation_rejected",
            status="skipped",
            detail={"tool_name": step.tool_name, "request_type": confirmation_request.request_type, "side_effect_level": step.side_effect_level, "finished_at": _utcnow()},
        )
        logger.info("arxiv_agent confirmation rejected: step_id=%s tool_name=%s", step.step_id, step.tool_name)

        record_confirmation_rejection(runtime, confirmation_request=confirmation_request)
        return self._build_turn_result(runtime)

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
