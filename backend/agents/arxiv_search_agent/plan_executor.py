from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from langgraph.types import interrupt
from tools.tool_registry import invoke_tool as invoke_backend_tool

from .observer import Observer
from .planner import PLAN_BUILDER_REGISTRY, GoalBuilder, build_plan_runtime
from .plan_validator import PlanValidator
from .replanner import Replanner
from .schemas import (
    AgentTurnResult,
    ArxivSearchSpec,
    ConfirmationDecisionOption,
    ConfirmationRequest,
    ExecutablePlan,
    ExecutionTrace,
    Goal,
    ObservationResult,
    PlanRuntime,
    PlanStep,
    StepCondition,
)
from .state import AgentState
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry

try:
    from .utils.paper_reference_resolver import _resolve_paper_reference
except Exception:  # pragma: no cover - 测试轻量导入场景下允许缺失
    _resolve_paper_reference = None


PlannerToolImplementation = Callable[[Dict[str, Any], AgentState, PlanRuntime, PlanStep], Any]
logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _normalize_step_output(step: PlanStep, tool_output: Any) -> Any:
    """优先把工具结果归一到 step.output_key 对应的数据，减少下游 binding 解析负担。"""
    if step.output_key is None:
        return tool_output
    if isinstance(tool_output, Mapping):
        if step.output_key in tool_output:
            return tool_output.get(step.output_key)
        if len(step.tool.output_schema or {}) == 1:
            only_key = next(iter(step.tool.output_schema.keys()))
            if only_key in tool_output:
                return tool_output.get(only_key)
    return tool_output


def _matches_declared_type(value: Any, declared_type: Any) -> bool:
    normalized_type = str(declared_type or "").strip().lower()
    if not normalized_type:
        return True
    if normalized_type == "dict":
        return isinstance(value, Mapping)
    if normalized_type == "list":
        return isinstance(value, list)
    if normalized_type == "str":
        return isinstance(value, str)
    if normalized_type == "bool":
        return isinstance(value, bool)
    if normalized_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    return True


def _validate_output_schema(step: PlanStep, raw_output: Any, normalized_output: Any) -> Optional[str]:
    schema = dict(step.tool.output_schema or {})
    if not schema:
        return None
    if len(schema) == 1:
        only_key, only_type = next(iter(schema.items()))
        if isinstance(raw_output, Mapping) and only_key in raw_output:
            if _matches_declared_type(raw_output.get(only_key), only_type):
                return None
            return f"output_schema_type_mismatch:{only_key}"
        # 这类 step 往往直接把“唯一产物本体”作为返回值，不再额外包一层同名字段。
        if step.output_key == only_key and _matches_declared_type(normalized_output, only_type):
            return None
        if not isinstance(raw_output, Mapping) and _matches_declared_type(normalized_output, only_type):
            return None
        if isinstance(raw_output, Mapping) and _matches_declared_type(raw_output, only_type):
            return None
        return "output_schema_invalid"
    if isinstance(raw_output, Mapping):
        for key, declared_type in schema.items():
            if key not in raw_output:
                return f"output_schema_missing_field:{key}"
            if not _matches_declared_type(raw_output.get(key), declared_type):
                return f"output_schema_type_mismatch:{key}"
        return None
    return "output_schema_invalid"


def _extract_papers(search_result: Any) -> List[Dict[str, Any]]:
    if isinstance(search_result, Mapping):
        papers = search_result.get("papers")
        if isinstance(papers, list):
            return [dict(item) for item in papers if isinstance(item, Mapping)]
    if isinstance(search_result, list):
        return [dict(item) for item in search_result if isinstance(item, Mapping)]
    return []


def _resolve_paper_reference_fallback(message: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    selected_paper = context.get("selected_paper")
    if isinstance(selected_paper, Mapping):
        return dict(selected_paper)
    for key in ("papers", "last_papers"):
        papers = context.get(key)
        if isinstance(papers, list) and papers and isinstance(papers[0], Mapping):
            return dict(papers[0])
    arxiv_id_match = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", message or "")
    if arxiv_id_match:
        return {"arxiv_id": arxiv_id_match.group(0), "query": message}
    return {"query": message}


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
        self._planner_tool_impls: Dict[str, PlannerToolImplementation] = {
            "normalize_request": self._normalize_request,
            "build_arxiv_search_spec": self._build_arxiv_search_spec,
            "search_arxiv": self._search_arxiv,
            "validate_arxiv_results": self._validate_arxiv_results,
            "rewrite_arxiv_query": self._rewrite_arxiv_query,
            "personalize_paper_results": self._personalize_paper_results,
            "synthesize_arxiv_response": self._synthesize_arxiv_response,
            "resolve_paper": self._resolve_paper,
            "check_paper_index": self._check_paper_index,
            "request_confirmation": self._request_confirmation,
            "parse_and_index_paper": self._parse_and_index_paper,
            "retrieve_paper_chunks": self._retrieve_paper_chunks,
            "rewrite_paper_query": self._rewrite_paper_query,
            "rerank_paper_chunks": self._rerank_paper_chunks,
            "validate_qa_evidence": self._validate_qa_evidence,
            "generate_paper_answer": self._generate_paper_answer,
            "verify_answer_grounding": self._verify_answer_grounding,
            "load_user_profile": self._load_user_profile,
            "load_candidate_papers": self._load_candidate_papers,
            "generate_recommendations": self._generate_recommendations,
            "validate_recommendations": self._validate_recommendations,
            "explain_recommendations": self._explain_recommendations,
            "resolve_preference_target": self._resolve_preference_target,
            "update_preference_store": self._update_preference_store,
            "update_interest_profile": self._update_interest_profile,
            "verify_preference_update": self._verify_preference_update,
            "synthesize_preference_response": self._synthesize_preference_response,
            "resolve_reading_list_action": self._resolve_reading_list_action,
            "update_reading_list_store": self._update_reading_list_store,
            "verify_reading_list_update": self._verify_reading_list_update,
            "synthesize_reading_list_response": self._synthesize_reading_list_response,
            "analyze_ambiguity": self._analyze_ambiguity,
            "generate_clarification": self._generate_clarification,
            "generate_fallback_response": self._generate_fallback_response,
        }

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
        """允许上层在已有 runtime 基础上续跑，便于后续接确认/重规划。"""
        return self._execute_runtime(runtime, state, allow_interrupt=False)

    def _execute_runtime(self, runtime: PlanRuntime, state: AgentState, *, allow_interrupt: bool) -> AgentTurnResult:
        plan = runtime.plan
        if plan is None:
            runtime.error = "missing_plan"
            runtime.turn_status = "failed"
            return AgentTurnResult(status="failed", error=runtime.error, plan=plan, outputs=dict(runtime.outputs), trace=list(runtime.trace), runtime=runtime)

        while True:
            executable = self._find_executable_step(runtime, state)
            if executable is None:
                break
            next_action = self._execute_step(executable, runtime, state, allow_interrupt=allow_interrupt)
            if next_action is not None:
                return next_action

        self._finalize_blocked_pending_steps(runtime)
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

    def _execute_step(self, step: PlanStep, runtime: PlanRuntime, state: AgentState, *, allow_interrupt: bool) -> Optional[AgentTurnResult]:
        runtime.step_status[step.step_id] = "running"
        started_at = _utcnow()
        resolved_input, missing_input = self._resolve_input_bindings(step, runtime, state)
        if missing_input:
            runtime.step_status[step.step_id] = "failed"
            runtime.error = f"missing_input:{step.step_id}:{missing_input}"
            self._append_trace(
                runtime,
                step,
                event="step_failed",
                status="failed",
                detail={"failure_reason": "missing_input", "missing_input": missing_input, "resolved_input": _safe_compact(resolved_input), "started_at": started_at, "finished_at": _utcnow()},
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

        self._append_trace(runtime, step, event="step_started", status="running", detail={"tool_name": step.tool_name, "resolved_input": _safe_compact(resolved_input), "started_at": started_at})

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
                self._append_trace(
                    runtime,
                    step,
                    event="step_failed",
                    status="failed",
                    detail={"failure_reason": "tool_execution_failed", "error": last_error, "resolved_input": _safe_compact(resolved_input), "started_at": started_at, "finished_at": _utcnow()},
                )
                return None

            normalized_output = _normalize_step_output(step, raw_output)
            schema_error = _validate_output_schema(step, raw_output, normalized_output)
            if schema_error:
                last_error = schema_error
                if attempts < max_attempts and bool(step.tool.can_retry or step.retry_policy):
                    continue
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"{schema_error}:{step.step_id}"
                self._append_trace(
                    runtime,
                    step,
                    event="step_failed",
                    status="failed",
                    detail={"failure_reason": "invalid_output", "error": schema_error, "raw_output": _safe_compact(raw_output), "started_at": started_at, "finished_at": _utcnow()},
                )
                return None

            if step.output_key and step.output_key in runtime.outputs:
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"duplicate_output_key:{step.output_key}"
                self._append_trace(runtime, step, event="step_failed", status="failed", detail={"failure_reason": "duplicate_output_key", "output_key": step.output_key})
                return None

            if any(not _evaluate_condition(condition, state, runtime.model_copy(update={"outputs": dict(runtime.outputs, **({step.output_key: normalized_output} if step.output_key else {}))})) for condition in list(step.postconditions or [])):
                runtime.step_status[step.step_id] = "failed"
                runtime.error = f"postcondition_failed:{step.step_id}"
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
            self._append_trace(
                runtime,
                step,
                event="step_observed",
                status="running",
                detail={
                    "observation_status": observation.status,
                    "observation_reason": observation.reason,
                    "suggested_action": observation.suggested_action,
                    "confidence": observation.confidence,
                },
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
                runtime.outputs[step.output_key] = normalized_output
                if (step.output_key == "final_answer" or step.tool_name == "verify_answer_grounding") and normalized_output is not None:
                    runtime.final_answer = str(normalized_output)

            runtime.step_status[step.step_id] = "success"
            self._append_trace(
                runtime,
                step,
                event="step_succeeded",
                status="success",
                detail={
                    "tool_name": step.tool_name,
                    "resolved_input": _safe_compact(resolved_input),
                    "raw_output": _safe_compact(raw_output),
                    "normalized_output": _safe_compact(normalized_output),
                    "started_at": started_at,
                    "finished_at": _utcnow(),
                },
            )
            return None

        runtime.step_status[step.step_id] = "failed"
        runtime.error = f"tool_execution_failed:{step.step_id}:{last_error or 'unknown_error'}"
        return None

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
        tool_impl = self._planner_tool_impls.get(step.tool_name)
        if tool_impl is None:
            raise ValueError(f"Unsupported planner tool implementation: {step.tool_name}")
        return tool_impl(resolved_input, state, runtime, step)

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
        )
        if replan_decision.updated_plan is not None and replan_decision.updated_runtime is not None:
            runtime.plan = replan_decision.updated_plan
            runtime.replan_counts = dict(replan_decision.updated_runtime.replan_counts or {})
            runtime.step_replan_counts = dict(replan_decision.updated_runtime.step_replan_counts or {})
            runtime.trace = list(replan_decision.updated_runtime.trace or [])
            runtime.step_status = dict(replan_decision.updated_runtime.step_status or runtime.step_status)
            runtime.pending_confirmation = replan_decision.updated_runtime.pending_confirmation
            # 低质量 final_answer 只作为触发重规划的观察对象，不能提前污染最终答案。
            if step.output_key and step.output_key not in runtime.outputs and step.output_key != "final_answer":
                runtime.outputs[step.output_key] = normalized_output
            runtime.step_status[step.step_id] = "success"
            self._append_trace(
                runtime,
                step,
                event="step_replanned",
                status="success",
                detail={
                    "observation_status": observation.status,
                    "observation_reason": observation.reason,
                    "raw_output": _safe_compact(raw_output),
                },
            )
            return None

        runtime.step_status[step.step_id] = "failed"
        fallback_reason = str(replan_decision.fallback_reason or observation.reason or observation.status)
        runtime.error = f"fallback:{step.step_id}:{fallback_reason}"
        fallback_answer = self._generate_fallback_response({"message": state.message or "", "fallback_reason": fallback_reason}, state, runtime, step)
        runtime.final_answer = fallback_answer
        runtime.outputs["final_answer"] = fallback_answer
        self._append_trace(
            runtime,
            step,
            event="replan_fallback",
            status="failed",
            detail={
                "observation_status": observation.status,
                "observation_reason": observation.reason,
                "fallback_reason": fallback_reason,
            },
        )
        runtime.turn_status = "fallback"
        return AgentTurnResult(
            status="fallback",
            final_answer=fallback_answer,
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
        confirmed_step_ids = context.get("confirmed_step_ids")
        if isinstance(confirmed_step_ids, list) and step.step_id in confirmed_step_ids:
            return False
        pending_action = state.pending_action if isinstance(state.pending_action, Mapping) else {}
        confirmation_decision = str(pending_action.get("decision") or "").strip().lower()
        if confirmation_decision == "approve" and pending_action.get("step_id") == step.step_id:
            return False
        if pending_action.get("status") in {"confirmed", "approved"} and pending_action.get("step_id") == step.step_id:
            return False
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
            reason=str(reason or pending_action_payload.get("reason") or "waiting_for_user_confirmation").strip() or None,
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
        self._append_trace(
            runtime,
            step,
            event="confirmation_requested",
            status="waiting_confirmation",
            detail={
                "tool_name": step.tool_name,
                "side_effect_level": step.side_effect_level,
                "allowed_decisions": [item.code for item in list(confirmation_request.allowed_decisions or [])],
                "started_at": started_at,
            },
        )
        logger.info(
            "arxiv_agent confirmation requested: step_id=%s tool_name=%s side_effect_level=%s allowed_decisions=%s",
            confirmation_request.step_id,
            confirmation_request.tool_name,
            confirmation_request.side_effect_level,
            [item.code for item in list(confirmation_request.allowed_decisions or [])],
        )

        if not allow_interrupt:
            return self._build_turn_result(runtime)

        resume_payload = interrupt(confirmation_request.model_dump())
        decision = self._normalize_confirmation_resume_payload(resume_payload)
        if decision == "approve":
            state.context = dict(state.context or {})
            confirmed_step_ids = list(state.context.get("confirmed_step_ids") or [])
            if step.step_id not in confirmed_step_ids:
                confirmed_step_ids.append(step.step_id)
            state.context["confirmed_step_ids"] = confirmed_step_ids
            state.pending_action = {
                "type": "tool_approval",
                "status": "approved",
                "decision": "approve",
                "step_id": step.step_id,
                "tool_name": step.tool_name,
            }
            runtime.step_status[step.step_id] = "pending"
            runtime.pending_confirmation = None
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
        state.pending_action = {
            "type": "tool_approval",
            "status": "cancelled",
            "decision": "reject",
            "step_id": step.step_id,
            "tool_name": step.tool_name,
        }
        self._append_trace(
            runtime,
            step,
            event="confirmation_rejected",
            status="skipped",
            detail={"tool_name": step.tool_name, "side_effect_level": step.side_effect_level, "finished_at": _utcnow()},
        )
        logger.info("arxiv_agent confirmation rejected: step_id=%s tool_name=%s", step.step_id, step.tool_name)

        if step.tool_name == "parse_and_index_paper":
            target_paper = dict(confirmation_request.target_paper or {})
            paper_label = str(target_paper.get("title") or target_paper.get("arxiv_id") or "该论文").strip()
            runtime.final_answer = f"已取消解析 {paper_label}，因此无法继续基于全文回答。"
            runtime.outputs["final_answer"] = runtime.final_answer
            return self._build_turn_result(runtime)

        return self._build_turn_result(runtime)

    def _normalize_request(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        search_spec = resolved_input.get("search_spec")
        if isinstance(search_spec, ArxivSearchSpec):
            search_spec_payload = search_spec.model_dump()
        elif isinstance(search_spec, Mapping):
            search_spec_payload = dict(search_spec)
        else:
            search_spec_payload = state.search_spec.model_dump() if state.search_spec is not None else None
        return {
            "intent": str(resolved_input.get("intent") or state.intent or "").strip() or "unsupported",
            "message": str(resolved_input.get("message") or state.message or "").strip(),
            "search_spec": search_spec_payload,
        }

    def _build_arxiv_search_spec(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        normalized_request = resolved_input.get("normalized_request") if isinstance(resolved_input.get("normalized_request"), Mapping) else resolved_input
        existing_spec = normalized_request.get("search_spec") if isinstance(normalized_request, Mapping) else None
        if isinstance(existing_spec, ArxivSearchSpec):
            spec = existing_spec
        elif isinstance(existing_spec, Mapping) and existing_spec:
            spec = ArxivSearchSpec.model_validate(existing_spec)
        elif state.search_spec is not None:
            spec = state.search_spec
        else:
            message = str((normalized_request or {}).get("message") or state.message or "").strip()
            spec = ArxivSearchSpec(intent="arxiv_search", query=message, max_results=10)
        return spec.model_dump()

    def _search_arxiv(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        search_spec = resolved_input.get("search_spec")
        if isinstance(search_spec, ArxivSearchSpec):
            tool_args = search_spec.model_dump(exclude_none=True)
        elif isinstance(search_spec, Mapping):
            tool_args = {key: value for key, value in dict(search_spec).items() if value is not None}
        else:
            raise ValueError("search_arxiv requires search_spec")
        tool_result = invoke_backend_tool("search_arxiv_structured", **tool_args)
        papers = _extract_papers((tool_result or {}).get("data") or {})
        return {
            "papers": papers,
            "tool_result": tool_result,
            "search_spec": tool_args,
        }

    def _validate_arxiv_results(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        arxiv_results = resolved_input.get("arxiv_results")
        papers = _extract_papers(arxiv_results)
        tool_result = arxiv_results.get("tool_result") if isinstance(arxiv_results, Mapping) else None
        warnings: List[str] = []
        if not papers:
            warnings.append("no_results")
        if isinstance(tool_result, Mapping) and not bool(tool_result.get("ok", False)):
            warnings.append("tool_failed")
        return {
            "ok": bool(papers) and "tool_failed" not in warnings,
            "result_count": len(papers),
            "warnings": warnings,
        }

    def _rewrite_arxiv_query(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        search_spec = resolved_input.get("search_spec")
        if isinstance(search_spec, Mapping):
            payload = dict(search_spec)
        elif state.search_spec is not None:
            payload = state.search_spec.model_dump()
        else:
            payload = {"intent": "arxiv_search", "query": state.message or "", "max_results": 10}
        if not payload.get("query") and state.message:
            payload["query"] = state.message
        payload["title_query"] = None
        payload["abstract_query"] = None
        return payload

    def _personalize_paper_results(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> List[Dict[str, Any]]:
        del runtime, step
        papers = _extract_papers(resolved_input.get("arxiv_results"))
        personalized = bool(resolved_input.get("user_memory_summary") or resolved_input.get("research_profile"))
        ranked: List[Dict[str, Any]] = []
        for index, paper in enumerate(papers, start=1):
            next_paper = dict(paper)
            next_paper["rank"] = index
            if personalized:
                next_paper.setdefault("reason", "matched_profile_context")
            ranked.append(next_paper)
        state.personalized_rerank_applied = personalized
        return ranked

    def _synthesize_arxiv_response(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del runtime, step
        ranked_papers = resolved_input.get("ranked_papers")
        if not isinstance(ranked_papers, list) or not ranked_papers:
            ranked_papers = _extract_papers(runtime.outputs.get("arxiv_results"))
        quality = resolved_input.get("arxiv_result_quality") if isinstance(resolved_input.get("arxiv_result_quality"), Mapping) else {}
        result_count = int(quality.get("result_count") or len(ranked_papers or []))
        if result_count <= 0:
            return "当前没有检索到合适的 arXiv 结果，建议收窄或改写查询后重试。"
        titles = [str((paper or {}).get("title") or "").strip() for paper in list(ranked_papers or [])[:3] if str((paper or {}).get("title") or "").strip()]
        title_summary = "；".join(titles) if titles else "已返回相关论文"
        if state.personalized_rerank_applied:
            return f"已检索到 {result_count} 篇相关 arXiv 论文，并结合用户上下文完成排序。优先关注：{title_summary}。"
        return f"已检索到 {result_count} 篇相关 arXiv 论文。优先关注：{title_summary}。"

    def _resolve_paper(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        message = str(resolved_input.get("message") or state.message or "").strip()
        context = state.context if isinstance(state.context, Mapping) else {}
        selected_paper = resolved_input.get("selected_paper")
        logger.info(
            "arxiv_agent resolve_paper: message=%s selected_paper_title=%s selected_paper_arxiv_id=%s context_keys=%s recent_paper_count=%s",
            message,
            str((selected_paper or {}).get("title") or "").strip() if isinstance(selected_paper, Mapping) else "",
            str((selected_paper or {}).get("arxiv_id") or "").strip() if isinstance(selected_paper, Mapping) else "",
            sorted(context.keys()),
            len(context.get("last_papers") or []) if isinstance(context.get("last_papers"), list) else 0,
        )
        if callable(_resolve_paper_reference):
            resolution = _resolve_paper_reference(message, context)
            if isinstance(resolution, Mapping):
                logger.info(
                    "arxiv_agent resolve_paper result: source=resolver arxiv_id=%s title=%s matched_by=%s",
                    str(resolution.get("arxiv_id") or "").strip(),
                    str(resolution.get("title") or "").strip(),
                    str(resolution.get("matched_by") or resolution.get("source") or "").strip(),
                )
                return dict(resolution)
        if isinstance(selected_paper, Mapping):
            logger.info(
                "arxiv_agent resolve_paper result: source=selected_paper arxiv_id=%s title=%s",
                str(selected_paper.get("arxiv_id") or "").strip(),
                str(selected_paper.get("title") or "").strip(),
            )
            return dict(selected_paper)
        fallback_resolution = _resolve_paper_reference_fallback(message, context)
        logger.info(
            "arxiv_agent resolve_paper result: source=fallback arxiv_id=%s title=%s matched_by=%s",
            str(fallback_resolution.get("arxiv_id") or "").strip(),
            str(fallback_resolution.get("title") or "").strip(),
            str(fallback_resolution.get("matched_by") or fallback_resolution.get("source") or "").strip(),
        )
        return fallback_resolution

    def _check_paper_index(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        paper_ref = resolved_input.get("paper_ref")
        arxiv_id = str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""
        logger.info(
            "arxiv_agent check_paper_index: arxiv_id=%s title=%s",
            arxiv_id,
            str((paper_ref or {}).get("title") or "").strip() if isinstance(paper_ref, Mapping) else "",
        )
        if not arxiv_id:
            return {"status": "missing", "has_index": False}
        tool_result = invoke_backend_tool("check_paper_qa_index", arxiv_id=arxiv_id)
        data = (tool_result or {}).get("data") if isinstance(tool_result, Mapping) else {}
        logger.info(
            "arxiv_agent check_paper_index result: arxiv_id=%s status=%s has_index=%s",
            arxiv_id,
            str((data or {}).get("status") or "").strip(),
            (data or {}).get("has_index"),
        )
        return dict(data or {"status": "unknown", "has_index": False, "tool_result": tool_result})

    def _request_confirmation(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        pending_action = resolved_input.get("pending_action")
        pending_state = state.pending_action if isinstance(state.pending_action, Mapping) else {}
        status = "confirmed" if pending_state.get("status") in {"confirmed", "approved"} else "waiting_confirmation"
        return {"status": status, "pending_action": pending_action}

    def _parse_and_index_paper(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        paper_ref = resolved_input.get("paper_reference") or resolved_input.get("paper_ref")
        arxiv_id = str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""
        if not arxiv_id:
            raise ValueError("parse_and_index_paper requires arxiv_id")
        tool_result = invoke_backend_tool("build_paper_qa_index", arxiv_id=arxiv_id)
        return dict((tool_result or {}).get("data") or {"tool_result": tool_result})

    def _retrieve_paper_chunks(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> List[Dict[str, Any]]:
        del runtime, step
        context = state.context if isinstance(state.context, Mapping) else {}
        cached_chunks = context.get("paper_chunks")
        if isinstance(cached_chunks, list):
            return [dict(item) for item in cached_chunks if isinstance(item, Mapping)]
        paper_ref = resolved_input.get("paper_ref")
        title = str((paper_ref or {}).get("title") or "").strip() if isinstance(paper_ref, Mapping) else ""
        message = str(resolved_input.get("message") or state.message or "").strip()
        return [{"chunk_id": "synthetic-1", "text": title or message, "score": 1.0}]

    def _rewrite_paper_query(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del runtime, step
        return str(resolved_input.get("message") or state.message or "").strip()

    def _rerank_paper_chunks(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> List[Dict[str, Any]]:
        del state, runtime, step
        chunks = resolved_input.get("retrieved_chunks")
        if not isinstance(chunks, list):
            return []
        normalized_chunks = [dict(item) for item in chunks if isinstance(item, Mapping)]
        normalized_chunks.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return normalized_chunks

    def _validate_qa_evidence(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        reranked_chunks = resolved_input.get("reranked_chunks")
        count = len(reranked_chunks) if isinstance(reranked_chunks, list) else 0
        return {"ok": count > 0, "chunk_count": count}

    def _generate_paper_answer(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del step
        paper_ref = runtime.outputs.get("paper_ref")
        arxiv_id = str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""
        question = str(resolved_input.get("message") or state.message or "").strip()
        if not arxiv_id:
            return {"answer": "", "sources": [], "tool_result": None, "error": "missing_arxiv_id"}
        tool_result = invoke_backend_tool(
            "answer_paper_question",
            arxiv_id=arxiv_id,
            question=question,
            stricter_grounding=bool(resolved_input.get("stricter_grounding") or False),
        )
        tool_data = dict((tool_result or {}).get("data") or {})
        tool_data["tool_result"] = tool_result
        return tool_data

    def _verify_answer_grounding(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del state, runtime, step
        draft_answer = resolved_input.get("draft_answer")
        if isinstance(draft_answer, Mapping):
            answer = str(draft_answer.get("answer") or "").strip()
            if answer:
                return answer
        return str(draft_answer or "").strip()

    def _load_user_profile(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        context = resolved_input.get("context") if isinstance(resolved_input.get("context"), Mapping) else {}
        profile = context.get("research_profile") if isinstance(context.get("research_profile"), Mapping) else {}
        return {
            "user_id": state.user_id,
            "research_profile": dict(profile),
            "user_memory_summary": context.get("user_memory_summary"),
            "message": state.message,
            "request_context": dict(context),
        }

    def _load_candidate_papers(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> List[Dict[str, Any]]:
        del runtime, step
        context = state.context if isinstance(state.context, Mapping) else {}
        for key in ("papers", "last_papers", "candidate_papers"):
            papers = context.get(key)
            if isinstance(papers, list):
                return [dict(item) for item in papers if isinstance(item, Mapping)]
        return []

    def _generate_recommendations(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        profile = resolved_input.get("recommendation_profile") if isinstance(resolved_input.get("recommendation_profile"), Mapping) else {}
        # 画像为空时 Replanner 会降级为“当前消息驱动”的推荐，避免把空画像继续传给推荐工具。
        message = profile.get("message") or state.message
        tool_result = invoke_backend_tool(
            "recommend_papers",
            user_id=state.user_id or "",
            message=message,
            user_memory_summary=profile.get("user_memory_summary"),
            research_profile=profile.get("research_profile"),
            request_context=profile.get("request_context"),
        )
        return {
            "recommendations": list((((tool_result or {}).get("data") or {}).get("recommendations") or [])),
            "tool_result": tool_result,
            "candidate_papers": resolved_input.get("candidate_papers"),
        }

    def _validate_recommendations(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        result = resolved_input.get("recommendation_result") if isinstance(resolved_input.get("recommendation_result"), Mapping) else {}
        recommendations = result.get("recommendations")
        return {
            "ok": isinstance(recommendations, list) and len(recommendations) > 0,
            "recommendations": list(recommendations or []),
        }

    def _explain_recommendations(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del state, runtime, step
        validated = resolved_input.get("validated_recommendations") if isinstance(resolved_input.get("validated_recommendations"), Mapping) else {}
        recommendations = list(validated.get("recommendations") or [])
        if not recommendations:
            return "当前没有生成可用的推荐结果，建议先补充偏好或改成明确主题搜索。"
        titles = [str((item or {}).get("title") or "").strip() for item in recommendations[:3] if isinstance(item, Mapping)]
        return f"已生成推荐结果，可优先阅读：{'；'.join([title for title in titles if title]) or '候选论文列表'}。"

    def _resolve_preference_target(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        return self._resolve_paper(resolved_input, state, runtime, step)

    def _update_preference_store(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        paper_reference = resolved_input.get("paper_reference")
        arxiv_id = str((paper_reference or {}).get("arxiv_id") or "").strip() if isinstance(paper_reference, Mapping) else ""
        message = str(resolved_input.get("message") or state.message or "")
        lowered = message.lower()
        liked = not any(token in lowered for token in ("不喜欢", "取消喜欢", "dislike", "remove like", "unlike"))
        tool_result = invoke_backend_tool(
            "record_paper_preference",
            user_id=state.user_id or "",
            arxiv_id=arxiv_id,
            liked=liked,
            paper=dict(paper_reference) if isinstance(paper_reference, Mapping) else None,
        )
        return {
            "arxiv_id": arxiv_id,
            "liked": liked,
            "tool_result": tool_result,
        }

    def _update_interest_profile(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        preference_action_result = resolved_input.get("preference_action_result")
        return {"synced": True, "preference_action_result": preference_action_result}

    def _verify_preference_update(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        action_result = resolved_input.get("preference_action_result")
        return {"ok": isinstance(action_result, Mapping) and bool(action_result.get("arxiv_id")), "detail": action_result}

    def _synthesize_preference_response(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del state, runtime, step
        verified = resolved_input.get("verified_preference_update") if isinstance(resolved_input.get("verified_preference_update"), Mapping) else {}
        detail = verified.get("detail") if isinstance(verified.get("detail"), Mapping) else {}
        if not verified.get("ok"):
            return "偏好更新未成功，请确认目标论文后重试。"
        return f"已更新论文偏好：{detail.get('arxiv_id') or '目标论文'}。"

    def _resolve_reading_list_action(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        message = str(resolved_input.get("message") or state.message or "").strip()
        action = "add"
        if any(token in message.lower() for token in ("移除", "删除", "remove", "delete")):
            action = "remove"
        paper_reference = _resolve_paper_reference_fallback(message, state.context if isinstance(state.context, Mapping) else {})
        return {"action": action, "paper_reference": paper_reference}

    def _update_reading_list_store(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        reading_list_action = resolved_input.get("reading_list_action") if isinstance(resolved_input.get("reading_list_action"), Mapping) else {}
        return {"ok": True, **reading_list_action}

    def _verify_reading_list_update(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del state, runtime, step
        result = resolved_input.get("reading_list_result") if isinstance(resolved_input.get("reading_list_result"), Mapping) else {}
        return {"ok": bool(result.get("ok")), "detail": result}

    def _synthesize_reading_list_response(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del state, runtime, step
        verified = resolved_input.get("verified_reading_list_update") if isinstance(resolved_input.get("verified_reading_list_update"), Mapping) else {}
        detail = verified.get("detail") if isinstance(verified.get("detail"), Mapping) else {}
        if not verified.get("ok"):
            return "阅读列表更新未成功，请确认操作对象后重试。"
        action = detail.get("action") or "update"
        paper_reference = detail.get("paper_reference") if isinstance(detail.get("paper_reference"), Mapping) else {}
        label = paper_reference.get("title") or paper_reference.get("arxiv_id") or "目标论文"
        return f"已完成阅读列表操作：{action} {label}。"

    def _analyze_ambiguity(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> Dict[str, Any]:
        del runtime, step
        message = str(resolved_input.get("message") or state.message or "").strip()
        return {"message": message, "missing_fields": ["topic"], "reason": "request_is_ambiguous"}

    def _generate_clarification(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del state, runtime, step
        missing_information = resolved_input.get("missing_information") if isinstance(resolved_input.get("missing_information"), Mapping) else {}
        missing_fields = list(missing_information.get("missing_fields") or [])
        if missing_fields:
            return f"我还缺少关键信息：{', '.join(str(item) for item in missing_fields)}。请补充后我再继续。"
        return "当前请求还不够明确，请补充目标主题、论文或操作对象。"

    def _generate_fallback_response(self, resolved_input: Dict[str, Any], state: AgentState, runtime: PlanRuntime, step: PlanStep) -> str:
        del state, runtime, step
        message = str(resolved_input.get("message") or "").strip()
        if message:
            return f"当前请求暂不在该 agent 的支持范围内：{message}。建议改成 arXiv 搜索、论文问答、推荐或偏好更新。"
        return "当前请求暂不在该 agent 的支持范围内。"
def run_agent_turn(state: AgentState, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> AgentTurnResult:
    """统一 Agent 单轮入口：构造目标、生成计划、校验计划并执行。

    主流程从这里进入后，PlanExecutor 会在同一份 runtime 中完成工具执行、质量观察和
    rule-based 重规划，避免旧节点链把计划、工具输出和 trace 分散到多套状态里。
    """
    goal = GoalBuilder.from_state(state)
    builder = PLAN_BUILDER_REGISTRY.get(goal.goal_type or "unsupported")
    plan = builder.build(goal, state, tool_registry)
    # 所有进入执行器的 plan 都先过统一校验，确保工具名、依赖拓扑和副作用声明一致。
    PlanValidator().validate(plan, tool_registry)
    state.goal = goal
    state.execution_plan = plan
    return PlanExecutor(tool_registry=tool_registry).execute(plan, state)


def run_agent_turn_in_graph(state: AgentState, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> AgentTurnResult:
    """图内执行入口：允许确认链路通过 LangGraph interrupt/resume 恢复。"""
    goal = GoalBuilder.from_state(state)
    builder = PLAN_BUILDER_REGISTRY.get(goal.goal_type or "unsupported")
    plan = builder.build(goal, state, tool_registry)
    PlanValidator().validate(plan, tool_registry)
    state.goal = goal
    state.execution_plan = plan
    runtime = build_plan_runtime(state, goal=plan.goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
    runtime.outputs = {}
    runtime.trace = []
    runtime.retry_counts = {}
    runtime.replan_counts = {}
    runtime.step_replan_counts = {}
    return PlanExecutor(tool_registry=tool_registry)._execute_runtime(runtime, state, allow_interrupt=True)


def execute_executable_plan(state: AgentState, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> AgentTurnResult:
    """便捷入口：从 AgentState 先构造计划，再立刻执行。"""
    return run_agent_turn(state, tool_registry=tool_registry)


__all__ = ["PlanExecutor", "run_agent_turn", "run_agent_turn_in_graph", "execute_executable_plan"]
