"""通用工具执行节点。

这个模块负责把 `AgentState.tool_call_request` 转成一次标准化工具执行：
1. 读取待执行工具请求；
2. 通过现有 tool_registry.invoke_tool 复用参数校验与执行逻辑；
3. 把结果回写到旧字段（tool_name/tool_args/tool_result/tool_calls）；
4. 同步产出新的 ToolObservation，作为阶段 2 的统一观察协议。

注意：
- 本节点不负责决定下一步；
- 不负责重试；
- 不负责选择工具；
- 不负责生成最终回答。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

_BACKEND_DIR = str(Path(__file__).resolve().parents[3])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:  # pragma: no cover - import path differs between backend cwd and package import
    from tools.tool_registry import invoke_tool
except ModuleNotFoundError:  # pragma: no cover
    from backend.tools.tool_registry import invoke_tool

from services.paper_qa.repair_actions import RETRY_WITH_EXPANDED_CONTEXT, RETRY_WITH_QUERY_REWRITE

from ..schemas import AgentToolCall, ToolObservation
from ..state import AgentState
from ..utils.result_utils import _extract_error_message, _result_mapping, _result_ok, _result_text, _to_plain_dict
from ..utils.state_utils import _append_step, _coerce_state, _get_next_executable_plan_step, _update_execution_plan_step
from .plan_step_mapping import build_tool_call_request_from_plan_step


def _build_result_ref(result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """从工具结果中提取一个轻量引用，避免 observation 过重。"""
    data = result.get("data")
    if isinstance(data, dict):
        compact: Dict[str, Any] = {}
        for key in ("status", "message", "arxiv_id", "title", "has_index", "remove_scope", "interest_profile_mode", "interest_cluster_count", "recall_mode"):
            value = data.get(key)
            if value not in (None, "", [], {}):
                compact[key] = value

        answer = str(data.get("answer") or "").strip()
        if answer:
            compact["answer_preview"] = answer[:200]

        for key in ("sources", "papers", "recommendations", "recommended_papers", "items"):
            value = data.get(key)
            if isinstance(value, list):
                compact[f"{key}_count"] = len(value)
                preview_items = []
                for item in value[:3]:
                    if isinstance(item, Mapping):
                        preview = {
                            "arxiv_id": item.get("arxiv_id") or item.get("id"),
                            "title": item.get("title"),
                        }
                        preview_items.append({k: v for k, v in preview.items() if v not in (None, "")})
                if preview_items:
                    compact[f"{key}_preview"] = preview_items
                break

        retrieval_debug = data.get("retrieval_debug")
        if isinstance(retrieval_debug, Mapping):
            compact["retrieval_debug_keys"] = sorted(str(key) for key in retrieval_debug.keys())
        qa_observation = data.get("qa_observation")
        if isinstance(qa_observation, Mapping):
            # observation 只摘取质量和建议，避免把完整 stage map 塞进工具观察摘要。
            compact["qa_observation"] = {
                key: qa_observation.get(key)
                for key in (
                    "retrieval_quality",
                    "retrieval_quality_reason",
                    "answer_quality",
                    "answer_quality_reason",
                    "missing_evidence_type",
                    "weak_source_reason",
                    "degraded_stages",
                    "observation_reason",
                    "recommended_repair_actions",
                )
                if qa_observation.get(key) not in (None, "", [], {})
            }

        return compact or {"data_keys": sorted(str(key) for key in data.keys())[:20]}
    if data is not None:
        return {"data": data}

    trace = _result_mapping(result, "trace") or {}
    if trace:
        compact_trace = {
            key: trace.get(key)
            for key in ("tool_name", "returned_count", "source", "timestamp")
            if trace.get(key) not in (None, "", [], {})
        }
        if compact_trace:
            return compact_trace
    return None


def _make_failed_result(tool_name: Optional[str], message: str, *, code: str) -> Dict[str, Any]:
    """为节点级失败分支构造统一的失败结果。"""
    return {
        "ok": False,
        "tool_name": tool_name,
        "summary": message,
        "data": None,
        "trace": {"tool_name": tool_name},
        "error": {"code": code, "message": message},
    }


def _derive_observation_details(tool_name: Optional[str], result: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive protocol-level observation hints from tool result data."""
    ok = _result_ok(result)
    if not ok:
        return {
            "is_sufficient": False,
            "next_action_hint": "inspect_tool_request_or_choose_alternative",
        }

    data = _result_mapping(result, "data") or {}
    normalized_tool_name = str(tool_name or "").strip()
    if normalized_tool_name == "check_paper_qa_index":
        has_index = bool(data.get("has_index")) or str(data.get("status") or "").strip().lower() == "indexed"
        return {
            "is_sufficient": has_index,
            "next_action_hint": None if has_index else "ask_user_confirmation",
        }

    if normalized_tool_name == "build_paper_qa_index":
        has_index = bool(data.get("has_index")) or str(data.get("status") or "").strip().lower() in {"indexed", "success"}
        return {
            "is_sufficient": has_index,
            "next_action_hint": None if has_index else "inspect_build_result_or_retry_index_build",
        }

    if normalized_tool_name == "answer_paper_question":
        answer = str(data.get("answer") or "").strip()
        qa_observation = data.get("qa_observation") if isinstance(data.get("qa_observation"), Mapping) else {}
        if not answer:
            return {
                "is_sufficient": False,
                "next_action_hint": "answer_with_available_context",
            }
        if qa_observation:
            answer_quality = str(qa_observation.get("answer_quality") or "").strip()
            retrieval_quality = str(qa_observation.get("retrieval_quality") or "").strip()
            answer_insufficient = str(qa_observation.get("answer_insufficient_evidence") or "").strip()
            repair_actions = qa_observation.get("recommended_repair_actions") if isinstance(qa_observation.get("recommended_repair_actions"), list) else []
            repair_hint = next((str(action) for action in repair_actions if str(action).strip()), "")
            if answer_quality in {"insufficient_evidence", "generation_failed"} or answer_insufficient == "yes":
                return {
                    "is_sufficient": False,
                    "next_action_hint": repair_hint or RETRY_WITH_EXPANDED_CONTEXT,
                }
            if retrieval_quality in {"failed", "weak"}:
                return {
                    "is_sufficient": False,
                    "next_action_hint": repair_hint or RETRY_WITH_QUERY_REWRITE,
                }
            degraded_stages = qa_observation.get("degraded_stages") if isinstance(qa_observation.get("degraded_stages"), list) else []
            if retrieval_quality == "partial" or answer_quality == "warning" or degraded_stages:
                # 有答案不代表链路完全健康；保留 sufficient，同时给 Agent 暴露可选的质量修复提示。
                return {
                    "is_sufficient": True,
                    "next_action_hint": repair_hint or "inspect_qa_observation_degraded",
                }
        return {
            "is_sufficient": bool(answer),
            "next_action_hint": None,
        }

    if normalized_tool_name == "recommend_papers":
        paper_candidates = []
        for key in ("recommended_papers", "recommendations", "items", "papers"):
            value = data.get(key)
            if isinstance(value, list):
                paper_candidates = value
                break
        personalization_signals = data.get("personalization_signals") if isinstance(data.get("personalization_signals"), Mapping) else {}
        has_behavior_history = bool(personalization_signals.get("has_behavior_history"))
        return {
            "is_sufficient": bool(paper_candidates),
            "next_action_hint": None if paper_candidates else ("adjust_recommendation_constraints_or_collect_more_preferences" if has_behavior_history else "fallback_to_normal_search"),
        }

    if normalized_tool_name in {"record_paper_preference", "remove_paper_preference", "record_user_paper_preference", "remove_user_paper_preference"}:
        return {
            "is_sufficient": ok,
            "next_action_hint": None if ok else "check_existing_preference_or_retry_mutation",
        }

    return {
        "is_sufficient": ok,
        "next_action_hint": None,
    }


def _append_observation(state: AgentState, observation: ToolObservation) -> None:
    """把 observation 追加到状态中。"""
    state.tool_observations = list(state.tool_observations or []) + [observation]


def _append_tool_call(state: AgentState, *, tool_name: Optional[str], arguments: Dict[str, Any], result: Mapping[str, Any]) -> None:
    """把工具结果同步写入旧版 AgentToolCall 轨迹，保持兼容。"""
    state.tool_calls = list(state.tool_calls or []) + [
        AgentToolCall(
            tool_name=str(tool_name or ""),
            arguments=dict(arguments or {}),
            status="success" if _result_ok(result) else "failed",
            summary=_result_text(result, "summary"),
            trace=_result_mapping(result, "trace"),
            error=_result_mapping(result, "error"),
        )
    ]


def _record_plan_step_observation(
    state: AgentState,
    *,
    plan_step_id: Optional[str],
    tool_name: Optional[str],
    observation: ToolObservation,
) -> None:
    """把计划步骤与最近一次 observation 的轻量关联写入 debug。"""
    if not plan_step_id:
        return

    debug = dict(state.debug or {})
    plan_step_observations = dict(debug.get("plan_step_observations", {}))
    plan_step_observations[plan_step_id] = {
        "tool_name": tool_name,
        "status": observation.status,
        "ok": observation.ok,
        "result_summary": observation.result_summary,
        "next_action_hint": observation.next_action_hint,
        "error": observation.error,
    }
    debug["plan_step_observations"] = plan_step_observations
    state.debug = debug


def execute_planned_tool_step(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """统一执行“计划步骤 -> 工具请求 -> 工具执行 -> observation 回写”闭环。

    当前实现先覆盖白名单式的计划步骤映射：
    - 如果上游已经准备好了 tool_call_request，则直接走 execute_tool；
    - 否则根据当前可执行计划步骤动态构造请求；
    - 若当前步骤无法映射为工具请求，则安全产出 skipped observation，并把步骤标记为 skipped。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.tool_call_request is None:
        request, mapping_debug = build_tool_call_request_from_plan_step(next_state)
        debug = dict(next_state.debug or {})
        debug["plan_step_mapping"] = dict(mapping_debug)
        history = list(debug.get("plan_step_mapping_history", []))
        history.append(dict(mapping_debug))
        debug["plan_step_mapping_history"] = history
        next_state.debug = debug

        if request is None:
            next_plan_step = _get_next_executable_plan_step(next_state)
            plan_step_id = str(getattr(next_plan_step, "step_id", "") or "").strip() or None
            reason = str(mapping_debug.get("reason") or "当前计划步骤未生成可执行工具请求").strip()
            next_state.warnings = list(next_state.warnings or []) + [f"工具执行已跳过：{reason}"]
            observation = ToolObservation(
                tool_name=None,
                ok=False,
                status="skipped",
                result_summary=reason,
                result_ref={
                    "plan_step_id": plan_step_id,
                    "step_type": mapping_debug.get("step_type"),
                    "intent": mapping_debug.get("intent"),
                },
                error={"code": "plan_step_mapping_skipped", "message": reason},
                is_sufficient=False,
                next_action_hint="inspect_tool_request_or_choose_alternative",
                raw_trace={"mapping_debug": dict(mapping_debug)},
            )
            _append_observation(next_state, observation)
            _record_plan_step_observation(next_state, plan_step_id=plan_step_id, tool_name=None, observation=observation)
            if plan_step_id:
                next_state = _update_execution_plan_step(next_state, step_id=plan_step_id, status="skipped")

            debug = dict(next_state.debug or {})
            debug["last_tool_execution"] = {
                "status": "skipped",
                "reason": reason,
                "mapping_debug": dict(mapping_debug),
                "observation": observation.model_dump(),
            }
            history = list(debug.get("tool_execution_history", []))
            history.append(
                {
                    "tool_name": None,
                    "status": observation.status,
                    "ok": observation.ok,
                    "plan_step_id": plan_step_id,
                }
            )
            debug["tool_execution_history"] = history
            next_state.debug = debug
            return _append_step(
                next_state,
                step="execute_planned_tool_step",
                status="skipped",
                action="根据计划步骤执行统一工具调用",
                inputs={"mapping_debug": dict(mapping_debug)},
                outputs={"reason": reason, "plan_step_id": plan_step_id},
            )

        next_state.tool_call_request = request
        next_state.tool_name = str(request.tool_name or "").strip() or None
        next_state.tool_args = dict(request.arguments or {})

    executed_state = execute_tool(next_state)
    if executed_state.steps and executed_state.steps[-1].step == "execute_tool":
        executed_state.steps[-1].step = "execute_planned_tool_step"
        executed_state.steps[-1].action = "根据计划步骤执行统一工具调用"
    return executed_state


def execute_tool(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """执行 `tool_call_request` 指定的工具，并把结果写回状态。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    request = next_state.tool_call_request
    if request is None:
        observation = ToolObservation(
            tool_name=None,
            ok=False,
            status="skipped",
            result_summary="tool_call_request is missing",
            result_ref=None,
            error={"code": "missing_tool_call_request", "message": "tool_call_request is missing"},
            is_sufficient=False,
            next_action_hint="inspect_tool_request_or_choose_alternative",
            raw_trace=None,
        )
        _append_observation(next_state, observation)
        debug = dict(next_state.debug or {})
        debug["last_tool_execution"] = {
            "status": "skipped",
            "reason": "missing_tool_call_request",
            "observation": observation.model_dump(),
        }
        next_state.debug = debug
        return _append_step(
            next_state,
            step="execute_tool",
            status="skipped",
            action="执行通用工具调用",
            inputs={},
            outputs={"reason": "tool_call_request 缺失，跳过执行"},
        )

    tool_name = str(request.tool_name or "").strip() or None
    arguments = dict(request.arguments or {})
    plan_step_id = str(request.plan_step_id or "").strip() or None
    next_state.tool_name = tool_name
    next_state.tool_args = arguments
    if plan_step_id:
        next_state = _update_execution_plan_step(next_state, step_id=plan_step_id, status="in_progress")

    if tool_name is None:
        result = _make_failed_result(None, "Tool name is missing", code="tool_name_missing")
    else:
        try:
            # 复用现有 tool_registry 的参数校验与执行逻辑，不在节点层重复实现。
            result = _to_plain_dict(invoke_tool(tool_name, **arguments))
        except Exception as exc:  # pragma: no cover - registry normally absorbs exceptions
            result = {
                "ok": False,
                "tool_name": tool_name,
                "summary": "Tool execution failed",
                "data": None,
                "trace": {"tool_name": tool_name, "inputs": arguments, "source": "execute_tool_node"},
                "error": {"code": "tool_execution_failed", "message": str(exc)},
            }

    next_state.tool_result = dict(result)

    observation_details = _derive_observation_details(tool_name, result)
    observation = ToolObservation(
        tool_name=tool_name,
        ok=_result_ok(result),
        status="success" if _result_ok(result) else "failed",
        result_summary=_result_text(result, "summary"),
        result_ref=_build_result_ref(result),
        error=_result_mapping(result, "error"),
        is_sufficient=bool(observation_details.get("is_sufficient")),
        next_action_hint=observation_details.get("next_action_hint"),
        raw_trace=_result_mapping(result, "trace"),
    )
    _append_observation(next_state, observation)
    _append_tool_call(next_state, tool_name=tool_name, arguments=arguments, result=result)

    debug = dict(next_state.debug or {})
    _record_plan_step_observation(next_state, plan_step_id=plan_step_id, tool_name=tool_name, observation=observation)
    debug = dict(next_state.debug or {})
    debug["last_tool_execution"] = {
        "request": request.model_dump(),
        "result": dict(result),
        "observation": observation.model_dump(),
    }
    history = list(debug.get("tool_execution_history", []))
    history.append(
        {
            "tool_name": tool_name,
            "status": observation.status,
            "ok": observation.ok,
            "plan_step_id": request.plan_step_id,
        }
    )
    debug["tool_execution_history"] = history
    next_state.debug = debug

    if plan_step_id:
        next_state = _update_execution_plan_step(
            next_state,
            step_id=plan_step_id,
            status="completed" if observation.ok else "failed",
        )

    error_message = _extract_error_message(result)
    return _append_step(
        next_state,
        step="execute_tool",
        status="success" if observation.ok else "failed",
        action="执行通用工具调用",
        inputs={
            "tool_name": tool_name,
            "arguments": arguments,
            "plan_step_id": plan_step_id,
        },
        outputs={
            "tool_status": observation.status,
            "tool_ok": observation.ok,
            "result_summary": observation.result_summary,
        },
        error=error_message,
    )
