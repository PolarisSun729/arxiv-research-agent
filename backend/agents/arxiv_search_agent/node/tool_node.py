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

from ..schemas import AgentToolCall, ToolObservation
from ..state import AgentState
from ..utils.result_utils import _extract_error_message, _result_mapping, _result_ok, _result_text, _to_plain_dict
from ..utils.state_utils import _append_step, _coerce_state


def _build_result_ref(result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """从工具结果中提取一个轻量引用，避免 observation 过重。"""
    data = result.get("data")
    if isinstance(data, dict):
        return data
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


def execute_tool(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """执行 `tool_call_request` 指定的工具，并把结果写回状态。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    request = next_state.tool_call_request
    if request is None:
        debug = dict(next_state.debug or {})
        debug["last_tool_execution"] = {"status": "skipped", "reason": "missing_tool_call_request"}
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
    next_state.tool_name = tool_name
    next_state.tool_args = arguments

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

    observation = ToolObservation(
        tool_name=tool_name,
        ok=_result_ok(result),
        status="success" if _result_ok(result) else "failed",
        result_summary=_result_text(result, "summary"),
        result_ref=_build_result_ref(result),
        error=_result_mapping(result, "error"),
        is_sufficient=_result_ok(result),
        next_action_hint=None if _result_ok(result) else "inspect_tool_request_or_choose_alternative",
        raw_trace=_result_mapping(result, "trace"),
    )
    _append_observation(next_state, observation)
    _append_tool_call(next_state, tool_name=tool_name, arguments=arguments, result=result)

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

    error_message = _extract_error_message(result)
    return _append_step(
        next_state,
        step="execute_tool",
        status="success" if observation.ok else "failed",
        action="执行通用工具调用",
        inputs={
            "tool_name": tool_name,
            "arguments": arguments,
            "plan_step_id": request.plan_step_id,
        },
        outputs={
            "tool_status": observation.status,
            "tool_ok": observation.ok,
            "result_summary": observation.result_summary,
        },
        error=error_message,
    )
