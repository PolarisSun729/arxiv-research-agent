from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def make_tool_result(
    *,
    ok: bool,
    tool_name: str,
    summary: str,
    data: Optional[Dict[str, Any]] = None,
    trace: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "tool_name": tool_name,
        "summary": summary,
        "data": data,
        "trace": trace or {},
        "error": error,
    }


def make_tool_trace(
    tool_name: str,
    *,
    inputs: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "tool_name": tool_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if inputs is not None:
        trace["inputs"] = inputs
    if source is not None:
        trace["source"] = source
    if notes is not None:
        trace["notes"] = notes
    return trace


def make_tool_error(code: str, message: str, detail: Any = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if detail is not None:
        error["detail"] = detail
    return error
