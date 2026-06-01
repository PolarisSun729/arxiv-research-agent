from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

from fastapi.responses import StreamingResponse
from pydantic import ValidationError

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:  # pragma: no cover - optional runtime dependency for LLM parsing
    from dependencies import get_generation_service as _get_generation_service
except Exception:  # pragma: no cover
    _get_generation_service = None

from .graph import END, route_after_parse
from .nodes import (
    build_search_tool_args,
    check_search_result,
    invoke_search_tool,
    parse_search_request,
    personalized_rank_and_annotate_papers,
    synthesize_response,
)
from .schemas import AgentStep, AgentStreamEvent, ArxivSearchRequest, ArxivSearchResponse
from .state import AgentState


def run_arxiv_search_agent(request: ArxivSearchRequest) -> ArxivSearchResponse:
    try:
        normalized_request = _coerce_request(request)
        generation_service = _resolve_generation_service()
        initial_state = AgentState(
            user_id=normalized_request.user_id,
            session_id=normalized_request.session_id,
            message=normalized_request.message,
        )
        final_state = _run_arxiv_search_pipeline(initial_state, generation_service=generation_service)
        return _state_to_response(final_state)
    except ValidationError as exc:
        return _build_error_response(
            message="请求参数校验失败",
            detail=str(exc),
            code="request_validation_error",
        )
    except Exception as exc:
        return _build_error_response(
            message="arXiv 搜索 Agent 运行失败",
            detail=str(exc),
            code="agent_runtime_error",
        )


def stream_arxiv_search_agent(request: ArxivSearchRequest) -> StreamingResponse:
    normalized_request = _coerce_request(request)

    def event_stream():
        run_id = str(uuid4())
        sequence = 1
        generation_service = _resolve_generation_service()
        try:
            current_state = AgentState(
                user_id=normalized_request.user_id,
                session_id=normalized_request.session_id,
                message=normalized_request.message,
            )
            yield _sse_event(
                _make_stream_event(
                    event_type="run_start",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "status": "started",
                        "request": {
                            "user_id": normalized_request.user_id,
                            "session_id": normalized_request.session_id,
                            "message": normalized_request.message,
                        },
                    },
                )
            )
            sequence += 1

            current_node = "parse_search_request"
            while current_node != END:
                step_name = current_node
                yield _sse_event(
                    _make_stream_event(
                        event_type="step_start",
                        sequence=_next_sequence(sequence),
                        run_id=run_id,
                        data={
                            "step": step_name,
                            "state": _compact_state(current_state),
                        },
                    )
                )
                sequence += 1

                tool_call_started = False
                tool_call_payload: Optional[Dict[str, Any]] = None
                if step_name == "invoke_search_tool" and _should_emit_tool_call(current_state):
                    tool_call_started = True
                    tool_call_payload = {
                        "tool_name": current_state.tool_name,
                        "arguments": _compact_tool_args(current_state.tool_args),
                    }
                    yield _sse_event(
                        _make_stream_event(
                            event_type="tool_call_start",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "step": step_name,
                                "tool_call": tool_call_payload,
                                "state": _compact_state(current_state),
                            },
                        )
                    )
                    sequence += 1

                next_state = _invoke_node(step_name, current_state, generation_service=generation_service)
                current_state = _coerce_state(next_state)

                latest_step = current_state.steps[-1].model_dump() if current_state.steps else None
                yield _sse_event(
                    _make_stream_event(
                        event_type="step_end",
                        sequence=_next_sequence(sequence),
                        run_id=run_id,
                        data={
                            "step": step_name,
                            "status": latest_step.get("status") if isinstance(latest_step, dict) else "success",
                            "step_detail": latest_step,
                            "state": _compact_state(current_state),
                        },
                    )
                )
                sequence += 1

                if tool_call_started:
                    latest_tool_call = current_state.tool_calls[-1].model_dump() if current_state.tool_calls else {
                        "tool_name": current_state.tool_name,
                        "arguments": _compact_tool_args(current_state.tool_args),
                        "status": "failed",
                        "summary": "工具调用未产出结果",
                        "trace": {},
                        "error": None,
                    }
                    yield _sse_event(
                        _make_stream_event(
                            event_type="tool_call_end",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "step": step_name,
                                "tool_call": latest_tool_call,
                                "state": _compact_state(current_state),
                            },
                        )
                    )
                    sequence += 1

                if step_name == "synthesize_response":
                    final_response = _state_to_response(current_state)
                    yield _sse_event(
                        _make_stream_event(
                            event_type="final_response",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "response": final_response.model_dump(),
                                "state": _compact_state(current_state),
                            },
                        )
                    )
                    sequence += 1
                    yield _sse_event(
                        _make_stream_event(
                            event_type="stream_end",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "status": "success",
                                "final_sequence": sequence,
                            },
                        )
                    )
                    return

                current_node = _next_node(step_name, current_state)

            final_response = _state_to_response(current_state)
            yield _sse_event(
                _make_stream_event(
                    event_type="final_response",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "response": final_response.model_dump(),
                        "state": _compact_state(current_state),
                    },
                )
            )
            sequence += 1
            yield _sse_event(
                _make_stream_event(
                    event_type="stream_end",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "status": "success",
                        "final_sequence": sequence,
                    },
                )
            )
        except Exception as exc:
            error_response = _build_error_response_from_state(
                current_state if "current_state" in locals() else None,
                message="arXiv 搜索 Agent 运行失败",
                detail=str(exc),
                code="agent_runtime_error",
            )
            yield _sse_event(
                _make_stream_event(
                    event_type="exception",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "message": "Agent 流式执行异常",
                        "detail": str(exc),
                        "response": error_response.model_dump(),
                    },
                )
            )
            sequence += 1
            yield _sse_event(
                _make_stream_event(
                    event_type="final_response",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "response": error_response.model_dump(),
                    },
                )
            )
            sequence += 1
            yield _sse_event(
                _make_stream_event(
                    event_type="stream_end",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "status": "error",
                        "final_sequence": sequence,
                    },
                )
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _run_arxiv_search_pipeline(initial_state: AgentState, *, generation_service: Optional[Any]) -> AgentState:
    current_state = initial_state.model_copy(deep=True)
    current_node = "parse_search_request"
    while current_node != END:
        current_state = _invoke_node(current_node, current_state, generation_service=generation_service)
        current_state = _coerce_state(current_state)
        if current_node == "synthesize_response":
            return current_state
        current_node = _next_node(current_node, current_state)
    return current_state


def _invoke_node(node_name: str, state: AgentState, *, generation_service: Optional[Any]) -> AgentState:
    if node_name == "parse_search_request":
        return parse_search_request(state, generation_service=generation_service)
    if node_name == "build_search_tool_args":
        return build_search_tool_args(state)
    if node_name == "invoke_search_tool":
        return invoke_search_tool(state)
    if node_name == "check_search_result":
        return check_search_result(state)
    if node_name == "personalized_rank_and_annotate_papers":
        return personalized_rank_and_annotate_papers(state)
    if node_name == "synthesize_response":
        return synthesize_response(state)
    return state


def _coerce_state(state: Any) -> AgentState:
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)


def _next_node(node_name: str, state: AgentState) -> str:
    if node_name == "parse_search_request":
        return {
            "arxiv_search": "build_search_tool_args",
            "unclear": "synthesize_response",
            "unsupported": "synthesize_response",
        }.get(route_after_parse(state), "synthesize_response")
    if node_name == "build_search_tool_args":
        return "invoke_search_tool"
    if node_name == "invoke_search_tool":
        return "check_search_result"
    if node_name == "check_search_result":
        return "personalized_rank_and_annotate_papers"
    if node_name == "personalized_rank_and_annotate_papers":
        return "synthesize_response"
    return END


def _should_emit_tool_call(state: AgentState) -> bool:
    return bool(state.intent == "arxiv_search" and state.tool_name and state.tool_args)


def _compact_state(state: AgentState) -> Dict[str, Any]:
    return {
        "intent": state.intent,
        "search_spec": state.search_spec.model_dump() if state.search_spec is not None else None,
        "tool_name": state.tool_name,
        "tool_args": _compact_tool_args(state.tool_args),
        "tool_call_count": len(state.tool_calls or []),
        "paper_count": len(state.papers or []),
        "warning_count": len(state.warnings or []),
        "next_actions": list(state.next_actions or []),
        "personalized_rerank_applied": bool(state.personalized_rerank_applied),
    }


def _compact_tool_args(tool_args: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(tool_args or {}).items()
        if value not in (None, "", [], {})
    }


def _make_stream_event(*, event_type: str, sequence: int, run_id: str, data: Dict[str, Any]) -> AgentStreamEvent:
    return AgentStreamEvent(
        event_type=event_type,  # type: ignore[arg-type]
        sequence=sequence,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        data=data,
    )


def _sse_event(event: AgentStreamEvent) -> str:
    return f"event: {event.event_type}\ndata: {json.dumps(event.model_dump(), ensure_ascii=False)}\n\n"


def _next_sequence(sequence: int) -> int:
    return sequence


def _coerce_request(request: ArxivSearchRequest | Dict[str, Any]) -> ArxivSearchRequest:
    if isinstance(request, ArxivSearchRequest):
        return request
    return ArxivSearchRequest.model_validate(dict(request))


def _resolve_generation_service() -> Optional[Any]:
    if _get_generation_service is None:
        return None
    try:
        return _get_generation_service()
    except Exception:
        return None


def _state_to_response(state: Any) -> ArxivSearchResponse:
    final_state = state if isinstance(state, AgentState) else AgentState.model_validate(state)
    return ArxivSearchResponse(
        intent=final_state.intent or "unsupported",
        answer=final_state.answer or "",
        search_spec=final_state.search_spec,
        plan=list(final_state.plan or []),
        tool_calls=list(final_state.tool_calls or []),
        papers=list(final_state.papers or []),
        warnings=list(final_state.warnings or []),
        next_actions=list(final_state.next_actions or []),
        steps=list(final_state.steps or []),
    )


def _build_error_response(*, message: str, detail: str, code: str) -> ArxivSearchResponse:
    return _build_error_response_from_state(None, message=message, detail=detail, code=code)


def _build_error_response_from_state(
    state: Optional[AgentState],
    *,
    message: str,
    detail: str,
    code: str,
) -> ArxivSearchResponse:
    warning = f"{code}: {detail}" if detail else code
    base_state = state.model_copy(deep=True) if isinstance(state, AgentState) else AgentState()
    base_state.intent = base_state.intent or "unsupported"
    base_state.answer = f"{message}。{detail}".strip("。")
    base_state.warnings = list(base_state.warnings or []) + [warning]
    base_state.next_actions = list(base_state.next_actions or []) or ["请修正输入后重试", "后续可以接入论文总结或 QA 功能"]
    base_state.steps = list(base_state.steps or []) + [
        AgentStep(
            step="agent_runtime",
            status="failed",
            action="Agent 在执行过程中发生异常并回退到错误响应",
            inputs={"message": message, "code": code},
            outputs={},
            error=detail or code,
        )
    ]
    return _state_to_response(base_state)


__all__ = ["run_arxiv_search_agent", "stream_arxiv_search_agent"]
