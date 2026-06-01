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

from .graph import build_arxiv_search_graph
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
            context=dict(normalized_request.context or {}),
            pending_action=(normalized_request.context or {}).get("pending_action") if isinstance(normalized_request.context, dict) else None,
        )
        graph = build_arxiv_search_graph(generation_service=generation_service)
        final_state = graph.invoke(initial_state.model_dump())
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
        current_state: Optional[AgentState] = None

        try:
            current_state = AgentState(
                user_id=normalized_request.user_id,
                session_id=normalized_request.session_id,
                message=normalized_request.message,
                context=dict(normalized_request.context or {}),
                pending_action=(normalized_request.context or {}).get("pending_action")
                if isinstance(normalized_request.context, dict)
                else None,
            )
            graph = build_arxiv_search_graph(generation_service=generation_service)

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

            for update in graph.stream(current_state.model_dump(), stream_mode="updates"):
                if not update:
                    continue

                step_name, step_payload = next(iter(update.items()))
                previous_state = current_state

                yield _sse_event(
                    _make_stream_event(
                        event_type="step_start",
                        sequence=_next_sequence(sequence),
                        run_id=run_id,
                        data={
                            "step": step_name,
                            "state": _compact_state(previous_state),
                        },
                    )
                )
                sequence += 1

                tool_call_started = False
                if step_name == "invoke_search_tool" and _should_emit_tool_call(previous_state):
                    tool_call_started = True
                    # LangGraph 负责执行节点，这里只补充可读的工具调用事件。
                    yield _sse_event(
                        _make_stream_event(
                            event_type="tool_call_start",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "step": step_name,
                                "tool_call": {
                                    "tool_name": previous_state.tool_name,
                                    "arguments": _compact_tool_args(previous_state.tool_args),
                                },
                                "state": _compact_state(previous_state),
                            },
                        )
                    )
                    sequence += 1

                current_state = _coerce_state(step_payload)
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
                        "summary": "工具调用未产生结果",
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
                current_state,
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


def _coerce_state(state: Any) -> AgentState:
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)


def _state_to_response(state: Any) -> ArxivSearchResponse:
    final_state = state if isinstance(state, AgentState) else AgentState.model_validate(state)
    return ArxivSearchResponse(
        intent=final_state.intent or "unsupported",
        intent_source=final_state.intent_source,
        fallback_reason=final_state.fallback_reason,
        llm_confidence=final_state.llm_confidence,
        answer=final_state.answer or "",
        search_spec=final_state.search_spec,
        pending_action=final_state.pending_action,
        paper_qa_result=final_state.paper_qa_result,
        preference_action_result=final_state.preference_action_result,
        plan=list(final_state.plan or []),
        tool_calls=list(final_state.tool_calls or []),
        papers=list(final_state.papers or []),
        warnings=list(final_state.warnings or []),
        next_actions=list(final_state.next_actions or []),
        steps=list(final_state.steps or []),
        debug=dict(final_state.debug or {}),
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
    base_state.answer = f"{message}: {detail}".strip()
    base_state.warnings = list(base_state.warnings or []) + [warning]
    base_state.next_actions = list(base_state.next_actions or []) or [
        "请修正输入后重试",
        "后续可以接入论文总结或 QA 功能",
    ]
    base_state.steps = list(base_state.steps or []) + [
        AgentStep(
            step="agent_runtime",
            status="failed",
            action="Agent 在执行过程中发生异常并返回错误响应",
            inputs={"message": message, "code": code},
            outputs={},
            error=detail or code,
        )
    ]
    return _state_to_response(base_state)


def _compact_state(state: Optional[AgentState]) -> Dict[str, Any]:
    if state is None:
        return {}
    return {
        "intent": state.intent,
        "intent_source": state.intent_source,
        "fallback_reason": state.fallback_reason,
        "llm_confidence": state.llm_confidence,
        "search_spec": state.search_spec.model_dump() if state.search_spec is not None else None,
        "pending_action": state.pending_action,
        "paper_qa_result": state.paper_qa_result,
        "preference_action_result": state.preference_action_result,
        "tool_name": state.tool_name,
        "tool_args": _compact_tool_args(state.tool_args),
        "tool_call_count": len(state.tool_calls or []),
        "paper_count": len(state.papers or []),
        "warning_count": len(state.warnings or []),
        "next_actions": list(state.next_actions or []),
        "personalized_rerank_applied": bool(state.personalized_rerank_applied),
        "debug": dict(state.debug or {}),
    }


def _compact_tool_args(tool_args: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(tool_args or {}).items()
        if value not in (None, "", [], {})
    }


def _should_emit_tool_call(state: Optional[AgentState]) -> bool:
    return bool(state and state.intent == "arxiv_search" and state.tool_name and state.tool_args)


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


__all__ = ["run_arxiv_search_agent", "stream_arxiv_search_agent"]
