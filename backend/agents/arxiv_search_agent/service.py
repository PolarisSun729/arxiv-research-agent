from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:  # pragma: no cover - optional runtime dependency for LLM parsing
    from dependencies import get_generation_service as _get_generation_service
except Exception:  # pragma: no cover
    _get_generation_service = None

from .graph import build_arxiv_search_graph
from .schemas import ArxivSearchRequest, ArxivSearchResponse
from .state import AgentState


def run_arxiv_search_agent(request: ArxivSearchRequest) -> ArxivSearchResponse:
    try:
        normalized_request = _coerce_request(request)
        generation_service = _resolve_generation_service()
        graph = build_arxiv_search_graph(generation_service=generation_service)
        initial_state = AgentState(
            user_id=normalized_request.user_id,
            session_id=normalized_request.session_id,
            message=normalized_request.message,
        )
        final_state = graph.invoke(initial_state)
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
    )


def _build_error_response(*, message: str, detail: str, code: str) -> ArxivSearchResponse:
    warning = f"{code}: {detail}" if detail else code
    return ArxivSearchResponse(
        intent="unsupported",
        answer=f"{message}。{detail}".strip("。"),
        search_spec=None,
        plan=[],
        tool_calls=[],
        papers=[],
        warnings=[warning],
        next_actions=["请修正输入后重试", "后续可以接入论文总结或 QA 功能"],
    )


__all__ = ["run_arxiv_search_agent"]
