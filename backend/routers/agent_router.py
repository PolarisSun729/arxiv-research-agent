from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

try:
    from agents.arxiv_search_agent import (
        ArxivSearchGraphResponse,
        ArxivSearchRequest,
        ArxivSearchResponse,
        export_arxiv_search_graph_mermaid,
        run_arxiv_search_agent,
        stream_arxiv_search_agent,
    )
except ModuleNotFoundError:  # pragma: no cover - package root differs by startup cwd
    from backend.agents.arxiv_search_agent import (
        ArxivSearchGraphResponse,
        ArxivSearchRequest,
        ArxivSearchResponse,
        export_arxiv_search_graph_mermaid,
        run_arxiv_search_agent,
        stream_arxiv_search_agent,
    )

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=ArxivSearchResponse)
async def agent_chat_endpoint(request: ArxivSearchRequest):
    logger.info("Running agent chat for user_id=%s session_id=%s", request.user_id, request.session_id)
    return run_arxiv_search_agent(request)


@router.post("/chat/stream")
async def agent_chat_stream_endpoint(request: ArxivSearchRequest) -> StreamingResponse:
    logger.info("Running agent chat stream for user_id=%s session_id=%s", request.user_id, request.session_id)
    return stream_arxiv_search_agent(request)


@router.get("/graph", response_model=ArxivSearchGraphResponse)
async def agent_graph_endpoint():
    """返回 arXiv Agent 的静态图结构，方便调试页或前端直接渲染。"""
    return export_arxiv_search_graph_mermaid()
