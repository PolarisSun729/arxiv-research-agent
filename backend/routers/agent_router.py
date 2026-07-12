from __future__ import annotations

"""arXiv Agent 对话与图结构路由。"""

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
except ModuleNotFoundError:  # pragma: no cover - 启动目录不同会改变包根路径
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
    """执行一次完整 Agent 对话，并一次性返回结果。"""
    logger.debug("Running agent chat for user_id=%s session_id=%s", request.user_id, request.session_id)
    return run_arxiv_search_agent(request)


@router.post("/chat/stream")
async def agent_chat_stream_endpoint(request: ArxivSearchRequest) -> StreamingResponse:
    """流式执行 Agent；交互恢复仍通过请求中的结构化 resume 字段完成。"""
    logger.debug("Running agent chat stream for user_id=%s session_id=%s", request.user_id, request.session_id)
    return stream_arxiv_search_agent(request)


@router.get("/graph", response_model=ArxivSearchGraphResponse)
async def agent_graph_endpoint():
    """返回静态图结构，供调试和可视化使用。"""
    return export_arxiv_search_graph_mermaid()
