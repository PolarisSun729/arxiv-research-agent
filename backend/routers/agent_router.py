from __future__ import annotations

"""Agent 对话路由。

该模块把 arXiv 搜索 Agent 的同步问答、流式输出、图结构导出能力
统一包装成 HTTP 接口，方便前端或调试工具直接调用。
"""

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
    """执行一次完整的 Agent 对话，并在结束后一次性返回结果。"""
    logger.debug("Running agent chat for user_id=%s session_id=%s", request.user_id, request.session_id)
    return run_arxiv_search_agent(request)


@router.post("/chat/stream")
async def agent_chat_stream_endpoint(request: ArxivSearchRequest) -> StreamingResponse:
    """以流式响应方式执行 Agent，对适合边生成边展示的前端场景更友好。"""
    logger.debug("Running agent chat stream for user_id=%s session_id=%s", request.user_id, request.session_id)
    return stream_arxiv_search_agent(request)


@router.get("/graph", response_model=ArxivSearchGraphResponse)
async def agent_graph_endpoint():
    """返回 arXiv Agent 的静态图结构，方便调试页或前端直接渲染。"""
    # 这里导出的通常是 Mermaid 等可视化结构，
    # 用于帮助开发者理解 Agent 内部节点与边的编排关系。
    return export_arxiv_search_graph_mermaid()
