from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agents.arxiv_search_agent import ArxivSearchRequest, ArxivSearchResponse, run_arxiv_search_agent, stream_arxiv_search_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/arxiv-search", response_model=ArxivSearchResponse)
async def arxiv_search_agent_endpoint(request: ArxivSearchRequest):
    logger.debug("Running arxiv search agent for user_id=%s session_id=%s", request.user_id, request.session_id)
    return run_arxiv_search_agent(request)


@router.post("/arxiv-search/stream")
async def arxiv_search_agent_stream_endpoint(request: ArxivSearchRequest) -> StreamingResponse:
    logger.debug("Running arxiv search agent stream for user_id=%s session_id=%s", request.user_id, request.session_id)
    return stream_arxiv_search_agent(request)
