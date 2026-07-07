from __future__ import annotations

"""Agent 对话路由。

该模块把 arXiv 搜索 Agent 的同步问答、流式输出、图结构导出能力
统一包装成 HTTP 接口，方便前端或调试工具直接调用。
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dependencies import (
    get_agent_qa_index_continuation_store,
    get_index_job_manager,
    get_paper_qa_index_store,
)
from services.storage.sqlite.shared import DEFAULT_USER_ID

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


class AgentQAIndexContinuationCreateRequest(BaseModel):
    """前端确认建索引后提交的 continuation 请求。"""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    loading_method: str = "docling"
    original_question: Optional[str] = None
    pending_action: Optional[Dict[str, Any]] = None
    resume_payload: Optional[Dict[str, Any]] = None


class AgentQAIndexContinuationStatusRequest(BaseModel):
    user_id: Optional[str] = None
    status: str
    error_message: Optional[str] = None


def _nested_mapping(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _resolve_continuation_identity(payload: AgentQAIndexContinuationCreateRequest) -> Dict[str, Any]:
    """从请求和 pending_action 镜像中提取最小恢复身份字段。"""
    pending_action = dict(payload.pending_action or {})
    confirmation = _nested_mapping(pending_action, "confirmation_request")
    target_paper = _nested_mapping(pending_action, "target_paper")
    session_id = _first_text(payload.session_id, pending_action.get("session_id"), confirmation.get("session_id"))
    arxiv_id = _first_text(payload.arxiv_id, pending_action.get("arxiv_id"), target_paper.get("arxiv_id"))
    pending_action_id = _first_text(pending_action.get("pending_action_id"), confirmation.get("pending_action_id"))
    step_id = _first_text(pending_action.get("step_id"), confirmation.get("step_id"))
    tool_name = _first_text(pending_action.get("tool_name"), confirmation.get("tool_name"), "parse_and_index_paper")
    original_question = _first_text(
        payload.original_question,
        pending_action.get("original_question"),
        confirmation.get("original_question"),
        pending_action.get("qa_question"),
    )
    return {
        "user_id": _first_text(payload.user_id, pending_action.get("user_id"), DEFAULT_USER_ID) or DEFAULT_USER_ID,
        "session_id": session_id,
        "arxiv_id": arxiv_id,
        "pending_action_id": pending_action_id,
        "step_id": step_id,
        "tool_name": tool_name,
        "original_question": original_question,
        "pending_action": pending_action,
    }


def _build_resume_payload(identity: Dict[str, Any], request_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """生成前端刷新后仍可复用的结构化 resume 载荷。"""
    resume_payload = dict(request_payload or {})
    resume_payload.setdefault("decision", "approve")
    resume_payload.setdefault("note", "索引构建完成，继续回答原问题")
    resume_payload.setdefault("step_id", identity.get("step_id") or None)
    resume_payload.setdefault("tool_name", identity.get("tool_name") or "parse_and_index_paper")
    resume_payload.setdefault("pending_action_id", identity.get("pending_action_id") or None)
    return resume_payload


def _job_status(job: Optional[Dict[str, Any]]) -> str:
    return str((job or {}).get("status") or "").strip().lower()


def _sync_continuation_status(
    continuation_store: Any,
    continuation: Dict[str, Any],
    job: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """根据 job 终态同步 continuation，避免前端刷新后看到过期 waiting 状态。"""
    current_status = str(continuation.get("status") or "").strip().lower()
    status = _job_status(job)
    if current_status in {"resumed", "cancelled"}:
        return continuation
    if status == "success" and current_status != "ready_to_resume":
        return continuation_store.update_status(
            continuation["job_id"],
            user_id=continuation.get("user_id") or DEFAULT_USER_ID,
            status="ready_to_resume",
            job_snapshot=job,
        ) or continuation
    if status in {"failed", "stale", "cancelled"} and current_status != "failed":
        return continuation_store.update_status(
            continuation["job_id"],
            user_id=continuation.get("user_id") or DEFAULT_USER_ID,
            status="failed",
            error_message=(job or {}).get("error_message") or "索引构建失败，请重试。",
            job_snapshot=job,
        ) or continuation
    return continuation


def _serialize_continuation(
    continuation_store: Any,
    continuation: Dict[str, Any],
    job: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    synced = _sync_continuation_status(continuation_store, continuation, job)
    return {
        **synced,
        "job": job,
    }


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


@router.post("/qa-index-continuations")
async def create_agent_qa_index_continuation(
    request: AgentQAIndexContinuationCreateRequest,
    index_job_manager=Depends(get_index_job_manager),
    continuation_store=Depends(get_agent_qa_index_continuation_store),
):
    """提交 QA 索引构建任务，并保存索引完成后的 Agent 继续回答上下文。"""
    identity = _resolve_continuation_identity(request)
    if not identity["session_id"]:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not identity["arxiv_id"]:
        raise HTTPException(status_code=400, detail="arxiv_id is required")
    # continuation 只记录恢复最小信息；实际索引执行仍由现有 job manager 负责幂等和进度落库。
    job = index_job_manager.submit_job(identity["arxiv_id"], request.loading_method)
    continuation = continuation_store.upsert_continuation(
        job_id=job["job_id"],
        user_id=identity["user_id"],
        session_id=identity["session_id"],
        arxiv_id=identity["arxiv_id"],
        pending_action_id=identity["pending_action_id"],
        step_id=identity["step_id"],
        tool_name=identity["tool_name"],
        original_question=identity["original_question"],
        resume_payload=_build_resume_payload(identity, request.resume_payload),
        pending_action=identity["pending_action"],
        job_snapshot=job,
        status="waiting_job",
    )
    if not continuation:
        raise HTTPException(status_code=500, detail="failed to save QA index continuation")
    return {
        "status": "submitted",
        "job": job,
        "continuation": _serialize_continuation(continuation_store, continuation, job),
    }


@router.get("/qa-index-continuations/active")
async def list_active_agent_qa_index_continuations(
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    continuation_store=Depends(get_agent_qa_index_continuation_store),
    paper_qa_index_store=Depends(get_paper_qa_index_store),
):
    """列出仍需前端轮询或继续 resume 的 Agent 建索引 continuation。"""
    normalized_user_id = _first_text(user_id, DEFAULT_USER_ID) or DEFAULT_USER_ID
    continuations = continuation_store.list_active_continuations(
        user_id=normalized_user_id,
        session_id=session_id,
        limit=limit,
    )
    results = []
    for continuation in continuations:
        job = paper_qa_index_store.get_paper_index_job(continuation["job_id"])
        results.append(_serialize_continuation(continuation_store, continuation, job))
    return {"continuations": results}


@router.post("/qa-index-continuations/{job_id}/status")
async def update_agent_qa_index_continuation_status(
    job_id: str,
    request: AgentQAIndexContinuationStatusRequest,
    continuation_store=Depends(get_agent_qa_index_continuation_store),
    paper_qa_index_store=Depends(get_paper_qa_index_store),
):
    """更新 continuation 终态，例如前端取消或已成功 resume。"""
    normalized_user_id = _first_text(request.user_id, DEFAULT_USER_ID) or DEFAULT_USER_ID
    job = paper_qa_index_store.get_paper_index_job(job_id)
    continuation = continuation_store.update_status(
        job_id,
        user_id=normalized_user_id,
        status=request.status,
        error_message=request.error_message,
        job_snapshot=job,
    )
    if not continuation:
        raise HTTPException(status_code=404, detail="QA index continuation not found")
    return {"continuation": _serialize_continuation(continuation_store, continuation, job)}


@router.get("/graph", response_model=ArxivSearchGraphResponse)
async def agent_graph_endpoint():
    """返回 arXiv Agent 的静态图结构，方便调试页或前端直接渲染。"""
    # 这里导出的通常是 Mermaid 等可视化结构，
    # 用于帮助开发者理解 Agent 内部节点与边的编排关系。
    return export_arxiv_search_graph_mermaid()
