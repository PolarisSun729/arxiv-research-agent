from __future__ import annotations

"""arXiv Agent 对话与图结构路由。"""

import logging
import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from dependencies import (
    RequestActor,
    get_agent_runtime_checkpoint_store,
    get_agent_session_store,
    get_agent_resume_run_manager,
    get_agent_work_store,
    get_agent_work_continuation_service,
    get_request_actor,
)
from services.storage.sqlite.stores.agent_work import AgentWorkConflict
from auth.ownership import prepare_agent_request
from utils.secret_redaction import redact_sensitive_value

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


def _agent_work_http_error(exc: AgentWorkConflict) -> HTTPException:
    reason = str(exc)
    if "owner_mismatch" in reason:
        return HTTPException(status_code=404, detail="Agent work item not found")
    if "missing" in reason:
        return HTTPException(status_code=404, detail=reason)
    return HTTPException(status_code=409, detail=reason)


def _resume_sse(event_name: str, payload: dict) -> str:
    # SSE 绕过普通 JSONResponse，恢复事件必须在逐条序列化时使用相同的脱敏边界。
    return f"event: {event_name}\ndata: {json.dumps(redact_sensitive_value(payload), ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ArxivSearchResponse)
async def agent_chat_endpoint(request: ArxivSearchRequest):
    """执行一次完整 Agent 对话，并一次性返回结果。"""
    from auth.context import current_auth
    if current_auth.get() is not None:
        prepare_agent_request(request, get_agent_session_store())
    logger.debug("Running agent chat for user_id=%s session_id=%s", request.user_id, request.session_id)
    return run_arxiv_search_agent(request)


@router.post("/chat/stream")
async def agent_chat_stream_endpoint(request: ArxivSearchRequest) -> StreamingResponse:
    """流式执行 Agent；交互恢复仍通过请求中的结构化 resume 字段完成。"""
    from auth.context import current_auth
    if current_auth.get() is not None:
        prepare_agent_request(request, get_agent_session_store())
    logger.debug("Running agent chat stream for user_id=%s session_id=%s", request.user_id, request.session_id)
    return stream_arxiv_search_agent(request)


@router.get("/graph", response_model=ArxivSearchGraphResponse)
async def agent_graph_endpoint():
    """返回静态图结构，供调试和可视化使用。"""
    return export_arxiv_search_graph_mermaid()


@router.post("/sessions/{session_id}/clear")
async def clear_agent_session(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    session_store=Depends(get_agent_session_store),
    checkpoint_store=Depends(get_agent_runtime_checkpoint_store),
    work_store=Depends(get_agent_work_store),
):
    """清空当前 Agent 对话；历史任务记录保留，但不再允许旧状态驱动后续会话。"""
    session = session_store.get_agent_session(session_id, user_id=actor.user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Agent session not found")

    # 先终止可取消的恢复链路，再清理轻量会话记忆，避免旧确认态在并发请求中重新写回。
    cancelled_continuations = work_store.cancel_session_continuations(
        user_id=actor.user_id,
        session_id=session_id,
    )
    cancelled_checkpoints = checkpoint_store.cancel_agent_runtime_checkpoints_for_session(
        user_id=actor.user_id,
        session_id=session_id,
    )
    if not session_store.clear_agent_session(session_id, user_id=actor.user_id):
        raise HTTPException(status_code=500, detail="Failed to clear agent session")
    return {
        "session_id": session_id,
        "status": "cleared",
        "cancelled_continuations": cancelled_continuations,
        "cancelled_checkpoints": cancelled_checkpoints,
    }


@router.get("/work-continuations/active")
async def list_active_agent_work_continuations(
    session_id: Optional[str] = Query(None),
    actor: RequestActor = Depends(get_request_actor),
    service=Depends(get_agent_work_continuation_service),
):
    """返回当前已认证用户可见的后台任务卡；响应不暴露 checkpoint、grant 或工具参数。"""
    return {"items": service.list_active(user_id=actor.user_id, session_id=session_id)}


@router.get("/work-continuations/{continuation_id}")
async def get_agent_work_continuation(
    continuation_id: str,
    session_id: str = Query(...),
    actor: RequestActor = Depends(get_request_actor),
    service=Depends(get_agent_work_continuation_service),
):
    try:
        return service.get(continuation_id, user_id=actor.user_id, session_id=session_id)
    except AgentWorkConflict as exc:
        raise _agent_work_http_error(exc) from exc


@router.post("/work-continuations/{continuation_id}/cancel")
async def cancel_agent_work_continuation(
    continuation_id: str,
    session_id: str = Query(...),
    actor: RequestActor = Depends(get_request_actor),
    service=Depends(get_agent_work_continuation_service),
):
    try:
        return service.cancel(continuation_id, user_id=actor.user_id, session_id=session_id)
    except AgentWorkConflict as exc:
        raise _agent_work_http_error(exc) from exc


@router.post("/work-continuations/{continuation_id}/resume/stream")
async def resume_agent_work_continuation_stream(
    continuation_id: str,
    session_id: str = Query(...),
    actor: RequestActor = Depends(get_request_actor),
    manager=Depends(get_agent_resume_run_manager),
) -> StreamingResponse:
    """创建或复用唯一 AgentResumeRun；传输断开不会停止后台恢复线程。"""
    try:
        claimed = manager.claim_and_start(
            continuation_id,
            user_id=actor.user_id,
            session_id=session_id,
        )
    except AgentWorkConflict as exc:
        raise _agent_work_http_error(exc) from exc

    async def event_stream():
        resume_run_id = str(claimed["resume_run_id"])
        yield _resume_sse(
            "run_start",
            {"resume_run_id": resume_run_id, "status": claimed.get("status")},
        )
        while True:
            run = manager.get(resume_run_id, user_id=actor.user_id, session_id=session_id)
            status = str(run.get("status") or "")
            if status == "completed":
                # final_response 已在同一事务中持久化后才会进入 completed，SSE 断线可改走 GET 取回同一结果。
                yield _resume_sse(
                    "final_response",
                    {"resume_run_id": resume_run_id, "response": run.get("final_response")},
                )
                # 只有 final_response 已经交给 ASGI 发送后才记录读取，避免发送前断线导致刷新时找不到持久结果。
                manager.mark_result_retrieved(resume_run_id)
                yield _resume_sse("stream_end", {"resume_run_id": resume_run_id, "status": "completed"})
                return
            if status in {"failed", "indeterminate"}:
                yield _resume_sse(
                    "exception",
                    {
                        "resume_run_id": resume_run_id,
                        "status": status,
                        "error_code": run.get("error_code"),
                        "message": run.get("error_message"),
                    },
                )
                yield _resume_sse("stream_end", {"resume_run_id": resume_run_id, "status": status})
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/resume-runs/{resume_run_id}")
async def get_agent_resume_run(
    resume_run_id: str,
    session_id: Optional[str] = Query(None),
    actor: RequestActor = Depends(get_request_actor),
    manager=Depends(get_agent_resume_run_manager),
):
    try:
        return manager.get(
            resume_run_id,
            user_id=actor.user_id,
            session_id=session_id,
            mark_retrieved=True,
        )
    except AgentWorkConflict as exc:
        raise _agent_work_http_error(exc) from exc
