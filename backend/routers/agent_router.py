"""arXiv 检索 Agent 的对话、会话管理与后台任务恢复路由。

本模块挂载在 /api/agent 下，分为三组接口：
1. 对话：/chat（一次性返回）与 /chat/stream（SSE 流式返回），以及调试用的 /graph；
2. 会话：/sessions/{session_id}/clear，清空一段 Agent 对话并作废其挂起的恢复链路；
3. 后台任务卡（work continuation）与恢复运行（resume run）：
   Agent 遇到需要用户确认或耗时较长的工具调用时，会把执行状态持久化为一张"任务卡"，
   前端稍后可查询、取消，或发起恢复；恢复在后台线程中执行，由 AgentResumeRun 记录进度与最终结果。

路由层只负责身份透传、错误码转换与 SSE 封装，具体编排都在 agents/ 与 services/ 中实现。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from agents.arxiv_search_agent import (
    ArxivSearchGraphResponse,
    ArxivSearchRequest,
    ArxivSearchResponse,
    export_arxiv_search_graph_mermaid,
    run_arxiv_search_agent,
    stream_arxiv_search_agent,
)
from auth.context import current_auth
from auth.ownership import prepare_agent_request
from core.responses import format_sse_event
from dependencies import (
    RequestActor,
    get_agent_resume_run_manager,
    get_agent_runtime_checkpoint_store,
    get_agent_session_store,
    get_agent_work_continuation_service,
    get_agent_work_store,
    get_request_actor,
)
from services.storage.sqlite.stores.agent_work import AgentWorkConflict

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

# 恢复流轮询后台 AgentResumeRun 状态的间隔（秒）。恢复本身在后台线程执行，这里只是观察者。
RESUME_POLL_INTERVAL_SECONDS = 0.5


def _agent_work_http_error(exc: AgentWorkConflict) -> HTTPException:
    """把任务卡/恢复运行的冲突异常映射为 HTTP 错误。

    - owner_mismatch：资源属于其他用户，按 404 处理，不向调用方暴露"资源存在但无权访问"；
    - missing：资源不存在，返回 404；
    - 其他（状态不允许、已被领取等）：返回 409，提示调用方当前状态不允许该操作。
    """
    reason = str(exc)
    if "owner_mismatch" in reason:
        return HTTPException(status_code=404, detail="Agent work item not found")
    if "missing" in reason:
        return HTTPException(status_code=404, detail=reason)
    return HTTPException(status_code=409, detail=reason)


def _prepare_authenticated_agent_request(request: ArxivSearchRequest) -> None:
    """已认证请求在进入 Agent 前校正 user_id / session_id 的归属。

    API Key 模式下没有用户身份（current_auth 为空），沿用请求体中的标识。
    """
    if current_auth.get() is not None:
        prepare_agent_request(request, get_agent_session_store())


# ---------------------------------------------------------------------------
# 对话
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ArxivSearchResponse)
async def agent_chat_endpoint(request: ArxivSearchRequest):
    """执行一次完整 Agent 对话，阻塞到图执行结束后一次性返回结果。"""
    _prepare_authenticated_agent_request(request)
    logger.debug("Running agent chat for user_id=%s session_id=%s", request.user_id, request.session_id)
    return run_arxiv_search_agent(request)


@router.post("/chat/stream")
async def agent_chat_stream_endpoint(request: ArxivSearchRequest) -> StreamingResponse:
    """以 SSE 流式执行 Agent，逐步推送节点事件与最终回答。

    需要用户确认的中断点（interrupt）不走单独接口：前端在下一次请求体中携带结构化 resume 字段即可继续。
    """
    _prepare_authenticated_agent_request(request)
    logger.debug("Running agent chat stream for user_id=%s session_id=%s", request.user_id, request.session_id)
    return stream_arxiv_search_agent(request)


@router.get("/graph", response_model=ArxivSearchGraphResponse)
async def agent_graph_endpoint():
    """返回 Agent 状态图的 Mermaid 描述，仅用于调试与可视化，不涉及用户数据。"""
    return export_arxiv_search_graph_mermaid()


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------


@router.post("/sessions/{session_id}/clear")
async def clear_agent_session(
    session_id: str,
    actor: RequestActor = Depends(get_request_actor),
    session_store=Depends(get_agent_session_store),
    checkpoint_store=Depends(get_agent_runtime_checkpoint_store),
    work_store=Depends(get_agent_work_store),
):
    """清空当前 Agent 对话。

    历史任务记录会保留用于审计，但该会话下所有可取消的任务卡和运行时 checkpoint 都会被作废，
    保证旧状态不能再驱动后续对话。
    """
    session = session_store.get_agent_session(session_id, user_id=actor.user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Agent session not found")

    # 顺序很重要：先终止可取消的恢复链路，再清理轻量会话记忆，
    # 避免并发中的恢复请求在清理之后又把旧的确认态写回会话。
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


# ---------------------------------------------------------------------------
# 后台任务卡（work continuation）
# ---------------------------------------------------------------------------


@router.get("/work-continuations/active")
async def list_active_agent_work_continuations(
    session_id: Optional[str] = Query(None),
    actor: RequestActor = Depends(get_request_actor),
    service=Depends(get_agent_work_continuation_service),
):
    """列出当前用户仍处于活跃状态的任务卡，可按 session_id 过滤。

    响应是面向前端展示的视图，不暴露 checkpoint、授权 grant 或工具调用参数。
    """
    return {"items": service.list_active(user_id=actor.user_id, session_id=session_id)}


@router.get("/work-continuations/{continuation_id}")
async def get_agent_work_continuation(
    continuation_id: str,
    session_id: str = Query(...),
    actor: RequestActor = Depends(get_request_actor),
    service=Depends(get_agent_work_continuation_service),
):
    """查询单张任务卡；session_id 必填，用于确认任务卡确实属于该会话。"""
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
    """取消一张任务卡。

    只有 submitting / waiting_job / ready_to_resume 状态可以取消；对已取消的任务卡重复取消是幂等的，
    其他状态（如已开始恢复或已完成）返回 409。
    """
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
    """恢复一张任务卡，并以 SSE 推送恢复进度与最终结果。

    同一任务卡只会创建或复用唯一的 AgentResumeRun（重复点击不会重复执行）。
    恢复在后台线程中进行，SSE 只是轮询观察者：客户端断开连接不会停止恢复，
    之后可以通过 GET /resume-runs/{resume_run_id} 取回同一份持久化结果。

    事件序列：run_start -> final_response | exception -> stream_end。
    """
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
        yield format_sse_event(
            "run_start",
            {"resume_run_id": resume_run_id, "status": claimed.get("status")},
        )
        while True:
            run = manager.get(resume_run_id, user_id=actor.user_id, session_id=session_id)
            status = str(run.get("status") or "")
            if status == "completed":
                # final_response 与 completed 状态在同一事务中持久化，SSE 断线后可改走 GET 取回同一结果。
                yield format_sse_event(
                    "final_response",
                    {"resume_run_id": resume_run_id, "response": run.get("final_response")},
                )
                # 只有 final_response 已交给 ASGI 发送后才标记"结果已读取"，
                # 避免发送前断线导致前端刷新时找不到持久结果。
                manager.mark_result_retrieved(resume_run_id)
                yield format_sse_event("stream_end", {"resume_run_id": resume_run_id, "status": "completed"})
                return
            if status in {"failed", "indeterminate"}:
                # indeterminate：恢复执行中途进程重启等情况，无法证明 checkpoint 是否已被消费，
                # 因此禁止自动重放，只把结果如实告知用户。
                yield format_sse_event(
                    "exception",
                    {
                        "resume_run_id": resume_run_id,
                        "status": status,
                        "error_code": run.get("error_code"),
                        "message": run.get("error_message"),
                    },
                )
                yield format_sse_event("stream_end", {"resume_run_id": resume_run_id, "status": status})
                return
            await asyncio.sleep(RESUME_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/resume-runs/{resume_run_id}")
async def get_agent_resume_run(
    resume_run_id: str,
    session_id: Optional[str] = Query(None),
    actor: RequestActor = Depends(get_request_actor),
    manager=Depends(get_agent_resume_run_manager),
):
    """查询恢复运行的状态与结果，供 SSE 断线后补取；读取到最终结果时会标记为已读取。"""
    try:
        return manager.get(
            resume_run_id,
            user_id=actor.user_id,
            session_id=session_id,
            mark_retrieved=True,
        )
    except AgentWorkConflict as exc:
        raise _agent_work_http_error(exc) from exc
