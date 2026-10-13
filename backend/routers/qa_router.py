"""论文 QA、会话与笔记相关路由。

该模块是后端中最核心的论文问答入口之一，负责：
1. QA 索引构建与状态查询；
2. 检索 trace 导出与诊断；
3. 论文聊天会话及消息管理；
4. 论文笔记的创建、编辑、导出；
5. 同步问答与流式问答；
6. 证据图片资源读取。

整体设计上，路由层尽量保持轻量，主要承担参数整理、对象序列化、
错误码转换和响应结构统一，复杂业务则下沉到 service 层实现。

关于 user_id：请求里的 user_id（查询参数或请求体）在 JWT 模式下由认证中间件
强制改写/校验为当前登录用户，因此路由层只需做空值兜底（_normalize_user_id），
无需再自行校验归属；单机无认证模式下缺省为配置中的默认用户。

接口一览（均挂载在 /api/paper/{arxiv_id} 下）：
- QA 索引：GET /qa-status、GET /qa-diagnose、POST /create-qa-index、
  GET /qa-index-jobs/latest、GET /qa-index-jobs/{job_id}；
- 检索 trace：GET /qa-trace/latest；
- 聊天会话：GET/POST /chat-sessions、GET /chat-sessions/recent、
  GET/DELETE /chat-sessions/{session_id}、GET /chat-sessions/{session_id}/messages、
  POST /chat-sessions/{session_id}/clear；
- 问答：POST /qa（非流式）、POST /qa/stream（SSE 流式）；
- 证据资源：GET /evidence-assets/{source_id}。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from dependencies import (
    get_enhanced_retrieval_service,
    get_index_job_manager,
    get_paper_chat_message_store,
    get_paper_chat_session_store,
    get_paper_qa_index_store,
    get_paper_qa_service,
    get_vector_store_service,
)
from core.errors import AppError, ErrorCode, error_response
from core.responses import RedactedJSONResponse, format_sse_event
from routers.qa_utils import build_qa_diagnostic, get_latest_retrieval_trace, sanitize_trace_slug
from services.paper_qa.qa_observation import build_error_qa_observation
from utils.config import get_default_user_id, get_qa_index_job_runtime_config
from utils.secret_redaction import redact_text
from auth.ownership import prepare_paper_qa

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper/{arxiv_id}", tags=["paper-qa"])

# 处于这些终态的索引任务允许用户重新提交；前端据此决定是否展示"重试"按钮。
QA_INDEX_JOB_RETRYABLE_STATUSES = {"failed", "stale", "cancelled"}
# 索引任务超时配置缺失时的兜底值（秒），与 IndexJobManager 的默认值保持一致。
DEFAULT_QA_INDEX_JOB_TIMEOUT_SECONDS = 30 * 60


class QaRequest(BaseModel):
    """论文问答请求体。

    除了问题本身，还允许调用方控制多种检索增强开关，
    如 query rewrite、HyDE、关键词检索、LLM rerank 等，
    便于调试不同召回/排序链路的效果。
    """
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    enable_context_expansion: Optional[bool] = None
    debug: Optional[bool] = None
    conversation_context: Optional[List[dict]] = None


class CreatePaperChatSessionRequest(BaseModel):
    """创建论文聊天会话的请求体。"""
    user_id: Optional[str] = None
    title: Optional[str] = None


def _normalize_user_id(value: Optional[str]) -> str:
    """标准化 user_id，确保系统内部总能拿到一个非空用户标识。"""
    return str(value or get_default_user_id()).strip() or get_default_user_id()


def _serialize_chat_session(chat_session: Optional[dict]) -> Optional[dict]:
    """把数据库中的 chat session 记录压缩成对外稳定的响应结构。"""
    if not chat_session:
        return None
    return {
        "session_id": chat_session.get("session_id"),
        "user_id": chat_session.get("user_id"),
        "arxiv_id": chat_session.get("arxiv_id"),
        "title": chat_session.get("title", ""),
        "created_at": chat_session.get("created_at"),
        "updated_at": chat_session.get("updated_at"),
        "message_count": chat_session.get("message_count", 0),
        "status": chat_session.get("status", "active"),
        "summary_updated_at": chat_session.get("summary_updated_at"),
        "summary_turn_count": chat_session.get("summary_turn_count", 0),
        "summary_last_turn_id": chat_session.get("summary_last_turn_id", ""),
    }


def _serialize_chat_message(message: dict) -> dict:
    """把聊天消息记录序列化成前端需要的字段集合。"""
    return {
        "message_id": message.get("message_id"),
        "turn_id": message.get("turn_id", ""),
        "session_id": message.get("session_id"),
        "role": message.get("role"),
        "content": message.get("content", ""),
        "sources": message.get("sources") or [],
        "retrieval_debug_snapshot": message.get("retrieval_debug_snapshot"),
        "contextualized_question": message.get("contextualized_question", ""),
        "question_contextualization": message.get("question_contextualization"),
        "status": message.get("status", "completed"),
        "created_at": message.get("created_at"),
    }


def _qa_index_job_timeout_seconds() -> int:
    """读取 QA 索引任务的超时秒数（配置项 QA_INDEX_JOB_CONFIG["timeout_seconds"]）。

    状态查询时的 stale 判定与响应里回显的 timeout_seconds 必须使用同一个值，
    否则前端看到的超时时间与后端实际判定不一致。
    """
    return int(get_qa_index_job_runtime_config().get("timeout_seconds") or DEFAULT_QA_INDEX_JOB_TIMEOUT_SECONDS)


def _serialize_qa_index_job(job: Optional[dict]) -> Optional[dict]:
    """序列化 QA 索引构建任务信息。

    存储层字段 attempt_count 对外改名为 attempt_no；另外派生出 retryable / stale 两个布尔值，
    前端无需了解状态枚举即可决定展示"重试"或"已失联"。
    """
    if not job:
        return None
    normalized_status = str(job.get("status") or "").strip().lower()
    timeout_seconds = _qa_index_job_timeout_seconds()
    return {
        "job_id": job.get("job_id"),
        "arxiv_id": job.get("arxiv_id"),
        "status": job.get("status"),
        "current_stage": job.get("current_stage"),
        "stage_message": job.get("stage_message"),
        "progress": job.get("progress"),
        "attempt_no": job.get("attempt_count"),
        "max_attempts": job.get("max_attempts"),
        "failure_code": job.get("failure_code"),
        "error_message": job.get("error_message"),
        "loading_method": job.get("loading_method"),
        "retryable": normalized_status in QA_INDEX_JOB_RETRYABLE_STATUSES,
        "stale": normalized_status == "stale",
        "heartbeat_at": job.get("heartbeat_at"),
        "timeout_seconds": timeout_seconds,
        "previous_job_id": job.get("previous_job_id"),
        "recovery_action": job.get("recovery_action"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


def _recover_stale_qa_index_jobs(paper_qa_index_store: Any, *, arxiv_id: str, job_id: Optional[str] = None) -> None:
    """查询前轻量执行 stale 自愈，避免前端轮询时长期看到不可恢复的 pending/running。

    worker 进程崩溃或重启后，任务状态会停留在 pending/running；这里把超时仍未更新的任务
    标记为 stale（可重试）。传入 job_id 时只检查该任务。
    """
    paper_qa_index_store.mark_stale_paper_index_jobs(
        arxiv_id=arxiv_id, job_id=job_id, timeout_seconds=_qa_index_job_timeout_seconds()
    )


# ---------------------------------------------------------------------------
# QA 索引：状态、诊断、trace 下载与构建任务
# ---------------------------------------------------------------------------


@router.get("/qa-status")
async def get_paper_qa_status(arxiv_id: str, paper_qa_service=Depends(get_paper_qa_service)):
    """获取指定论文的 QA 能力状态。

    常用于前端判断该论文是否已经完成索引构建、是否可以直接发起问答。
    """
    try:
        return paper_qa_service.get_qa_status(arxiv_id)
    except Exception as exc:
        logger.error("Error getting paper QA status: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qa-diagnose")
async def diagnose_paper_qa(
    arxiv_id: str,
    sample_limit: int = Query(3, ge=0, le=20),
    paper_qa_index_store=Depends(get_paper_qa_index_store),
    vector_store_service=Depends(get_vector_store_service),
):
    """输出 QA 索引诊断信息，用于排查索引和向量库状态。

    对比 SQLite 中记录的索引元数据与 Milvus 中的真实集合，并抽样返回最多 sample_limit 个 chunk，
    具体字段见 qa_utils.build_qa_diagnostic。
    """
    try:
        return build_qa_diagnostic(
            paper_qa_index_store=paper_qa_index_store,
            vector_store_service=vector_store_service,
            arxiv_id=arxiv_id,
            sample_limit=sample_limit,
        )
    except Exception as exc:
        logger.error("Error diagnosing paper QA: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qa-trace/latest")
async def download_latest_qa_trace(
    arxiv_id: str,
    format: str = Query("md"),
    trace_name: Optional[str] = Query(None),
    enhanced_retrieval_service=Depends(get_enhanced_retrieval_service),
):
    """下载最近一次或指定名称的检索 trace 文件。

    - format：md（人读报告）或 json（结构化数据），其他值返回 400；
    - trace_name：指定 trace 文件名（只能是纯文件名），不传则取最近一次；
    找不到文件返回 404。最终路径还会再校验一次必须位于该论文的 trace 目录内。
    """
    try:
        normalized_format = str(format or "md").strip().lower()
        if normalized_format not in {"md", "json"}:
            raise HTTPException(status_code=400, detail="format must be md or json")

        # trace 目录按 arxiv_id 的安全 slug 分桶，避免特殊字符污染路径结构。
        trace_root = Path(str(enhanced_retrieval_service.trace_export_dir))
        paper_dir = trace_root / sanitize_trace_slug(arxiv_id)
        trace_file = None

        if trace_name:
            # 只允许文件名本身，拒绝包含目录跳转成分的输入，避免路径穿越风险。
            safe_name = Path(str(trace_name)).name
            if safe_name != trace_name:
                raise HTTPException(status_code=400, detail="Invalid trace_name")
            candidate = paper_dir / safe_name
            if candidate.exists() and candidate.is_file():
                trace_file = candidate
        else:
            # 未指定 trace_name 时，默认返回最近一次导出的 trace。
            trace_file = get_latest_retrieval_trace(enhanced_retrieval_service, arxiv_id, normalized_format)

        if trace_file is None:
            raise HTTPException(status_code=404, detail="No retrieval trace found for this paper")

        if trace_file.suffix.lower() != f".{normalized_format}" or not trace_file.resolve().is_relative_to(paper_dir.resolve()):
            raise HTTPException(status_code=400, detail="Invalid trace file")

        # 历史 trace 可能在脱敏规则上线前生成；下载时再次处理，且不修改磁盘上的原文件。
        filename = f"{sanitize_trace_slug(arxiv_id)}_retrieval_trace.{normalized_format}"
        headers = {"Content-Disposition": f"attachment; filename*=utf-8''{quote(filename)}"}
        text = trace_file.read_text(encoding="utf-8")
        if normalized_format == "json":
            return RedactedJSONResponse(content=json.loads(text), headers=headers)
        return Response(content=redact_text(text), media_type="text/markdown", headers=headers)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error downloading QA trace: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/create-qa-index")
async def create_paper_qa_index(
    arxiv_id: str,
    loading_method: str = Query("docling"),
    sync: bool = Query(False),
    index_job_manager=Depends(get_index_job_manager),
    paper_qa_service=Depends(get_paper_qa_service),
):
    """为指定论文创建 QA 索引。

    支持同步构建和异步提交两种模式：
    - sync=true 时直接阻塞到构建完成；
    - 否则提交后台任务并立即返回 job 信息，前端随后轮询 /qa-index-jobs/{job_id}。

    loading_method 指定 PDF 解析方式（默认 docling）。业务失败（AppError）不抛 HTTP 异常，
    而是通过 error_response 返回统一的结构化错误体，便于前端展示失败阶段和是否可恢复。
    """
    try:
        if sync:
            # 同步模式一般用于调试或管理后台主动触发，便于立即拿到构建结果。
            return paper_qa_service.build_qa_index(arxiv_id, loading_method=loading_method)

        # 异步模式更适合生产环境，避免请求长时间阻塞。
        job = index_job_manager.submit_job(arxiv_id, loading_method)
        normalized_job_status = str(job.get("status") or "pending").strip().lower()
        # 回显任务管理器实际使用的超时；没有该属性的实现（如测试替身）返回 None。
        timeout_seconds = getattr(index_job_manager, "timeout_seconds", None)
        # 兼容旧响应字段，同时附加可重试和恢复信息，前端无需改变轮询方式即可展示失败原因。
        return {
            "status": "submitted",
            "job_id": job.get("job_id"),
            "arxiv_id": job.get("arxiv_id") or arxiv_id,
            "job_status": normalized_job_status or "pending",
            "current_stage": job.get("current_stage") or "pending",
            "stage_message": job.get("stage_message"),
            "progress": job.get("progress") if job.get("progress") is not None else 0,
            "attempt_no": job.get("attempt_count"),
            "max_attempts": job.get("max_attempts"),
            "failure_code": job.get("failure_code"),
            "message": "QA index job submitted",
            "retryable": normalized_job_status in QA_INDEX_JOB_RETRYABLE_STATUSES,
            "stale": normalized_job_status == "stale",
            "heartbeat_at": job.get("heartbeat_at"),
            "timeout_seconds": timeout_seconds,
            "previous_job_id": job.get("previous_job_id"),
            "recovery_action": job.get("recovery_action"),
        }
    except AppError as exc:
        logger.warning(
            "QA index creation failed: code=%s arxiv_id=%s stage=%s recoverable=%s detail=%s",
            exc.code,
            arxiv_id,
            exc.context.get("stage"),
            exc.recoverable,
            exc.detail,
        )
        return error_response(exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error creating QA index: arxiv_id=%s code=%s", arxiv_id, ErrorCode.QA_INDEX_BUILD_FAILED)
        return error_response(
            AppError(
                ErrorCode.QA_INDEX_BUILD_FAILED,
                detail=exc,
                context={"arxiv_id": arxiv_id, "stage": "create_paper_qa_index"},
            )
        )


@router.get("/qa-index-jobs/latest")
async def get_latest_paper_qa_index_job(
    arxiv_id: str,
    paper_qa_index_store=Depends(get_paper_qa_index_store),
):
    """获取指定论文最近一次 QA 索引任务；从未提交过任务时返回 404。"""
    try:
        _recover_stale_qa_index_jobs(paper_qa_index_store, arxiv_id=arxiv_id)
        job = paper_qa_index_store.get_latest_paper_index_job(arxiv_id)
        if not job:
            raise HTTPException(status_code=404, detail="QA index job not found")
        return _serialize_qa_index_job(job)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting latest QA index job: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qa-index-jobs/{job_id}")
async def get_paper_qa_index_job(
    arxiv_id: str,
    job_id: str,
    paper_qa_index_store=Depends(get_paper_qa_index_store),
):
    """按任务 ID 获取某次 QA 索引构建任务详情。

    任务必须属于路径中的论文，否则同样返回 404，避免借用其他论文的 job_id 读取任务信息。
    """
    try:
        _recover_stale_qa_index_jobs(paper_qa_index_store, arxiv_id=arxiv_id, job_id=job_id)
        job = paper_qa_index_store.get_paper_index_job(job_id)
        if not job or job.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="QA index job not found")
        return _serialize_qa_index_job(job)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting QA index job: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 聊天会话
# ---------------------------------------------------------------------------


@router.get("/chat-sessions")
async def list_paper_chat_sessions(
    arxiv_id: str,
    user_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    paper_chat_session_store=Depends(get_paper_chat_session_store),
):
    """列出某篇论文下、某个用户的聊天会话列表（最多 limit 条）。"""
    try:
        sessions = paper_chat_session_store.list_paper_chat_sessions(arxiv_id=arxiv_id, user_id=_normalize_user_id(user_id), limit=limit)
        return {"items": [_serialize_chat_session(item) for item in sessions]}
    except Exception as exc:
        logger.error("Error listing paper chat sessions: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chat-sessions/recent")
async def get_recent_paper_chat_session(
    arxiv_id: str,
    user_id: Optional[str] = Query(None),
    paper_chat_session_store=Depends(get_paper_chat_session_store),
):
    """获取最近一次论文聊天会话，便于前端恢复上下文；没有会话时 item 为 null。"""
    try:
        session = paper_chat_session_store.get_recent_paper_chat_session(arxiv_id=arxiv_id, user_id=_normalize_user_id(user_id))
        return {"item": _serialize_chat_session(session)}
    except Exception as exc:
        logger.error("Error getting recent paper chat session: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat-sessions")
async def create_paper_chat_session(
    arxiv_id: str,
    payload: CreatePaperChatSessionRequest = Body(default=CreatePaperChatSessionRequest()),
    paper_chat_session_store=Depends(get_paper_chat_session_store),
):
    """创建一个新的论文聊天会话。"""
    try:
        session = paper_chat_session_store.create_paper_chat_session(
            arxiv_id=arxiv_id,
            user_id=_normalize_user_id(payload.user_id),
            # 标题允许为空；若前端不传，后续也可以由系统根据首轮问题自动生成。
            title=(payload.title or "").strip() or None,
        )
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create chat session")
        return {"item": _serialize_chat_session(session)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error creating paper chat session: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chat-sessions/{session_id}")
async def get_paper_chat_session(
    arxiv_id: str,
    session_id: str,
    user_id: Optional[str] = Query(None),
    paper_chat_session_store=Depends(get_paper_chat_session_store),
):
    """获取单个聊天会话详情，并校验该会话确实属于当前论文。"""
    try:
        session = paper_chat_session_store.get_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return {"item": _serialize_chat_session(session)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting paper chat session: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chat-sessions/{session_id}/messages")
async def get_paper_chat_messages(
    arxiv_id: str,
    session_id: str,
    user_id: Optional[str] = Query(None),
    paper_chat_session_store=Depends(get_paper_chat_session_store),
    paper_chat_message_store=Depends(get_paper_chat_message_store),
):
    """获取某个聊天会话下的全部消息。"""
    try:
        session = paper_chat_session_store.get_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        messages = paper_chat_message_store.list_paper_chat_messages(session_id, user_id=_normalize_user_id(user_id))
        return {
            "session": _serialize_chat_session(session),
            "items": [_serialize_chat_message(item) for item in messages],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting paper chat messages: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat-sessions/{session_id}/clear")
async def clear_paper_chat_session(
    arxiv_id: str,
    session_id: str,
    payload: Optional[dict] = Body(default=None),
    paper_chat_session_store=Depends(get_paper_chat_session_store),
    paper_chat_message_store=Depends(get_paper_chat_message_store),
):
    """清空指定聊天会话中的消息，但保留会话本身。

    注意 user_id 放在请求体中（而非查询参数），请求体可以为空。
    """
    try:
        user_id = _normalize_user_id((payload or {}).get("user_id"))
        session = paper_chat_session_store.get_paper_chat_session(session_id, user_id=user_id)
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        # clear 后再重新读取一次，保证返回给前端的是最新 message_count 等状态。
        paper_chat_message_store.clear_paper_chat_session(session_id, user_id=user_id)
        refreshed = paper_chat_session_store.get_paper_chat_session(session_id, user_id=user_id)
        return {"item": _serialize_chat_session(refreshed)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error clearing paper chat session: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/chat-sessions/{session_id}")
async def delete_paper_chat_session(
    arxiv_id: str,
    session_id: str,
    user_id: Optional[str] = Query(None),
    paper_chat_session_store=Depends(get_paper_chat_session_store),
    paper_chat_message_store=Depends(get_paper_chat_message_store),
):
    """删除整个聊天会话（连同其中的全部消息）。"""
    try:
        session = paper_chat_session_store.get_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        deleted = paper_chat_message_store.delete_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
        return {"status": "success", "deleted": deleted}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting paper chat session: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 问答与证据资源
# ---------------------------------------------------------------------------


@router.post("/qa")
async def qa_paper(arxiv_id: str, payload: QaRequest, paper_qa_service=Depends(get_paper_qa_service)):
    """执行一次非流式论文问答，返回完整答案。

    prepare_paper_qa 放在 try 之外：它负责会话归属校验和配额扣减，
    其抛出的 HTTP 错误（404/429 等）必须原样返回，不能被下面的兜底包装掉。
    业务失败（AppError）通过 error_response 返回统一的结构化错误体。
    """
    prepare_paper_qa(arxiv_id, payload)
    try:
        return paper_qa_service.answer_question(arxiv_id, payload)
    except AppError as exc:
        logger.warning(
            "Paper QA failed: code=%s arxiv_id=%s user_id=%s session_id=%s stage=%s recoverable=%s detail=%s",
            exc.code,
            arxiv_id,
            payload.user_id,
            payload.session_id,
            exc.context.get("stage"),
            exc.recoverable,
            exc.detail,
        )
        return error_response(exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in QA: arxiv_id=%s code=%s", arxiv_id, ErrorCode.UNKNOWN_ERROR)
        return error_response(
            AppError(
                ErrorCode.UNKNOWN_ERROR,
                detail=exc,
                context={"arxiv_id": arxiv_id, "user_id": payload.user_id, "session_id": payload.session_id, "stage": "qa_paper"},
            )
        )


@router.get("/evidence-assets/{source_id}")
async def get_evidence_asset(
    arxiv_id: str,
    source_id: str,
    paper_qa_service=Depends(get_paper_qa_service),
):
    """按论文和稳定 source_id 返回图片，不允许客户端传入任意服务器路径。"""
    try:
        asset_path = paper_qa_service.resolve_evidence_asset_path(arxiv_id, source_id)
        suffix = Path(asset_path).suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "application/octet-stream")
        return FileResponse(asset_path, media_type=media_type)
    except FileNotFoundError:
        # 对外统一返回 404，避免泄漏索引路径、资产路径或论文是否存在的内部细节。
        raise HTTPException(status_code=404, detail="evidence asset not found")
    except Exception as exc:
        logger.warning("Evidence asset lookup failed: arxiv_id=%s source_id=%s error=%s", arxiv_id, source_id, exc)
        raise HTTPException(status_code=404, detail="evidence asset not found")


@router.post("/qa/stream")
async def qa_paper_stream(
    arxiv_id: str,
    payload: QaRequest,
    paper_qa_service=Depends(get_paper_qa_service),
):
    """执行流式论文问答，把证据研究引擎的阶段事件转换为 SSE 推送给前端。"""
    prepare_paper_qa(arxiv_id, payload)
    question = payload.question.strip()
    logger.debug("QA stream request for paper: %s, question: %s", arxiv_id, question)

    def event_stream():
        """把证据研究引擎的事件映射为前端约定的 SSE 事件。

        映射关系：
        - research_started             -> meta（status=started）
        - retrieval_completed          -> progress（stage=retrieval）
        - draft_created                -> progress（stage=draft）
        - claim_verification_completed -> progress（stage=verification）
        - research_completed           -> progress（stage=completed）
        - done                         -> done（携带最终答案与序列化后的 chat_session）
        其余内部事件不对外推送；出错时推送一条 error 事件后结束。

        这是同步生成器，StreamingResponse 会放到线程池中迭代，不会阻塞事件循环。
        流式响应不经过 JSONResponse 的脱敏，因此每一帧都通过 format_sse_event 脱敏。
        """
        try:
            for event in paper_qa_service.answer_question_with_research_stream(arxiv_id, payload):
                event_type = event.get("event", "unknown")
                event_data = event.get("data", {})

                # 映射研究引擎事件到 SSE 事件
                if event_type == "research_started":
                    yield format_sse_event("meta", {
                        "status": "started",
                        "arxiv_id": event_data.get("arxiv_id"),
                        "question": event_data.get("question"),
                    })
                elif event_type == "retrieval_completed":
                    yield format_sse_event("progress", {
                        "stage": "retrieval",
                        "retrieval_count": event_data.get("retrieval_count", 0),
                        "new_candidate_count": event_data.get("new_candidate_count", 0),
                    })
                elif event_type == "draft_created":
                    yield format_sse_event("progress", {
                        "stage": "draft",
                        "draft_attempt": event_data.get("draft_attempt", 0),
                        "claim_count": event_data.get("claim_count", 0),
                    })
                elif event_type == "claim_verification_completed":
                    yield format_sse_event("progress", {
                        "stage": "verification",
                        "supported_count": event_data.get("supported_count", 0),
                    })
                elif event_type == "research_completed":
                    yield format_sse_event("progress", {
                        "stage": "completed",
                        "outcome": event_data.get("outcome"),
                    })
                elif event_type == "done":
                    event_data = dict(event_data)
                    event_data["chat_session"] = _serialize_chat_session(event_data.get("chat_session") or {})
                    yield format_sse_event("done", event_data)
        except AppError as exc:
            # 服务只抛结构化异常；路由是唯一 error 出口，避免一轮失败发送两次错误。
            yield format_sse_event("error", exc.to_payload())
        except Exception:
            logger.exception("Error in research stream: arxiv_id=%s", arxiv_id)
            observation = build_error_qa_observation(
                error_code=ErrorCode.UNKNOWN_ERROR, error_stage="research_stream",
                error_reason="research_execution_failed",
            )
            # 最后一道兜底也不能返回异常字符串，其中可能包含 provider 响应或服务器路径。
            yield format_sse_event("error", AppError(
                ErrorCode.UNKNOWN_ERROR, detail={"stage": "research_stream"},
                context={"arxiv_id": arxiv_id, "qa_observation": observation},
            ).to_payload())

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # SSE 需要禁用缓存和代理缓冲，否则前端可能收不到实时增量。
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
