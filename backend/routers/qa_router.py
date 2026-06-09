from __future__ import annotations

"""论文 QA、会话与笔记相关路由。

该模块是后端中最核心的论文问答入口之一，负责：
1. QA 索引构建与状态查询；
2. 检索 trace 导出与诊断；
3. 论文聊天会话及消息管理；
4. 论文笔记的创建、编辑、导出；
5. 同步问答与流式问答。

整体设计上，路由层尽量保持轻量，主要承担参数整理、对象序列化、
错误码转换和响应结构统一，复杂业务则下沉到 service 层实现。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from dependencies import (
    get_database_service,
    get_enhanced_retrieval_service,
    get_generation_service,
    get_index_job_manager,
    get_memory_service,
    get_paper_qa_service,
    get_vector_store_service,
)
from core.errors import AppError, ErrorCode, error_response
from routers.qa_utils import build_qa_diagnostic, get_latest_retrieval_trace, sanitize_trace_slug
from utils.config import get_default_user_id, get_qa_index_job_runtime_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper/{arxiv_id}", tags=["paper-qa"])
QA_INDEX_JOB_RETRYABLE_STATUSES = {"failed", "stale", "cancelled"}


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


class PaperNoteRequest(BaseModel):
    """创建论文笔记的请求体。"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_message_id: Optional[str] = None
    source_turn_id: Optional[str] = None
    title: Optional[str] = None
    content: str
    note_type: Optional[str] = None
    source_chunk_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    include_in_profile: Optional[bool] = False


class UpdatePaperNoteRequest(BaseModel):
    """更新论文笔记的请求体。

    与创建不同，这里大部分字段都允许为空，表示按需局部更新。
    """
    user_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    note_type: Optional[str] = None
    source_chunk_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    include_in_profile: Optional[bool] = None


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


def _serialize_qa_index_job(job: Optional[dict]) -> Optional[dict]:
    """序列化 QA 索引构建任务信息。"""
    if not job:
        return None
    normalized_status = str(job.get("status") or "").strip().lower()
    timeout_seconds = int(get_qa_index_job_runtime_config().get("timeout_seconds") or 0)
    return {
        "job_id": job.get("job_id"),
        "arxiv_id": job.get("arxiv_id"),
        "status": job.get("status"),
        "current_stage": job.get("current_stage"),
        "progress": job.get("progress"),
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


def _recover_stale_qa_index_jobs(db_service: Any, *, arxiv_id: str, job_id: Optional[str] = None) -> None:
    """查询前轻量执行 stale 自愈，避免前端轮询时长期看到不可恢复的 pending/running。"""
    marker = getattr(db_service, "mark_stale_paper_index_jobs", None)
    if not callable(marker):
        return
    timeout_seconds = int(get_qa_index_job_runtime_config().get("timeout_seconds") or 1800)
    marker(arxiv_id=arxiv_id, job_id=job_id, timeout_seconds=timeout_seconds)


def _serialize_paper_note(note: Optional[dict], db_service=None) -> Optional[dict]:
    """序列化论文笔记，并按需补齐其关联的对话来源信息。"""
    if not note:
        return None
    linked_message = None
    if db_service and note.get("source_message_id"):
        # 如果笔记来源于某条 assistant 回复，这里顺手把原消息取出来，
        # 用于补齐 turn_id 和 sources，方便前端回溯笔记出处。
        linked_message = db_service.get_paper_chat_message(
            note.get("source_message_id"),
            user_id=note.get("user_id") or _normalize_user_id(None),
        )
    return {
        "note_id": note.get("note_id"),
        "user_id": note.get("user_id"),
        "arxiv_id": note.get("arxiv_id"),
        "session_id": note.get("session_id"),
        "source_message_id": note.get("source_message_id"),
        "source_turn_id": linked_message.get("turn_id") if linked_message else None,
        "title": note.get("title", ""),
        "content": note.get("content", ""),
        "note_type": note.get("note_type", "custom"),
        "source_chunk_ids": note.get("source_chunk_ids") or [],
        "tags": note.get("tags") or [],
        "include_in_profile": bool(note.get("include_in_profile", False)),
        "created_at": note.get("created_at"),
        "updated_at": note.get("updated_at"),
        "sources": linked_message.get("sources") if linked_message else [],
    }


def _build_notes_markdown(arxiv_id: str, notes: List[dict], paper_title: Optional[str] = None) -> str:
    """把笔记列表导出为 Markdown 文本。

    导出结果面向用户阅读，因此内容会按“标题 -> 元信息 -> 正文 -> 关联来源”的顺序组织。
    """
    lines = [f"# {paper_title or arxiv_id} 阅读笔记", "", f"- arXiv ID: {arxiv_id}", f"- 导出时间: {datetime.now().isoformat()}", ""]
    for index, note in enumerate(notes, start=1):
        lines.append(f"## {index}. {note.get('title') or '未命名笔记'}")
        lines.append("")
        lines.append(f"- 类型: {note.get('note_type') or 'custom'}")
        lines.append(f"- 创建时间: {note.get('created_at') or '-'}")
        if note.get("tags"):
            lines.append(f"- 标签: {', '.join(note.get('tags') or [])}")
        if note.get("source_chunk_ids"):
            lines.append(f"- Source Chunk IDs: {', '.join(note.get('source_chunk_ids') or [])}")
        lines.append("")
        lines.append(note.get("content") or "")
        lines.append("")
        sources = note.get("sources") or []
        if sources:
            lines.append("### 关联 Sources")
            lines.append("")
            for source in sources:
                # 这里保留页码、章节路径、父 chunk id，便于把笔记和原文片段重新对应起来。
                lines.append(
                    f"- page={source.get('page_number') or '-'} | section={source.get('section_path') or '-'} | chunk={source.get('parent_chunk_id') or '-'}"
                )
                preview = str(source.get("content") or "").strip()
                if preview:
                    lines.append(f"  - {preview}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


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
    db_service=Depends(get_database_service),
    vector_store_service=Depends(get_vector_store_service),
):
    """输出 QA 索引诊断信息，用于排查索引和向量库状态。"""
    try:
        return build_qa_diagnostic(
            db_service=db_service,
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
    """下载最近一次或指定名称的检索 trace 文件。"""
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

        return FileResponse(
            path=str(trace_file),
            filename=f"{arxiv_id}_retrieval_trace.{normalized_format}",
            media_type="application/json" if normalized_format == "json" else "text/markdown",
        )
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
    - 否则提交后台任务并立即返回 job 信息。
    """
    try:
        if sync:
            # 同步模式一般用于调试或管理后台主动触发，便于立即拿到构建结果。
            return paper_qa_service.build_qa_index(arxiv_id, loading_method=loading_method)

        # 异步模式更适合生产环境，避免请求长时间阻塞。
        job = index_job_manager.submit_job(arxiv_id, loading_method)
        normalized_job_status = str(job.get("status") or "pending").strip().lower()
        timeout_seconds = getattr(index_job_manager, "timeout_seconds", None)
        # 兼容旧响应字段，同时附加可重试和恢复信息，前端无需改变轮询方式即可展示失败原因。
        return {
            "status": "submitted",
            "job_id": job.get("job_id"),
            "arxiv_id": job.get("arxiv_id") or arxiv_id,
            "job_status": normalized_job_status or "pending",
            "current_stage": job.get("current_stage") or "pending",
            "progress": job.get("progress") if job.get("progress") is not None else 0,
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
    db_service=Depends(get_database_service),
):
    """获取指定论文最近一次 QA 索引任务。"""
    try:
        _recover_stale_qa_index_jobs(db_service, arxiv_id=arxiv_id)
        job = db_service.get_latest_paper_index_job(arxiv_id)
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
    db_service=Depends(get_database_service),
):
    """按任务 ID 获取某次 QA 索引构建任务详情。"""
    try:
        _recover_stale_qa_index_jobs(db_service, arxiv_id=arxiv_id, job_id=job_id)
        job = db_service.get_paper_index_job(job_id)
        if not job or job.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="QA index job not found")
        return _serialize_qa_index_job(job)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting QA index job: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chat-sessions")
async def list_paper_chat_sessions(
    arxiv_id: str,
    user_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db_service=Depends(get_database_service),
):
    """列出某篇论文下、某个用户的聊天会话列表。"""
    try:
        sessions = db_service.list_paper_chat_sessions(arxiv_id=arxiv_id, user_id=_normalize_user_id(user_id), limit=limit)
        return {"items": [_serialize_chat_session(item) for item in sessions]}
    except Exception as exc:
        logger.error("Error listing paper chat sessions: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chat-sessions/recent")
async def get_recent_paper_chat_session(
    arxiv_id: str,
    user_id: Optional[str] = Query(None),
    db_service=Depends(get_database_service),
):
    """获取最近一次论文聊天会话，便于前端恢复上下文。"""
    try:
        session = db_service.get_recent_paper_chat_session(arxiv_id=arxiv_id, user_id=_normalize_user_id(user_id))
        return {"item": _serialize_chat_session(session)}
    except Exception as exc:
        logger.error("Error getting recent paper chat session: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat-sessions")
async def create_paper_chat_session(
    arxiv_id: str,
    payload: CreatePaperChatSessionRequest = Body(default=CreatePaperChatSessionRequest()),
    db_service=Depends(get_database_service),
):
    """创建一个新的论文聊天会话。"""
    try:
        session = db_service.create_paper_chat_session(
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
    db_service=Depends(get_database_service),
):
    """获取单个聊天会话详情，并校验该会话确实属于当前论文。"""
    try:
        session = db_service.get_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
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
    db_service=Depends(get_database_service),
):
    """获取某个聊天会话下的全部消息。"""
    try:
        session = db_service.get_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        messages = db_service.list_paper_chat_messages(session_id, user_id=_normalize_user_id(user_id))
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
    db_service=Depends(get_database_service),
):
    """清空指定聊天会话中的消息，但保留会话本身。"""
    try:
        user_id = _normalize_user_id((payload or {}).get("user_id"))
        session = db_service.get_paper_chat_session(session_id, user_id=user_id)
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        # clear 后再重新读取一次，保证返回给前端的是最新 message_count 等状态。
        db_service.clear_paper_chat_session(session_id, user_id=user_id)
        refreshed = db_service.get_paper_chat_session(session_id, user_id=user_id)
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
    db_service=Depends(get_database_service),
):
    """删除整个聊天会话。"""
    try:
        session = db_service.get_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
        deleted = db_service.delete_paper_chat_session(session_id, user_id=_normalize_user_id(user_id))
        return {"status": "success", "deleted": deleted}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting paper chat session: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/notes")
async def list_paper_notes(
    arxiv_id: str,
    user_id: Optional[str] = Query(None),
    note_type: Optional[str] = Query(None),
    db_service=Depends(get_database_service),
):
    """列出某篇论文下的笔记，可按 note_type 过滤。"""
    try:
        notes = db_service.list_paper_notes(
            arxiv_id=arxiv_id,
            user_id=_normalize_user_id(user_id),
            note_type=note_type,
        )
        return {"items": [_serialize_paper_note(item, db_service=db_service) for item in notes]}
    except Exception as exc:
        logger.error("Error listing paper notes: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/notes")
async def create_paper_note(
    arxiv_id: str,
    payload: PaperNoteRequest,
    db_service=Depends(get_database_service),
    memory_service=Depends(get_memory_service),
):
    """创建一条论文笔记，并可选同步更新用户研究画像。"""
    try:
        user_id = _normalize_user_id(payload.user_id)
        source_message_id = (payload.source_message_id or "").strip() or None
        if not source_message_id and payload.session_id and payload.source_turn_id:
            # 如果前端没有直接传 message_id，但给了 session + turn，
            # 就回查对应的 assistant 消息，建立笔记与问答来源的关联。
            linked_message = db_service.get_paper_chat_message_by_turn(
                payload.session_id,
                payload.source_turn_id,
                role="assistant",
                user_id=user_id,
            )
            source_message_id = linked_message.get("message_id") if linked_message else None

        note = db_service.create_paper_note(
            user_id=user_id,
            arxiv_id=arxiv_id,
            session_id=(payload.session_id or "").strip() or None,
            source_message_id=source_message_id,
            title=(payload.title or "").strip() or None,
            content=payload.content,
            note_type=payload.note_type or "custom",
            source_chunk_ids=payload.source_chunk_ids or [],
            tags=payload.tags or [],
            include_in_profile=bool(payload.include_in_profile),
        )
        if not note:
            raise HTTPException(status_code=500, detail="Failed to create paper note")

        return {"item": _serialize_paper_note(note, db_service=db_service)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error creating paper note: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/notes/{note_id}")
async def update_paper_note(
    arxiv_id: str,
    note_id: str,
    payload: UpdatePaperNoteRequest,
    db_service=Depends(get_database_service),
    memory_service=Depends(get_memory_service),
):
    """更新指定笔记，并在需要时重新同步用户画像。"""
    try:
        user_id = _normalize_user_id(payload.user_id)
        current = db_service.get_paper_note(note_id, user_id=user_id)
        if not current or current.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Paper note not found")

        note = db_service.update_paper_note(
            note_id,
            user_id=user_id,
            title=payload.title,
            content=payload.content,
            note_type=payload.note_type,
            source_chunk_ids=payload.source_chunk_ids,
            tags=payload.tags,
            include_in_profile=payload.include_in_profile,
        )
        if not note:
            raise HTTPException(status_code=500, detail="Failed to update paper note")

        return {"item": _serialize_paper_note(note, db_service=db_service)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error updating paper note: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/notes/{note_id}")
async def delete_paper_note(
    arxiv_id: str,
    note_id: str,
    user_id: Optional[str] = Query(None),
    db_service=Depends(get_database_service),
):
    """删除指定论文笔记。"""
    try:
        normalized_user_id = _normalize_user_id(user_id)
        current = db_service.get_paper_note(note_id, user_id=normalized_user_id)
        if not current or current.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Paper note not found")
        deleted = db_service.delete_paper_note(note_id, user_id=normalized_user_id)
        return {"status": "success", "deleted": deleted}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting paper note: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/notes/export")
async def export_paper_notes_markdown(
    arxiv_id: str,
    user_id: Optional[str] = Query(None),
    db_service=Depends(get_database_service),
):
    """把某篇论文的全部笔记导出为 Markdown 文件下载。"""
    try:
        normalized_user_id = _normalize_user_id(user_id)
        notes = db_service.list_paper_notes(arxiv_id=arxiv_id, user_id=normalized_user_id)
        serialized_notes = [_serialize_paper_note(item, db_service=db_service) for item in notes]
        paper = db_service.get_paper(arxiv_id)
        markdown = _build_notes_markdown(arxiv_id, serialized_notes, paper_title=(paper or {}).get("title"))
        filename = f"{sanitize_trace_slug(arxiv_id)}_notes.md"
        return StreamingResponse(
            # 这里直接把内存中的 markdown bytes 作为流返回，避免额外生成临时文件。
            iter([markdown.encode("utf-8")]),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.error("Error exporting paper notes: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/qa")
async def qa_paper(arxiv_id: str, payload: QaRequest, paper_qa_service=Depends(get_paper_qa_service)):
    """执行一次非流式论文问答。"""
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


@router.post("/qa/stream")
async def qa_paper_stream(
    arxiv_id: str,
    payload: QaRequest,
    paper_qa_service=Depends(get_paper_qa_service),
    generation_service=Depends(get_generation_service),
):
    """执行流式论文问答，并以 SSE 持续向前端推送事件。"""
    question = payload.question.strip()
    logger.debug("QA stream request for paper: %s, question: %s", arxiv_id, question)

    def sse_event(event_name: str, data: dict) -> str:
        """格式化单条 SSE 消息。"""
        return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_stream():
        """生成 SSE 事件流。

        事件大致分为三类：
        1. meta：流开始时发送上下文与调试信息；
        2. delta：模型逐段生成答案；
        3. done / error：结束态事件。
        """
        try:
            # 上下文构建也放在 SSE 生成器内，确保未建索引、检索异常等前置失败能返回统一 error 事件。
            _, search_results, qa_context, retrieval_debug = paper_qa_service.build_qa_context(arxiv_id, payload)
            # 优先复用 ContextPackBuilder 产出的 source_payload，旧服务或测试替身没有该字段时再走兼容包装。
            source_payload = qa_context.get("source_payload") or paper_qa_service.build_source_payload(search_results)
            contextualized_question = str(qa_context.get("generation_question", question) or question).strip() or question
            question_contextualization = qa_context.get("question_contextualization", {}) or {}
            chat_session = qa_context.get("chat_session", {}) or {}

            yield sse_event(
                "meta",
                {
                    "status": "started",
                    "arxiv_id": arxiv_id,
                    "question": question,
                    "session_id": chat_session.get("session_id"),
                    "chat_session": _serialize_chat_session(chat_session),
                    "original_question": question,
                    "contextualized_question": contextualized_question,
                    "used_short_term_memory": bool(question_contextualization.get("used_short_term_memory", False)),
                    "question_contextualization": question_contextualization,
                    "sources": source_payload,
                    "image_inputs": qa_context["image_inputs"],
                    "asset_metadata": qa_context["asset_metadata"],
                    "retrieval_debug": retrieval_debug,
                },
            )

            for chunk in generation_service.stream_qwen_responses(
                query=contextualized_question,
                context=qa_context["text_context"],
                # 纸面 QA 的最终答案属于高质量生成任务，明确走大模型。
                task_type="paper_qa_final_answer",
                image_inputs=qa_context["image_inputs"],
                # 仅把 figure 类型的资源元信息传给多模态生成层，避免无关资产干扰回答。
                asset_metadata=[item for item in qa_context["asset_metadata"] if item.get("chunk_type") == "figure"],
            ):
                if chunk.get("type") == "delta":
                    # delta 事件只承载增量文本，适合前端逐字/逐段渲染。
                    yield sse_event("delta", {"delta": chunk.get("delta", "")})
                elif chunk.get("type") == "completed":
                    final_answer = chunk.get("answer", "") or ""
                    verification_debug = {}
                    verifier = getattr(paper_qa_service, "evidence_verifier", None)
                    if verifier is not None:
                        # 流式生成结束后做同一套轻量校验，避免 SSE 路径绕过证据闭环。
                        verification_debug = verifier.verify(
                            answer=final_answer,
                            sources=source_payload,
                            cited_source_ids=[],
                            claims=[],
                            generation_insufficient_evidence=False,
                        )
                        final_answer = verifier.apply_answer_guardrail(final_answer, verification_debug)
                    if isinstance(retrieval_debug, dict):
                        retrieval_debug["verification"] = verification_debug
                    # 回答生成结束后，把本轮问答、来源和调试快照统一持久化，
                    # 这样后续会话恢复、笔记关联、问题追踪都有完整上下文。
                    persisted_turn = paper_qa_service.persist_completed_turn(
                        chat_session=chat_session,
                        question=question,
                        answer=final_answer,
                        source_payload=source_payload,
                        retrieval_debug=retrieval_debug if isinstance(retrieval_debug, dict) else None,
                        contextualized_question=contextualized_question,
                        question_contextualization=question_contextualization,
                    )
                    yield sse_event(
                        "done",
                        {
                            "status": "success",
                            "answer": final_answer,
                            "session_id": chat_session.get("session_id"),
                            "chat_session": _serialize_chat_session(persisted_turn.get("chat_session", chat_session)),
                            "turn_id": persisted_turn.get("turn_id"),
                            "original_question": question,
                            "contextualized_question": contextualized_question,
                            "used_short_term_memory": bool(question_contextualization.get("used_short_term_memory", False)),
                            "question_contextualization": question_contextualization,
                            "sources": source_payload,
                            "image_inputs": qa_context["image_inputs"],
                            "asset_metadata": qa_context["asset_metadata"],
                            "verification_debug": verification_debug,
                            "retrieval_debug": retrieval_debug,
                            "usage": chunk.get("usage"),
                        },
                    )
                    return

            # 极端情况下模型流没有显式 completed 事件，仍返回一个空答案的 done，
            # 保证前端能收到结束信号，不会一直处于 loading 状态。
            verification_debug = {}
            verifier = getattr(paper_qa_service, "evidence_verifier", None)
            fallback_answer = ""
            if verifier is not None:
                # 没有 completed 事件意味着答案为空，仍记录校验结果，便于前端区分生成失败和证据不足。
                verification_debug = verifier.verify(answer="", sources=source_payload, cited_source_ids=[], claims=[])
                fallback_answer = verifier.apply_answer_guardrail("", verification_debug)
                if isinstance(retrieval_debug, dict):
                    retrieval_debug["verification"] = verification_debug
            yield sse_event(
                "done",
                {
                    "status": "success",
                    "answer": fallback_answer,
                    "session_id": chat_session.get("session_id"),
                    "chat_session": _serialize_chat_session(chat_session),
                    "original_question": question,
                    "contextualized_question": contextualized_question,
                    "used_short_term_memory": bool(question_contextualization.get("used_short_term_memory", False)),
                    "question_contextualization": question_contextualization,
                    "sources": source_payload,
                    "image_inputs": qa_context["image_inputs"],
                    "asset_metadata": qa_context["asset_metadata"],
                    "verification_debug": verification_debug,
                    "retrieval_debug": retrieval_debug,
                    "usage": None,
                },
            )
        except AppError as exc:
            logger.warning(
                "Paper QA stream failed: code=%s arxiv_id=%s user_id=%s session_id=%s stage=%s recoverable=%s detail=%s",
                exc.code,
                arxiv_id,
                payload.user_id,
                payload.session_id,
                exc.context.get("stage"),
                exc.recoverable,
                exc.detail,
            )
            yield sse_event("error", exc.to_payload())
        except Exception as exc:
            logger.exception("Error in QA stream: arxiv_id=%s code=%s", arxiv_id, ErrorCode.LLM_GENERATION_FAILED)
            # SSE 场景下不能直接抛异常中断连接，因此把错误包装成统一 error 事件返回。
            stream_error = AppError(
                ErrorCode.LLM_GENERATION_FAILED,
                detail=exc,
                context={"arxiv_id": arxiv_id, "user_id": payload.user_id, "session_id": payload.session_id, "stage": "qa_stream"},
            )
            yield sse_event("error", stream_error.to_payload())

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
