from __future__ import annotations

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
    get_paper_qa_service,
    get_vector_store_service,
)
from routers.qa_utils import build_qa_diagnostic, get_latest_retrieval_trace, sanitize_trace_slug
from utils.config import get_default_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper/{arxiv_id}", tags=["paper-qa"])


class QaRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    debug: Optional[bool] = None
    conversation_context: Optional[List[dict]] = None


class CreatePaperChatSessionRequest(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = None


class PaperNoteRequest(BaseModel):
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
    user_id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    note_type: Optional[str] = None
    source_chunk_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    include_in_profile: Optional[bool] = None


def _normalize_user_id(value: Optional[str]) -> str:
    return str(value or get_default_user_id()).strip() or get_default_user_id()


def _serialize_chat_session(chat_session: Optional[dict]) -> Optional[dict]:
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
    if not job:
        return None
    return {
        "job_id": job.get("job_id"),
        "arxiv_id": job.get("arxiv_id"),
        "status": job.get("status"),
        "current_stage": job.get("current_stage"),
        "progress": job.get("progress"),
        "error_message": job.get("error_message"),
        "loading_method": job.get("loading_method"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


def _merge_unique_strings(existing: List[Any], incoming: List[Any], limit: int = 20) -> List[str]:
    seen: List[str] = []
    for item in [*(existing or []), *(incoming or [])]:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen[:limit]


def _serialize_paper_note(note: Optional[dict], db_service=None) -> Optional[dict]:
    if not note:
        return None
    linked_message = None
    if db_service and note.get("source_message_id"):
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


def _apply_note_to_profile(db_service, note: dict) -> None:
    if not note or not note.get("include_in_profile"):
        return
    user_id = note.get("user_id") or _normalize_user_id(None)
    current = db_service.get_user_research_profile(user_id=user_id)
    tags = [str(item).strip() for item in (note.get("tags") or []) if str(item).strip()]
    note_type = str(note.get("note_type") or "").strip()
    updated_profile: Dict[str, Any] = {
        "positive_topics": _merge_unique_strings(current.get("positive_topics") or [], tags, limit=30),
        "recent_topics": _merge_unique_strings(current.get("recent_topics") or [], tags, limit=30),
        "common_question_types": _merge_unique_strings(
            current.get("common_question_types") or [],
            [note_type] if note_type else [],
            limit=20,
        ),
        "representative_papers": _merge_unique_strings(
            current.get("representative_papers") or [],
            [note.get("arxiv_id")] if note.get("arxiv_id") else [],
            limit=20,
        ),
    }
    db_service.patch_user_research_profile(user_id=user_id, profile=updated_profile)


def _build_notes_markdown(arxiv_id: str, notes: List[dict], paper_title: Optional[str] = None) -> str:
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
    try:
        normalized_format = str(format or "md").strip().lower()
        if normalized_format not in {"md", "json"}:
            raise HTTPException(status_code=400, detail="format must be md or json")

        trace_root = Path(str(enhanced_retrieval_service.trace_export_dir))
        paper_dir = trace_root / sanitize_trace_slug(arxiv_id)
        trace_file = None

        if trace_name:
            safe_name = Path(str(trace_name)).name
            if safe_name != trace_name:
                raise HTTPException(status_code=400, detail="Invalid trace_name")
            candidate = paper_dir / safe_name
            if candidate.exists() and candidate.is_file():
                trace_file = candidate
        else:
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
    try:
        if sync:
            return paper_qa_service.build_qa_index(arxiv_id, loading_method=loading_method)

        job = index_job_manager.submit_job(arxiv_id, loading_method)
        return {
            "status": "submitted",
            "job_id": job.get("job_id"),
            "arxiv_id": job.get("arxiv_id") or arxiv_id,
            "job_status": job.get("status") or "pending",
            "current_stage": job.get("current_stage") or "pending",
            "progress": job.get("progress") if job.get("progress") is not None else 0,
            "message": "QA index job submitted",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error creating QA index: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qa-index-jobs/latest")
async def get_latest_paper_qa_index_job(
    arxiv_id: str,
    db_service=Depends(get_database_service),
):
    try:
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
    try:
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
    try:
        session = db_service.create_paper_chat_session(
            arxiv_id=arxiv_id,
            user_id=_normalize_user_id(payload.user_id),
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
    try:
        user_id = _normalize_user_id((payload or {}).get("user_id"))
        session = db_service.get_paper_chat_session(session_id, user_id=user_id)
        if not session or session.get("arxiv_id") != arxiv_id:
            raise HTTPException(status_code=404, detail="Chat session not found")
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
):
    try:
        user_id = _normalize_user_id(payload.user_id)
        source_message_id = (payload.source_message_id or "").strip() or None
        if not source_message_id and payload.session_id and payload.source_turn_id:
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

        if payload.include_in_profile:
            _apply_note_to_profile(db_service, note)

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
):
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

        if note.get("include_in_profile"):
            _apply_note_to_profile(db_service, note)

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
    try:
        normalized_user_id = _normalize_user_id(user_id)
        notes = db_service.list_paper_notes(arxiv_id=arxiv_id, user_id=normalized_user_id)
        serialized_notes = [_serialize_paper_note(item, db_service=db_service) for item in notes]
        paper = db_service.get_paper(arxiv_id)
        markdown = _build_notes_markdown(arxiv_id, serialized_notes, paper_title=(paper or {}).get("title"))
        filename = f"{sanitize_trace_slug(arxiv_id)}_notes.md"
        return StreamingResponse(
            iter([markdown.encode("utf-8")]),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.error("Error exporting paper notes: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/qa")
async def qa_paper(arxiv_id: str, payload: QaRequest, paper_qa_service=Depends(get_paper_qa_service)):
    try:
        return paper_qa_service.answer_question(arxiv_id, payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error in QA: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/qa/stream")
async def qa_paper_stream(
    arxiv_id: str,
    payload: QaRequest,
    paper_qa_service=Depends(get_paper_qa_service),
    generation_service=Depends(get_generation_service),
):
    question = payload.question.strip()
    logger.info("QA stream request for paper: %s, question: %s", arxiv_id, question)

    _, search_results, qa_context, retrieval_debug = paper_qa_service.build_qa_context(arxiv_id, payload)
    source_payload = paper_qa_service.build_source_payload(search_results)
    contextualized_question = str(qa_context.get("generation_question", question) or question).strip() or question
    question_contextualization = qa_context.get("question_contextualization", {}) or {}
    chat_session = qa_context.get("chat_session", {}) or {}

    def sse_event(event_name: str, data: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_stream():
        try:
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
                asset_metadata=[item for item in qa_context["asset_metadata"] if item.get("chunk_type") == "figure"],
            ):
                if chunk.get("type") == "delta":
                    yield sse_event("delta", {"delta": chunk.get("delta", "")})
                elif chunk.get("type") == "completed":
                    persisted_turn = paper_qa_service.persist_completed_turn(
                        chat_session=chat_session,
                        question=question,
                        answer=chunk.get("answer", "") or "",
                        source_payload=source_payload,
                        retrieval_debug=retrieval_debug if isinstance(retrieval_debug, dict) else None,
                        contextualized_question=contextualized_question,
                        question_contextualization=question_contextualization,
                    )
                    yield sse_event(
                        "done",
                        {
                            "status": "success",
                            "answer": chunk.get("answer", ""),
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
                            "retrieval_debug": retrieval_debug,
                            "usage": chunk.get("usage"),
                        },
                    )
                    return

            yield sse_event(
                "done",
                {
                    "status": "success",
                    "answer": "",
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
                    "usage": None,
                },
            )
        except Exception as exc:
            logger.error("Error in QA stream: %s", str(exc))
            yield sse_event("error", {"status": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
