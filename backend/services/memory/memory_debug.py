from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.memory.memory_models import MemoryDebugPayload


def _build_profile_debug(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """统计用户画像的关键装载信息，便于快速判断画像数据是否可用。"""
    loaded = profile is not None
    profile = profile or {}
    return {
        "loaded": loaded,
        "positive_topic_count": len(profile.get("positive_topics") or []),
        "negative_topic_count": len(profile.get("negative_topics") or []),
        "recent_topic_count": len(profile.get("recent_topics") or []),
        "preferred_category_count": len(profile.get("preferred_categories") or []),
        "has_preferred_answer_style": bool(profile.get("preferred_answer_style")),
    }


def _build_preference_debug(preference_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """整理偏好摘要的调试指标，方便观察点赞/点踩与动作记忆是否加载成功。"""
    loaded = preference_summary is not None
    preference_summary = preference_summary or {}
    counts = dict(preference_summary.get("counts") or {})
    return {
        "loaded": loaded,
        "liked_count": counts.get("liked_papers", len(preference_summary.get("liked_papers") or [])),
        "disliked_count": counts.get("disliked_papers", len(preference_summary.get("disliked_papers") or [])),
        "paper_action_types": sorted((preference_summary.get("paper_actions") or {}).keys()),
        "has_interest_vector": preference_summary.get("interest_vector") is not None,
    }


def _build_notes_debug(paper_notes: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """汇总论文笔记的装载情况与笔记类型分布。"""
    loaded = paper_notes is not None
    paper_notes = paper_notes or []
    return {
        "loaded": loaded,
        "note_count": len(paper_notes),
        "note_types": sorted({str(note.get("note_type") or "custom") for note in paper_notes}),
    }


def _build_chat_debug(paper_chat_history: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总论文对话历史的会话与消息规模，便于定位短期记忆来源。"""
    loaded = paper_chat_history is not None
    paper_chat_history = paper_chat_history or {}
    selected_session = paper_chat_history.get("selected_session") or {}
    return {
        "loaded": loaded,
        "session_count": len(paper_chat_history.get("sessions") or []),
        "message_count": len(paper_chat_history.get("messages") or []),
        "total_messages": int(paper_chat_history.get("total_messages") or 0),
        "selected_session_id": selected_session.get("session_id"),
    }


def build_memory_debug_payload(
    *,
    user_id: str,
    arxiv_id: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None,
    preference_summary: Optional[Dict[str, Any]] = None,
    paper_notes: Optional[List[Dict[str, Any]]] = None,
    paper_chat_history: Optional[Dict[str, Any]] = None,
    frontend_context: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造统一的记忆调试载荷，汇总后端装载源与前端上下文键集合。"""
    payload = MemoryDebugPayload(
        user_id=user_id,
        arxiv_id=arxiv_id,
        loaded_sources={
            "user_profile": _build_profile_debug(user_profile),
            "preference_summary": _build_preference_debug(preference_summary),
            "paper_notes": _build_notes_debug(paper_notes),
            "paper_chat_history": _build_chat_debug(paper_chat_history),
        },
        frontend_context_keys=sorted((frontend_context or {}).keys()),
        extra=dict(extra or {}),
    )
    return payload.to_dict()
