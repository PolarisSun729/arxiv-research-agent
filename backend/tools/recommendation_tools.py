from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from dependencies import get_recommendation_service as get_dependency_recommendation_service
from .tool_result import make_tool_error, make_tool_result, make_tool_trace
from utils.config import get_default_user_id


@lru_cache(maxsize=1)
def _get_recommendation_service():
    return get_dependency_recommendation_service()


def recommend_papers(
    user_id: str = None,
    top_n: int = 10,
    max_age_months: int = 6,
    message: Optional[str] = None,
    topic_hint: Optional[str] = None,
    user_memory_summary: Optional[str] = None,
    research_profile: Optional[Dict[str, Any]] = None,
    request_context: Optional[Dict[str, Any]] = None,
    candidate_papers: Optional[list[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    tool_name = "recommend_papers"
    user_id = str(user_id or get_default_user_id()).strip() or get_default_user_id()
    trace_inputs = {
        "user_id": user_id,
        "top_n": top_n,
        "max_age_months": max_age_months,
        "topic_hint": topic_hint,
        "message": message,
        "has_user_memory_summary": bool(str(user_memory_summary or "").strip()),
        "has_research_profile": bool(research_profile),
        "request_context_keys": sorted((request_context or {}).keys()) if isinstance(request_context, dict) else [],
        "candidate_paper_count": len(candidate_papers or []),
    }
    try:
        service = _get_recommendation_service()
        recommend = getattr(service, "recommend_papers_with_context", None) or service.recommend_papers
        result = recommend(
            user_id=user_id,
            top_n=top_n,
            max_age_months=max_age_months,
            message=message,
            topic_hint=topic_hint,
            user_memory_summary=user_memory_summary,
            research_profile=research_profile,
            request_context=request_context,
            candidate_papers=candidate_papers,
        )
        payload = dict(result or {})
        paper_actions = payload.get("paper_actions") if isinstance(payload.get("paper_actions"), dict) else {}
        payload["personalization_signals"] = {
            "used_user_memory_summary": bool(str(user_memory_summary or "").strip()),
            "used_research_profile": bool(research_profile),
            "used_request_topic": bool(str(topic_hint or message or "").strip()),
            "interest_profile_mode": payload.get("interest_profile_mode"),
            "interest_cluster_count": payload.get("interest_cluster_count", 0),
            "has_behavior_history": any(bool(paper_actions.get(key)) for key in paper_actions),
            "recall_mode": payload.get("recall_mode"),
            "used_agent_context_candidates": bool(candidate_papers),
            "cold_start_mode": bool((payload.get("recommendation_context") or {}).get("cold_start")),
        }
        if payload["personalization_signals"]["cold_start_mode"]:
            payload["fallback_record"] = {
                "code": "recommendation_cold_start",
                "stage": "recommendation",
                "category": "context_cold_start",
                "message": "推荐已退化为当前请求驱动的冷启动路径。",
                "raw_reason": "recommendation_cold_start",
                "source": "recommendation_tools.recommend_papers",
                "resolution": "continue_with_defaults",
                "detail": {
                    "used_request_topic": payload["personalization_signals"]["used_request_topic"],
                    "used_research_profile": payload["personalization_signals"]["used_research_profile"],
                },
            }
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已为用户 {user_id} 生成推荐结果",
            data=payload,
            trace=make_tool_trace(
                tool_name,
                inputs=trace_inputs,
                source="recommendation_service",
                notes={
                    # trace 需要标明是否真正消费了会话上下文，避免 recommendation 看起来 Agent 化、实际上仍是旧服务直出。
                    "uses_interest_vector": not bool((payload.get("recommendation_context") or {}).get("cold_start")),
                    "uses_historical_preferences": True,
                    "uses_research_profile": bool(research_profile) or bool(payload.get("research_profile")),
                    "uses_request_context": bool(request_context),
                    "uses_agent_context_candidates": bool(candidate_papers),
                },
            ),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="推荐论文失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="recommendation_service"),
            error=make_tool_error("recommendation_failed", str(exc)),
        )


def record_paper_preference(
    user_id: str,
    arxiv_id: str,
    liked: bool,
    paper: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tool_name = "record_paper_preference"
    trace_inputs = {"user_id": user_id, "arxiv_id": arxiv_id, "liked": liked, "paper": paper}
    try:
        result = _get_recommendation_service().record_user_paper_preference(
            user_id=user_id,
            arxiv_id=arxiv_id,
            liked=liked,
            paper_payload=paper,
        )
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary="已记录论文偏好",
            data=result,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="recommendation_service"),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="记录论文偏好失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="recommendation_service"),
            error=make_tool_error("record_preference_failed", str(exc)),
        )


def remove_user_paper_preference(
    user_id: str,
    arxiv_id: str,
    remove_scope: str = "both",
) -> Dict[str, Any]:
    tool_name = "remove_user_paper_preference"
    normalized_scope = str(remove_scope or "both").strip().lower() or "both"
    trace_inputs = {"user_id": user_id, "arxiv_id": arxiv_id, "remove_scope": normalized_scope}
    try:
        service = _get_recommendation_service()
        removed_liked = False
        removed_disliked = False
        if normalized_scope in {"both", "liked"}:
            removed_liked = bool(service.db_service.remove_liked_paper(user_id=user_id, arxiv_id=arxiv_id))
        if normalized_scope in {"both", "disliked"}:
            removed_disliked = bool(service.db_service.remove_disliked_paper(user_id=user_id, arxiv_id=arxiv_id))

        success = removed_liked or removed_disliked
        data = {
            "removed_liked": removed_liked,
            "removed_disliked": removed_disliked,
            "remove_scope": normalized_scope,
            "message": "已取消偏好标记" if success else "未找到可取消的偏好标记",
        }
        return make_tool_result(
            ok=success,
            tool_name=tool_name,
            summary=str(data["message"]),
            data=data,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="recommendation_service"),
            error=None if success else make_tool_error("remove_preference_not_found", "未找到可移除的喜欢/不喜欢标记"),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="取消论文偏好失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="recommendation_service"),
            error=make_tool_error("remove_preference_failed", str(exc)),
        )
