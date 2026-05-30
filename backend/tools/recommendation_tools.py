from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from dependencies import get_recommendation_service as get_dependency_recommendation_service
from .tool_result import make_tool_error, make_tool_result, make_tool_trace


@lru_cache(maxsize=1)
def _get_recommendation_service():
    return get_dependency_recommendation_service()


def recommend_papers(user_id: str = "local_user", top_n: int = 10, max_age_months: int = 6) -> Dict[str, Any]:
    tool_name = "recommend_papers"
    trace_inputs = {"user_id": user_id, "top_n": top_n, "max_age_months": max_age_months}
    try:
        result = _get_recommendation_service().recommend_papers(
            user_id=user_id,
            top_n=top_n,
            max_age_months=max_age_months,
        )
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已为用户 {user_id} 生成推荐结果",
            data=result,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="recommendation_service"),
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
