from __future__ import annotations

from typing import Any, Mapping

from .schemas import PlanStep
from .tool_adapters.models import ToolExecutionResult


def project_tool_result(step: PlanStep, result: ToolExecutionResult) -> Any:
    """把 ToolExecutionResult 投影成 step.output_key 对应的运行时输出。

    Adapter 的真实输出已经由 output_model 校验；这里仅保留少量历史 output_key 兼容映射，
    避免下游 binding 在一次性迁移前读不到旧字段。新增工具应优先让 OutputModel 直接包含
    step.output_key 同名字段，而不是继续扩展执行器。
    """
    data = _model_to_plain(result.data)
    if not isinstance(data, Mapping):
        return data
    if step.output_key and step.output_key in data:
        return data.get(step.output_key)
    keys = _LEGACY_OUTPUT_PROJECTION.get(str(step.output_key or ""))
    if keys:
        projected = {key: data.get(key) for key in keys if key in data}
        if step.output_key in {"candidate_papers", "final_answer"} and len(projected) == 1:
            return next(iter(projected.values()))
        return projected
    return dict(data)


def _model_to_plain(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return value


_LEGACY_OUTPUT_PROJECTION = {
    "normalized_request": ["intent", "message", "search_spec"],
    "search_spec": [
        "intent",
        "query",
        "title_query",
        "abstract_query",
        "categories",
        "submitted_days_ago",
        "max_results",
        "sort_by",
        "sort_order",
        "field_operator",
        "category_operator",
        "reasoning_summary",
    ],
    "arxiv_result_quality": ["ok", "result_count", "warnings"],
    "paper_ref": ["arxiv_id", "title", "query", "matched_by", "source"],
    "paper_index_status": ["status", "has_index", "tool_result"],
    "index_build_result": ["status", "has_index", "tool_result"],
    "paper_qa_result": ["status", "answer", "sources", "retrieval_debug", "qa_observation", "error", "arxiv_id", "question", "tool_result"],
    "recommendation_profile": ["user_id", "research_profile", "user_memory_summary", "message", "request_context"],
    "candidate_papers": ["candidate_papers"],
    "recommendation_result": ["recommendations", "tool_result", "candidate_papers"],
    "validated_recommendations": ["ok", "recommendations"],
    "paper_reference": ["arxiv_id", "title", "query", "matched_by", "source"],
    "preference_action_result": ["status", "action", "label", "arxiv_id", "liked", "title", "message", "paper", "error", "tool_result"],
    "verified_preference_update": ["ok", "detail"],
    "missing_information": ["message", "missing_fields", "reason"],
    "final_answer": ["final_answer"],
}


__all__ = ["project_tool_result"]
