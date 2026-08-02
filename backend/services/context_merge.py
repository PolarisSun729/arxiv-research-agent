from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Set


BACKEND_AUTHORITY_FIELDS: Set[str] = {
    "user_memory_summary",
    "research_profile",
    "paper_qa_result",
    "active_arxiv_id",
    "arxiv_id",
    "active_paper_session_id",
    "selected_paper",
    "last_papers",
    "last_tool_calls_summary",
    "backend_memory",
    "runtime_state",
    "plan_runtime",
    "checkpoint",
    "runtime_checkpoint",
    "interaction",
}

FRONTEND_CONTEXT_ALLOWLIST: Set[str] = {
    "page_state",
    "current_page",
    "current_view",
    "ui_tab",
    "selected_tab",
    "active_tab",
    "debug",
    "top_k",
    "enable_query_rewrite",
    "enable_hyde",
    "enable_keyword_search",
    "enable_llm_rerank",
    "enable_context_expansion",
    "frontend_visible_paper",
    "recommendation_topic",
    "topic_hint",
    "top_n",
    "recommendation_top_n",
    "max_age_months",
    "recommendation_max_age_months",
}


@dataclass
class ContextMergeResult:
    """描述一次上下文合并结果，供业务入口和 debug 共同复用。"""

    merged_context: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def merge_backend_authoritative_context(
    *,
    backend_context: Optional[Mapping[str, Any]] = None,
    frontend_context: Optional[Mapping[str, Any]] = None,
    frontend_allowlist: Optional[Iterable[str]] = None,
    backend_authority_fields: Optional[Iterable[str]] = None,
) -> ContextMergeResult:
    """按“后端状态权威、前端只补充白名单”的规则合并上下文。

    Agent 和 QA 入口都经过这里统一定义权威边界：后端记忆字段先落入 merged_context，
    前端字段只有在白名单内才会补充；
    前端论文状态只能通过 frontend_visible_paper 传入，避免和后端权威 selected_paper/arxiv_id 重名。
    """

    backend_payload = dict(backend_context or {})
    frontend_payload = dict(frontend_context or {}) if isinstance(frontend_context, Mapping) else {}
    allowlist = set(frontend_allowlist or FRONTEND_CONTEXT_ALLOWLIST)
    authority_fields = set(backend_authority_fields or BACKEND_AUTHORITY_FIELDS)
    merged_context: Dict[str, Any] = dict(backend_payload)
    accepted_frontend_fields: Dict[str, str] = {}
    ignored_frontend_fields: Dict[str, str] = {}

    for key, value in frontend_payload.items():
        if key == "frontend_visible_paper":
            # 前端当前可见论文只作为本轮目标解析候选，不能改写后端已经确认的焦点。
            merged_context[key] = _validate_frontend_visible_paper(value)
            accepted_frontend_fields[key] = key
            continue
        if key in authority_fields:
            ignored_frontend_fields[key] = "backend_authoritative"
            continue
        if key in allowlist:
            merged_context[key] = value
            accepted_frontend_fields[key] = key
            continue
        ignored_frontend_fields[key] = "not_allowlisted"

    debug = {
        "policy": "backend_authoritative_frontend_allowlist",
        "backend_loaded_fields": sorted(backend_payload.keys()),
        "frontend_received_fields": sorted(frontend_payload.keys()),
        "frontend_accepted_fields": accepted_frontend_fields,
        "frontend_ignored_fields": ignored_frontend_fields,
        "backend_authority_fields": sorted(field for field in authority_fields if field in backend_payload),
    }
    return ContextMergeResult(merged_context=merged_context, debug=debug)


def _validate_frontend_visible_paper(value: Any) -> Dict[str, Any]:
    """校验前端论文候选的唯一结构，阻止标量 ID 再次覆盖论文对象。"""
    if not isinstance(value, Mapping):
        raise ValueError("frontend_visible_paper must be an object")
    arxiv_id = value.get("arxiv_id")
    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        raise ValueError("frontend_visible_paper.arxiv_id is required")
    title = value.get("title")
    if title is not None and not isinstance(title, str):
        raise ValueError("frontend_visible_paper.title must be a string")
    return dict(value)
