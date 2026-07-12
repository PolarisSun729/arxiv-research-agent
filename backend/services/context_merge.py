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
    "visible_paper_id",
    "visible_arxiv_id",
    "frontend_visible_paper_id",
    "frontend_selected_paper",
    "recommendation_topic",
    "topic_hint",
    "top_n",
    "recommendation_top_n",
    "max_age_months",
    "recommendation_max_age_months",
}

FRONTEND_PAPER_CANDIDATE_ALIASES: Set[str] = {
    "selected_paper",
    "arxiv_id",
    "active_arxiv_id",
    "paper_id",
    "arxivId",
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

    Agent 和 QA 入口都可能继续接收旧版 context，因此这里集中处理覆盖边界：
    后端记忆字段先落入 merged_context，前端字段只有在白名单内才会补充；
    对 selected_paper/arxiv_id 这类 UI 当前可见论文，只保留为候选值，避免回写成权威论文状态。
    """

    backend_payload = dict(backend_context or {})
    frontend_payload = dict(frontend_context or {}) if isinstance(frontend_context, Mapping) else {}
    allowlist = set(frontend_allowlist or FRONTEND_CONTEXT_ALLOWLIST)
    authority_fields = set(backend_authority_fields or BACKEND_AUTHORITY_FIELDS)
    merged_context: Dict[str, Any] = dict(backend_payload)
    accepted_frontend_fields: Dict[str, str] = {}
    ignored_frontend_fields: Dict[str, str] = {}

    for key, value in frontend_payload.items():
        if key in FRONTEND_PAPER_CANDIDATE_ALIASES:
            candidate_key = "frontend_visible_paper"
            # 前端当前可见论文只能作为本轮解析候选，不能改写 active_arxiv_id / selected_paper 等后端权威状态。
            merged_context[candidate_key] = value
            accepted_frontend_fields[key] = candidate_key
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
