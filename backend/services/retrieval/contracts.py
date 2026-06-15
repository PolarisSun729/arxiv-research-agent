from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalOptions:
    """单次检索请求的运行时开关，保持对外 API 的轻量输入契约。"""

    top_k: Optional[int] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_table_structured_route: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    enable_context_expansion: Optional[bool] = None
    debug: Optional[bool] = None
    memory_context: Optional[Dict[str, Any]] = None


@dataclass
class QueryProfile:
    """query planning 阶段输出的标准画像，供召回、融合和 rerank 共享。"""

    original_query: str
    normalized_query: str
    language: str
    intent_profile: Any
    tokens: List[str]
    keywords: List[str]
    intent_tags: List[str]
    question_type: str
    intent_summary: str
    paper_terms: List[str]
    ambiguity_score: float
    semantic_query: str
    evidence_query: str
    keyword_query: str
    section_preferences: List[str]
    query_plan: Dict[str, Any]


@dataclass
class QueryPlanResult:
    """QueryPlanner 的完整输出；不包含任何召回结果。"""

    intent_profile: Any
    query_profile: QueryProfile
    query_views: Dict[str, Any]
    rerank_query: str


@dataclass
class RouteRetrievalResult:
    """RouteRetriever 的输出；只描述各路由召回和路由级 debug。"""

    routes: Dict[str, List[Dict[str, Any]]]
    hyde_text: str
    hyde_debug: Dict[str, Any]
    keyword_debug: Dict[str, Any]
    memory_debug: Dict[str, Any]


@dataclass
class FusionResult:
    """融合阶段输出；保留 raw 和 fused 两级排序，供 trace/debug 复用。"""

    deduped_routes: Dict[str, List[Dict[str, Any]]]
    raw_retrieval_top_n: List[Dict[str, Any]]
    fused_results: List[Dict[str, Any]]
    fused_top_n: List[Dict[str, Any]]


@dataclass
class RerankResult:
    """rerank 阶段输出；失败时 chunks 会保持 fused 顺序。"""

    final_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    rerank_debug: Dict[str, Any]
    route_metric: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalPipelineResult:
    """RetrievalPipeline 的最终结果，格式与 EnhancedRetrievalService 旧入口兼容。"""

    chunks: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None
    trace_export: Optional[Dict[str, str]] = None

    def to_response(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"chunks": self.chunks}
        if self.debug is not None:
            payload["debug"] = self.debug
        if self.trace_export:
            payload["trace_export"] = self.trace_export
            if isinstance(payload.get("debug"), dict):
                payload["debug"]["trace_export"] = self.trace_export
        return payload
