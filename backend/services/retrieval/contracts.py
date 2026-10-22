from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def legacy_body_index_id(chunk_ref: Any) -> str:
    """为没有 index 字段的旧 chunk-level 数据合成 body index ID（格式 "<chunk_ref>:body:legacy"）。

    各层对 chunk_ref 的取值优先级不同（parent/original/chunk_id 的顺序各有讲究），
    因此这里只统一 ID 格式，chunk_ref 由调用方按自身语义选择后传入。
    """
    return f"{chunk_ref}:body:legacy"


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
    # debug 模式默认还会把检索 trace 落盘；只需要内存中 debug 摘要的调用方显式传 False 跳过写盘。
    export_trace: Optional[bool] = None
    # 检索调试页要按 chunk 对齐各路召回与各阶段名次，需要原始 chunk 而不是 debug 摘要；只在显式要求时返回。
    include_stage_chunks: Optional[bool] = None
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
    ambiguity_score: float
    section_preferences: List[str]


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
    stage_chunks: Optional[Dict[str, Any]] = None

    def to_response(self) -> Dict[str, Any]:
        """转换为接口响应字典；trace_export 同时挂到顶层和 debug 下，兼容两种读取方式。"""
        payload: Dict[str, Any] = {"chunks": self.chunks}
        if self.debug is not None:
            payload["debug"] = self.debug
        if self.stage_chunks is not None:
            payload["stage_chunks"] = self.stage_chunks
        if self.trace_export:
            payload["trace_export"] = self.trace_export
            if isinstance(payload.get("debug"), dict):
                payload["debug"]["trace_export"] = self.trace_export
        return payload
