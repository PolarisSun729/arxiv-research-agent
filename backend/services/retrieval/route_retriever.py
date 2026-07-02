from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from services.retrieval.collection_profile import CollectionRetrievalProfile
from services.retrieval.contracts import QueryProfile, RetrievalOptions
from services.retrieval.execution import QueryEmbeddingBatcher, RouteExecutionSupport
from services.retrieval.index_hit_aggregator import IndexHitAggregator
from services.retrieval.retrieval_index import CollectionRetrievalIndex, KEYWORD_FIELD_WEIGHTS, VECTOR_ROUTE_NAMES
from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

logger = logging.getLogger(__name__)


class RouteRetriever:
    """Run retrieval routes without fusion or rerank."""

    def __init__(
        self,
        *,
        embedding_service: Any,
        vector_store_service: Any,
        generation_service: Any,
        query_tools: Any,
        fusion_service: Any,
        table_structured_retriever: Any,
        route_confidence_builder: Any,
        structural_bonus_builder: Any,
        chunk_normalizer: Any,
        memory_flag_reader: Any,
        collection_profile_provider: Any,
        collection_retrieval_index_provider: Any,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.query_tools = query_tools
        self.fusion_service = fusion_service
        self.table_structured_retriever = table_structured_retriever
        self.route_confidence_builder = route_confidence_builder
        self.structural_bonus_builder = structural_bonus_builder
        self.chunk_normalizer = chunk_normalizer
        self.memory_flag_reader = memory_flag_reader
        self.collection_profile_provider = collection_profile_provider
        self.collection_retrieval_index_provider = collection_retrieval_index_provider
        self.embedding_batcher = QueryEmbeddingBatcher(embedding_service=self.embedding_service)
        self.index_hit_aggregator = IndexHitAggregator()
        self.route_executor = RouteExecutionSupport(
            timeouts={
                "default": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_default_seconds", 8),
                "vector_original": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_vector_original_seconds", 8),
                "vector_rewrite": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_vector_rewrite_seconds", 8),
                "vector_hyde": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_vector_hyde_seconds", 8),
                "keyword": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_keyword_seconds", 4),
                "table_structured": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_table_structured_seconds", 3),
                "memory_context": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_memory_context_seconds", 3),
            },
            max_workers=ENHANCED_RETRIEVAL_CONFIG.get("route_max_workers", 4),
        )
        self._keyword_backend_config = str(
            ENHANCED_RETRIEVAL_CONFIG.get("keyword_backend", "bm25s") or "bm25s"
        ).strip().lower()
        self._keyword_backend_init_fallback_reason = ""
        # keyword backend 会在初始化阶段就可能降级；保留配置值和降级原因，便于 benchmark/debug 判断实际跑的是哪套 BM25。
        self._keyword_backend = self._initialize_keyword_backend()

    def _initialize_keyword_backend(self) -> Any:
        """Initialize keyword backend based on config, with fallback to internal_bm25."""
        from services.retrieval.keyword_backend import InternalBM25Backend
        from services.retrieval.bm25s_backend import BM25sBackend

        backend_config = self._keyword_backend_config

        # Always create internal backend as fallback
        internal_backend = InternalBM25Backend(
            query_tools=self.query_tools,
            route_confidence_builder=self.route_confidence_builder,
            structural_bonus_builder=self.structural_bonus_builder,
            fusion_service=self.fusion_service,
        )

        if backend_config == "internal_bm25":
            logger.info("Using internal_bm25 keyword backend")
            return internal_backend

        # Try to use bm25s backend
        try:
            bm25s_backend = BM25sBackend(
                query_tools=self.query_tools,
                route_confidence_builder=self.route_confidence_builder,
                structural_bonus_builder=self.structural_bonus_builder,
                fusion_service=self.fusion_service,
            )

            # Check if bm25s is actually available
            if bm25s_backend._bm25s_available:
                logger.info("Using bm25s keyword backend")
                return bm25s_backend
            else:
                self._keyword_backend_init_fallback_reason = (
                    f"bm25s_not_available: {bm25s_backend._import_error}"
                )
                logger.warning(
                    "bm25s backend configured but not available, falling back to internal_bm25: %s",
                    bm25s_backend._import_error,
                )
                return internal_backend
        except Exception as exc:
            self._keyword_backend_init_fallback_reason = f"bm25s_initialization_failed: {exc}"
            logger.warning("Failed to initialize bm25s backend, falling back to internal_bm25: %s", exc)
            return internal_backend

    def _keyword_backend_debug(self, backend_name: str) -> Dict[str, Any]:
        """输出 keyword backend 的真实运行状态，避免 bm25s 初始化降级后指标被误读。"""
        return {
            "keyword_backend": backend_name,
            "keyword_backend_config": self._keyword_backend_config,
            "keyword_backend_init_fallback": bool(self._keyword_backend_init_fallback_reason),
            "keyword_backend_init_fallback_reason": self._keyword_backend_init_fallback_reason,
        }


    def build_route_bundle(
        self,
        *,
        collection_name: str,
        user_query: str,
        query_profile: QueryProfile,
        query_views: Dict[str, Any],
        options: RetrievalOptions,
        enable_hyde: bool,
        enable_keyword_search: bool,
        enable_table_structured_route: bool,
        recall_candidate_limit: int,
        collection_profile: Optional[CollectionRetrievalProfile] = None,
        collection_retrieval_index: Optional[CollectionRetrievalIndex] = None,
    ) -> Dict[str, Any]:
        if collection_profile is None:
            # 兼容直接调用 route 层的测试入口；正常 pipeline 会提前构建一次并复用。
            collection_profile = self.collection_profile_provider.get_profile(collection_name)
        if collection_retrieval_index is None:
            # keyword/memory 共享同一份 collection 索引，避免两个 route 各自全量读取 chunk。
            collection_retrieval_index = self.collection_retrieval_index_provider.get_index(
                collection_name,
                collection_profile=collection_profile,
            )
        intent_profile = query_profile.intent_profile
        hyde_text = ""
        hyde_debug: Dict[str, Any] = {"enabled": enable_hyde, "text": "", "source_queries": [], "focus_queries": [], "confidence": 0.0}
        if enable_hyde:
            hyde_text = self.generate_hyde_document(user_query, query_profile, query_views["selected_queries"])
            hyde_debug = {
                "enabled": True,
                "text": hyde_text,
                "source_queries": [user_query, *query_views["selected_queries"]],
                "focus_queries": query_views["selected_queries"][:2] if query_views["selected_queries"] else [user_query],
                "confidence": self.route_confidence_builder(
                    "vector_hyde",
                    query_profile,
                    hyde_text or user_query,
                    route_queries=query_views["selected_queries"] or [user_query],
                    intent_profile=intent_profile,
                ),
            }

        default_embedding_config = self.embedding_service.get_default_embedding_config()
        embedding_provider = str(collection_profile.embedding_provider or default_embedding_config.provider)
        embedding_model = str(collection_profile.embedding_model or default_embedding_config.model_name)
        embedding_dimension = int(collection_profile.vector_dimension) if collection_profile.vector_dimension else None
        rewrite_queries = query_views["selected_queries"] if query_views["enabled"] and query_views["selected_queries"] else []
        view_queries = query_views.get("view_queries", {}) or {}
        # batch 输入显式覆盖 original / semantic / evidence / keyword / rewrite / HyDE 视图；
        # QueryEmbeddingBatcher 会统一归一化去重，避免不同 route 各自请求 embedding。
        batch_queries = [
            user_query,
            view_queries.get("original", ""),
            view_queries.get("semantic", ""),
            view_queries.get("evidence", ""),
            view_queries.get("keywords", ""),
            *rewrite_queries,
            hyde_text,
        ]
        embedding_batch = self.embedding_batcher.build(
            batch_queries,
            provider=embedding_provider,
            model=embedding_model,
            dimension=embedding_dimension,
            batch_size=ENHANCED_RETRIEVAL_CONFIG.get("query_embedding_batch_size", 20),
        )
        embedding_map = embedding_batch["embeddings"]
        route_metrics: Dict[str, Any] = {}
        routes: Dict[str, List[Dict[str, Any]]] = {}
        original_result = self.route_executor.run(
            "vector_original",
            lambda: self.vector_retrieve(
                collection_name=collection_name,
                query=user_query,
                top_k=recall_candidate_limit,
                route_name="vector_original",
                source_query=user_query,
                query_profile=query_profile,
                route_queries=[user_query],
                collection_profile=collection_profile,
                embedding_map=embedding_map,
            ),
            required=True,
            cache_hit=bool(collection_profile.cache_hit),
        )
        routes["vector_original"] = original_result.results
        route_metrics["vector_original"] = original_result.to_metric()

        if rewrite_queries:
            rewrite_hits: List[Dict[str, Any]] = []
            rewrite_route_results = []
            for query in rewrite_queries:
                rewrite_result = self.route_executor.run(
                    "vector_rewrite",
                    lambda query=query: self.vector_retrieve(
                        collection_name=collection_name,
                        query=query,
                        top_k=recall_candidate_limit,
                        route_name="vector_rewrite",
                        source_query=query,
                        query_profile=query_profile,
                        route_queries=rewrite_queries,
                        collection_profile=collection_profile,
                        embedding_map=embedding_map,
                    ),
                    required=False,
                    cache_hit=bool(collection_profile.cache_hit),
                )
                rewrite_route_results.append(rewrite_result)
                rewrite_hits.extend(rewrite_result.results)
            routes["vector_rewrite"] = self.fusion_service.dedupe_preserve_order(rewrite_hits)
            route_metrics["vector_rewrite"] = self.combine_route_metrics("vector_rewrite", rewrite_route_results, len(routes["vector_rewrite"]))
        else:
            routes["vector_rewrite"] = []
            route_metrics["vector_rewrite"] = {"enabled": False, "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": 0, "error": "", "fallback_reason": "disabled", "cache_hit": bool(collection_profile.cache_hit), "timeout": False}

        if hyde_text:
            hyde_result = self.route_executor.run(
                "vector_hyde",
                lambda: self.vector_retrieve(
                    collection_name=collection_name,
                    query=hyde_text,
                    top_k=recall_candidate_limit,
                    route_name="vector_hyde",
                    source_query="hyde",
                    query_profile=query_profile,
                    route_queries=[hyde_text],
                    collection_profile=collection_profile,
                    embedding_map=embedding_map,
                ),
                required=False,
                cache_hit=bool(collection_profile.cache_hit),
            )
            routes["vector_hyde"] = hyde_result.results
            route_metrics["vector_hyde"] = hyde_result.to_metric()
        else:
            routes["vector_hyde"] = []
            route_metrics["vector_hyde"] = {"enabled": bool(enable_hyde), "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": 0, "error": "", "fallback_reason": "no_hyde_text", "cache_hit": bool(collection_profile.cache_hit), "timeout": False}

        if enable_keyword_search:
            # keyword route 以用户原问和 planner 的 keyword_query 为主，rewrite/semantic/evidence 只做辅助召回；
            # 否则通用改写词（如 method/section/framework）会在 BM25 中等权累加并压过真实问题焦点。
            keyword_query_views = self.build_keyword_query_views(
                user_query=user_query,
                query_profile=query_profile,
                query_views=query_views,
            )
            keyword_debug_holder: Dict[str, Any] = {}

            def run_keyword_route() -> List[Dict[str, Any]]:
                keyword_payload = self.keyword_retrieve(
                    collection_name=collection_name,
                    query_views=keyword_query_views,
                    top_k=recall_candidate_limit,
                    query_profile=query_profile,
                    retrieval_index=collection_retrieval_index,
                )
                keyword_debug_holder.update(keyword_payload["debug"])
                return keyword_payload["results"]

            keyword_exec = self.route_executor.run(
                "keyword",
                run_keyword_route,
                required=False,
                cache_hit=bool(collection_retrieval_index.cache_hit),
            )
            routes["keyword"] = keyword_exec.results
            keyword_result = {"debug": {**collection_retrieval_index.to_keyword_debug(len(routes["keyword"]), full_scan_used=False), **keyword_debug_holder}}
            route_metrics["keyword"] = keyword_exec.to_metric()
        else:
            routes["keyword"] = []
            keyword_query_views = []
            keyword_result = {"debug": collection_retrieval_index.to_keyword_debug(0, full_scan_used=False)}
            route_metrics["keyword"] = {"enabled": False, "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": 0, "error": "", "fallback_reason": "disabled", "cache_hit": bool(collection_retrieval_index.cache_hit), "timeout": False}

        if enable_table_structured_route:
            table_structured_debug_holder: Dict[str, Any] = {}

            def run_table_structured_route() -> List[Dict[str, Any]]:
                table_payload = self.table_structured_retrieve(
                    user_query=user_query,
                    query_profile=query_profile,
                    top_k=recall_candidate_limit,
                    retrieval_index=collection_retrieval_index,
                )
                table_structured_debug_holder.update(table_payload["debug"])
                return table_payload["results"]

            table_exec = self.route_executor.run(
                "table_structured",
                run_table_structured_route,
                required=False,
                cache_hit=bool(collection_retrieval_index.cache_hit),
            )
            routes["table_structured"] = table_exec.results
            table_structured_debug = dict(table_structured_debug_holder)
            route_metrics["table_structured"] = table_exec.to_metric()
        else:
            routes["table_structured"] = []
            table_structured_debug = {
                "enabled": False,
                "triggered_terms": [],
                "candidate_tables": [],
                "matched_tables": [],
                "matched_cells": [],
                "reason": "disabled",
            }
            route_metrics["table_structured"] = {"enabled": False, "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": 0, "error": "", "fallback_reason": "disabled", "cache_hit": bool(collection_retrieval_index.cache_hit), "timeout": False}

        memory_context = options.memory_context or {}
        memory_retrieval_enabled = bool(self.memory_flag_reader("enable_memory_aware_retrieval", True))
        if memory_retrieval_enabled:
            try:
                memory_debug_holder: Dict[str, Any] = {}

                def run_memory_route() -> List[Dict[str, Any]]:
                    memory_payload = self.memory_retrieve(
                        collection_name=collection_name,
                        memory_context=memory_context,
                        top_k=recall_candidate_limit,
                        query_profile=query_profile,
                        retrieval_index=collection_retrieval_index,
                    )
                    memory_debug_holder.update(memory_payload["debug"])
                    return memory_payload["results"]

                memory_exec = self.route_executor.run(
                    "memory_context",
                    run_memory_route,
                    required=False,
                    cache_hit=bool(collection_retrieval_index.cache_hit),
                )
                routes["memory_context"] = memory_exec.results
                memory_result = {"debug": {**collection_retrieval_index.to_memory_debug(lookup_mode="skipped", exact_hit_count=0, fallback_used=False, candidate_count=0), **memory_debug_holder}}
                memory_fallback_reason = None
                route_metrics["memory_context"] = memory_exec.to_metric()
            except Exception as exc:
                logger.warning("Memory-aware retrieval failed, skipping memory route: %s", exc)
                routes["memory_context"] = []
                memory_fallback_reason = str(exc)
                memory_result = {"debug": collection_retrieval_index.to_memory_debug(lookup_mode="error", exact_hit_count=0, fallback_used=False, candidate_count=0)}
                route_metrics["memory_context"] = {"enabled": True, "applied": False, "status": "error", "latency_ms": 0.0, "candidate_count": 0, "error": str(exc), "fallback_reason": str(exc), "cache_hit": bool(collection_retrieval_index.cache_hit), "timeout": False}
        else:
            routes["memory_context"] = []
            memory_fallback_reason = "disabled by runtime config"
            memory_result = {"debug": collection_retrieval_index.to_memory_debug(lookup_mode="disabled", exact_hit_count=0, fallback_used=False, candidate_count=0)}
            route_metrics["memory_context"] = {"enabled": False, "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": 0, "error": "", "fallback_reason": "disabled by runtime config", "cache_hit": bool(collection_retrieval_index.cache_hit), "timeout": False}

        keyword_debug = {
            "enabled": enable_keyword_search,
            "queries": [item["query"] for item in keyword_query_views],
            "query_views": keyword_query_views,
            "selected_rewrite_queries": query_views["selected_queries"],
            "query_details": self.query_tools.build_query_term_details([item["query"] for item in keyword_query_views]),
            "keywords": self.query_tools.build_query_keywords([item["query"] for item in keyword_query_views]),
            **keyword_result["debug"],
        }
        if not enable_keyword_search:
            # collection index 可能仍被 memory/table route 复用；这里标记的是 keyword route 本身没有消费 BM25。
            keyword_debug["keyword_route_index_source"] = "keyword_route_disabled"
            keyword_debug["keyword_index_fallback_used"] = False
            sparse_index_debug = dict(keyword_debug.get("sparse_index") or {})
            sparse_index_debug["load_source"] = "keyword_route_disabled"
            sparse_index_debug["keyword_route_hit_count"] = 0
            sparse_index_debug["fallback_count"] = 0
            keyword_debug["sparse_index"] = sparse_index_debug
        else:
            sparse_index_debug = dict(keyword_debug.get("sparse_index") or {})
            if sparse_index_debug and not sparse_index_debug.get("backend"):
                # runtime fallback 的实际 backend 由 keyword route 执行后才知道，这里补齐到 sparse trace 摘要。
                sparse_index_debug["backend"] = str(keyword_debug.get("keyword_backend") or "")
                keyword_debug["sparse_index"] = sparse_index_debug
        table_structured_debug = {
            "enabled": bool(enable_table_structured_route and table_structured_debug.get("enabled", False)),
            **table_structured_debug,
        }
        memory_debug = {
            "enabled": memory_retrieval_enabled,
            "applied": bool(memory_retrieval_enabled and memory_context.get("enabled", False) and routes["memory_context"]),
            "reason": str(memory_context.get("reason", "") or ""),
            "referenced_turn_ids": memory_context.get("referenced_turn_ids", []) or [],
            "referenced_source_ids": memory_context.get("referenced_source_ids", []) or [],
            "query_keywords": memory_context.get("query_keywords", []) or [],
            "candidates": memory_context.get("candidates", []) or [],
            "route_result_count": len(routes["memory_context"]),
            "fallback_reason": memory_fallback_reason or memory_context.get("fallback_reason"),
            **memory_result["debug"],
        }
        return {
            "routes": routes,
            "hyde_text": hyde_text,
            "hyde_debug": hyde_debug,
            "keyword_debug": keyword_debug,
            "table_structured_debug": table_structured_debug,
            "memory_debug": memory_debug,
            "collection_profile": collection_profile.to_debug() if collection_profile else None,
            "route_metrics": route_metrics,
            "embedding_batch": embedding_batch["debug"],
        }

    def build_keyword_query_views(
        self,
        *,
        user_query: str,
        query_profile: QueryProfile,
        query_views: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """统一管理 keyword route 的 query view 来源、去重与权重，避免 rewrite 数量直接放大 BM25。"""
        raw_views: List[Dict[str, Any]] = []
        view_queries = query_views.get("view_queries", {}) or {}
        raw_views.append({"query": user_query, "source": "original", "source_index": 0})
        raw_views.append({"query": query_profile.keyword_query, "source": "keyword", "source_index": 0})
        raw_views.append({"query": query_profile.evidence_query, "source": "evidence", "source_index": 0})
        raw_views.append({"query": query_profile.semantic_query, "source": "semantic", "source_index": 0})
        for idx, query in enumerate(query_views.get("selected_queries", []) or []):
            raw_views.append({"query": query, "source": "rewrite", "source_index": idx})
        hyde_query = str(view_queries.get("hyde", "") or "").strip()
        if hyde_query:
            raw_views.append({"query": hyde_query, "source": "hyde", "source_index": 0})

        kept: List[Dict[str, Any]] = []
        for row in raw_views:
            query = str(row.get("query", "") or "").strip()
            if not query:
                continue
            normalized = self.normalize_keyword_query_view(query)
            if not normalized:
                continue
            duplicate_reason = ""
            merged_into = ""
            duplicate_of = ""
            for existing in kept:
                if normalized == existing["normalized"]:
                    duplicate_reason = "same_normalized_query"
                    merged_into = existing["view_id"]
                    duplicate_of = existing["query"]
                    # 去重不是简单丢弃：把被合并 view 挂到主 view 上，debug 才能解释 rewrite 数量为何没有继续抬分。
                    existing.setdefault("merged_views", []).append(
                        {
                            "query": query,
                            "normalized": normalized,
                            "source": str(row.get("source", "rewrite") or "rewrite"),
                            "source_index": int(row.get("source_index", 0) or 0),
                            "reason": duplicate_reason,
                        }
                    )
                    break
                similarity = self.keyword_query_similarity(normalized, existing["normalized"])
                if similarity >= 0.88:
                    duplicate_reason = f"high_similarity:{similarity:.2f}"
                    merged_into = existing["view_id"]
                    duplicate_of = existing["query"]
                    # 高相似 query 只贡献一次 BM25 召回，避免同一关键词因多个 rewrite 重复投票。
                    existing.setdefault("merged_views", []).append(
                        {
                            "query": query,
                            "normalized": normalized,
                            "source": str(row.get("source", "rewrite") or "rewrite"),
                            "source_index": int(row.get("source_index", 0) or 0),
                            "reason": duplicate_reason,
                            "similarity": round(float(similarity), 4),
                        }
                    )
                    break
            if duplicate_reason:
                row["normalized"] = normalized
                row["selected"] = False
                row["reason"] = duplicate_reason
                row["merged_into"] = merged_into
                row["duplicate_of"] = duplicate_of
                continue
            source = str(row.get("source", "rewrite") or "rewrite")
            source_index = int(row.get("source_index", 0) or 0)
            view_id = f"{source}:{source_index}"
            kept.append(
                {
                    "view_id": view_id,
                    "query": query,
                    "normalized": normalized,
                    "source": source,
                    "source_index": source_index,
                    "selected": True,
                    "reason": "kept",
                    "merged_views": [],
                }
            )

        rewrite_count = sum(1 for row in kept if row["source"] == "rewrite")
        for row in kept:
            row["weight"] = self.keyword_query_view_weight(row["source"], rewrite_count=rewrite_count)
        return kept

    def normalize_keyword_query_view(self, query: str) -> str:
        tokens = self.filter_keyword_query_tokens(self.query_tools.tokenize_for_keyword_search(query))
        expanded_tokens = self.expand_keyword_query_tokens(tokens)
        seen: List[str] = []
        for token in expanded_tokens:
            normalized = str(token or "").strip().lower()
            if normalized and normalized not in seen:
                seen.append(normalized)
        return " ".join(seen)

    def keyword_query_similarity(self, left: str, right: str) -> float:
        left_tokens = set(str(left or "").split())
        right_tokens = set(str(right or "").split())
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return overlap / union if union else 0.0

    @staticmethod
    def keyword_rrf_k() -> int:
        """keyword route 自己的 query-view 融合使用较小的 RRF 常数，放大少量高质量 view 的区分度。"""
        return 12

    @staticmethod
    def keyword_query_view_weight(source: str, *, rewrite_count: int) -> float:
        """控制 query view 融合权重，rewrite 越多时单条 rewrite 权重越低，避免数量膨胀。"""
        normalized = str(source or "rewrite").strip().lower()
        if normalized == "original":
            return 1.22
        if normalized == "evidence":
            return 0.78
        if normalized == "semantic":
            return 0.48
        if normalized == "keyword":
            return 0.9
        if normalized == "hyde":
            return 0.34
        if normalized == "rewrite":
            return min(0.3, 0.65 / max(rewrite_count, 1))
        return 0.3

    @staticmethod
    def build_matched_terms(term_traces: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把跨 query view 聚合的命中词整理成可解释列表，按贡献分降序。"""
        matched = [
            {
                "token": str(trace.get("token", "")),
                "idf": float(trace.get("idf", 0.0) or 0.0),
                "term_score": float(trace.get("best_term_score", 0.0) or 0.0),
                "fields": list(trace.get("fields", []) or []),
                "query_sources": list(trace.get("query_sources", []) or []),
            }
            for trace in term_traces.values()
        ]
        matched.sort(key=lambda item: (item["term_score"], item["idf"]), reverse=True)
        return matched

    # IDF 低于此阈值的词在当前 collection 里近乎人人都有，命中它基本不解释相关性。
    LOW_IDF_THRESHOLD = 0.6
    ASSET_NOISE_FIELDS = {"asset_caption", "asset_aux"}

    def detect_keyword_noise_flags(
        self,
        *,
        matched_terms: List[Dict[str, Any]],
        matched_fields: List[str],
        query_profile: QueryProfile,
        chunk: Dict[str, Any],
    ) -> List[str]:
        """轻量噪声识别：标记低信息量命中、图表 OCR 误召回、与意图不符的字段命中。"""
        flags: List[str] = []
        if not matched_terms:
            flags.append("no_informative_terms")
            return flags

        informative_terms = [term for term in matched_terms if float(term.get("idf", 0.0)) >= self.LOW_IDF_THRESHOLD]
        if not informative_terms:
            # 全部命中词都是 collection 内泛词（低 IDF），BM25 高分多半是噪声堆出来的。
            flags.append("only_low_idf_terms")

        non_garbled = [term for term in matched_terms if self._is_informative_token(term.get("token", ""))]
        if len(non_garbled) < len(matched_terms):
            flags.append("garbled_tokens")

        is_figure_query = query_profile.question_type == "figure_table" or "figure_table" in (query_profile.intent_tags or [])
        asset_only_terms = [
            term
            for term in matched_terms
            if term.get("fields") and all(field in self.ASSET_NOISE_FIELDS for field in term.get("fields", []))
        ]
        if asset_only_terms and not is_figure_query:
            # 非图表问题里命中词只来自图表 caption/OCR，通常是表格残片误召回。
            flags.append("asset_field_only_hit")

        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        if chunk_type in {"figure", "table"} and not is_figure_query and informative_terms:
            top_term = informative_terms[0]
            if top_term.get("fields") and all(field in self.ASSET_NOISE_FIELDS for field in top_term.get("fields", [])):
                flags.append("intent_field_mismatch")

        section_path = self._normalize_field_text(str(chunk.get("section_path", "") or ""))
        section_title = self._normalize_field_text(str(chunk.get("section_title", "") or ""))
        only_section_hit = matched_fields and all(field in {"section_title", "section_path"} for field in matched_fields)
        if only_section_hit and max(len(section_path), len(section_title)) <= 3:
            flags.append("short_section_header_only")

        return flags

    def adjust_keyword_route_confidence(
        self,
        base_confidence: float,
        *,
        matched_terms: List[Dict[str, Any]],
        noise_flags: List[str],
        query_profile: QueryProfile,
    ) -> float:
        """结合实际 BM25 命中质量调节 keyword route confidence，而不是只看 query profile。

        - overview/summary 问题降低 BM25 影响，避免泛词 chunk 压过向量召回；
        - method/experiment/dataset/comparison/limitation 问题维持或略增，让精确术语补充向量；
        - 命中质量差（只命中低 IDF/噪声字段）时显著降权。
        """
        main_intent = self.query_intent_bucket(query_profile)
        factor = 1.0
        if main_intent in {"summary", "other"}:
            factor *= 0.82
        elif main_intent in {"method", "experiment", "dataset", "comparison", "limitation"}:
            factor *= 1.08
        elif main_intent == "figure_table":
            factor *= 1.0

        # 命中质量：以最高 IDF 命中词为代表，越是只命中泛词越要降权。
        max_idf = max((float(term.get("idf", 0.0) or 0.0) for term in matched_terms), default=0.0)
        if max_idf < self.LOW_IDF_THRESHOLD:
            factor *= 0.55
        elif max_idf < 1.2:
            factor *= 0.85

        penalty_flags = {
            "only_low_idf_terms": 0.6,
            "no_informative_terms": 0.5,
            "garbled_tokens": 0.85,
            "asset_field_only_hit": 0.6,
            "intent_field_mismatch": 0.7,
            "short_section_header_only": 0.8,
        }
        for flag in noise_flags:
            factor *= penalty_flags.get(flag, 1.0)

        floor = float(ENHANCED_RETRIEVAL_CONFIG.get("route_default_floor", 0.2))
        return max(floor * 0.5, min(1.0, float(base_confidence) * factor))

    def query_intent_bucket(self, query_profile: QueryProfile) -> str:
        bucketizer = getattr(self.query_tools, "legacy_intent_bucket", None)
        raw_intent = ""
        if query_profile.intent_profile is not None:
            raw_intent = str(getattr(query_profile.intent_profile, "main_intent", "") or "")
        raw_intent = raw_intent or str(query_profile.question_type or "other")
        if callable(bucketizer):
            return bucketizer(raw_intent)
        return str(raw_intent or "other").strip().lower()

    @staticmethod
    def _normalize_field_text(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    def _is_informative_token(self, token: str) -> bool:
        """优先复用 query_tools 的 token 质量判断，缺失时退回本地最小乱码检测。"""
        checker = getattr(self.query_tools, "is_informative_keyword_token", None)
        if callable(checker):
            return bool(checker(token))
        normalized = str(token or "").strip().lower()
        if len(normalized) <= 1:
            return False
        if re.search(r"[��]", normalized):
            return False
        if re.search(r"(.)\1{4,}", normalized):
            return False
        return True

    def generate_hyde_document(self, user_query: str, query_profile: QueryProfile, rewritten_queries: List[str]) -> str:
        if self.generation_service is not None:
            try:
                text = self.generation_service.generate_hyde_document(user_query)
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass
        return self.heuristic_hyde_document(query_profile, rewritten_queries)

    @staticmethod
    def combine_route_metrics(route_name: str, route_results: List[Any], candidate_count: int) -> Dict[str, Any]:
        if not route_results:
            return {"enabled": False, "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": 0, "error": "", "fallback_reason": "disabled", "cache_hit": None, "timeout": False}
        statuses = [item.status for item in route_results]
        errors = [item.error for item in route_results if item.error]
        fallback_reasons = [item.fallback_reason for item in route_results if item.fallback_reason]
        return {
            "enabled": True,
            "applied": candidate_count > 0,
            "status": "ok" if all(status == "ok" for status in statuses) else "partial",
            "latency_ms": sum(float(item.latency_ms or 0.0) for item in route_results),
            "candidate_count": candidate_count,
            "error": "; ".join(errors),
            "fallback_reason": "; ".join(fallback_reasons),
            "cache_hit": route_results[0].cache_hit,
            "timeout": any(bool(item.timeout) for item in route_results),
        }

    def vector_retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        route_name: str,
        source_query: str,
        query_profile: QueryProfile,
        route_queries: List[str],
        collection_profile: CollectionRetrievalProfile,
        embedding_map: Dict[str, List[float]],
    ) -> List[Dict[str, Any]]:
        normalized_query = QueryEmbeddingBatcher.normalize_query(query)
        embedding = embedding_map.get(normalized_query)
        if embedding is None:
            raise RuntimeError(f"Missing batch embedding for route={route_name} query={normalized_query[:80]}")
        # 向量库现在一条记录对应一个 retrieval index；多取少量 index hit 再聚合，避免 top_k 被同一 chunk 占满。
        index_hit_top_k = max(top_k, top_k * 3)
        results = self.vector_store_service.search_similar_vectors(
            collection_name=collection_name,
            query_vector=embedding,
            top_k=index_hit_top_k,
        )
        route_confidence = self.route_confidence_builder(route_name, query_profile, source_query, route_queries=route_queries, intent_profile=query_profile.intent_profile)
        return self.normalize_route_results(results, route_name, source_query, route_confidence, query_profile, top_k=top_k)

    def keyword_retrieve(self, collection_name: str, query_views: List[Dict[str, Any]], top_k: int, query_profile: QueryProfile, retrieval_index: CollectionRetrievalIndex) -> Dict[str, Any]:
        """Keyword retrieval with backend selection and automatic fallback."""
        from services.retrieval.keyword_backend import InternalBM25Backend

        # Try configured backend first
        try:
            result = self._keyword_backend.retrieve(
                query_views=query_views,
                top_k=top_k,
                query_profile=query_profile,
                retrieval_index=retrieval_index,
            )

            # If backend succeeded, return results with backend info
            if not result.fallback_reason:
                debug = dict(result.debug)
                debug.update(self._keyword_backend_debug(result.backend_name))
                debug["keyword_backend_fallback"] = False
                aggregated_results = self.index_hit_aggregator.aggregate(
                    result.results,
                    route_name="keyword",
                    top_k=top_k,
                )
                debug["index_hit_aggregation"] = {
                    "applied": True,
                    "input_count": len(result.results),
                    "output_count": len(aggregated_results),
                    "max_matched_indexes": self.index_hit_aggregator.max_matched_indexes,
                }
                return {"results": aggregated_results, "debug": debug}

            # Backend returned with fallback reason - log and fallback
            logger.warning(
                "Keyword backend %s failed, falling back to internal_bm25: %s",
                result.backend_name,
                result.fallback_reason,
            )

        except Exception as exc:
            logger.warning("Keyword backend execution failed, falling back to internal_bm25: %s", exc, exc_info=True)

        # Fallback to internal_bm25
        internal_backend = InternalBM25Backend(
            query_tools=self.query_tools,
            route_confidence_builder=self.route_confidence_builder,
            structural_bonus_builder=self.structural_bonus_builder,
            fusion_service=self.fusion_service,
        )

        fallback_result = internal_backend.retrieve(
            query_views=query_views,
            top_k=top_k,
            query_profile=query_profile,
            retrieval_index=retrieval_index,
        )
        aggregated_results = self.index_hit_aggregator.aggregate(
            fallback_result.results,
            route_name="keyword",
            top_k=top_k,
        )

        debug = dict(fallback_result.debug)
        debug.update(self._keyword_backend_debug("internal_bm25"))
        debug["keyword_backend_fallback"] = True
        debug["keyword_backend_fallback_reason"] = getattr(
            self._keyword_backend, "backend_name", lambda: "unknown"
        )() + " failed"
        debug["index_hit_aggregation"] = {
            "applied": True,
            "input_count": len(fallback_result.results),
            "output_count": len(aggregated_results),
            "max_matched_indexes": self.index_hit_aggregator.max_matched_indexes,
        }
        return {"results": aggregated_results, "debug": debug}

    def expand_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        expander = getattr(self.query_tools, "expand_keyword_query_tokens", None)
        if callable(expander):
            return expander(tokens)
        return tokens

    def filter_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        extractor = getattr(self.query_tools, "extract_query_keywords", None)
        if callable(extractor):
            return extractor(tokens, limit=max(len(tokens), 1))
        return tokens

    def keyword_field_weights_for_query(self, query_profile: QueryProfile) -> Dict[str, float]:
        """按问题类型动态调节字段权重，普通问题不让图表 OCR/caption 与正文等权竞争。"""
        weights = dict(KEYWORD_FIELD_WEIGHTS)
        main_intent = "other"
        if query_profile.intent_profile is not None:
            main_intent = str(getattr(query_profile.intent_profile, "main_intent", "") or "other")
        main_intent = str(query_profile.question_type or main_intent or "other").strip().lower()
        if main_intent == "figure_table" or "figure_table" in (query_profile.intent_tags or []):
            weights["asset_caption"] = 1.15
            weights["asset_aux"] = 0.72
        else:
            weights["asset_caption"] = min(weights.get("asset_caption", 0.0), 0.12)
            weights["asset_aux"] = min(weights.get("asset_aux", 0.0), 0.05)
        return weights

    def keyword_field_weights_for_document(
        self,
        base_weights: Dict[str, float],
        chunk: Dict[str, Any],
        query_profile: QueryProfile,
    ) -> Dict[str, float]:
        """按 chunk 类型二次调权，非图表问题下表格/图片 chunk 只作为弱补充候选。"""
        weights = dict(base_weights)
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        is_asset_chunk = chunk_type in {"figure", "table"}
        is_figure_query = query_profile.question_type == "figure_table" or "figure_table" in (query_profile.intent_tags or [])
        if is_asset_chunk and not is_figure_query:
            weights["body"] = min(weights.get("body", 1.0), 0.38)
            weights["section_title"] = min(weights.get("section_title", 1.0), 0.35)
            weights["section_path"] = min(weights.get("section_path", 1.0), 0.25)
            weights["asset_caption"] = min(weights.get("asset_caption", 0.0), 0.08)
            weights["asset_aux"] = min(weights.get("asset_aux", 0.0), 0.03)
        return weights

    @staticmethod
    def weight_keyword_document_fields(
        field_token_counts: Dict[str, Counter],
        query_token_counter: Counter,
        field_weights: Dict[str, float],
    ) -> tuple[Counter, int, Counter, Dict[str, Counter]]:
        """把字段级 token 合成为本次查询的 BM25 计数，同时保留命中字段用于诊断。

        返回 token_field_hits（query token -> {字段: 加权计数}），让 matched_terms trace
        能解释每个命中词到底来自正文还是图表 OCR 等字段。
        """
        doc_counts = Counter()
        match_fields = Counter()
        token_field_hits: Dict[str, Counter] = {}
        doc_length = 0.0
        query_tokens = set(query_token_counter)
        for field_name, field_counts in field_token_counts.items():
            weight = float(field_weights.get(field_name, 1.0) or 0.0)
            if weight <= 0:
                continue
            field_length = sum(field_counts.values())
            doc_length += field_length * weight
            for token, count in field_counts.items():
                weighted_count = count * weight
                doc_counts[token] += weighted_count
                if token in query_tokens:
                    match_fields[field_name] += weighted_count
                    token_field_hits.setdefault(token, Counter())[field_name] += weighted_count
        return doc_counts, max(int(round(doc_length)), 1), match_fields, token_field_hits

    def memory_retrieve(self, collection_name: str, memory_context: Dict[str, Any], top_k: int, query_profile: QueryProfile, retrieval_index: CollectionRetrievalIndex) -> Dict[str, Any]:
        if not memory_context or not bool(memory_context.get("enabled", False)):
            return {"results": [], "debug": retrieval_index.to_memory_debug(lookup_mode="disabled", exact_hit_count=0, fallback_used=False, candidate_count=0)}
        candidates = [item for item in (memory_context.get("candidates", []) or []) if isinstance(item, dict)]
        if not candidates:
            return {"results": [], "debug": retrieval_index.to_memory_debug(lookup_mode="no_candidates", exact_hit_count=0, fallback_used=False, candidate_count=0)}
        query_keywords = memory_context.get("query_keywords", []) or query_profile.keywords or []
        route_results: List[Dict[str, Any]] = []
        matched_doc_ids: List[int] = []
        lookup_modes: List[str] = []
        fallback_used = False
        for candidate in candidates:
            doc_ids, mode = self.lookup_memory_candidate(candidate, retrieval_index)
            if not doc_ids:
                doc_ids = self.memory_fallback_doc_ids(candidate, query_keywords, retrieval_index, top_k=top_k)
                if doc_ids:
                    mode = "limited_keyword_fallback"
                    fallback_used = True
            if mode:
                lookup_modes.append(mode)
            for doc_id in doc_ids:
                if doc_id not in matched_doc_ids:
                    matched_doc_ids.append(doc_id)
                chunk = retrieval_index.documents[doc_id].chunk
                route_results.append(self.build_memory_route_chunk(candidate, chunk, memory_context, query_keywords, query_profile))
        ranked = sorted(route_results, key=lambda item: (float(item.get("memory_score", 0.0) or 0.0), float(item.get("route_confidence", 0.0) or 0.0), float(item.get("structural_bonus", 0.0) or 0.0)), reverse=True)
        deduped = self.fusion_service.dedupe_route_results(ranked)
        for rank, item in enumerate(deduped[:top_k], start=1):
            item["route_rank"] = rank
        lookup_mode = "none"
        if lookup_modes:
            lookup_mode = "mixed" if len(set(lookup_modes)) > 1 else lookup_modes[0]
        return {
            "results": deduped[:top_k],
            "debug": retrieval_index.to_memory_debug(
                lookup_mode=lookup_mode,
                exact_hit_count=len(matched_doc_ids),
                fallback_used=fallback_used,
                candidate_count=len(route_results),
                full_scan_used=False,
            ),
        }

    def lookup_memory_candidate(self, candidate: Dict[str, Any], retrieval_index: CollectionRetrievalIndex) -> tuple[List[int], str]:
        """Lookup memory candidate by stable source/chunk metadata."""
        for key, mapping, mode in (
            ("source_id", retrieval_index.by_chunk_id, "source_id"),
            ("chunk_id", retrieval_index.by_chunk_id, "chunk_id"),
            ("parent_chunk_id", retrieval_index.by_parent_chunk_id, "parent_chunk_id"),
            ("original_chunk_id", retrieval_index.by_original_chunk_id, "original_chunk_id"),
        ):
            value = str(candidate.get(key, "") or "").strip()
            if value and mapping.get(value):
                return list(mapping[value]), mode

        page_number = str(candidate.get("page_number", "") or "").strip()
        section_path = str(candidate.get("section_path", "") or "").strip()
        if page_number and section_path:
            doc_ids = retrieval_index.by_page_section.get((page_number, section_path.lower()), [])
            if doc_ids:
                return list(doc_ids), "page_section"
        if page_number and retrieval_index.by_page_number.get(page_number):
            return list(retrieval_index.by_page_number[page_number]), "page_number"
        if section_path and retrieval_index.by_section_path.get(section_path):
            return list(retrieval_index.by_section_path[section_path]), "section_path"
        return [], "none"

    def memory_fallback_doc_ids(
        self,
        candidate: Dict[str, Any],
        query_keywords: List[str],
        retrieval_index: CollectionRetrievalIndex,
        *,
        top_k: int,
    ) -> List[int]:
        # 兜底只使用倒排表上的小范围候选，不允许退回 collection 全量 chunk 扫描。
        fallback_terms = [
            *query_keywords[:8],
            *self.query_tools.tokenize_for_keyword_search(str(candidate.get("content_preview", "") or ""))[:8],
            *self.query_tools.tokenize_for_keyword_search(str(candidate.get("memory_reason", "") or ""))[:4],
        ]
        candidate_ids: List[int] = []
        for token in fallback_terms:
            for doc_id in retrieval_index.postings.get(token, set()):
                if doc_id not in candidate_ids:
                    candidate_ids.append(doc_id)
                if len(candidate_ids) >= max(1, top_k):
                    return candidate_ids
        return candidate_ids

    def build_memory_route_chunk(
        self,
        candidate: Dict[str, Any],
        chunk: Dict[str, Any],
        memory_context: Dict[str, Any],
        query_keywords: List[str],
        query_profile: QueryProfile,
    ) -> Dict[str, Any]:
        memory_score = self.memory_candidate_score(candidate, chunk, query_keywords, query_profile)
        route_confidence = min(
            0.78,
            self.route_confidence_builder(
                "memory_context",
                query_profile,
                str(candidate.get("content_preview", "") or query_profile.original_query),
                route_queries=[str(candidate.get("content_preview", "") or query_profile.original_query)],
                intent_profile=query_profile.intent_profile,
            ) + min(0.18, memory_score * 0.15),
        )
        memory_chunk = dict(chunk)
        memory_chunk.update(
            {
                "retrieval_route": "memory_context",
                "source_query": query_profile.original_query,
                "route_score": float(memory_score),
                "normalized_route_score": float(min(1.0, memory_score / 1.4)),
                "route_confidence": float(route_confidence),
                "structural_bonus": float(self.structural_bonus_builder(memory_chunk, query_profile)),
                "memory_score": float(memory_score),
                "memory_reason": str(candidate.get("memory_reason", "") or memory_context.get("reason", "")),
                "source_turn_id": str(candidate.get("source_turn_id", "") or ""),
                "is_recent_turn": bool(candidate.get("is_recent_turn", False)),
                "memory_match_type": str(candidate.get("match_type", "metadata") or "metadata"),
                "memory_reference_strength": float(candidate.get("reference_strength", 0.0) or 0.0),
            }
        )
        return memory_chunk

    def normalize_route_results(
        self,
        results: List[Dict[str, Any]],
        route_name: str,
        source_query: str,
        route_confidence: float,
        query_profile: QueryProfile,
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []
        normalized_scores = self.normalize_scores([float(item.get("score", 0.0) or 0.0) for item in results])
        normalized = []
        for rank, (item, normalized_score) in enumerate(zip(results, normalized_scores), start=1):
            chunk = self.chunk_normalizer(item)
            chunk["retrieval_route"] = route_name
            chunk["source_query"] = source_query
            chunk["route_rank"] = rank
            chunk["route_score"] = float(item.get("score", 0.0) or 0.0)
            chunk["normalized_route_score"] = float(normalized_score)
            chunk["route_confidence"] = float(route_confidence)
            chunk["structural_bonus"] = float(self.structural_bonus_builder(chunk, query_profile))
            normalized.append(chunk)
        if route_name in VECTOR_ROUTE_NAMES:
            # dense route 命中的是 index 向量；聚合后再进入 RRF，保证下游继续按 chunk candidate 工作。
            return self.index_hit_aggregator.aggregate(
                normalized,
                route_name=route_name,
                top_k=top_k if top_k is not None else len(normalized),
            )
        return normalized

    def heuristic_hyde_document(self, query_profile: QueryProfile, rewritten_queries: List[str]) -> str:
        focus = ", ".join(rewritten_queries[:2]) if rewritten_queries else query_profile.keyword_query
        preferred_sections = ", ".join(query_profile.section_preferences[:3]) if query_profile.section_preferences else "relevant sections"
        return (
            "The paper likely contains a passage that answers the question with concrete evidence, terminology, "
            f"and section-level details. The most useful chunk is probably near {preferred_sections}. "
            f"Relevant terms may include: {focus}. The passage should help retrieve the specific evidence needed to answer: "
            f"{query_profile.original_query}"
        )

    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(min_score, max_score):
            return [1.0 for _ in scores]
        scale = max_score - min_score
        return [(score - min_score) / scale for score in scores]

    @staticmethod
    def memory_candidate_matches(candidate: Dict[str, Any], chunk: Dict[str, Any]) -> bool:
        candidate_ids = {str(candidate.get(key, "") or "").strip() for key in ("chunk_id", "parent_chunk_id", "original_chunk_id", "source_id")}
        chunk_ids = {str(chunk.get(key, "") or "").strip() for key in ("chunk_id", "parent_chunk_id", "original_chunk_id")}
        candidate_ids.discard("")
        chunk_ids.discard("")
        if candidate_ids and candidate_ids & chunk_ids:
            return True
        candidate_source = str(candidate.get("source", "") or "").strip().lower()
        candidate_section = str(candidate.get("section_path", "") or "").strip().lower()
        candidate_page = str(candidate.get("page_number", "") or "").strip()
        chunk_source = str(chunk.get("source", "") or "").strip().lower()
        chunk_section = str(chunk.get("section_path", "") or "").strip().lower()
        chunk_page = str(chunk.get("page_number", "") or "").strip()
        if candidate_source and candidate_section and candidate_source == chunk_source and candidate_section == chunk_section:
            return True
        return bool(candidate_source and candidate_page and candidate_source == chunk_source and candidate_page == chunk_page)

    def memory_candidate_score(self, candidate: Dict[str, Any], chunk: Dict[str, Any], query_keywords: List[str], query_profile: QueryProfile) -> float:
        score = 0.25
        if bool(candidate.get("is_recent_turn", False)):
            score += 0.28
        score += min(0.24, float(candidate.get("reference_strength", 0.0) or 0.0) * 0.24)
        candidate_ids = {str(candidate.get(key, "") or "").strip() for key in ("chunk_id", "parent_chunk_id", "original_chunk_id", "source_id")}
        chunk_ids = {str(chunk.get(key, "") or "").strip() for key in ("chunk_id", "parent_chunk_id", "original_chunk_id")}
        if {item for item in candidate_ids if item} & {item for item in chunk_ids if item}:
            score += 0.34
        chunk_terms_text = " ".join([str(chunk.get("content", "") or ""), str(chunk.get("section_path", "") or ""), str(chunk.get("asset_summary", "") or "")]).lower()
        overlap = sum(1 for keyword in query_keywords[:10] if str(keyword or "").strip().lower() in chunk_terms_text)
        if query_keywords:
            score += min(0.28, overlap / max(len(query_keywords[:10]), 1) * 0.28)
        score += max(0.0, float(self.structural_bonus_builder(chunk, query_profile)))
        return min(1.4, score)

    def table_structured_retrieve(
        self,
        *,
        user_query: str,
        query_profile: QueryProfile,
        top_k: int,
        retrieval_index: CollectionRetrievalIndex,
    ) -> Dict[str, Any]:
        """对结构化表格索引执行精确匹配，并转成统一 route 结果格式。"""
        payload = self.table_structured_retriever.retrieve(
            user_query=user_query,
            query_profile=query_profile,
            retrieval_index=retrieval_index,
            top_k=top_k,
        )
        raw_results = list(payload.get("results") or [])
        route_confidence = self.route_confidence_builder(
            "table_structured",
            query_profile,
            user_query,
            route_queries=[user_query, query_profile.evidence_query, query_profile.keyword_query],
            intent_profile=query_profile.intent_profile,
        )
        return {
            "results": self.normalize_route_results(raw_results, "table_structured", user_query, route_confidence, query_profile),
            "debug": payload.get("debug") or {},
        }

    @staticmethod
    def bm25_score(
        token_counter: Counter,
        doc_counts: Counter,
        doc_freqs: Dict[str, int],
        total_docs: int,
        avg_doc_length: float,
        content: str,
        *,
        doc_length: int,
    ) -> float:
        return RouteRetriever.bm25_score_detailed(
            token_counter,
            doc_counts,
            doc_freqs,
            total_docs,
            avg_doc_length,
            content,
            doc_length=doc_length,
        )["score"]

    @staticmethod
    def bm25_score_detailed(
        token_counter: Counter,
        doc_counts: Counter,
        doc_freqs: Dict[str, int],
        total_docs: int,
        avg_doc_length: float,
        content: str,
        *,
        doc_length: int,
        token_field_hits: Optional[Dict[str, Counter]] = None,
    ) -> Dict[str, Any]:
        """返回 BM25 分数及每个命中 token 的 idf/tf/贡献分，用于可解释 trace 与噪声识别。"""
        doc_len = max(doc_length, 1)
        k1 = ENHANCED_RETRIEVAL_CONFIG["bm25_k1"]
        b = ENHANCED_RETRIEVAL_CONFIG["bm25_b"]
        boost_tokens = {"method", "methods", "dataset", "datasets", "baseline", "ablation", "limitation", "limitations"}
        score = 0.0
        content_lower = content.lower()
        term_details: List[Dict[str, Any]] = []
        for token, qtf in token_counter.items():
            if token not in doc_counts:
                continue
            df = max(doc_freqs.get(token, 0), 1)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = doc_counts[token]
            term_score = qtf * idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / max(avg_doc_length, 1.0)))))
            boost = 0.0
            if token in boost_tokens and token in content_lower[:300]:
                boost = ENHANCED_RETRIEVAL_CONFIG["bm25_token_boost"]
            score += term_score + boost
            fields = []
            if token_field_hits and token in token_field_hits:
                fields = [field for field, _ in token_field_hits[token].most_common(4)]
            term_details.append(
                {
                    "token": token,
                    "idf": float(idf),
                    "tf": float(tf),
                    "qtf": float(qtf),
                    "df": int(df),
                    "term_score": float(term_score + boost),
                    "boosted": bool(boost > 0),
                    "fields": fields,
                }
            )
        term_details.sort(key=lambda item: item["term_score"], reverse=True)
        return {"score": float(score), "term_details": term_details}
