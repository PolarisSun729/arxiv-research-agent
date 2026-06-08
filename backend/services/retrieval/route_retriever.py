from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from services.retrieval.collection_profile import CollectionRetrievalProfile
from services.retrieval.contracts import QueryProfile, RetrievalOptions
from services.retrieval.execution import QueryEmbeddingBatcher, RouteExecutionSupport
from services.retrieval.retrieval_index import CollectionRetrievalIndex
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
        self.route_confidence_builder = route_confidence_builder
        self.structural_bonus_builder = structural_bonus_builder
        self.chunk_normalizer = chunk_normalizer
        self.memory_flag_reader = memory_flag_reader
        self.collection_profile_provider = collection_profile_provider
        self.collection_retrieval_index_provider = collection_retrieval_index_provider
        self.embedding_batcher = QueryEmbeddingBatcher(embedding_service=self.embedding_service)
        self.route_executor = RouteExecutionSupport(
            timeouts={
                "default": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_default_seconds", 8),
                "vector_original": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_vector_original_seconds", 8),
                "vector_rewrite": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_vector_rewrite_seconds", 8),
                "vector_hyde": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_vector_hyde_seconds", 8),
                "keyword": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_keyword_seconds", 4),
                "memory_context": ENHANCED_RETRIEVAL_CONFIG.get("route_timeout_memory_context_seconds", 3),
            },
            max_workers=ENHANCED_RETRIEVAL_CONFIG.get("route_max_workers", 4),
        )

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
            keyword_queries = [user_query, *query_views["selected_queries"], query_profile.semantic_query, query_profile.evidence_query]
            keyword_debug_holder: Dict[str, Any] = {}

            def run_keyword_route() -> List[Dict[str, Any]]:
                keyword_payload = self.keyword_retrieve(
                    collection_name=collection_name,
                    queries=keyword_queries,
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
            keyword_queries = []
            keyword_result = {"debug": collection_retrieval_index.to_keyword_debug(0, full_scan_used=False)}
            route_metrics["keyword"] = {"enabled": False, "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": 0, "error": "", "fallback_reason": "disabled", "cache_hit": bool(collection_retrieval_index.cache_hit), "timeout": False}

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
            "queries": keyword_queries,
            "selected_rewrite_queries": query_views["selected_queries"],
            "query_details": self.query_tools.build_query_term_details(keyword_queries),
            "keywords": self.query_tools.build_query_keywords(keyword_queries),
            **keyword_result["debug"],
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
            "memory_debug": memory_debug,
            "collection_profile": collection_profile.to_debug() if collection_profile else None,
            "route_metrics": route_metrics,
            "embedding_batch": embedding_batch["debug"],
        }

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
        results = self.vector_store_service.search_similar_vectors(collection_name=collection_name, query_vector=embedding, top_k=top_k)
        route_confidence = self.route_confidence_builder(route_name, query_profile, source_query, route_queries=route_queries, intent_profile=query_profile.intent_profile)
        return self.normalize_route_results(results, route_name, source_query, route_confidence, query_profile)

    def keyword_retrieve(self, collection_name: str, queries: List[str], top_k: int, query_profile: QueryProfile, retrieval_index: CollectionRetrievalIndex) -> Dict[str, Any]:
        if not retrieval_index.documents:
            return {"results": [], "debug": retrieval_index.to_keyword_debug(0, full_scan_used=False)}
        candidate_scores: Dict[int, float] = {}
        query_signatures: List[str] = []
        for query in queries:
            tokens = self.query_tools.tokenize_for_keyword_search(query)
            if not tokens:
                continue
            query_signatures.append(query)
            token_counter = Counter(tokens)
            # 只遍历倒排表命中的 doc，避免每次 query 都对 collection 全量 chunk 重算 BM25。
            candidate_doc_ids = set()
            for token in token_counter:
                candidate_doc_ids.update(retrieval_index.postings.get(token, set()))
            for doc_id in candidate_doc_ids:
                document = retrieval_index.documents[doc_id]
                candidate_scores[doc_id] = candidate_scores.get(doc_id, 0.0) + self.bm25_score(
                    token_counter,
                    document.token_counts,
                    retrieval_index.document_frequency,
                    len(retrieval_index.documents),
                    retrieval_index.avgdl,
                    document.content_prefix,
                    doc_length=document.doc_length,
                )
        ranked = [(doc_id, score) for doc_id, score in candidate_scores.items() if score > 0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return {"results": [], "debug": retrieval_index.to_keyword_debug(0, full_scan_used=False)}
        normalized_scores = self.normalize_scores([score for _, score in ranked])
        route_confidence = self.route_confidence_builder("keyword", query_profile, " | ".join(query_signatures) if query_signatures else query_profile.original_query, route_queries=query_signatures or [query_profile.keyword_query], intent_profile=query_profile.intent_profile)
        results: List[Dict[str, Any]] = []
        for rank, ((doc_id, score), normalized_score) in enumerate(zip(ranked[:top_k], normalized_scores[:top_k])):
            chunk = dict(retrieval_index.documents[doc_id].chunk)
            chunk["retrieval_route"] = "keyword"
            chunk["source_query"] = " | ".join(query_signatures)
            chunk["route_rank"] = rank + 1
            chunk["route_score"] = float(score)
            chunk["normalized_route_score"] = float(normalized_score)
            chunk["route_confidence"] = float(route_confidence)
            chunk["structural_bonus"] = float(self.structural_bonus_builder(chunk, query_profile))
            results.append(chunk)
        return {
            "results": results,
            "debug": retrieval_index.to_keyword_debug(len(ranked), full_scan_used=False),
        }

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

    def normalize_route_results(self, results: List[Dict[str, Any]], route_name: str, source_query: str, route_confidence: float, query_profile: QueryProfile) -> List[Dict[str, Any]]:
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
        doc_len = max(doc_length, 1)
        k1 = ENHANCED_RETRIEVAL_CONFIG["bm25_k1"]
        b = ENHANCED_RETRIEVAL_CONFIG["bm25_b"]
        score = 0.0
        content_lower = content.lower()
        for token, qtf in token_counter.items():
            if token not in doc_counts:
                continue
            df = max(doc_freqs.get(token, 0), 1)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = doc_counts[token]
            score += qtf * idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / max(avg_doc_length, 1.0)))))
            if token in {"method", "methods", "dataset", "datasets", "baseline", "ablation", "limitation", "limitations"} and token in content_lower[:300]:
                score += ENHANCED_RETRIEVAL_CONFIG["bm25_token_boost"]
        return score
