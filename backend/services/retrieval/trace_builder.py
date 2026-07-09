from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.retrieval.contracts import QueryProfile
from services.retrieval.table_evidence_formatter import summarize_table_evidence_debug
from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

logger = logging.getLogger(__name__)


class RetrievalTraceBuilder:
    """统一构造 retrieval debug 与 trace export，避免各阶段各自拼 debug。"""

    def __init__(
        self,
        *,
        trace_export_enabled: bool,
        trace_export_dir: Path,
        rrf_k: int,
        route_weights: Dict[str, float],
        route_confidence_builder: Any,
        route_weights_builder: Any,
    ) -> None:
        self.trace_export_enabled = trace_export_enabled
        self.trace_export_dir = trace_export_dir
        self.rrf_k = rrf_k
        self.route_weights = dict(route_weights or {})
        self.route_confidence_builder = route_confidence_builder
        self.route_weights_builder = route_weights_builder

    def build_debug(
        self,
        *,
        user_query: str,
        intent_profile: Any,
        query_profile: QueryProfile,
        query_views: Dict[str, Any],
        rerank_query: str,
        hyde_text: str,
        hyde_debug: Dict[str, Any],
        keyword_debug: Dict[str, Any],
        table_structured_debug: Dict[str, Any],
        memory_debug: Dict[str, Any],
        collection_profile: Dict[str, Any],
        route_metrics: Dict[str, Any],
        embedding_batch: Dict[str, Any],
        routes: Dict[str, List[Dict[str, Any]]],
        deduped_routes: Dict[str, List[Dict[str, Any]]],
        raw_retrieval_top30: List[Dict[str, Any]],
        fused_top30: List[Dict[str, Any]],
        reranked_top30: List[Dict[str, Any]],
        final_context_top15: List[Dict[str, Any]],
        final_results: List[Dict[str, Any]],
        rerank_debug: Dict[str, Any],
        context_expansion: Dict[str, Any],
        context_budget: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造保持旧字段兼容的 retrieval_debug，并按阶段组织 stages。"""
        effective_top_k = int(config["effective_top_k"])
        asset_type_counts = {
            "raw_retrieval_top30": self.count_chunk_types(raw_retrieval_top30),
            "fused_top30": self.count_chunk_types(fused_top30),
            "reranked_top30": self.count_chunk_types(reranked_top30),
            "final_context_top15": self.count_chunk_types(final_context_top15),
        }
        memory_retrieval_enabled = bool(memory_debug.get("enabled", False))
        multi_index_debug = self.build_multi_index_debug(
            routes=deduped_routes,
            raw_retrieval_top30=raw_retrieval_top30,
            fused_top30=fused_top30,
            reranked_top30=reranked_top30,
            final_context_top15=final_context_top15,
            config=config,
        )
        sparse_index_debug = self.build_sparse_index_debug(keyword_debug)
        return {
            "original_query": user_query,
            "original_question": user_query,
            "intent_profile": self.debug_intent_profile(intent_profile),
            "query_profile": self.debug_query_profile(query_profile),
            "query_plan": query_profile.query_plan,
            "query_views": query_views,
            "rewritten_queries": query_views["selected_queries"],
            "rerank_query": rerank_query,
            "hyde_text": hyde_text,
            "query_rewrite": query_views["rewrite_debug"],
            "hyde": hyde_debug,
            "collection_profile": collection_profile,
            "embedding_batch": embedding_batch,
            "route_metrics": route_metrics,
            "sparse_index": sparse_index_debug,
            "keyword_search": keyword_debug,
            "table_structured": table_structured_debug,
            "memory": {
                **memory_debug,
                "final_context_hits": [
                    self.debug_chunk_item(item)
                    for item in final_results[:effective_top_k]
                    if "memory_context" in (item.get("matched_routes", []) or [item.get("retrieval_route")])
                ],
            },
            "routes": {
                route_name: [self.debug_chunk_item(item) for item in route_results]
                for route_name, route_results in deduped_routes.items()
            },
            "multi_index": multi_index_debug,
            "stages": {
                "raw_retrieval_top30": [self.debug_chunk_item(item) for item in raw_retrieval_top30],
                "fused_top30": [self.debug_chunk_item(item) for item in fused_top30],
                "reranked_top30": [self.debug_chunk_item(item) for item in reranked_top30],
                "final_context_top15": [self.debug_chunk_item(item) for item in final_context_top15],
            },
            "final_chunks": [self.debug_chunk_item(item) for item in final_results],
            "config": {
                **config,
                "top_k": effective_top_k,
                "candidate_k": config["recall_candidate_limit"],
                "final_context_top_k": effective_top_k,
                "enable_memory_aware_retrieval": memory_retrieval_enabled,
            },
            "asset_type_counts": asset_type_counts,
            "fusion": {
                "algorithm": "pure_rrf",
                "rrf_k": self.rrf_k,
                "route_weights": self.route_weights_builder(intent_profile),
                "dedupe_per_route": True,
                "route_confidence": {
                    route_name: self.route_confidence_builder(
                        route_name,
                        query_profile,
                        (route_results[0]["source_query"] if route_results else user_query),
                        route_queries=self.collect_route_queries(route_results, query_views["selected_queries"], user_query),
                        intent_profile=intent_profile,
                    )
                    for route_name, route_results in routes.items()
                },
            },
            "llm_rerank": rerank_debug,
            "context_expansion": context_expansion,
            "context_budget": context_budget,
            "intent": self.debug_intent_profile(intent_profile),
        }

    def export_trace(
        self,
        *,
        original_question: str,
        user_query: str,
        collection_name: str,
        paper_context: Dict[str, Any],
        options: Dict[str, Any],
        query_profile: QueryProfile,
        intent_profile: Any,
        rerank_query: str,
        query_views: Dict[str, Any],
        hyde_debug: Dict[str, Any],
        table_structured_debug: Dict[str, Any],
        keyword_debug: Dict[str, Any],
        collection_profile: Dict[str, Any],
        route_metrics: Dict[str, Any],
        embedding_batch: Dict[str, Any],
        routes: Dict[str, List[Dict[str, Any]]],
        raw_retrieval_top30: List[Dict[str, Any]],
        fused_results: List[Dict[str, Any]],
        reranked_results: List[Dict[str, Any]],
        final_results: List[Dict[str, Any]],
        context_expansion: Dict[str, Any],
        context_budget: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """导出 JSON/Markdown trace；失败不能影响检索主流程。"""
        if not self.trace_export_enabled:
            return None

        try:
            paper_id = self.sanitize_trace_slug(str(paper_context.get("arxiv_id", "") or collection_name or "query"))
            export_dir = self.trace_export_dir / paper_id
            export_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            short_id = uuid.uuid4().hex[:8]
            slug = self.sanitize_trace_slug(user_query)
            base_name = f"{stamp}_{collection_name}_{slug}_{short_id}"
            json_path = export_dir / f"{base_name}.json"
            md_path = export_dir / f"{base_name}.md"

            multi_index_debug = self.build_multi_index_debug(
                routes=routes,
                raw_retrieval_top30=raw_retrieval_top30,
                fused_top30=fused_results[:30],
                reranked_top30=reranked_results[:30],
                final_context_top15=final_results,
                config=options,
            )
            sparse_index_debug = self.build_sparse_index_debug(keyword_debug)
            payload = {
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "arxiv_id": str(paper_context.get("arxiv_id", "") or ""),
                "collection_name": collection_name,
                "original_question": original_question,
                "user_query": user_query,
                "rerank_query": rerank_query,
                "intent_profile": self.normalize_trace_value(self.debug_intent_profile(intent_profile)),
                "paper_context": self.normalize_trace_value(paper_context),
                "options": self.normalize_trace_value(options),
                "query_profile": self.normalize_trace_value(self.debug_query_profile(query_profile)),
                "query_views": self.normalize_trace_value(query_views),
                "collection_profile": self.normalize_trace_value(collection_profile),
                "embedding_batch": self.normalize_trace_value(embedding_batch),
                "route_metrics": self.normalize_trace_value(route_metrics),
                "sparse_index": self.normalize_trace_value(sparse_index_debug),
                "keyword_search": self.normalize_trace_value(keyword_debug),
                "multi_index": self.normalize_trace_value(multi_index_debug),
                "context_expansion": self.normalize_trace_value(context_expansion),
                "context_budget": self.normalize_trace_value(context_budget),
                "steps": [
                    {"step": "query_profile", "result": self.normalize_trace_value(self.debug_query_profile(query_profile))},
                    {"step": "intent_profile", "result": self.normalize_trace_value(self.debug_intent_profile(intent_profile))},
                    {"step": "query_rewrite", "result": self.normalize_trace_value(query_views.get("rewrite_debug", {}))},
                    {"step": "original_question", "result": self.normalize_trace_value(original_question)},
                    {"step": "rerank_query", "result": self.normalize_trace_value(rerank_query)},
                    {"step": "hyde", "result": self.normalize_trace_value(hyde_debug)},
                    {"step": "table_structured", "result": self.normalize_trace_value(table_structured_debug)},
                    {"step": "keyword_search", "result": self.normalize_trace_value(keyword_debug)},
                    {"step": "sparse_index", "result": self.normalize_trace_value(sparse_index_debug)},
                    {"step": "collection_profile", "result": self.normalize_trace_value(collection_profile)},
                    {"step": "embedding_batch", "result": self.normalize_trace_value(embedding_batch)},
                    {"step": "route_metrics", "result": self.normalize_trace_value(route_metrics)},
                    {
                        "step": "routes",
                        "result": {
                            route_name: [self.normalize_trace_value(self.debug_chunk_item(item)) for item in route_results]
                            for route_name, route_results in routes.items()
                        },
                    },
                    {
                        "step": "multi_index",
                        "result": self.normalize_trace_value(multi_index_debug),
                    },
                    {
                        "step": "raw_retrieval_top30",
                        "result": [self.normalize_trace_value(self.debug_chunk_item(item)) for item in raw_retrieval_top30],
                    },
                    {
                        "step": "fused_top30",
                        "result": [self.normalize_trace_value(self.debug_chunk_item(item)) for item in fused_results[:30]],
                    },
                    {
                        "step": "reranked_top30",
                        "result": [self.normalize_trace_value(self.debug_chunk_item(item)) for item in reranked_results[:30]],
                    },
                    {
                        "step": "context_expansion",
                        "result": self.normalize_trace_value(context_expansion),
                    },
                    {
                        "step": "context_budget",
                        "result": self.normalize_trace_value(context_budget),
                    },
                    {
                        "step": "final_context_top15",
                        "result": [self.normalize_trace_value(self.debug_chunk_item(item)) for item in final_results],
                    },
                ],
            }

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            with md_path.open("w", encoding="utf-8") as f:
                f.write(self.render_retrieval_trace_text(payload))
            return {"json": str(json_path), "md": str(md_path)}
        except Exception as exc:  # pragma: no cover - trace export should never break retrieval
            logger.warning("Failed to export retrieval trace: %s", exc)
            return None

    def debug_query_profile(self, query_profile: Optional[QueryProfile]) -> Optional[Dict[str, Any]]:
        if query_profile is None:
            return None
        return {
            "original_query": query_profile.original_query,
            "normalized_query": query_profile.normalized_query,
            "language": query_profile.language,
            "intent_profile": self.debug_intent_profile(query_profile.intent_profile),
            "tokens": query_profile.tokens,
            "keywords": query_profile.keywords,
            "intent_tags": query_profile.intent_tags,
            "question_type": query_profile.question_type,
            "intent_summary": query_profile.intent_summary,
            "paper_terms": query_profile.paper_terms,
            "ambiguity_score": query_profile.ambiguity_score,
            "semantic_query": query_profile.semantic_query,
            "evidence_query": query_profile.evidence_query,
            "keyword_query": query_profile.keyword_query,
            "section_preferences": query_profile.section_preferences,
            "query_plan": query_profile.query_plan,
        }

    def debug_intent_profile(self, intent_profile: Any) -> Optional[Dict[str, Any]]:
        if intent_profile is None:
            return None
        return {
            "original_query": intent_profile.original_query,
            "normalized_query": intent_profile.normalized_query,
            "language": intent_profile.language,
            "main_intent": intent_profile.main_intent,
            "sub_intents": intent_profile.sub_intents,
            "confidence": intent_profile.confidence,
            "ambiguity_score": intent_profile.ambiguity_score,
            "intent_summary": intent_profile.intent_summary,
            "preferred_sections": intent_profile.preferred_sections,
            "route_weights": intent_profile.route_weights,
            "rewrite_count": intent_profile.rewrite_count,
            "use_keyword_search": intent_profile.use_keyword_search,
            "use_hyde": intent_profile.use_hyde,
            "rewrite_focus": intent_profile.rewrite_focus,
            "rerank_focus": intent_profile.rerank_focus,
            "fallback_reason": intent_profile.fallback_reason,
            "source": intent_profile.source,
        }

    def debug_chunk_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        preview = item.get("content", "")[: ENHANCED_RETRIEVAL_CONFIG["rerank_document_preview_limit"]].replace("\n", " ").strip()
        return {
            "chunk_id": item.get("chunk_id"),
            "original_chunk_id": item.get("original_chunk_id"),
            "index_id": item.get("index_id", ""),
            "index_type": item.get("index_type", ""),
            "retrieval_index_id": item.get("retrieval_index_id", ""),
            "retrieval_index_type": item.get("retrieval_index_type", ""),
            "retrieval_index_text_preview": self.short_text_preview(item.get("retrieval_index_text", ""), 180),
            "retrieval_index_weight": item.get("retrieval_index_weight", 1.0),
            "matched_index_id": item.get("matched_index_id", ""),
            "matched_index_type": item.get("matched_index_type", ""),
            "matched_index_text_preview": self.short_text_preview(item.get("matched_index_text", ""), 180),
            "matched_index_score": item.get("matched_index_score"),
            "matched_indexes": item.get("matched_indexes", []),
            "best_matched_index": item.get("best_matched_index", {}),
            "matched_index_types": item.get("matched_index_types", []),
            "index_aggregation_applied": item.get("index_aggregation_applied"),
            "index_aggregation_strategy": item.get("index_aggregation_strategy"),
            "index_aggregation_bonus": item.get("index_aggregation_bonus"),
            "index_aggregation_hit_count": item.get("index_aggregation_hit_count"),
            "index_aggregation_distinct_type_count": item.get("index_aggregation_distinct_type_count"),
            "index_aggregation_best_rank": item.get("index_aggregation_best_rank"),
            "chunk_type": item.get("chunk_type", "text"),
            "asset_kind": item.get("asset_kind", ""),
            "asset_path": item.get("asset_path", ""),
            "asset_summary": item.get("asset_summary", ""),
            "asset_summary_preview": self.short_text_preview(item.get("asset_summary", ""), 160),
            "asset_preview_text": item.get("asset_preview_text", ""),
            "asset_section_match_type": item.get("asset_section_match_type", ""),
            "asset_section_match_confidence": item.get("asset_section_match_confidence", 0.0),
            "asset_section_match_reason": item.get("asset_section_match_reason", ""),
            "asset_section_match_is_heuristic": item.get("asset_section_match_is_heuristic", False),
            "asset_section_match_allow_embedding": item.get("asset_section_match_allow_embedding", False),
            "table_id": item.get("table_id", ""),
            "table_evidence": self._table_evidence_debug(item.get("table_evidence")),
            "page_number": item.get("page_number"),
            "page_range": item.get("page_range"),
            "score": item.get("score"),
            "route_score": item.get("route_score"),
            "normalized_route_score": item.get("normalized_route_score"),
            "route_confidence": item.get("route_confidence"),
            "structural_bonus": item.get("structural_bonus"),
            "retrieval_route": item.get("retrieval_route"),
            "route_rank": item.get("route_rank"),
            "matched_routes": item.get("matched_routes", []),
            "route_scores": item.get("route_scores", {}),
            "source_query": item.get("source_query"),
            "source_queries": item.get("source_queries", []),
            # keyword route 的 BM25 可解释字段，便于在 debug 里直接看到“为什么召回”。
            "bm25_raw_score": item.get("bm25_raw_score"),
            "bm25_fused_score": item.get("bm25_fused_score"),
            "keyword_best_raw_bm25_score": item.get("keyword_best_raw_bm25_score"),
            "keyword_base_route_confidence": item.get("keyword_base_route_confidence"),
            "keyword_match_fields": item.get("keyword_match_fields", []),
            "keyword_matched_terms": item.get("keyword_matched_terms", []),
            "keyword_query_sources": item.get("keyword_query_sources", []),
            "keyword_query_contributions": item.get("keyword_query_contributions", []),
            "keyword_noise_flags": item.get("keyword_noise_flags", []),
            "keyword_hit_asset_field": item.get("keyword_hit_asset_field"),
            "memory_score": item.get("memory_score"),
            "memory_reason": item.get("memory_reason"),
            "source_turn_id": item.get("source_turn_id"),
            "is_recent_turn": item.get("is_recent_turn"),
            "memory_match_type": item.get("memory_match_type"),
            "memory_reference_strength": item.get("memory_reference_strength"),
            "rerank_text": item.get("rerank_text", ""),
            "rerank_text_preview": self.short_text_preview(item.get("rerank_text", ""), 160),
            "final_context_uses_original_chunk": item.get("final_context_uses_original_chunk"),
            "context_role": item.get("context_role"),
            "context_budget_score": item.get("context_budget_score"),
            "context_budget_reason": item.get("context_budget_reason"),
            "expansion_score": item.get("expansion_score"),
            "expansion_source_anchor_ids": item.get("expansion_source_anchor_ids", []),
            "relationship_types": item.get("relationship_types", []),
            "expansion_reasons": item.get("expansion_reasons", []),
            "final_context_reason": item.get("final_context_reason"),
            "subchunk_label": item.get("subchunk_label"),
            "section_tags": item.get("section_tags", []),
            "content": item.get("content", ""),
            "preview": preview,
        }

    @staticmethod
    def _table_evidence_debug(evidence: Any) -> Dict[str, Any]:
        if not isinstance(evidence, dict) or not evidence:
            return {}
        try:
            return summarize_table_evidence_debug(evidence)
        except Exception as exc:
            # trace 只记录坏 payload，不在 debug 展示阶段吞掉上游证据问题。
            return {"invalid_table_evidence": True, "error": str(exc)}

    @staticmethod
    def count_chunk_types(chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {"text": 0, "figure": 0, "table": 0}
        for chunk in chunks:
            chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
            counts[chunk_type] = counts.get(chunk_type, 0) + 1
        return counts

    @staticmethod
    def mark_final_context_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        marked_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            marked = dict(chunk)
            marked["final_context_uses_original_chunk"] = True
            marked_chunks.append(marked)
        return marked_chunks

    def log_retrieval_stage(self, stage_name: str, chunks: List[Dict[str, Any]]) -> None:
        """统一记录阶段快照，避免 pipeline 和 rerank 各自输出不同格式。"""
        logger.debug("%s count=%d", stage_name, len(chunks))
        for idx, chunk in enumerate(chunks[:10], start=1):
            logger.debug(
                "%s[%d] chunk_id=%s chunk_type=%s asset_kind=%s route=%s route_rank=%s route_score=%s fused_score=%s rerank_score=%s final_context_uses_original_chunk=%s key=%s",
                stage_name,
                idx,
                chunk.get("chunk_id"),
                chunk.get("chunk_type"),
                chunk.get("asset_kind"),
                chunk.get("retrieval_route"),
                chunk.get("route_rank"),
                chunk.get("route_score"),
                chunk.get("score"),
                chunk.get("llm_rerank_score"),
                chunk.get("final_context_uses_original_chunk"),
                self.chunk_unique_key(chunk),
            )
            if stage_name == "final_context_top15":
                logger.debug(
                    "final_context[%d] chunk_id=%s chunk_type=%s asset_kind=%s rerank_score=%s uses_original_chunk=%s original_chunk_preview=%s asset_summary_preview=%s rerank_preview=%s",
                    idx,
                    chunk.get("chunk_id"),
                    chunk.get("chunk_type"),
                    chunk.get("asset_kind"),
                    chunk.get("llm_rerank_score"),
                    chunk.get("final_context_uses_original_chunk"),
                    self.short_text_preview(chunk.get("content", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                    self.short_text_preview(chunk.get("asset_summary", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                    self.short_text_preview(chunk.get("rerank_text", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                )

    @staticmethod
    def collect_route_queries(
        route_results: List[Dict[str, Any]],
        fallback_queries: List[str],
        user_query: str,
    ) -> List[str]:
        queries = [item.get("source_query", "") for item in route_results if item.get("source_query")]
        if queries:
            return queries
        return fallback_queries or [user_query]

    @staticmethod
    def chunk_unique_key(item: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(item.get("chunk_type", "text")),
                str(item.get("asset_kind", "")),
                str(item.get("asset_path", "")),
                str(item.get("source", "")),
                str(item.get("original_chunk_id", item.get("parent_chunk_id", item.get("chunk_id", 0)))),
                str(item.get("content_part_label", "")),
                str(item.get("page_range", "")),
                str(item.get("order_index", "")),
            ]
        )

    def build_fusion_trace(
        self,
        routes: Dict[str, List[Dict[str, Any]]],
        fused_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "algorithm": "pure_rrf",
            "rrf_k": self.rrf_k,
            "route_weights": self.route_weights,
            "route_counts": {route_name: len(route_results) for route_name, route_results in routes.items()},
            "final_count": len(fused_results),
            "dedupe_per_route": True,
        }

    @staticmethod
    def build_sparse_index_debug(keyword_debug: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        keyword_debug = dict(keyword_debug or {})
        sparse_index = dict(keyword_debug.get("sparse_index") or {})
        if sparse_index:
            return sparse_index
        artifact = dict(keyword_debug.get("retrieval_index_artifact") or {})
        nested_sparse = dict(artifact.get("sparse_index_artifact") or {})
        load_source = str(keyword_debug.get("keyword_route_index_source") or artifact.get("keyword_route_index_source") or "")
        if not load_source:
            load_source = "keyword_route_disabled" if not keyword_debug.get("enabled", True) else "runtime_build_fallback"
        # trace 只输出稳定摘要；完整 artifact/debug 仍保留在 keyword_search 里用于深挖。
        return {
            "load_source": load_source,
            "build_id": str(nested_sparse.get("build_id") or artifact.get("build_id") or keyword_debug.get("keyword_index_build_id") or ""),
            "index_version": str(nested_sparse.get("index_version") or artifact.get("index_version") or keyword_debug.get("keyword_index_version") or ""),
            "source_type": str(nested_sparse.get("source_type") or artifact.get("source_type") or ""),
            "backend": str(nested_sparse.get("backend") or artifact.get("backend") or ""),
            "schema_version": str(nested_sparse.get("schema_version") or artifact.get("schema_version") or ""),
            "manifest_file": str(nested_sparse.get("sparse_index_manifest_file") or artifact.get("sparse_index_manifest_file") or ""),
            "document_count": int(keyword_debug.get("keyword_index_document_count") or 0),
            "keyword_route_hit_count": int(keyword_debug.get("keyword_candidate_count") or 0),
            "load_time_ms": round(float(keyword_debug.get("keyword_index_build_time") or 0.0) * 1000.0, 3) if load_source == "persistent_sparse_artifact" else 0.0,
            "fallback_count": 1 if bool(keyword_debug.get("keyword_index_fallback_used")) else 0,
            "artifact_stale_reason": str(nested_sparse.get("reason") or ""),
        }

    def build_multi_index_debug(
        self,
        *,
        routes: Dict[str, List[Dict[str, Any]]],
        raw_retrieval_top30: List[Dict[str, Any]],
        fused_top30: List[Dict[str, Any]],
        reranked_top30: List[Dict[str, Any]],
        final_context_top15: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """汇总 multi-index 召回解释，帮助判断问题发生在 index、聚合、fusion 还是 rerank。"""
        raw_index_hits: Dict[str, List[Dict[str, Any]]] = {}
        chunk_aggregation: Dict[str, List[Dict[str, Any]]] = {}
        route_hit_distribution: Dict[str, int] = {}
        index_type_counter: Counter[str] = Counter()

        for route_name, route_results in (routes or {}).items():
            flattened_hits: List[Dict[str, Any]] = []
            aggregation_rows: List[Dict[str, Any]] = []
            route_type_counter: Counter[str] = Counter()
            for candidate in route_results:
                matched_rows = self.collect_matched_index_rows(candidate)
                aggregation_rows.append(self.debug_index_aggregation_item(candidate, route_name=route_name))
                for row in matched_rows:
                    hit = self.debug_index_hit_item(candidate, row, route_name=route_name)
                    flattened_hits.append(hit)
                    index_type = str(hit.get("matched_index_type") or "body")
                    route_type_counter[index_type] += 1
                    index_type_counter[index_type] += 1
            raw_index_hits[route_name] = flattened_hits[:30]
            chunk_aggregation[route_name] = aggregation_rows[:30]
            route_hit_distribution[route_name] = len(route_results)

        return {
            "enabled": bool(
                config.get("enable_multi_index_embedding", True)
                or config.get("enable_index_level_bm25", True)
            ),
            "config": {
                "enable_multi_index_embedding": config.get("enable_multi_index_embedding"),
                "enable_index_level_bm25": config.get("enable_index_level_bm25"),
                "enable_generated_question_index": config.get("enable_generated_question_index"),
                "enable_chunk_level_retrieval_fallback": config.get("enable_chunk_level_retrieval_fallback"),
            },
            "route_hit_distribution": route_hit_distribution,
            "index_type_contribution": dict(index_type_counter),
            "raw_index_hits": raw_index_hits,
            "chunk_aggregation": chunk_aggregation,
            "stage_index_type_contribution": {
                "raw_retrieval_top30": self.count_matched_index_types(raw_retrieval_top30),
                "fused_top30": self.count_matched_index_types(fused_top30),
                "reranked_top30": self.count_matched_index_types(reranked_top30),
                "final_context_top15": self.count_matched_index_types(final_context_top15),
            },
        }

    def debug_index_hit_item(
        self,
        candidate: Dict[str, Any],
        row: Dict[str, Any],
        *,
        route_name: str,
    ) -> Dict[str, Any]:
        return {
            "retrieval_route": str(route_name or row.get("retrieval_route") or ""),
            "chunk_id": candidate.get("chunk_id"),
            "matched_index_id": row.get("matched_index_id") or row.get("index_id"),
            "matched_index_type": row.get("matched_index_type") or row.get("index_type"),
            "matched_index_text_preview": self.short_text_preview(
                row.get("matched_index_text") or row.get("index_text") or "",
                180,
            ),
            "index_level_score": row.get("matched_index_score"),
            "weighted_index_score": row.get("weighted_index_score"),
            "chunk_level_aggregated_score": candidate.get("route_score"),
            "source_query": row.get("source_query") or candidate.get("source_query"),
            "source_queries": row.get("source_queries") or candidate.get("source_queries", []),
            "hit_rank": row.get("hit_rank"),
        }

    def debug_index_aggregation_item(
        self,
        candidate: Dict[str, Any],
        *,
        route_name: str,
    ) -> Dict[str, Any]:
        best_index = candidate.get("best_matched_index") or {}
        if not isinstance(best_index, dict):
            best_index = {}
        return {
            "retrieval_route": route_name,
            "chunk_id": candidate.get("chunk_id"),
            "route_score": candidate.get("route_score"),
            "normalized_route_score": candidate.get("normalized_route_score"),
            "aggregation_applied": candidate.get("index_aggregation_applied"),
            "aggregation_strategy": candidate.get("index_aggregation_strategy"),
            "aggregation_bonus": candidate.get("index_aggregation_bonus"),
            "matched_index_count": candidate.get("index_aggregation_hit_count", len(candidate.get("matched_indexes", []) or [])),
            "matched_index_types": candidate.get("matched_index_types", []),
            "best_matched_index": {
                "matched_index_id": best_index.get("matched_index_id") or best_index.get("index_id"),
                "matched_index_type": best_index.get("matched_index_type") or best_index.get("index_type"),
                "matched_index_score": best_index.get("matched_index_score"),
                "weighted_index_score": best_index.get("weighted_index_score"),
                "matched_index_text_preview": self.short_text_preview(
                    best_index.get("matched_index_text") or best_index.get("index_text") or "",
                    180,
                ),
            },
            "source_queries": candidate.get("source_queries", []),
        }

    def collect_matched_index_rows(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = [row for row in (item.get("matched_indexes") or []) if isinstance(row, dict)]
        if rows:
            return [dict(row) for row in rows]
        if item.get("matched_index_id") or item.get("retrieval_index_id") or item.get("index_id"):
            # 旧 chunk-level 或聚合前结果只有主命中字段时，也转换成统一 index-hit 行，保证 trace 不断层。
            return [
                {
                    "matched_index_id": item.get("matched_index_id") or item.get("retrieval_index_id") or item.get("index_id"),
                    "matched_index_type": item.get("matched_index_type") or item.get("retrieval_index_type") or item.get("index_type"),
                    "matched_index_text": item.get("matched_index_text") or item.get("retrieval_index_text") or item.get("index_text"),
                    "matched_index_score": item.get("matched_index_score") or item.get("route_score") or item.get("score"),
                    "source_query": item.get("source_query"),
                }
            ]
        return []

    def count_matched_index_types(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for chunk in chunks or []:
            rows = self.collect_matched_index_rows(chunk)
            if rows:
                for row in rows:
                    counts[str(row.get("matched_index_type") or row.get("index_type") or "body")] += 1
            elif chunk.get("matched_index_type") or chunk.get("index_type"):
                counts[str(chunk.get("matched_index_type") or chunk.get("index_type") or "body")] += 1
        return dict(counts)

    def render_retrieval_trace_text(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("# Retrieval Trace")
        lines.append("")
        lines.append(f"- exported_at: {payload.get('exported_at', '')}")
        lines.append(f"- collection_name: {payload.get('collection_name', '')}")
        lines.append("")
        lines.append("## User Query")
        lines.append("")
        lines.append("```text")
        lines.append(self.normalize_trace_newlines(str(payload.get("user_query", ""))))
        lines.append("```")
        lines.append("")
        lines.append("## Options")
        for key, value in (payload.get("options") or {}).items():
            lines.append(f"- {key}: {value}")

        for step in payload.get("steps", []):
            lines.append("")
            lines.append(f"## {step.get('step', '')}")
            lines.append("")
            lines.append("```text")
            lines.append(self.format_trace_block(step.get("result")))
            lines.append("```")

        return "\n".join(lines).rstrip() + "\n"

    def format_trace_block(self, value: Any, indent: int = 0) -> str:
        pad = "  " * indent
        if isinstance(value, dict):
            if not value:
                return f"{pad}{{}}"
            lines: List[str] = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.append(self.format_trace_block(item, indent + 1))
                else:
                    lines.append(f"{pad}{key}: {self.normalize_trace_newlines(str(item))}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return f"{pad}[]"
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.append(self.format_trace_block(item, indent + 1))
                else:
                    lines.append(f"{pad}- {self.normalize_trace_newlines(str(item))}")
            return "\n".join(lines)
        return f"{pad}{self.normalize_trace_newlines(str(value))}"

    def normalize_trace_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.normalize_trace_newlines(value)
        if isinstance(value, dict):
            return {key: self.normalize_trace_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.normalize_trace_value(item) for item in value]
        return value

    @staticmethod
    def normalize_trace_newlines(text: str) -> str:
        if not text:
            return text
        return text.replace("\r\n", "\n").replace("\\r\\n", "\n").replace("\\n", "\n")

    @staticmethod
    def short_text_preview(text: Any, limit: int = ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit]

    @staticmethod
    def sanitize_trace_slug(text: str, max_length: int = ENHANCED_RETRIEVAL_CONFIG["sanitize_trace_slug_max_length"]) -> str:
        slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        if not slug:
            slug = "query"
        return slug[:max_length]
