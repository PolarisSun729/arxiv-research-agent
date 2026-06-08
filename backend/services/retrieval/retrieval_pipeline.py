from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.retrieval.contracts import RetrievalOptions, RetrievalPipelineResult, RerankResult
from services.retrieval.execution import RouteExecutionSupport
logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """串联完整检索链路，是 PaperQAService 未来可直接调用的检索入口。"""

    def __init__(
        self,
        *,
        query_planner: Any,
        route_retriever: Any,
        fusion_service: Any,
        rerank_service: Any,
        trace_builder: Any,
        collection_resolver: Any,
        collection_profile_provider: Any,
        collection_retrieval_index_provider: Any,
        option_resolver: Any,
        memory_flag_reader: Any,
        retrieval_config: Dict[str, Any],
        enhanced_config: Dict[str, Any],
    ) -> None:
        self.query_planner = query_planner
        self.route_retriever = route_retriever
        self.fusion_service = fusion_service
        self.rerank_service = rerank_service
        self.trace_builder = trace_builder
        self.collection_resolver = collection_resolver
        self.collection_profile_provider = collection_profile_provider
        self.collection_retrieval_index_provider = collection_retrieval_index_provider
        self.option_resolver = option_resolver
        self.memory_flag_reader = memory_flag_reader
        self.retrieval_config = retrieval_config
        self.enhanced_config = enhanced_config
        self.rerank_executor = RouteExecutionSupport(
            timeouts={
                "rerank": self.enhanced_config.get("route_timeout_rerank_seconds", 12),
                "default": self.enhanced_config.get("route_timeout_default_seconds", 8),
            },
            max_workers=1,
        )

    def retrieve(
        self,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        options: Optional[RetrievalOptions] = None,
    ) -> Dict[str, Any]:
        options = options or RetrievalOptions()
        runtime = self._resolve_runtime_options(options)
        normalized_collection_name = self.collection_resolver(collection_name)
        # collection 画像在单次检索开始时统一解析，后续 vector/keyword/memory route 共用同一份元信息。
        collection_profile = self.collection_profile_provider.get_profile(
            normalized_collection_name,
            index_record=paper_context,
        )
        collection_retrieval_index = self.collection_retrieval_index_provider.get_index(
            normalized_collection_name,
            collection_profile=collection_profile,
            index_record=paper_context,
        )

        query_bundle = self.query_planner.build_query_bundle(
            user_query=user_query,
            collection_name=normalized_collection_name,
            paper_context=paper_context,
            enable_query_rewrite=runtime["enable_query_rewrite"],
        )
        intent_profile = query_bundle["intent_profile"]
        query_profile = query_bundle["query_profile"]
        query_views = query_bundle["query_views"]
        rerank_query = query_bundle["rerank_query"]

        route_bundle = self.route_retriever.build_route_bundle(
            collection_name=normalized_collection_name,
            user_query=user_query,
            query_profile=query_profile,
            query_views=query_views,
            options=options,
            enable_hyde=runtime["enable_hyde"],
            enable_keyword_search=runtime["enable_keyword_search"],
            recall_candidate_limit=runtime["recall_candidate_limit"],
            collection_profile=collection_profile,
            collection_retrieval_index=collection_retrieval_index,
        )
        routes = route_bundle["routes"]
        memory_debug = route_bundle["memory_debug"]
        collection_profile_debug = route_bundle.get("collection_profile") or collection_profile.to_debug()
        route_metrics = dict(route_bundle.get("route_metrics") or {})
        embedding_batch_debug = route_bundle.get("embedding_batch") or {}

        fused_limit = runtime["rrf_candidate_limit"] if runtime["enable_llm_rerank"] else runtime["effective_top_k"]
        fusion_result = self.fusion_service.fuse(
            routes,
            fused_limit=fused_limit,
            raw_limit=runtime["recall_candidate_limit"],
            fused_stage_limit=runtime["rrf_candidate_limit"],
            query_profile=query_profile,
        )
        self.trace_builder.log_retrieval_stage("raw_retrieval_top30", fusion_result.raw_retrieval_top_n)
        self.trace_builder.log_retrieval_stage("fused_top30", fusion_result.fused_top_n)

        rerank_result = self._rerank_or_passthrough(
            enable_llm_rerank=runtime["enable_llm_rerank"],
            rerank_query=rerank_query,
            fused_results=fusion_result.fused_results,
            effective_top_k=runtime["effective_top_k"],
            rerank_candidate_limit=runtime["rerank_candidate_limit"],
            rrf_candidate_limit=runtime["rrf_candidate_limit"],
            query_profile=query_profile,
            original_question=user_query,
        )

        final_context_top15 = rerank_result.final_results[: runtime["effective_top_k"]]
        route_metrics["rerank"] = rerank_result.route_metric or {}
        debug = None
        if runtime["debug_enabled"]:
            debug = self.trace_builder.build_debug(
                user_query=user_query,
                intent_profile=intent_profile,
                query_profile=query_profile,
                query_views=query_views,
                rerank_query=rerank_query,
                hyde_text=route_bundle["hyde_text"],
                hyde_debug=route_bundle["hyde_debug"],
                keyword_debug=route_bundle["keyword_debug"],
                memory_debug=memory_debug,
                collection_profile=collection_profile_debug,
                route_metrics=route_metrics,
                embedding_batch=embedding_batch_debug,
                routes=routes,
                deduped_routes=fusion_result.deduped_routes,
                raw_retrieval_top30=fusion_result.raw_retrieval_top_n,
                fused_top30=fusion_result.fused_top_n,
                reranked_top30=rerank_result.reranked_results[: runtime["rrf_candidate_limit"]],
                final_context_top15=final_context_top15,
                final_results=rerank_result.final_results,
                rerank_debug=rerank_result.rerank_debug,
                config={
                    "requested_top_k": runtime["requested_top_k"],
                    "effective_top_k": runtime["effective_top_k"],
                    "recall_candidate_limit": runtime["recall_candidate_limit"],
                    "rrf_candidate_limit": runtime["rrf_candidate_limit"],
                    "rerank_candidate_limit": runtime["rerank_candidate_limit"],
                    "default_final_context_top_k": runtime["default_final_context_top_k"],
                    "max_final_context_top_k": runtime["max_final_context_top_k"],
                    "enable_query_rewrite": runtime["enable_query_rewrite"],
                    "enable_hyde": runtime["enable_hyde"],
                    "enable_keyword_search": runtime["enable_keyword_search"],
                    "enable_llm_rerank": runtime["enable_llm_rerank"],
                    "memory_source_boost_weight": float(
                        self.memory_flag_reader(
                            "memory_source_boost_weight",
                            self.enhanced_config.get("memory_source_boost_weight", 0.12),
                        )
                    ),
                },
            )

        trace_export = self.trace_builder.export_trace(
            original_question=user_query,
            user_query=user_query,
            collection_name=normalized_collection_name,
            paper_context=paper_context or {},
            options={
                "top_k": runtime["effective_top_k"],
                "requested_top_k": runtime["requested_top_k"],
                "default_final_context_top_k": runtime["default_final_context_top_k"],
                "max_final_context_top_k": runtime["max_final_context_top_k"],
                "candidate_k": runtime["recall_candidate_limit"],
                "enable_query_rewrite": runtime["enable_query_rewrite"],
                "enable_hyde": runtime["enable_hyde"],
                "enable_keyword_search": runtime["enable_keyword_search"],
                "enable_llm_rerank": runtime["enable_llm_rerank"],
                "debug": runtime["debug_enabled"],
            },
            query_profile=query_profile,
            intent_profile=intent_profile,
            rerank_query=rerank_query,
            query_views=query_views,
            hyde_debug=route_bundle["hyde_debug"],
            collection_profile=collection_profile_debug,
            route_metrics=route_metrics,
            embedding_batch=embedding_batch_debug,
            routes=fusion_result.deduped_routes,
            raw_retrieval_top30=fusion_result.raw_retrieval_top_n,
            fused_results=fusion_result.fused_results,
            reranked_results=rerank_result.reranked_results,
            final_results=rerank_result.final_results,
        )

        return RetrievalPipelineResult(
            chunks=rerank_result.final_results,
            debug=debug,
            trace_export=trace_export,
        ).to_response()

    def _rerank_or_passthrough(
        self,
        *,
        enable_llm_rerank: bool,
        rerank_query: str,
        fused_results: list[dict[str, Any]],
        effective_top_k: int,
        rerank_candidate_limit: int,
        rrf_candidate_limit: int,
        query_profile: Any,
        original_question: str,
    ) -> RerankResult:
        reranked_results = fused_results
        final_results = fused_results
        rerank_debug: Dict[str, Any] = {
            "enabled": enable_llm_rerank,
            "applied": False,
            "mode": "passthrough",
            "reason": "disabled",
            "input_chunks": len(fused_results),
            "output_chunks": len(fused_results),
        }
        if enable_llm_rerank:
            logger.debug("*" * 50)
            logger.debug("Entering LLM rerank module with %d candidate chunks", len(fused_results))
            logger.debug(
                "LLM rerank configuration: model=%s, provider=%s, batch_size=%d, candidate_limit=%d, fallback_local=%s",
                getattr(self.rerank_service, "llm_rerank_model_name_or_path", ""),
                getattr(self.rerank_service, "llm_rerank_provider", ""),
                getattr(self.rerank_service, "llm_rerank_batch_size", 0),
                rerank_candidate_limit,
                getattr(self.rerank_service, "llm_rerank_fallback_local", False),
            )
            logger.debug("*" * 50)
            rerank_exec = self.rerank_executor.run(
                "rerank",
                lambda: [
                    self.rerank_service.llm_rerank(
                        rerank_query,
                        fused_results,
                        effective_top_k,
                        query_profile=query_profile,
                        original_question=original_question,
                        candidate_limit=rerank_candidate_limit,
                    )
                ],
                required=False,
                cache_hit=None,
            )
            if rerank_exec.status == "ok" and rerank_exec.results:
                rerank_payload = rerank_exec.results[0]
                reranked_results = rerank_payload.get("reranked_chunks", rerank_payload["chunks"])
                final_results = self.trace_builder.mark_final_context_chunks(rerank_payload["chunks"])
                self.trace_builder.log_retrieval_stage("reranked_top30", reranked_results[:rrf_candidate_limit])
                self.trace_builder.log_retrieval_stage("final_context_top15", final_results[:effective_top_k])
                rerank_debug = rerank_payload["debug"]
            else:
                # rerank 是增强阶段，失败或超时时保持 fused 顺序，避免整条 QA 被外部 rerank 拖垮。
                reranked_results = fused_results
                final_results = self.trace_builder.mark_final_context_chunks(fused_results[:effective_top_k])
                self.trace_builder.log_retrieval_stage("reranked_top30", reranked_results[:rrf_candidate_limit])
                self.trace_builder.log_retrieval_stage("final_context_top15", final_results[:effective_top_k])
                rerank_debug = {
                    **rerank_debug,
                    "mode": "fallback_fused",
                    "reason": rerank_exec.fallback_reason or rerank_exec.error or "rerank_failed",
                    "applied": False,
                    "input_chunks": len(fused_results),
                    "output_chunks": len(final_results),
                }
            rerank_metric = rerank_exec.to_metric()
            # rerank wrapper 返回的是单个 payload，metric 里的候选数要按实际 chunk 数修正，
            # 否则 debug 会误导排查者以为 rerank 只处理了 1 条候选。
            rerank_metric["candidate_count"] = len(reranked_results)
        else:
            self.trace_builder.log_retrieval_stage("reranked_top30", reranked_results[:rrf_candidate_limit])
            final_results = self.trace_builder.mark_final_context_chunks(fused_results[:effective_top_k])
            self.trace_builder.log_retrieval_stage("final_context_top15", final_results[:effective_top_k])
            rerank_metric = {"enabled": False, "applied": False, "status": "disabled", "latency_ms": 0.0, "candidate_count": len(final_results), "error": "", "fallback_reason": "disabled", "cache_hit": None, "timeout": False}
        return RerankResult(
            final_results=final_results,
            reranked_results=reranked_results,
            rerank_debug=rerank_debug,
            route_metric=rerank_metric,
        )

    def _resolve_runtime_options(self, options: RetrievalOptions) -> Dict[str, Any]:
        default_final_context_top_k = max(1, int(self.enhanced_config["final_context_top_k"]))
        max_final_context_top_k = max(
            1,
            int(self.enhanced_config.get("max_final_context_top_k", default_final_context_top_k)),
        )
        requested_top_k = options.top_k
        effective_top_k = requested_top_k if requested_top_k is not None else default_final_context_top_k
        effective_top_k = min(max_final_context_top_k, max(1, int(effective_top_k)))
        return {
            "requested_top_k": requested_top_k,
            "effective_top_k": effective_top_k,
            "default_final_context_top_k": default_final_context_top_k,
            "max_final_context_top_k": max_final_context_top_k,
            "enable_query_rewrite": self.option_resolver(options.enable_query_rewrite, self.retrieval_config["enable_query_rewrite"]),
            "enable_hyde": self.option_resolver(options.enable_hyde, self.retrieval_config["enable_hyde"]),
            "enable_keyword_search": self.option_resolver(options.enable_keyword_search, self.retrieval_config["enable_keyword_search"]),
            "enable_llm_rerank": self.option_resolver(options.enable_llm_rerank, self.retrieval_config.get("enable_llm_rerank", False)),
            "debug_enabled": self.option_resolver(options.debug, self.retrieval_config["debug"]),
            "recall_candidate_limit": self.enhanced_config["recall_candidate_limit"],
            "rrf_candidate_limit": self.enhanced_config["rrf_candidate_limit"],
            "rerank_candidate_limit": self.enhanced_config["rerank_candidate_limit"],
        }
