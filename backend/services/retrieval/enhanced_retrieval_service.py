import json
import math
import re
import logging
import threading
import uuid
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import requests

from services.embedding.embedding_service import EmbeddingService
from services.retrieval.query_planner import QueryPlanner
from services.retrieval.rerank_service import RerankService
from services.retrieval.route_retriever import RouteRetriever
from services.storage.vector_store_service import VectorStoreService
from utils.config import RETRIEVAL_CONFIG, get_enhanced_retrieval_runtime_config, get_memory_runtime_config
from utils.model_utils import get_huggingface_model_path
from services.intent.intent_service import IntentProfile, IntentService

try:  # pragma: no cover - optional dependency import is environment dependent
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

if TYPE_CHECKING:
    from services.llm.generation_service import GenerationService
else:
    try:
        from services.llm.generation_service import GenerationService
    except Exception:  # pragma: no cover - optional dependency fallback
        GenerationService = Any  # type: ignore


ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()
MEMORY_RUNTIME_CONFIG = get_memory_runtime_config()
QUERY_VIEW_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_view_limit"]
QUERY_PLAN_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_plan_limit"]

logger = logging.getLogger(__name__)

EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "please",
    "should",
    "summarize",
    "summarise",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
    "would",
    "you",
}

ZH_STOPWORDS = {
    "的",
    "了",
    "吗",
    "呢",
    "请",
    "这篇",
    "论文",
    "本文",
    "该文",
    "这个",
    "那个",
    "如何",
    "什么",
    "总结",
    "说明",
    "一下",
}

INTENT_RULES = {
    "summary": {
        "keywords": [
            "summary",
            "summarize",
            "summarise",
            "overview",
            "contribution",
            "contributions",
            "finding",
            "findings",
            "核心",
            "总结",
            "贡献",
            "概述",
            "要点",
        ],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "method": {
        "keywords": [
            "method",
            "methods",
            "approach",
            "framework",
            "architecture",
            "model",
            "training",
            "implementation",
            "方法",
            "模型",
            "框架",
            "架构",
            "训练",
            "实现",
        ],
        "preferred_sections": ["method", "approach", "model", "architecture"],
    },
    "experiment": {
        "keywords": [
            "experiment",
            "experiments",
            "evaluation",
            "result",
            "results",
            "benchmark",
            "benchmarks",
            "metric",
            "metrics",
            "实验",
            "结果",
            "评估",
            "基准",
        ],
        "preferred_sections": ["experiment", "results", "evaluation", "ablation"],
    },
    "comparison": {
        "keywords": [
            "baseline",
            "baselines",
            "compare",
            "comparison",
            "ablation",
            "compared",
            "对比",
            "比较",
            "基线",
            "消融",
        ],
        "preferred_sections": ["experiment", "results", "ablation"],
    },
    "limitation": {
        "keywords": [
            "limitation",
            "limitations",
            "weakness",
            "future work",
            "failure",
            "局限",
            "限制",
            "不足",
            "未来工作",
        ],
        "preferred_sections": ["conclusion", "discussion", "limitations", "appendix"],
    },
    "definition": {
        "keywords": [
            "definition",
            "define",
            "what is",
            "problem setup",
            "formulation",
            "定义",
            "概念",
            "任务定义",
            "问题设定",
        ],
        "preferred_sections": ["introduction", "background", "method"],
    },
    "dataset": {
        "keywords": [
            "dataset",
            "datasets",
            "corpus",
            "data",
            "training set",
            "测试集",
            "数据集",
            "语料",
        ],
        "preferred_sections": ["experiment", "dataset", "data"],
    },
}

SECTION_TAG_RULES = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "method": ["method", "methods", "approach", "framework", "architecture", "model"],
    "experiment": ["experiment", "experiments", "evaluation", "results"],
    "ablation": ["ablation"],
    "conclusion": ["conclusion", "summary", "discussion", "future work"],
    "appendix": ["appendix"],
    "prompt": ["prompt", "instruction"],
    "figure": ["figure", "fig."],
    "table": ["table"],
    "references": ["references", "bibliography"],
}

NOISY_SECTION_TAGS = {"appendix", "prompt", "figure", "table", "references"}

QUESTION_TYPE_RULES = {
    "method_flow": {
        "keywords": [
            "method",
            "methods",
            "approach",
            "framework",
            "workflow",
            "pipeline",
            "algorithm",
            "model",
            "architecture",
            "training",
            "inference",
            "流程",
            "方法",
            "框架",
            "模型",
            "算法",
        ],
        "preferred_sections": ["method", "approach", "model", "architecture", "introduction"],
    },
    "experiment_setup": {
        "keywords": [
            "experiment",
            "experiments",
            "evaluation",
            "dataset",
            "baseline",
            "metric",
            "implementation",
            "setup",
            "setting",
            "实验",
            "数据集",
            "baseline",
            "指标",
        ],
        "preferred_sections": ["experiment", "evaluation", "results", "ablation"],
    },
    "results_analysis": {
        "keywords": [
            "result",
            "results",
            "performance",
            "comparison",
            "ablation",
            "analysis",
            "effect",
            "improve",
            "效果",
            "结果",
            "对比",
            "分析",
        ],
        "preferred_sections": ["results", "evaluation", "ablation", "discussion"],
    },
    "contribution": {
        "keywords": [
            "contribution",
            "contributions",
            "novel",
            "novelty",
            "idea",
            "propose",
            "proposed",
            "创新",
            "贡献",
            "提出",
        ],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "limitation": {
        "keywords": [
            "limitation",
            "limitations",
            "weakness",
            "future work",
            "failure",
            "constraint",
            "不足",
            "局限",
            "未来工作",
        ],
        "preferred_sections": ["discussion", "conclusion", "limitations", "appendix"],
    },
    "dataset": {
        "keywords": [
            "dataset",
            "datasets",
            "corpus",
            "benchmark",
            "data",
            "training data",
            "语料",
            "数据",
            "基准",
        ],
        "preferred_sections": ["experiment", "dataset", "data", "setup"],
    },
    "metric": {
        "keywords": [
            "metric",
            "metrics",
            "measure",
            "evaluation",
            "formula",
            "objective",
            "指标",
            "公式",
            "评价",
        ],
        "preferred_sections": ["method", "experiment", "evaluation"],
    },
    "figure_table": {
        "keywords": [
            "figure",
            "fig.",
            "table",
            "chart",
            "diagram",
            "图",
            "表",
        ],
        "preferred_sections": ["figure", "table", "appendix", "results"],
    },
    "summary": {
        "keywords": [
            "summary",
            "summarize",
            "overview",
            "paper",
            "whole paper",
            "总体",
            "总结",
            "概述",
        ],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "other": {
        "keywords": [],
        "preferred_sections": ["abstract", "introduction", "method", "experiment", "conclusion"],
    },
}


@dataclass
class RetrievalOptions:
    top_k: Optional[int] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    debug: Optional[bool] = None
    memory_context: Optional[Dict[str, Any]] = None


@dataclass
class QueryProfile:
    original_query: str
    normalized_query: str
    language: str
    intent_profile: IntentProfile
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


class EnhancedRetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        generation_service: Optional[GenerationService] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.intent_service = IntentService(generation_service=self.generation_service)
        self.rrf_k = RETRIEVAL_CONFIG["rrf_k"]
        self.route_weights = RETRIEVAL_CONFIG["route_weights"]
        self.candidate_multiplier = RETRIEVAL_CONFIG["candidate_multiplier"]
        self.llm_rerank_model_name_or_path = RETRIEVAL_CONFIG.get(
            "llm_rerank_model_name_or_path", "00-models/Qwen3-VL-Reranker-2B"
        )
        self.llm_rerank_prompt = RETRIEVAL_CONFIG.get(
            "llm_rerank_prompt", "Retrieve text relevant to the user's query."
        )
        self.llm_rerank_provider = str(RETRIEVAL_CONFIG.get("llm_rerank_provider", "dashscope")).strip().lower()
        self.llm_rerank_api_key = str(RETRIEVAL_CONFIG.get("llm_rerank_api_key", "")).strip()
        self.llm_rerank_base_url = str(
            RETRIEVAL_CONFIG.get(
                "llm_rerank_base_url",
                "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
            )
        ).strip()
        self.llm_rerank_fallback_local = bool(RETRIEVAL_CONFIG.get("llm_rerank_fallback_local", True))
        self.llm_rerank_model_name = str(
            RETRIEVAL_CONFIG.get("llm_rerank_model_name_or_path", "qwen3-rerank")
        ).strip()
        self.llm_rerank_local_model_name_or_path = str(
            RETRIEVAL_CONFIG.get("llm_rerank_local_model_name_or_path", "00-models/Qwen3-VL-Reranker-2B")
        ).strip()
        self.llm_rerank_batch_size = int(RETRIEVAL_CONFIG.get("llm_rerank_batch_size", 8))
        self.llm_rerank_candidate_limit = int(RETRIEVAL_CONFIG.get("llm_rerank_candidate_limit", 24))
        self.llm_rerank_max_doc_chars = int(RETRIEVAL_CONFIG.get("llm_rerank_max_doc_chars", 4096))
        self.trace_export_enabled = bool(RETRIEVAL_CONFIG.get("trace_export_enabled", True))
        self.trace_export_dir = Path(str(RETRIEVAL_CONFIG.get("trace_export_dir", "temp/retrieval-traces")))
        self._llm_reranker = None
        self._llm_reranker_lock = threading.Lock()
        self._llm_reranker_path: Optional[str] = None
        self._llm_reranker_device: Optional[str] = None
        self._llm_reranker_error: Optional[str] = None
        self.memory_runtime_config = get_memory_runtime_config()
        self.query_planner = QueryPlanner(self)
        self.route_retriever = RouteRetriever(self)
        self.rerank_service = RerankService(self)

    def _memory_flag(self, key: str, default: Any = None) -> Any:
        return self.memory_runtime_config.get(key, default)

    def enhanced_retrieve(
        self,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        options: Optional[RetrievalOptions] = None,
    ) -> Dict[str, Any]:
        options = options or RetrievalOptions()
        default_final_context_top_k = max(1, int(ENHANCED_RETRIEVAL_CONFIG["final_context_top_k"]))
        max_final_context_top_k = max(
            1,
            int(ENHANCED_RETRIEVAL_CONFIG.get("max_final_context_top_k", default_final_context_top_k)),
        )
        requested_top_k = options.top_k
        effective_top_k = requested_top_k if requested_top_k is not None else default_final_context_top_k
        effective_top_k = min(max_final_context_top_k, max(1, int(effective_top_k)))
        enable_query_rewrite = self._resolve_option(
            options.enable_query_rewrite, RETRIEVAL_CONFIG["enable_query_rewrite"]
        )
        enable_hyde = self._resolve_option(options.enable_hyde, RETRIEVAL_CONFIG["enable_hyde"])
        enable_keyword_search = self._resolve_option(
            options.enable_keyword_search, RETRIEVAL_CONFIG["enable_keyword_search"]
        )
        enable_llm_rerank = self._resolve_option(
            options.enable_llm_rerank, RETRIEVAL_CONFIG.get("enable_llm_rerank", False)
        )
        debug_enabled = self._resolve_option(options.debug, RETRIEVAL_CONFIG["debug"])
        recall_candidate_limit = ENHANCED_RETRIEVAL_CONFIG["recall_candidate_limit"]
        rrf_candidate_limit = ENHANCED_RETRIEVAL_CONFIG["rrf_candidate_limit"]
        rerank_candidate_limit = ENHANCED_RETRIEVAL_CONFIG["rerank_candidate_limit"]
        final_context_top_k = default_final_context_top_k

        normalized_collection_name = self._resolve_collection_name(collection_name)
        query_bundle = self.query_planner.build_query_bundle(
            user_query=user_query,
            collection_name=normalized_collection_name,
            paper_context=paper_context,
            enable_query_rewrite=enable_query_rewrite,
        )
        intent_profile = query_bundle["intent_profile"]
        query_profile = query_bundle["query_profile"]
        query_views = query_bundle["query_views"]
        planned_rerank_query = query_bundle["rerank_query"]

        route_bundle = self.route_retriever.build_route_bundle(
            collection_name=normalized_collection_name,
            user_query=user_query,
            query_profile=query_profile,
            query_views=query_views,
            options=options,
            enable_hyde=enable_hyde,
            enable_keyword_search=enable_keyword_search,
            recall_candidate_limit=recall_candidate_limit,
        )
        routes = route_bundle["routes"]
        hyde_text = route_bundle["hyde_text"]
        hyde_debug = route_bundle["hyde_debug"]
        keyword_debug = route_bundle["keyword_debug"]
        memory_debug = route_bundle["memory_debug"]
        memory_retrieval_enabled = bool(memory_debug.get("enabled", False))

        fused_limit = rrf_candidate_limit if enable_llm_rerank else effective_top_k
        deduped_routes = {
            route_name: self._dedupe_route_results(route_results)
            for route_name, route_results in routes.items()
        }
        raw_retrieval_top30 = self._build_raw_retrieval_top_n(deduped_routes, limit=recall_candidate_limit)
        self._log_retrieval_stage("raw_retrieval_top30", raw_retrieval_top30)

        fused_results = self._fuse_routes(deduped_routes, fused_limit, query_profile)
        fused_top30 = fused_results[:rrf_candidate_limit]
        self._log_retrieval_stage("fused_top30", fused_top30)

        reranked_results = fused_results
        final_results = fused_results
        rerank_query = planned_rerank_query
        rerank_debug: Dict[str, Any] = {
            "enabled": enable_llm_rerank,
            "applied": False,
            "mode": "passthrough",
            "reason": "disabled",
            "input_chunks": len(fused_results),
            "output_chunks": len(fused_results),
        }
        if enable_llm_rerank:
            # 打印log信息，说明进入了rerank模块
            logger.debug("*" * 50)
            logger.debug("Entering LLM rerank module with %d candidate chunks", len(fused_results))
            logger.debug("LLM rerank configuration: model=%s, provider=%s, batch_size=%d, candidate_limit=%d, fallback_local=%s",
                self.llm_rerank_model_name_or_path,
                self.llm_rerank_provider,
                self.llm_rerank_batch_size,
                rerank_candidate_limit,
                self.llm_rerank_fallback_local,
            )
            logger.debug("*" * 50)
            rerank_result = self.rerank_service.llm_rerank(
                rerank_query,
                fused_results,
                effective_top_k,
                query_profile=query_profile,
                original_question=user_query,
                candidate_limit=rerank_candidate_limit,
            )
            reranked_results = rerank_result.get("reranked_chunks", rerank_result["chunks"])
            final_results = self._mark_final_context_chunks(rerank_result["chunks"])
            self._log_retrieval_stage("reranked_top30", reranked_results[:rrf_candidate_limit])
            self._log_retrieval_stage("final_context_top15", final_results[:effective_top_k])
            rerank_debug = rerank_result["debug"]
        else:
            self._log_retrieval_stage("reranked_top30", reranked_results[:rrf_candidate_limit])
            final_results = self._mark_final_context_chunks(fused_results[:effective_top_k])
            self._log_retrieval_stage("final_context_top15", final_results[:effective_top_k])

        asset_type_counts = {
            "raw_retrieval_top30": self._count_chunk_types(raw_retrieval_top30),
            "fused_top30": self._count_chunk_types(fused_top30),
            "reranked_top30": self._count_chunk_types(reranked_results[:rrf_candidate_limit]),
            "final_context_top15": self._count_chunk_types(final_results[:effective_top_k]),
        }

        result: Dict[str, Any] = {"chunks": final_results}

        if debug_enabled:
            result["debug"] = {
                "original_query": user_query,
                "original_question": user_query,
                "intent_profile": self._debug_intent_profile(intent_profile),
                "query_profile": self._debug_query_profile(query_profile),
                "query_plan": query_profile.query_plan,
                "query_views": query_views,
                "rewritten_queries": query_views["selected_queries"],
                "rerank_query": rerank_query,
                "hyde_text": hyde_text,
                "query_rewrite": query_views["rewrite_debug"],
                "hyde": hyde_debug,
                "keyword_search": keyword_debug,
                "memory": {
                    **memory_debug,
                    "final_context_hits": [
                        self._debug_chunk_item(item)
                        for item in final_results[:effective_top_k]
                        if "memory_context" in (item.get("matched_routes", []) or [item.get("retrieval_route")])
                    ],
                },
                "routes": {
                    route_name: [self._debug_chunk_item(item) for item in route_results]
                    for route_name, route_results in deduped_routes.items()
                },
                "stages": {
                    "raw_retrieval_top30": [self._debug_chunk_item(item) for item in raw_retrieval_top30],
                    "fused_top30": [self._debug_chunk_item(item) for item in fused_top30],
                    "reranked_top30": [self._debug_chunk_item(item) for item in reranked_results[:rrf_candidate_limit]],
                    "final_context_top15": [self._debug_chunk_item(item) for item in final_results[:effective_top_k]],
                },
                "final_chunks": [self._debug_chunk_item(item) for item in final_results],
                "config": {
                    "requested_top_k": requested_top_k,
                    "effective_top_k": effective_top_k,
                    "top_k": effective_top_k,
                    "candidate_k": recall_candidate_limit,
                    "rrf_candidate_limit": rrf_candidate_limit,
                    "rerank_candidate_limit": rerank_candidate_limit,
                    "final_context_top_k": effective_top_k,
                    "default_final_context_top_k": default_final_context_top_k,
                    "max_final_context_top_k": max_final_context_top_k,
                    "enable_query_rewrite": enable_query_rewrite,
                    "enable_hyde": enable_hyde,
                    "enable_keyword_search": enable_keyword_search,
                    "enable_llm_rerank": enable_llm_rerank,
                    "enable_memory_aware_retrieval": memory_retrieval_enabled,
                    "memory_source_boost_weight": float(self._memory_flag("memory_source_boost_weight", ENHANCED_RETRIEVAL_CONFIG.get("memory_source_boost_weight", 0.12))),
                },
                "asset_type_counts": asset_type_counts,
                "fusion": {
                    "algorithm": "pure_rrf",
                    "rrf_k": self.rrf_k,
                    "route_weights": self._route_weights_for_intent(intent_profile),
                    "dedupe_per_route": True,
                    "route_confidence": {
                        route_name: self._route_confidence(
                            route_name,
                            query_profile,
                            (route_results[0]["source_query"] if route_results else user_query),
                            route_queries=self._collect_route_queries(route_results, query_views["selected_queries"], user_query),
                            intent_profile=intent_profile,
                        )
                        for route_name, route_results in routes.items()
                    },
                },
                "llm_rerank": rerank_debug,
                "intent": self._debug_intent_profile(intent_profile),
            }

        trace_export = self._export_retrieval_trace(
            original_question=user_query,
            user_query=user_query,
            collection_name=normalized_collection_name,
            paper_context=paper_context or {},
            options={
                "top_k": effective_top_k,
                "requested_top_k": requested_top_k,
                "default_final_context_top_k": default_final_context_top_k,
                "max_final_context_top_k": max_final_context_top_k,
                "candidate_k": recall_candidate_limit,
                "enable_query_rewrite": enable_query_rewrite,
                "enable_hyde": enable_hyde,
                "enable_keyword_search": enable_keyword_search,
                "enable_llm_rerank": enable_llm_rerank,
                "debug": debug_enabled,
            },
            query_profile=query_profile,
            intent_profile=intent_profile,
            rerank_query=rerank_query,
            query_views=query_views,
            hyde_debug=hyde_debug,
            routes=deduped_routes,
            raw_retrieval_top30=raw_retrieval_top30,
            fused_results=fused_results,
            reranked_results=reranked_results,
            final_results=final_results,
            rerank_debug=rerank_debug,
            final_context_top_k=effective_top_k,
        )
        if trace_export:
            result["trace_export"] = trace_export
        if trace_export and debug_enabled and "debug" in result:
            result["debug"]["trace_export"] = trace_export

        return result

    def _llm_rerank_impl(
        self,
        rerank_query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        query_profile: Optional[QueryProfile] = None,
        original_question: Optional[str] = None,
        candidate_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        rerank_limit = min(len(chunks), max(1, int(candidate_limit or self.llm_rerank_candidate_limit)))
        rerank_limit = max(0, rerank_limit)
        limited_chunks = chunks[: max(1, top_k)]
        original_question = original_question or rerank_query

        if not chunks:
            return {
                "chunks": [],
                "reranked_chunks": [],
                "debug": {
                    "enabled": True,
                    "applied": False,
                    "mode": "passthrough",
                    "reason": "no_candidates",
                    "model_path": None,
                    "input_chunks": 0,
                    "output_chunks": 0,
                    "candidate_limit": rerank_limit,
                    "query": rerank_query,
                    "original_question": original_question,
                    "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                },
            }

        candidate_chunks = [dict(chunk) for chunk in chunks[:rerank_limit]]
        rerank_documents = [
            self.rerank_service.build_rerank_document_text(chunk)
            for chunk in candidate_chunks
        ]
        self._log_rerank_inputs(original_question, rerank_query, candidate_chunks, rerank_documents)

        if self.llm_rerank_provider == "dashscope":
            remote_result = self._rerank_with_dashscope(
                user_query=rerank_query,
                chunks=chunks,
                candidate_chunks=candidate_chunks,
                rerank_documents=rerank_documents,
                rerank_limit=rerank_limit,
                top_k=top_k,
                query_profile=query_profile,
            )
            if remote_result is not None:
                return remote_result
            if not self.llm_rerank_fallback_local:
                return {
                    "chunks": limited_chunks,
                    "reranked_chunks": limited_chunks,
                    "debug": {
                        "enabled": True,
                        "applied": False,
                        "mode": "remote_failed",
                        "reason": self._llm_reranker_error or "remote_rerank_unavailable",
                        "provider": "dashscope",
                        "model_path": None,
                        "device": None,
                        "input_chunks": len(chunks),
                        "output_chunks": len(limited_chunks),
                        "candidate_limit": rerank_limit,
                        "batch_size": self.llm_rerank_batch_size,
                        "query": rerank_query,
                        "original_question": original_question,
                        "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                    },
                }

        reranker = self._load_llm_reranker()
        if reranker is None:
            return {
                "chunks": limited_chunks,
                "reranked_chunks": limited_chunks,
                "debug": {
                    "enabled": True,
                    "applied": False,
                    "mode": "passthrough",
                    "reason": self._llm_reranker_error or "llm_reranker_unavailable",
                    "model_path": self._llm_reranker_path,
                    "device": self._llm_reranker_device,
                    "input_chunks": len(chunks),
                        "output_chunks": len(limited_chunks),
                        "candidate_limit": rerank_limit,
                    "batch_size": self.llm_rerank_batch_size,
                    "query": rerank_query,
                    "original_question": original_question,
                    "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                },
            }

        pairs = [(rerank_query, document_text) for document_text in rerank_documents]

        try:
            raw_scores = reranker.predict(
                pairs,
                prompt=self.llm_rerank_prompt,
                batch_size=self.llm_rerank_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self._llm_reranker_device,
            )
            rerank_scores = torch.sigmoid(torch.as_tensor(raw_scores, dtype=torch.float32)).tolist()
            self._log_rerank_raw_scores(
                provider="local",
                raw_results=[{"returned_index": idx, "relevance_score": float(score)} for idx, score in enumerate(rerank_scores, start=1)],
            )
        except Exception as exc:  # pragma: no cover - model/runtime failures are environment dependent
            logger.exception("Failed to rerank chunks with Qwen3-VL-Reranker")
            return {
                "chunks": limited_chunks,
                "reranked_chunks": limited_chunks,
                "debug": {
                    "enabled": True,
                    "applied": False,
                    "mode": "passthrough",
                    "reason": f"inference_error: {exc}",
                    "model_path": self._llm_reranker_path,
                    "device": self._llm_reranker_device,
                    "input_chunks": len(chunks),
                        "output_chunks": len(limited_chunks),
                        "candidate_limit": rerank_limit,
                    "batch_size": self.llm_rerank_batch_size,
                    "query": rerank_query,
                    "original_question": original_question,
                    "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                },
            }

        mapped_candidates: List[Dict[str, Any]] = []
        candidate_debug: List[Dict[str, Any]] = []
        for idx, (chunk, score, document_text) in enumerate(zip(candidate_chunks, rerank_scores, rerank_documents), start=1):
            rerank_score = float(score)
            fused_score = float(chunk.get("score", 0.0) or 0.0)
            reranked_chunk = dict(chunk)
            reranked_chunk["fusion_score"] = fused_score
            reranked_chunk["fusion_rank"] = int(chunk.get("fusion_rank", chunk.get("route_rank", idx)) or idx)
            reranked_chunk["llm_rerank_score"] = rerank_score
            reranked_chunk["llm_rerank_model"] = self._llm_reranker_path
            reranked_chunk["llm_rerank_prompt"] = self.llm_rerank_prompt
            reranked_chunk["llm_rerank_input_rank"] = idx
            reranked_chunk["llm_rerank_returned_index"] = idx
            reranked_chunk["llm_rerank_document_preview"] = document_text[: ENHANCED_RETRIEVAL_CONFIG["rerank_document_preview_limit"]]
            reranked_chunk["llm_rerank_used_compressed_text"] = bool(reranked_chunk.get("rerank_text"))
            reranked_chunk["score"] = rerank_score
            mapped_candidates.append(
                {
                    "rerank_rank": None,
                    "returned_index": idx,
                    "chunk_id": reranked_chunk.get("chunk_id"),
                    "page_number": reranked_chunk.get("page_number"),
                    "fusion_rank": reranked_chunk.get("fusion_rank"),
                    "fusion_score": fused_score,
                    "rerank_score": rerank_score,
                    "text_preview": document_text[: ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]],
                    "chunk": reranked_chunk,
                }
            )
            candidate_debug.append(
                {
                    "input_rank": idx,
                    "chunk_id": reranked_chunk.get("chunk_id"),
                    "page_number": reranked_chunk.get("page_number"),
                    "fusion_score": fused_score,
                    "rerank_score": rerank_score,
                    "rerank_text_preview": self._short_text_preview(
                        reranked_chunk.get("rerank_text", ""),
                        ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"],
                    ),
                    "rerank_document_preview": reranked_chunk.get("llm_rerank_document_preview", "")[
                        : ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]
                    ],
                    "section_tags": reranked_chunk.get("section_tags", []),
                    "source_query": reranked_chunk.get("source_query", ""),
                }
            )

        mapped_candidates.sort(key=lambda item: float(item.get("rerank_score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(mapped_candidates, start=1):
            item["rerank_rank"] = rank
            item["chunk"]["llm_rerank_rank"] = rank
        self._log_rerank_mapped_results(provider="local", mapped_results=mapped_candidates)
        ranked_candidates = [item["chunk"] for item in mapped_candidates]

        tail_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[rerank_limit:], start=rerank_limit + 1):
            tail_chunk = dict(chunk)
            tail_chunk["fusion_score"] = float(tail_chunk.get("score", 0.0) or 0.0)
            tail_chunk["fusion_rank"] = int(tail_chunk.get("fusion_rank", tail_chunk.get("route_rank", idx)) or idx)
            tail_chunk["llm_rerank_score"] = None
            tail_chunk["llm_rerank_model"] = self._llm_reranker_path
            tail_chunk["llm_rerank_prompt"] = self.llm_rerank_prompt
            tail_chunk["llm_rerank_input_rank"] = idx
            tail_chunk["llm_rerank_rank"] = None
            tail_chunks.append(tail_chunk)

        reranked_chunks = ranked_candidates + tail_chunks
        final_chunks = reranked_chunks[: max(1, top_k)]

        return {
            "chunks": final_chunks,
            "reranked_chunks": reranked_chunks,
            "debug": {
                "enabled": True,
                "applied": True,
                "mode": "cross_encoder",
                "reason": "ok",
                "model_path": self._llm_reranker_path,
                "device": self._llm_reranker_device,
                "batch_size": self.llm_rerank_batch_size,
                "candidate_limit": rerank_limit,
                "input_chunks": len(chunks),
                "output_chunks": len(final_chunks),
                "query": rerank_query,
                "original_question": original_question,
                "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                "candidate_scores": candidate_debug[: min(ENHANCED_RETRIEVAL_CONFIG["candidate_debug_limit"], len(candidate_debug))],
            },
        }

    def _rerank_with_dashscope(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]],
        candidate_chunks: List[Dict[str, Any]],
        rerank_documents: List[str],
        rerank_limit: int,
        top_k: int,
        query_profile: Optional[QueryProfile] = None,
    ) -> Optional[Dict[str, Any]]:
        api_key = self.llm_rerank_api_key
        if not api_key:
            self._llm_reranker_error = "dashscope_api_key_not_set"
            return None

        documents = [doc[: self.llm_rerank_max_doc_chars] for doc in rerank_documents]
        payload: Dict[str, Any] = {
            "model": self.llm_rerank_model_name or "qwen3-rerank",
            "query": user_query,
            "documents": documents,
            "top_n": min(len(documents), max(1, rerank_limit)),
        }
        if self.llm_rerank_prompt:
            payload["instruct"] = self.llm_rerank_prompt

        try:
            response = requests.post(
                self.llm_rerank_base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self._llm_reranker_error = f"dashscope_rerank_request_failed: {exc}"
            logger.warning("DashScope rerank failed, will %s", "fallback to local model" if self.llm_rerank_fallback_local else "stop")
            return None

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            self._llm_reranker_error = f"dashscope_rerank_invalid_response: {data}"
            return None
        self._log_rerank_raw_scores(provider="dashscope", raw_results=results)

        score_by_index: Dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score"))
            try:
                if index is None or score is None:
                    continue
                score_by_index[int(index)] = float(score)
            except Exception:
                continue

        if not score_by_index:
            self._llm_reranker_error = f"dashscope_rerank_empty_scores: {data}"
            return None

        self._llm_reranker_path = self.llm_rerank_model_name
        self._llm_reranker_device = "remote"
        self._llm_reranker_error = None

        ranked_candidates: List[Dict[str, Any]] = []
        candidate_debug: List[Dict[str, Any]] = []
        for idx, (chunk, document_text) in enumerate(zip(candidate_chunks, rerank_documents), start=1):
            returned_index = idx if idx in score_by_index else (idx - 1 if (idx - 1) in score_by_index else None)
            rerank_score = float(score_by_index.get(idx - 1, score_by_index.get(idx, 0.0)))
            fused_score = float(chunk.get("score", 0.0) or 0.0)
            reranked_chunk = dict(chunk)
            reranked_chunk["fusion_score"] = fused_score
            reranked_chunk["fusion_rank"] = int(chunk.get("fusion_rank", chunk.get("route_rank", idx)) or idx)
            reranked_chunk["llm_rerank_score"] = rerank_score
            reranked_chunk["llm_rerank_model"] = self.llm_rerank_model_name
            reranked_chunk["llm_rerank_prompt"] = self.llm_rerank_prompt
            reranked_chunk["llm_rerank_input_rank"] = idx
            reranked_chunk["llm_rerank_returned_index"] = returned_index
            reranked_chunk["llm_rerank_document_preview"] = document_text[: ENHANCED_RETRIEVAL_CONFIG["rerank_document_preview_limit"]]
            reranked_chunk["llm_rerank_used_compressed_text"] = bool(reranked_chunk.get("rerank_text"))
            reranked_chunk["score"] = rerank_score
            ranked_candidates.append(reranked_chunk)
            candidate_debug.append(
                {
                    "input_rank": idx,
                    "chunk_id": reranked_chunk.get("chunk_id"),
                    "page_number": reranked_chunk.get("page_number"),
                    "fusion_score": fused_score,
                    "rerank_score": rerank_score,
                    "rerank_text_preview": self._short_text_preview(
                        reranked_chunk.get("rerank_text", ""),
                        ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"],
                    ),
                    "rerank_document_preview": reranked_chunk.get("llm_rerank_document_preview", "")[
                        : ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]
                    ],
                    "section_tags": reranked_chunk.get("section_tags", []),
                    "source_query": reranked_chunk.get("source_query", ""),
                }
            )

        ranked_candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(ranked_candidates, start=1):
            item["llm_rerank_rank"] = rank
        mapped_candidates = []
        for rank, item in enumerate(ranked_candidates, start=1):
            mapped_candidates.append(
                {
                    "rerank_rank": rank,
                    "returned_index": item.get("llm_rerank_returned_index", item.get("llm_rerank_input_rank", rank)),
                    "chunk_id": item.get("chunk_id"),
                    "page_number": item.get("page_number"),
                    "fusion_rank": item.get("fusion_rank"),
                    "fusion_score": item.get("fusion_score"),
                    "rerank_score": item.get("llm_rerank_score"),
                    "text_preview": item.get("llm_rerank_document_preview", "")[
                        : ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]
                    ],
                }
            )
        self._log_rerank_mapped_results(provider="dashscope", mapped_results=mapped_candidates)

        tail_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[rerank_limit:], start=rerank_limit + 1):
            tail_chunk = dict(chunk)
            tail_chunk["fusion_score"] = float(tail_chunk.get("score", 0.0) or 0.0)
            tail_chunk["fusion_rank"] = int(tail_chunk.get("fusion_rank", tail_chunk.get("route_rank", idx)) or idx)
            tail_chunk["llm_rerank_score"] = None
            tail_chunk["llm_rerank_model"] = self.llm_rerank_model_name
            tail_chunk["llm_rerank_prompt"] = self.llm_rerank_prompt
            tail_chunk["llm_rerank_input_rank"] = idx
            tail_chunk["llm_rerank_rank"] = None
            tail_chunks.append(tail_chunk)

        reranked_chunks = ranked_candidates + tail_chunks
        final_chunks = reranked_chunks[: max(1, top_k)]

        return {
            "chunks": final_chunks,
            "reranked_chunks": reranked_chunks,
            "debug": {
                "enabled": True,
                "applied": True,
                "mode": "dashscope",
                "reason": "ok",
                "provider": "dashscope",
                "model_path": self.llm_rerank_model_name,
                "device": "remote",
                "batch_size": self.llm_rerank_batch_size,
                "candidate_limit": rerank_limit,
                "input_chunks": len(chunks),
                "output_chunks": len(final_chunks),
                "query": user_query,
                "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                "candidate_scores": candidate_debug[: min(ENHANCED_RETRIEVAL_CONFIG["candidate_debug_limit"], len(candidate_debug))],
            },
        }

    def _load_llm_reranker(self) -> Optional[Any]:
        if self._llm_reranker is not None:
            return self._llm_reranker

        with self._llm_reranker_lock:
            if self._llm_reranker is not None:
                return self._llm_reranker

            if CrossEncoder is None:
                self._llm_reranker_error = "sentence_transformers_not_available"
                logger.warning("sentence_transformers is not available, llm rerank is disabled")
                return None

            resolved_path = self._resolve_reranker_model_path()
            if not resolved_path:
                self._llm_reranker_error = "llm_reranker_model_path_not_found"
                logger.warning("Could not resolve reranker model path")
                return None

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_kwargs = {
                    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                }
                self._llm_reranker = CrossEncoder(
                    resolved_path,
                    device=device,
                    trust_remote_code=True,
                    local_files_only=Path(resolved_path).exists(),
                    default_prompt_name="query",
                    model_kwargs=model_kwargs,
                )
                self._llm_reranker_path = resolved_path
                self._llm_reranker_device = device
                self._llm_reranker_error = None
                logger.info("Loaded reranker model from %s on %s", resolved_path, device)
                return self._llm_reranker
            except Exception as exc:  # pragma: no cover - loading depends on local environment
                self._llm_reranker_error = str(exc)
                self._llm_reranker_path = resolved_path
                self._llm_reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.exception("Failed to load reranker model from %s", resolved_path)
                return None

    def _resolve_reranker_model_path(self) -> Optional[str]:
        raw_path = str(self.llm_rerank_local_model_name_or_path or "").strip()
        if not raw_path:
            return None

        repo_root = Path(__file__).resolve().parents[2]
        candidates = []

        raw_candidate = Path(raw_path).expanduser()
        candidates.append(raw_candidate)
        if not raw_candidate.is_absolute():
            candidates.append((repo_root / raw_candidate).expanduser())

        hf_candidate = get_huggingface_model_path(raw_path)
        if hf_candidate:
            candidates.append(Path(hf_candidate).expanduser())

        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(candidate.resolve())
            except Exception:
                continue

        if raw_candidate.is_absolute():
            return str(raw_candidate)
        return hf_candidate

    def _legacy_intent_bucket(self, intent: str) -> str:
        intent = str(intent or "other").strip().lower() or "other"
        aliases = {
            "contribution": "summary",
            "paper_overview": "summary",
            "method_flow": "method",
            "implementation_detail": "method",
            "definition": "method",
            "experiment_setup": "experiment",
            "result_analysis": "experiment",
            "comparison": "comparison",
            "dataset": "dataset",
            "limitation": "limitation",
            "figure_table": "figure_table",
            "other": "other",
            "summary": "summary",
            "method": "method",
            "experiment": "experiment",
            "results_analysis": "experiment",
        }
        return aliases.get(intent, intent)

    def _limit_rerank_text(self, text: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip()

    def _resolve_option(self, runtime_value: Optional[bool], default_value: bool) -> bool:
        return default_value if runtime_value is None else bool(runtime_value)

    def _resolve_collection_name(self, collection_name: str) -> str:
        resolver = getattr(self.vector_store_service, "resolve_collection_name", None)
        if callable(resolver):
            return resolver(collection_name)
        return collection_name

    def _build_paper_context(
        self,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        sample_limit: int = ENHANCED_RETRIEVAL_CONFIG["sample_limit"],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "title": "",
            "abstract": "",
            "section_titles": [],
            "candidate_terms": [],
            "source_samples": [],
        }

        if paper_context:
            merged["title"] = str(paper_context.get("title", "") or "").strip()
            merged["abstract"] = str(paper_context.get("abstract", "") or "").strip()
            merged["section_titles"] = [
                str(item).strip()
                for item in (paper_context.get("section_titles", []) or [])
                if str(item).strip()
            ]
            merged["candidate_terms"] = [
                str(item).strip()
                for item in (paper_context.get("candidate_terms", []) or [])
                if str(item).strip()
            ]

        try:
            sample_chunks = self.vector_store_service.get_all_chunks(collection_name, limit=sample_limit)
        except Exception:
            sample_chunks = []

        section_titles = list(merged["section_titles"])
        source_samples: List[str] = []
        abstract_candidates: List[str] = []
        for chunk in sample_chunks:
            title = str(chunk.get("title", "") or "").strip()
            content = str(chunk.get("content", "") or "").strip()
            section_title = str(
                chunk.get("section_title")
                or chunk.get("content_part_label")
                or chunk.get("subchunk_label")
                or ""
            ).strip()
            if not merged["title"] and title:
                merged["title"] = title
            if section_title:
                normalized_section = self._normalize_query_text(section_title)
                if normalized_section and normalized_section not in {self._normalize_query_text(item) for item in section_titles}:
                    section_titles.append(section_title)
            if content and len(source_samples) < ENHANCED_RETRIEVAL_CONFIG["source_sample_limit"]:
                source_samples.append(content[: ENHANCED_RETRIEVAL_CONFIG["source_sample_primary_limit"]])
            section_tags = chunk.get("section_tags", []) or []
            if any(tag == "abstract" for tag in section_tags) and content:
                abstract_candidates.append(content)
            elif any(tag in {"introduction", "conclusion", "method", "experiment"} for tag in section_tags) and content:
                source_samples.append(content[: ENHANCED_RETRIEVAL_CONFIG["source_sample_secondary_limit"]])

        if not merged["abstract"] and abstract_candidates:
            merged["abstract"] = max(abstract_candidates, key=len)[:1800]

        merged["section_titles"] = self._dedupe_list(section_titles)
        merged["candidate_terms"] = self._merge_candidate_terms(
            merged["candidate_terms"],
            [
                merged["title"],
                merged["abstract"][:900],
                " ".join(merged["section_titles"][:20]),
                " ".join(source_samples[: ENHANCED_RETRIEVAL_CONFIG["source_sample_limit"]]),
            ],
        )
        merged["source_samples"] = source_samples[: ENHANCED_RETRIEVAL_CONFIG["source_sample_limit"]]
        return merged

    def _merge_candidate_terms(self, existing_terms: List[str], texts: List[str], limit: int = ENHANCED_RETRIEVAL_CONFIG["merge_candidate_terms_limit"]) -> List[str]:
        terms: List[str] = []
        for item in existing_terms:
            if str(item).strip():
                terms.append(str(item).strip())
        for text in texts:
            if not text:
                continue
            for token in self._extract_paper_terms_from_text(text):
                if token not in terms:
                    terms.append(token)
                if len(terms) >= limit:
                    return terms[:limit]
        return terms[:limit]

    def _extract_paper_terms_from_text(self, text: str, limit: int = ENHANCED_RETRIEVAL_CONFIG["extract_paper_terms_limit"]) -> List[str]:
        tokens = self._tokenize_for_keyword_search(text)
        filtered = [token for token in tokens if token not in EN_STOPWORDS and token not in ZH_STOPWORDS]
        seen: List[str] = []
        for token in filtered:
            if token not in seen:
                seen.append(token)
        return seen[:limit]

    def _build_query_plan(
        self,
        user_query: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> Dict[str, Any]:
        query_plan: Dict[str, Any] = {}
        llm_error: Optional[str] = None
        intent_payload = intent_profile.to_dict() if intent_profile is not None else None

        if self.generation_service is not None:
            try:
                if hasattr(self.generation_service, "plan_queries_for_retrieval"):
                    query_plan = self.generation_service.plan_queries_for_retrieval(
                        question=user_query,
                        max_queries=QUERY_PLAN_LIMIT,
                        paper_context=paper_context,
                        intent_profile=intent_payload,
                    )
                else:
                    rewrites = self.generation_service.rewrite_query_for_retrieval(
                        question=user_query,
                        max_queries=QUERY_PLAN_LIMIT,
                        paper_context=paper_context,
                    )
                    query_plan = {
                        "question_type": "other",
                        "intent_summary": "",
                        "paper_terms": paper_context.get("candidate_terms", [])[: ENHANCED_RETRIEVAL_CONFIG["paper_terms_preview_limit"]],
                        "preferred_sections": [],
                        "rewrite_queries": [
                            {
                                "query": query,
                                "focus": "retrieval",
                                "channels": ["vector", "keyword"],
                            }
                            for query in rewrites
                        ],
                        "main_intent": intent_profile.main_intent if intent_profile else "other",
                        "sub_intents": intent_profile.sub_intents if intent_profile else [],
                        "intent_summary": intent_profile.intent_summary if intent_profile else "",
                        "preferred_sections": intent_profile.preferred_sections if intent_profile else [],
                    }
            except Exception as exc:  # pragma: no cover - remote model failures are environment dependent
                llm_error = str(exc)

        if not query_plan:
            query_plan = self._heuristic_query_plan(user_query, paper_context, intent_profile=intent_profile)
        else:
            if not isinstance(query_plan, dict):
                query_plan = {}
            query_plan = self._normalize_query_plan(query_plan, user_query, paper_context, intent_profile=intent_profile)

        if llm_error and not query_plan.get("llm_error"):
            query_plan["llm_error"] = llm_error
        return query_plan

    def _normalize_query_plan(
        self,
        query_plan: Dict[str, Any],
        user_query: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> Dict[str, Any]:
        normalized = dict(query_plan or {})
        rewrite_queries = self._extract_plan_queries(normalized)
        if not rewrite_queries:
            normalized = self._heuristic_query_plan(user_query, paper_context, intent_profile=intent_profile)
            rewrite_queries = self._extract_plan_queries(normalized)
        normalized["rewrite_queries"] = rewrite_queries
        normalized["question_type"] = str(normalized.get("question_type", "other")).strip() or "other"
        normalized["intent_summary"] = str(normalized.get("intent_summary", "")).strip()
        normalized["paper_terms"] = [
            str(item).strip()
            for item in (normalized.get("paper_terms", []) or [])
            if str(item).strip()
        ]
        normalized["paper_terms"] = self._dedupe_list(normalized["paper_terms"])
        normalized["preferred_sections"] = [
            str(item).strip()
            for item in (normalized.get("preferred_sections", []) or [])
            if str(item).strip()
        ]
        normalized["preferred_sections"] = self._dedupe_list(normalized["preferred_sections"])
        normalized["paper_title"] = str(paper_context.get("title", "") or "").strip()
        normalized["paper_abstract"] = str(paper_context.get("abstract", "") or "").strip()
        normalized["section_titles"] = [
            str(item).strip()
            for item in (paper_context.get("section_titles", []) or [])
            if str(item).strip()
        ]
        if intent_profile is not None:
            normalized["question_type"] = intent_profile.main_intent
            normalized["main_intent"] = intent_profile.main_intent
            normalized["sub_intents"] = list(intent_profile.sub_intents)
            normalized["intent_confidence"] = intent_profile.confidence
            normalized["intent_fallback_reason"] = intent_profile.fallback_reason
            normalized["preferred_sections"] = self._dedupe_list(
                [*intent_profile.preferred_sections, *normalized.get("preferred_sections", [])]
            )
            normalized["route_weights"] = dict(intent_profile.route_weights)
            normalized["rewrite_count"] = intent_profile.rewrite_count
            normalized["use_keyword_search"] = intent_profile.use_keyword_search
            normalized["use_hyde"] = intent_profile.use_hyde
        return normalized

    def _heuristic_query_plan(
        self,
        user_query: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> Dict[str, Any]:
        normalized_query = self._normalize_query_text(user_query)
        tokens = self._tokenize_for_keyword_search(user_query)
        if intent_profile is None:
            intent_profile = self._build_intent_profile(user_query, paper_context=paper_context)
        intent_tags = list(intent_profile.sub_intents)
        question_type = self._legacy_intent_bucket(intent_profile.main_intent or self._classify_question_type(normalized_query, intent_tags))
        paper_terms = paper_context.get("candidate_terms", []) or []
        paper_terms = [str(item).strip() for item in paper_terms if str(item).strip()]
        section_titles = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
        preferred_sections = self._preferred_sections_for_question_type(question_type, intent_tags)
        term_focus = self._compact_terms(paper_terms, limit=5)
        section_focus = self._compact_terms(section_titles, limit=4)
        type_terms = self._query_type_terms(question_type)

        def join_parts(parts: List[str]) -> str:
            return self._dedupe_terms([part for part in parts if part]).strip() or user_query.strip()

        templates_by_intent = {
            "summary": [
                ([*term_focus[:3], "summary", "overview", "contribution"], "overview"),
                ([*term_focus[:3], "abstract", "introduction", "conclusion"], "paper arc"),
            ],
            "method": [
                ([*term_focus[:3], "method", "framework", "architecture"], "method overview"),
                ([*term_focus[:3], "training", "inference", "implementation"], "technical details"),
                ([*section_focus[:2], "approach", "model", "pipeline"], "paper structure"),
            ],
            "experiment": [
                ([*term_focus[:3], "experiment", "dataset", "baseline"], "setup"),
                ([*term_focus[:3], "evaluation", "metric", "implementation"], "evaluation details"),
                ([*section_focus[:2], "ablation", "results", "benchmark"], "experiment sections"),
            ],
            "comparison": [
                ([*term_focus[:3], "results", "performance", "comparison"], "results"),
                ([*term_focus[:3], "baseline", "ablation", "effect"], "comparison evidence"),
                ([*section_focus[:2], "table", "figure", "result"], "tables and figures"),
            ],
            "dataset": [
                ([*term_focus[:3], "dataset", "corpus", "benchmark"], "data source"),
                ([*term_focus[:3], "data split", "training data", "evaluation"], "data splits"),
                ([*section_focus[:2], "dataset", "setup", "experiment"], "dataset section"),
            ],
            "limitation": [
                ([*term_focus[:3], "limitation", "future work", "constraint"], "limitations"),
                ([*term_focus[:3], "failure case", "assumption", "weakness"], "failure cases"),
                ([*section_focus[:2], "discussion", "appendix", "future work"], "discussion"),
            ],
            "figure_table": [
                ([*term_focus[:3], "figure", "table", "diagram"], "visuals"),
                ([*term_focus[:3], "figure", "table", "result"], "figure or table caption"),
                ([*section_focus[:2], "appendix", "results", "experiment"], "visual evidence"),
            ],
            "other": [
                ([*term_focus[:4], *type_terms[:2]], "semantic"),
                ([*term_focus[:3], *section_focus[:2], *type_terms[2:4]], "section-aware"),
                ([user_query, *term_focus[:2], *type_terms[:3]], "query expansion"),
            ],
        }
        template_items = templates_by_intent.get(question_type, templates_by_intent["other"])
        queries = [
            {
                "query": join_parts(parts),
                "focus": focus,
                "channels": ["vector", "keyword"],
            }
            for parts, focus in template_items
        ]

        rewrite_limit = intent_profile.rewrite_count if intent_profile else QUERY_VIEW_LIMIT
        return {
            "question_type": question_type,
            "intent_summary": intent_profile.intent_summary if intent_profile else self._summarize_intent(question_type, intent_tags, term_focus),
            "paper_terms": term_focus,
            "preferred_sections": preferred_sections,
            "rewrite_queries": queries[:rewrite_limit],
            "paper_title": str(paper_context.get("title", "") or "").strip(),
            "paper_abstract": str(paper_context.get("abstract", "") or "").strip(),
            "section_titles": section_titles,
            "main_intent": question_type,
            "sub_intents": intent_tags,
            "intent_confidence": intent_profile.confidence if intent_profile else None,
            "intent_fallback_reason": intent_profile.fallback_reason if intent_profile else "",
            "route_weights": intent_profile.route_weights if intent_profile else self.route_weights,
            "rewrite_count": intent_profile.rewrite_count if intent_profile else len(queries),
        }

    def _extract_plan_queries(self, query_plan: Dict[str, Any]) -> List[str]:
        rewrites: List[str] = []
        raw_queries = query_plan.get("rewrite_queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = query_plan.get("queries", [])
        if not isinstance(raw_queries, list):
            return rewrites

        for item in raw_queries:
            if isinstance(item, dict):
                query = str(item.get("query", "")).strip()
                if query:
                    rewrites.append(query)
            elif isinstance(item, str) and item.strip():
                rewrites.append(item.strip())
        return self._dedupe_list(rewrites)[:QUERY_VIEW_LIMIT]

    def _build_query_views_from_plan(
        self,
        user_query: str,
        query_plan: Dict[str, Any],
        query_profile: Optional[QueryProfile],
        keywords: List[str],
        intent_tags: List[str],
        language: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> Tuple[str, str, str]:
        plan_queries = self._extract_plan_queries(query_plan)
        paper_terms = [str(item).strip() for item in (query_plan.get("paper_terms", []) or []) if str(item).strip()]
        section_titles = [str(item).strip() for item in (query_plan.get("section_titles", []) or []) if str(item).strip()]
        question_type = str((intent_profile.main_intent if intent_profile else query_plan.get("question_type", "other")) or "other").strip() or "other"

        if not paper_terms:
            paper_terms = self._extract_paper_terms_from_text(
                " ".join(
                    [
                        str(paper_context.get("title", "") or ""),
                        str(paper_context.get("abstract", "") or ""),
                        " ".join(section_titles),
                    ]
                ),
                limit=6,
            )

        if plan_queries:
            semantic_query = plan_queries[0]
            evidence_query = plan_queries[1] if len(plan_queries) > 1 else plan_queries[0]
            keyword_query = plan_queries[2] if len(plan_queries) > 2 else " ".join(
                self._dedupe_list([*paper_terms[:4], *keywords[:4], question_type])
            ).strip()
        else:
            semantic_query = self._build_semantic_query(user_query, keywords + paper_terms, intent_tags, intent_profile=intent_profile)
            evidence_query = self._build_evidence_query(keywords + paper_terms, intent_tags, language, intent_profile=intent_profile)
            keyword_query = self._build_keyword_query(keywords + paper_terms, intent_tags, intent_profile=intent_profile)

        if not semantic_query:
            semantic_query = self._build_semantic_query(user_query, keywords + paper_terms, intent_tags, intent_profile=intent_profile)
        if not evidence_query:
            evidence_query = self._build_evidence_query(keywords + paper_terms, intent_tags, language, intent_profile=intent_profile)
        if not keyword_query:
            keyword_query = self._build_keyword_query(keywords + paper_terms, intent_tags, intent_profile=intent_profile)

        return semantic_query, evidence_query, keyword_query

    def _preferred_sections_for_question_type(self, question_type: str, intent_tags: List[str]) -> List[str]:
        question_type_alias = {
            "contribution": "contribution",
            "paper_overview": "paper_overview",
            "method": "method_flow",
            "method_flow": "method_flow",
            "experiment": "experiment_setup",
            "experiment_setup": "experiment_setup",
            "result_analysis": "results_analysis",
            "results_analysis": "results_analysis",
            "comparison": "results_analysis",
            "dataset": "dataset",
            "definition": "summary",
            "implementation_detail": "method_flow",
            "figure_table": "figure_table",
            "summary": "summary",
            "limitation": "limitation",
        }
        rule_key = question_type_alias.get(question_type, question_type)
        preferred = list(QUESTION_TYPE_RULES.get(rule_key, QUESTION_TYPE_RULES["other"]).get("preferred_sections", []))
        preferred.extend(self._preferred_section_tags(intent_tags))
        return self._dedupe_list(preferred)[:6]

    def _classify_question_type(self, normalized_query: str, intent_tags: List[str]) -> str:
        query_text = normalized_query.lower()
        for question_type, spec in QUESTION_TYPE_RULES.items():
            if question_type == "other":
                continue
            if any(keyword.lower() in query_text for keyword in spec.get("keywords", [])):
                return question_type
        if "summary" in intent_tags:
            return "summary"
        return "other"

    def _query_type_terms(self, question_type: str) -> List[str]:
        question_type_alias = {
            "contribution": "contribution",
            "paper_overview": "summary",
            "method": "method_flow",
            "method_flow": "method_flow",
            "experiment": "experiment_setup",
            "experiment_setup": "experiment_setup",
            "result_analysis": "results_analysis",
            "results_analysis": "results_analysis",
            "comparison": "results_analysis",
            "dataset": "dataset",
            "definition": "summary",
            "implementation_detail": "method_flow",
            "figure_table": "figure_table",
            "summary": "summary",
            "limitation": "limitation",
        }
        spec = QUESTION_TYPE_RULES.get(question_type_alias.get(question_type, question_type), QUESTION_TYPE_RULES["other"])
        return self._dedupe_list([str(item).strip() for item in spec.get("keywords", []) if str(item).strip()])

    def _preferred_section_tags_from_plan(self, query_plan: Dict[str, Any], intent_tags: List[str]) -> List[str]:
        preferred = [
            str(item).strip()
            for item in (query_plan.get("preferred_sections", []) or [])
            if str(item).strip()
        ]
        question_type = self._legacy_intent_bucket(query_plan.get("question_type", "other"))
        preferred.extend(QUESTION_TYPE_RULES.get(question_type, QUESTION_TYPE_RULES["other"]).get("preferred_sections", []))
        preferred.extend(self._preferred_section_tags(intent_tags))
        return self._dedupe_list(preferred)[:6]

    def _compact_terms(self, terms: List[str], limit: int = ENHANCED_RETRIEVAL_CONFIG["compact_terms_limit"]) -> List[str]:
        compacted: List[str] = []
        for term in terms:
            normalized = str(term).strip()
            if not normalized:
                continue
            if normalized not in compacted:
                compacted.append(normalized)
            if len(compacted) >= limit:
                break
        return compacted

    def _summarize_intent(self, question_type: str, intent_tags: List[str], paper_terms: List[str]) -> str:
        question_type_alias = {
            "contribution": "contribution",
            "paper_overview": "summary",
            "method": "method_flow",
            "method_flow": "method_flow",
            "experiment": "experiment_setup",
            "experiment_setup": "experiment_setup",
            "result_analysis": "results_analysis",
            "results_analysis": "results_analysis",
            "comparison": "results_analysis",
            "dataset": "dataset",
            "definition": "summary",
            "implementation_detail": "method_flow",
            "figure_table": "figure_table",
            "summary": "summary",
            "limitation": "limitation",
        }
        question_type = question_type_alias.get(question_type, question_type)
        if question_type == "paper_overview":
            return "understand the paper overview and key ideas"
        if question_type == "contribution":
            return "understand the paper's main contribution and novelty"
        if question_type == "method_flow":
            return "understand the method flow and paper-specific implementation details"
        if question_type == "experiment_setup":
            return "understand the experimental setup, datasets, baselines, and evaluation details"
        if question_type == "results_analysis":
            return "understand the results, comparison, and ablation analysis"
        if question_type == "limitation":
            return "understand the limitations and future work"
        if question_type == "dataset":
            return "understand the dataset or benchmark used in the paper"
        if question_type == "metric":
            return "understand the metric, formula, or evaluation protocol"
        if question_type == "figure_table":
            return "find the relevant figure or table and interpret it"
        if question_type == "summary":
            return "summarize the paper around its main ideas and findings"
        if question_type == "definition":
            return "understand the definition or concept being asked about"
        if question_type == "implementation_detail":
            return "understand the implementation details and training settings"
        if intent_tags:
            return f"understand the paper with focus on {', '.join(intent_tags[:3])}"
        if paper_terms:
            return f"retrieve evidence around {', '.join(paper_terms[:3])}"
        return "retrieve the most relevant paper evidence"

    def _dedupe_list(self, items: List[str]) -> List[str]:
        seen_normalized: List[str] = []
        unique: List[str] = []
        for item in items:
            raw = str(item).strip()
            normalized = self._normalize_query_text(raw)
            if normalized and normalized not in seen_normalized:
                seen_normalized.append(normalized)
                unique.append(raw)
        return unique

    def _build_intent_profile(
        self,
        user_query: str,
        paper_context: Optional[Dict[str, Any]] = None,
    ) -> IntentProfile:
        return self.intent_service.build_intent_profile(user_query, paper_context=paper_context or {})

    def _build_query_profile(
        self,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        intent_profile: Optional[IntentProfile] = None,
    ) -> QueryProfile:
        normalized_query = self._normalize_query_text(user_query)
        tokens = self._tokenize_for_keyword_search(user_query)
        keywords = self._extract_query_keywords(tokens)
        language = self._detect_language(user_query, tokens)
        paper_context_payload = self._build_paper_context(collection_name, paper_context=paper_context)
        intent_profile = intent_profile or self._build_intent_profile(user_query, paper_context=paper_context)
        query_plan = self._build_query_plan(user_query, paper_context_payload, intent_profile=intent_profile)
        question_type = str(intent_profile.main_intent or query_plan.get("question_type", "other")).strip() or "other"
        intent_tags = list(intent_profile.sub_intents)
        intent_summary = str(intent_profile.intent_summary or query_plan.get("intent_summary", "")).strip()
        paper_terms = [str(item).strip() for item in query_plan.get("paper_terms", []) if str(item).strip()]
        section_preferences = self._preferred_section_tags_from_plan(query_plan, intent_tags)
        section_preferences = self._dedupe_list([*intent_profile.preferred_sections, *section_preferences])
        ambiguity_score = intent_profile.ambiguity_score
        semantic_query, evidence_query, keyword_query = self._build_query_views_from_plan(
            user_query=user_query,
            query_plan=query_plan,
            query_profile=None,
            keywords=keywords,
            intent_tags=intent_tags,
            language=language,
            paper_context=paper_context_payload,
            intent_profile=intent_profile,
        )
        return QueryProfile(
            original_query=user_query,
            normalized_query=normalized_query,
            language=language,
            intent_profile=intent_profile,
            tokens=tokens,
            keywords=keywords,
            intent_tags=intent_tags,
            question_type=question_type,
            intent_summary=intent_summary,
            paper_terms=paper_terms,
            ambiguity_score=ambiguity_score,
            semantic_query=semantic_query,
            evidence_query=evidence_query,
            keyword_query=keyword_query,
            section_preferences=section_preferences,
            query_plan=query_plan,
        )

    def _build_query_views(
        self,
        user_query: str,
        query_profile: QueryProfile,
        enable_query_rewrite: bool,
    ) -> Dict[str, Any]:
        query_plan = query_profile.query_plan or {}
        plan_queries = self._extract_plan_queries(query_plan)
        fallback_rewrites = self._heuristic_query_rewrites(query_profile)
        llm_rewrites: List[str] = []
        llm_error: Optional[str] = None

        if plan_queries:
            llm_rewrites = plan_queries[:QUERY_VIEW_LIMIT]
        elif enable_query_rewrite and self.generation_service is not None:
            try:
                llm_rewrites = self.generation_service.rewrite_query_for_retrieval(
                    user_query,
                    max_queries=QUERY_VIEW_LIMIT,
                    paper_context={
                        "title": query_plan.get("paper_title", ""),
                        "abstract": query_plan.get("paper_abstract", ""),
                        "section_titles": query_plan.get("section_titles", []),
                        "candidate_terms": query_profile.paper_terms,
                    },
                    intent_profile=query_profile.intent_profile.to_dict(),
                )
            except Exception as exc:  # pragma: no cover - remote model failures are environment dependent
                llm_error = str(exc)

        candidate_sources = [
            ("core", query_profile.original_query),
            ("core", query_profile.semantic_query),
            ("core", query_profile.evidence_query),
            ("core", query_profile.keyword_query),
        ]
        if enable_query_rewrite:
            candidate_sources.extend(("plan", query) for query in llm_rewrites)
            candidate_sources.extend(("heuristic", query) for query in fallback_rewrites)

        candidate_rows: List[Dict[str, Any]] = []
        seen = set()
        selected_queries: List[str] = []

        for idx, (source, query) in enumerate(candidate_sources):
            stripped = str(query).strip()
            normalized = self._normalize_query_text(stripped)
            row = {
                "query": stripped,
                "source": source,
                "source_index": idx,
                "normalized": normalized,
                "selected": False,
                "reason": "kept",
            }
            if not normalized:
                row["reason"] = "empty"
                candidate_rows.append(row)
                continue
            if normalized == query_profile.normalized_query:
                row["reason"] = "same_as_original"
                candidate_rows.append(row)
                continue
            if normalized in seen:
                row["reason"] = "duplicate"
                candidate_rows.append(row)
                continue
            seen.add(normalized)
            candidate_rows.append(row)
            if len(selected_queries) < QUERY_VIEW_LIMIT:
                row["selected"] = True
                selected_queries.append(stripped)
            else:
                row["reason"] = "trimmed_to_top_k"

        if not selected_queries:
            selected_queries.append(query_profile.semantic_query)

        rewrite_debug = {
            "enabled": enable_query_rewrite,
            "original_query": user_query,
            "intent_profile": self._debug_intent_profile(query_profile.intent_profile),
            "query_plan": query_plan,
            "model_queries": llm_rewrites,
            "heuristic_queries": fallback_rewrites,
            "selected_queries": selected_queries,
            "selected_keywords": self._build_query_keywords(selected_queries),
            "selected_query_details": self._build_query_term_details(selected_queries),
            "candidates": candidate_rows,
            "llm_error": llm_error,
            "view_queries": {
                "original": query_profile.original_query,
                "semantic": query_profile.semantic_query,
                "evidence": query_profile.evidence_query,
                "keywords": query_profile.keyword_query,
            },
        }

        return {
            "enabled": enable_query_rewrite,
            "original_query": user_query,
            "query_plan": query_plan,
            "model_queries": llm_rewrites,
            "heuristic_queries": fallback_rewrites,
            "selected_queries": selected_queries,
            "selected_keywords": self._build_query_keywords(selected_queries),
            "selected_query_details": self._build_query_term_details(selected_queries),
            "candidates": candidate_rows,
            "llm_error": llm_error,
            "view_queries": {
                "original": query_profile.original_query,
                "semantic": query_profile.semantic_query,
                "evidence": query_profile.evidence_query,
                "keywords": query_profile.keyword_query,
            },
            "rewrite_debug": rewrite_debug,
        }

    def _normalize_route_results(
        self,
        results: List[Dict[str, Any]],
        route_name: str,
        source_query: str,
        route_confidence: float,
        query_profile: QueryProfile,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        raw_scores = [float(item.get("score", 0.0) or 0.0) for item in results]
        normalized_scores = self._normalize_scores(raw_scores)
        normalized = []
        for rank, (item, normalized_score) in enumerate(zip(results, normalized_scores), start=1):
            chunk = self._normalize_chunk(item)
            chunk["retrieval_route"] = route_name
            chunk["source_query"] = source_query
            chunk["route_rank"] = rank
            chunk["route_score"] = float(item.get("score", 0.0) or 0.0)
            chunk["normalized_route_score"] = float(normalized_score)
            chunk["route_confidence"] = float(route_confidence)
            chunk["structural_bonus"] = float(self._compute_structural_bonus(chunk, query_profile))
            normalized.append(chunk)
        return normalized

    def _fuse_routes(
        self,
        routes: Dict[str, List[Dict[str, Any]]],
        top_k: int,
        query_profile: QueryProfile,
    ) -> List[Dict[str, Any]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        route_weights = self._route_weights_for_intent(query_profile.intent_profile)
        for route_name, route_results in routes.items():
            weight = route_weights.get(route_name, self.route_weights.get(route_name, 1.0))
            for rank, item in enumerate(route_results):
                chunk_key = self._chunk_unique_key(item)
                route_confidence = float(item.get("route_confidence", 1.0) or 1.0)
                entry = aggregated.setdefault(
                    chunk_key,
                    {
                        **item,
                        "score": 0.0,
                        "matched_routes": [],
                        "route_scores": {},
                        "route_confidences": {},
                        "source_queries": [],
                    },
                )
                vote = weight * route_confidence * (1.0 / (self.rrf_k + rank + 1))
                entry["score"] += vote
                entry["matched_routes"].append(route_name)
                entry["route_scores"][route_name] = item.get("route_score")
                entry["route_confidences"][route_name] = route_confidence
                if item.get("source_query") and item["source_query"] not in entry["source_queries"]:
                    entry["source_queries"].append(item["source_query"])

        fused = sorted(
            aggregated.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                max(item.get("route_scores", {}).values() or [0.0]),
                max(item.get("route_confidences", {}).values() or [0.0]),
            ),
            reverse=True,
        )

        return fused[:top_k]

    def _normalize_chunk(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = item.get("metadata", {})
        chunk = dict(item)
        chunk["content"] = (
            item.get("content")
            or item.get("text")
            or metadata.get("content")
            or metadata.get("text")
            or ""
        )
        chunk["text"] = chunk["content"]
        chunk["source"] = item.get("source") or metadata.get("source", "")
        chunk["document_name"] = item.get("document_name") or metadata.get("document_name", "")
        chunk["chunk_id"] = item.get("chunk_id") or metadata.get("chunk_id", 0)
        chunk["chunk_index"] = item.get("chunk_index") or metadata.get("chunk_index", 0)
        chunk["parent_chunk_id"] = item.get("parent_chunk_id") or metadata.get("parent_chunk_id", chunk["chunk_id"])
        chunk["original_chunk_id"] = item.get("original_chunk_id") or metadata.get("original_chunk_id", chunk["parent_chunk_id"])
        chunk["page_number"] = item.get("page_number") or metadata.get("page_number", "")
        chunk["page_start"] = item.get("page_start") or metadata.get("page_start")
        chunk["page_end"] = item.get("page_end") or metadata.get("page_end")
        chunk["page_range"] = item.get("page_range") or metadata.get("page_range", "")
        chunk["subchunk_label"] = item.get("subchunk_label") or metadata.get("subchunk_label", "")
        chunk["content_part_label"] = item.get("content_part_label") or metadata.get("content_part_label", "")
        chunk["title"] = item.get("title") or metadata.get("title", "")
        chunk["authors"] = item.get("authors") or metadata.get("authors", "")
        chunk["categories"] = item.get("categories") or metadata.get("categories", "")
        chunk["published_date"] = item.get("published_date") or metadata.get("published_date", "")
        chunk["url"] = item.get("url") or metadata.get("url", "")
        chunk["chunk_type"] = item.get("chunk_type") or metadata.get("chunk_type", "text")
        chunk["asset_kind"] = item.get("asset_kind") or metadata.get("asset_kind", "")
        chunk["asset_path"] = item.get("asset_path") or metadata.get("asset_path", "")
        chunk["asset_abs_path"] = item.get("asset_abs_path") or metadata.get("asset_abs_path", "")
        chunk["asset_summary"] = item.get("asset_summary") or metadata.get("asset_summary", "")
        chunk["asset_preview_text"] = item.get("asset_preview_text") or metadata.get("asset_preview_text", "")
        chunk["asset_caption"] = item.get("asset_caption") or metadata.get("asset_caption", "")
        chunk["asset_rows"] = item.get("asset_rows") or metadata.get("asset_rows", 0)
        chunk["asset_columns"] = item.get("asset_columns") or metadata.get("asset_columns", 0)
        chunk["order_index"] = item.get("order_index") or metadata.get("order_index", 0)
        chunk["section_tags"] = self._detect_section_tags(chunk["content"])
        return chunk

    def _generate_hyde_document(
        self,
        user_query: str,
        query_profile: QueryProfile,
        rewritten_queries: List[str],
    ) -> str:
        if self.generation_service is not None:
            try:
                text = self.generation_service.generate_hyde_document(user_query)
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass
        return self._heuristic_hyde_document(query_profile, rewritten_queries)

    def _heuristic_query_rewrites(self, query_profile: QueryProfile) -> List[str]:
        rewrites: List[str] = []
        if query_profile.semantic_query:
            rewrites.append(query_profile.semantic_query)
        if query_profile.evidence_query and query_profile.evidence_query not in rewrites:
            rewrites.append(query_profile.evidence_query)
        if query_profile.keyword_query and query_profile.keyword_query not in rewrites:
            rewrites.append(query_profile.keyword_query)

        question_type = query_profile.question_type or "other"
        paper_terms = query_profile.paper_terms[:4]

        type_templates = {
            "method_flow": [
                "method framework algorithm training inference",
                "architecture component pipeline implementation",
            ],
            "experiment_setup": [
                "experiment dataset baseline metric implementation",
                "evaluation setup data split benchmark",
            ],
            "results_analysis": [
                "results performance comparison ablation analysis",
                "result table figure effect improvement",
            ],
            "contribution": [
                "main contribution novel proposed method",
                "key idea summary contribution overview",
            ],
            "limitation": [
                "limitations future work failure cases",
                "discussion constraints assumptions weaknesses",
            ],
            "dataset": [
                "dataset corpus benchmark data split",
                "training data evaluation dataset",
            ],
            "metric": [
                "metric formula evaluation objective",
                "score measure evaluation protocol",
            ],
            "figure_table": [
                "figure table diagram caption",
                "table figure result appendix",
            ],
            "summary": [
                "abstract introduction conclusion summary",
                "main findings key contribution overview",
            ],
            "other": [
                "paper evidence section relevant passages",
                "retrieval relevant chunks academic paper",
            ],
        }

        for template in type_templates.get(question_type, type_templates["other"]):
            terms = [*paper_terms, template]
            rewrites.append(" ".join(self._dedupe_list(terms)))

        for intent in query_profile.intent_tags:
            if intent == "summary":
                rewrites.append(" ".join(self._dedupe_list([*paper_terms, "abstract", "introduction", "conclusion"])))
            elif intent == "method":
                rewrites.append(" ".join(self._dedupe_list([*paper_terms, "method", "framework", "architecture", "implementation"])))
            elif intent == "experiment":
                rewrites.append(" ".join(self._dedupe_list([*paper_terms, "experiment", "evaluation", "benchmark", "ablation"])))
            elif intent == "comparison":
                rewrites.append(" ".join(self._dedupe_list([*paper_terms, "baseline", "comparison", "ablation"])))
            elif intent == "limitation":
                rewrites.append(" ".join(self._dedupe_list([*paper_terms, "limitations", "future work", "failure cases"])))
            elif intent == "definition":
                rewrites.append(" ".join(self._dedupe_list([*paper_terms, "problem formulation", "task setup"])))
            elif intent == "dataset":
                rewrites.append(" ".join(self._dedupe_list([*paper_terms, "dataset", "corpus", "benchmark", "data"])))

        return self._dedupe_list(rewrites)[: QUERY_VIEW_LIMIT]

    def _heuristic_hyde_document(
        self,
        query_profile: QueryProfile,
        rewritten_queries: List[str],
    ) -> str:
        focus = ", ".join(rewritten_queries[:2]) if rewritten_queries else query_profile.keyword_query
        preferred_sections = ", ".join(query_profile.section_preferences[:3]) if query_profile.section_preferences else "relevant sections"
        return (
            "The paper likely contains a passage that answers the question with concrete evidence, terminology, "
            f"and section-level details. The most useful chunk is probably near {preferred_sections}. "
            f"Relevant terms may include: {focus}. The passage should help retrieve the specific evidence needed to answer: "
            f"{query_profile.original_query}"
        )

    def _build_semantic_query(
        self,
        user_query: str,
        keywords: List[str],
        intent_tags: List[str],
        intent_profile: Optional[IntentProfile] = None,
    ) -> str:
        parts: List[str] = []
        if intent_profile and intent_profile.rewrite_focus:
            parts.extend(intent_profile.rewrite_focus[:4])
        if intent_tags:
            parts.extend(self._intent_to_terms(intent_tags))
        parts.extend(keywords[:6])
        if not parts:
            parts.extend(self._tokenize_for_keyword_search(user_query)[:6])
        return self._dedupe_terms(parts) or self._normalize_query_text(user_query)

    def _build_evidence_query(self, keywords: List[str], intent_tags: List[str], language: str, intent_profile: Optional[IntentProfile] = None) -> str:
        parts = self._intent_to_evidence_terms(intent_tags)
        if intent_profile and intent_profile.rerank_focus:
            parts.extend(intent_profile.rerank_focus[:4])
        parts.extend(keywords[:4])
        if language == "zh":
            parts.extend(["论文", "证据", "段落"])
        else:
            parts.extend(["paper", "evidence", "passage"])
        return self._dedupe_terms(parts)

    def _build_keyword_query(self, keywords: List[str], intent_tags: List[str], intent_profile: Optional[IntentProfile] = None) -> str:
        parts = keywords[: ENHANCED_RETRIEVAL_CONFIG["keyword_parts_limit"]] + self._intent_to_terms(intent_tags)
        if intent_profile and intent_profile.rewrite_focus:
            parts.extend(intent_profile.rewrite_focus[:4])
        if not parts:
            parts = keywords[: ENHANCED_RETRIEVAL_CONFIG["keyword_parts_limit"]]
        return self._dedupe_terms(parts)

    def _detect_language(self, user_query: str, tokens: List[str]) -> str:
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_query))
        has_latin = any(re.search(r"[a-zA-Z]", token) for token in tokens)
        if has_cjk and has_latin:
            return "mixed"
        if has_cjk:
            return "zh"
        if has_latin:
            return "en"
        return "unknown"

    def _detect_intent_tags(self, normalized_query: str, tokens: List[str]) -> List[str]:
        query_text = " ".join(tokens + [normalized_query])
        detected: List[str] = []
        for intent, spec in INTENT_RULES.items():
            if any(keyword.lower() in query_text for keyword in spec["keywords"]):
                detected.append(intent)
        return detected

    def _estimate_ambiguity(self, keywords: List[str], intent_tags: List[str], language: str, user_query: str) -> float:
        content_weight = min(1.0, len(keywords) / ENHANCED_RETRIEVAL_CONFIG["extract_query_keywords_limit"])
        intent_weight = min(1.0, len(intent_tags) / 3.0)
        length_weight = min(1.0, len(user_query.strip()) / 50.0)
        language_weight = ENHANCED_RETRIEVAL_CONFIG["language_weight_zh_mixed"] if language in {"zh", "mixed"} else 0.0
        specificity = min(
            1.0,
            ENHANCED_RETRIEVAL_CONFIG["specificity_content_weight"] * content_weight
            + ENHANCED_RETRIEVAL_CONFIG["specificity_intent_weight"] * intent_weight
            + ENHANCED_RETRIEVAL_CONFIG["specificity_length_weight"] * length_weight
            + language_weight,
        )
        return max(ENHANCED_RETRIEVAL_CONFIG["specificity_floor"], min(1.0, 1.0 - specificity))

    def _preferred_section_tags(self, intent_tags: List[str]) -> List[str]:
        preferred: List[str] = []
        for intent in intent_tags:
            preferred.extend(INTENT_RULES.get(intent, {}).get("preferred_sections", []))
        return self._dedupe_terms(preferred).split()

    def _intent_to_terms(self, intent_tags: List[str]) -> List[str]:
        terms: List[str] = []
        for intent in intent_tags:
            if intent == "summary":
                terms.extend(["main", "contributions", "key", "findings", "summary"])
            elif intent == "method":
                terms.extend(["proposed", "method", "approach", "architecture", "implementation"])
            elif intent == "experiment":
                terms.extend(["experimental", "results", "evaluation", "benchmarks"])
            elif intent == "comparison":
                terms.extend(["baseline", "comparison", "ablation", "competing"])
            elif intent == "limitation":
                terms.extend(["limitations", "future", "work", "failure", "cases"])
            elif intent == "definition":
                terms.extend(["definition", "formulation", "problem", "setup"])
            elif intent == "dataset":
                terms.extend(["datasets", "corpus", "data", "splits"])
        return self._dedupe_terms(terms).split()

    def _intent_to_evidence_terms(self, intent_tags: List[str]) -> List[str]:
        terms: List[str] = []
        for intent in intent_tags:
            if intent == "summary":
                terms.extend(["abstract", "introduction", "conclusion"])
            elif intent == "method":
                terms.extend(["method", "approach", "architecture", "model"])
            elif intent == "experiment":
                terms.extend(["experiment", "results", "evaluation", "ablation"])
            elif intent == "comparison":
                terms.extend(["baseline", "comparison", "ablation", "results"])
            elif intent == "limitation":
                terms.extend(["limitations", "discussion", "future work"])
            elif intent == "definition":
                terms.extend(["background", "definition", "problem setup"])
            elif intent == "dataset":
                terms.extend(["dataset", "corpus", "data"])
        return self._dedupe_terms(terms).split()

    def _extract_query_keywords(self, tokens: List[str], limit: int = ENHANCED_RETRIEVAL_CONFIG["extract_query_keywords_limit"]) -> List[str]:
        filtered = [token for token in tokens if token not in EN_STOPWORDS and token not in ZH_STOPWORDS]
        if not filtered:
            filtered = [token for token in tokens if len(token) > 1]
        seen = []
        for token in filtered:
            if token not in seen:
                seen.append(token)
        return seen[:limit]

    def _tokenize_for_keyword_search(self, text: str) -> List[str]:
        lowered = text.lower()
        english_tokens = re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", lowered)
        chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
        return english_tokens + chinese_tokens

    def _build_query_keywords(self, queries: List[str], limit: int = ENHANCED_RETRIEVAL_CONFIG["build_query_keywords_limit"]) -> List[str]:
        token_counter: Counter = Counter()
        for query in queries:
            token_counter.update(self._tokenize_for_keyword_search(query))
        return [token for token, _ in token_counter.most_common(limit)]

    def _build_query_term_details(self, queries: List[str]) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for query in queries:
            tokens = self._tokenize_for_keyword_search(query)
            unique_tokens = list(dict.fromkeys(tokens))
            details.append(
                {
                    "query": query,
                    "keywords": unique_tokens,
                    "keyword_count": len(unique_tokens),
                }
            )
        return details

    def _normalize_query_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _dedupe_terms(self, terms: List[str]) -> str:
        seen = []
        for term in terms:
            normalized = self._normalize_query_text(str(term))
            if normalized and normalized not in seen:
                seen.append(normalized)
        return " ".join(seen)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if math.isclose(min_score, max_score):
            return [1.0 for _ in scores]
        scale = max_score - min_score
        return [(score - min_score) / scale for score in scores]

    def _route_weights_for_intent(self, intent_profile: Optional[IntentProfile]) -> Dict[str, float]:
        if intent_profile is None:
            weights = dict(self.route_weights)
        else:
            weights = dict(intent_profile.route_weights or self.route_weights)
        weights.setdefault("memory_context", 0.42)
        return weights

    def _route_confidence(
        self,
        route_name: str,
        query_profile: QueryProfile,
        source_query: str,
        route_queries: Optional[List[str]] = None,
        intent_profile: Optional[IntentProfile] = None,
    ) -> float:
        route_queries = route_queries or [source_query]
        source_text = " ".join(route_queries) if route_queries else source_query
        similarity = self._query_similarity(query_profile.normalized_query, source_text)
        ambiguity = query_profile.ambiguity_score
        main_intent = self._legacy_intent_bucket(intent_profile.main_intent if intent_profile else query_profile.question_type)
        if route_name == "vector_original":
            base = ENHANCED_RETRIEVAL_CONFIG["query_weight_base_summary_other"] if main_intent in {"summary", "other"} else ENHANCED_RETRIEVAL_CONFIG["query_weight_base_default"]
        elif route_name == "vector_rewrite":
            base = ENHANCED_RETRIEVAL_CONFIG["query_weight_base_ambiguous_keyword"] + 0.18 * ambiguity
            if main_intent in {"method", "experiment", "comparison", "dataset"}:
                base += ENHANCED_RETRIEVAL_CONFIG["route_focus_bonus"]
        elif route_name == "vector_hyde":
            base = ENHANCED_RETRIEVAL_CONFIG["query_weight_base_ambiguous_other"] + 0.25 * ambiguity
            if main_intent == "summary":
                base += ENHANCED_RETRIEVAL_CONFIG["route_summary_bonus"]
        elif route_name == "keyword":
            base = ENHANCED_RETRIEVAL_CONFIG["query_weight_base_keyword"] + 0.18 * min(1.0, len(query_profile.keywords) / ENHANCED_RETRIEVAL_CONFIG["extract_query_keywords_limit"])
            if main_intent in {"method", "experiment", "figure_table"}:
                base += ENHANCED_RETRIEVAL_CONFIG["route_keyword_bonus"]
        elif route_name == "memory_context":
            base = 0.42 + 0.2 * ambiguity
            if main_intent in {"method", "experiment", "comparison", "figure_table", "dataset"}:
                base += 0.08
        else:
            base = ENHANCED_RETRIEVAL_CONFIG["query_weight_base_fallback"]
        return max(
            ENHANCED_RETRIEVAL_CONFIG["route_default_floor"],
            min(
                1.0,
                base
                * (
                    ENHANCED_RETRIEVAL_CONFIG["route_confidence_multiplier"]
                    + ENHANCED_RETRIEVAL_CONFIG["route_confidence_similarity_weight"] * similarity
                ),
            ),
        )

    def _query_similarity(self, left: str, right: str) -> float:
        left_tokens = set(self._tokenize_for_keyword_search(left))
        right_tokens = set(self._tokenize_for_keyword_search(right))
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return intersection / union if union else 0.0

    def _compute_structural_bonus(self, chunk: Dict[str, Any], query_profile: QueryProfile) -> float:
        section_tags = set(chunk.get("section_tags", []) or [])
        if not section_tags:
            section_tags = set(self._detect_section_tags(chunk.get("content", "")))
        preferred = set(query_profile.section_preferences)
        bonus = 0.0
        if preferred:
            bonus += ENHANCED_RETRIEVAL_CONFIG["section_bonus_weight"] * len(section_tags & preferred)
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        noisy_tags = set(section_tags & NOISY_SECTION_TAGS)
        main_intent = self._legacy_intent_bucket(query_profile.intent_profile.main_intent if query_profile.intent_profile else query_profile.question_type)
        if chunk_type in {"figure", "table"} and (main_intent == "figure_table" or "figure_table" in query_profile.intent_tags):
            noisy_tags -= {"figure", "table"}
            bonus += ENHANCED_RETRIEVAL_CONFIG["figure_table_bonus_weight"]
        if noisy_tags:
            bonus -= ENHANCED_RETRIEVAL_CONFIG["noisy_section_penalty_weight"] * len(noisy_tags)
        if "abstract" in section_tags and (main_intent == "summary" or "paper_overview" in query_profile.intent_tags or "contribution" in query_profile.intent_tags):
            bonus += ENHANCED_RETRIEVAL_CONFIG["preferred_section_bonus_weight"]
        if "conclusion" in section_tags and (main_intent == "summary" or "paper_overview" in query_profile.intent_tags or "contribution" in query_profile.intent_tags):
            bonus += ENHANCED_RETRIEVAL_CONFIG["section_path_bonus_weight"]
        return max(-0.05, min(0.12, bonus))

    def _detect_section_tags(self, content: str) -> List[str]:
        prefix = self._normalize_query_text(content[:900])
        tags: List[str] = []
        for tag, patterns in SECTION_TAG_RULES.items():
            if any(pattern in prefix for pattern in patterns):
                tags.append(tag)
        return tags

    def _bm25_score(
        self,
        token_counter: Counter,
        doc_tokens: List[str],
        doc_freqs: Dict[str, int],
        total_docs: int,
        avg_doc_length: float,
        content: str,
    ) -> float:
        doc_counts = Counter(doc_tokens)
        doc_len = max(len(doc_tokens), 1)
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
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_length, 1.0)))
            score += qtf * idf * (numerator / denominator)

            if token in {"method", "methods", "dataset", "datasets", "baseline", "ablation", "limitation", "limitations"}:
                if token in content_lower[:300]:
                    score += ENHANCED_RETRIEVAL_CONFIG["bm25_token_boost"]

        return score

    def _dedupe_preserve_order(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for item in items:
            key = self._chunk_unique_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _dedupe_route_results(self, route_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in route_results:
            key = self._chunk_unique_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(item))
        return deduped

    def _build_raw_retrieval_top_n(
        self,
        routes: Dict[str, List[Dict[str, Any]]],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        raw_results: List[Dict[str, Any]] = []
        for route_results in routes.values():
            for item in route_results:
                raw_results.append(dict(item))
                if len(raw_results) >= limit:
                    return raw_results[:limit]
        return raw_results[:limit]

    def _log_retrieval_stage(self, stage_name: str, chunks: List[Dict[str, Any]]) -> None:
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
                self._chunk_unique_key(chunk),
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
                    self._short_text_preview(chunk.get("content", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                    self._short_text_preview(chunk.get("asset_summary", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                    self._short_text_preview(chunk.get("rerank_text", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                )

    def _short_text_preview(self, text: Any, limit: int = ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit]

    def _log_rerank_inputs(
        self,
        original_question: str,
        rerank_query: str,
        candidate_chunks: List[Dict[str, Any]],
        rerank_documents: List[str],
    ) -> None:
        logger.debug("original_question=%s", original_question)
        logger.debug("actual_rerank_query=%s", rerank_query)
        logger.debug("document_count=%d", len(rerank_documents))
        for idx, (chunk, document_text) in enumerate(zip(candidate_chunks, rerank_documents), start=1):
            logger.debug(
                "rerank_input[%d] input_index=%s chunk_id=%s chunk_type=%s asset_kind=%s page=%s fusion_rank=%s fusion_score=%s raw_chunk_preview=%s asset_summary_preview=%s rerank_text_preview=%s clean_document_preview=%s",
                idx,
                idx,
                chunk.get("chunk_id"),
                chunk.get("chunk_type"),
                chunk.get("asset_kind"),
                chunk.get("page_number") or chunk.get("page_range"),
                chunk.get("fusion_rank"),
                chunk.get("fusion_score", chunk.get("score")),
                self._short_text_preview(chunk.get("content", "") or chunk.get("text", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                self._short_text_preview(chunk.get("asset_summary", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                self._short_text_preview(chunk.get("rerank_text", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
                self._short_text_preview(document_text, ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
            )

    def _log_rerank_raw_scores(self, provider: str, raw_results: List[Any]) -> None:
        logger.debug("rerank_raw_results provider=%s count=%d", provider, len(raw_results))
        for idx, item in enumerate(raw_results[:20], start=1):
            if isinstance(item, dict):
                returned_index = item.get("index", item.get("document_index", idx))
                relevance_score = item.get("relevance_score", item.get("score"))
                document_text = item.get("document", item.get("text", ""))
            else:
                returned_index = idx
                relevance_score = item
                document_text = ""
            logger.debug(
                "rerank_raw_result[%d] returned_index=%s relevance_score=%s document_preview=%s",
                idx,
                returned_index,
                relevance_score,
                self._short_text_preview(document_text, ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]) if document_text else "",
            )

    def _log_rerank_mapped_results(self, provider: str, mapped_results: List[Dict[str, Any]]) -> None:
        logger.debug("rerank_mapped_results provider=%s count=%d", provider, len(mapped_results))
        for item in mapped_results[:20]:
            logger.debug(
                "rerank_mapped_result rerank_rank=%s returned_index=%s chunk_id=%s page=%s fusion_rank=%s fusion_score=%s rerank_score=%s text_preview=%s",
                item.get("rerank_rank"),
                item.get("returned_index"),
                item.get("chunk_id"),
                item.get("page_number") or item.get("page_range"),
                item.get("fusion_rank"),
                item.get("fusion_score"),
                item.get("rerank_score"),
                self._short_text_preview(item.get("text_preview", ""), ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
            )

    def _collect_route_queries(
        self,
        route_results: List[Dict[str, Any]],
        fallback_queries: List[str],
        user_query: str,
    ) -> List[str]:
        queries = [item.get("source_query", "") for item in route_results if item.get("source_query")]
        if queries:
            return queries
        return fallback_queries or [user_query]

    def _chunk_unique_key(self, item: Dict[str, Any]) -> str:
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

    def _debug_query_profile(self, query_profile: Optional[QueryProfile]) -> Optional[Dict[str, Any]]:
        if query_profile is None:
            return None
        return {
            "original_query": query_profile.original_query,
            "normalized_query": query_profile.normalized_query,
            "language": query_profile.language,
            "intent_profile": self._debug_intent_profile(query_profile.intent_profile),
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

    def _debug_intent_profile(self, intent_profile: Optional[IntentProfile]) -> Optional[Dict[str, Any]]:
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

    def _debug_chunk_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        preview = item.get("content", "")[: ENHANCED_RETRIEVAL_CONFIG["rerank_document_preview_limit"]].replace("\n", " ").strip()
        return {
            "chunk_id": item.get("chunk_id"),
            "original_chunk_id": item.get("original_chunk_id"),
            "chunk_type": item.get("chunk_type", "text"),
            "asset_kind": item.get("asset_kind", ""),
            "asset_path": item.get("asset_path", ""),
            "asset_summary": item.get("asset_summary", ""),
            "asset_summary_preview": self._short_text_preview(item.get("asset_summary", ""), 160),
            "asset_preview_text": item.get("asset_preview_text", ""),
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
            "memory_score": item.get("memory_score"),
            "memory_reason": item.get("memory_reason"),
            "source_turn_id": item.get("source_turn_id"),
            "is_recent_turn": item.get("is_recent_turn"),
            "memory_match_type": item.get("memory_match_type"),
            "memory_reference_strength": item.get("memory_reference_strength"),
            "rerank_text": item.get("rerank_text", ""),
            "rerank_text_preview": self._short_text_preview(item.get("rerank_text", ""), 160),
            "final_context_uses_original_chunk": item.get("final_context_uses_original_chunk"),
            "subchunk_label": item.get("subchunk_label"),
            "section_tags": item.get("section_tags", []),
            "content": item.get("content", ""),
            "preview": preview,
        }

    def _count_chunk_types(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {"text": 0, "figure": 0, "table": 0}
        for chunk in chunks:
            chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
            counts[chunk_type] = counts.get(chunk_type, 0) + 1
        return counts

    def _mark_final_context_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        marked_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            marked = dict(chunk)
            marked["final_context_uses_original_chunk"] = True
            marked_chunks.append(marked)
        return marked_chunks

    def _export_retrieval_trace(
        self,
        original_question: str,
        user_query: str,
        collection_name: str,
        paper_context: Dict[str, Any],
        options: Dict[str, Any],
        query_profile: QueryProfile,
        intent_profile: IntentProfile,
        rerank_query: str,
        query_views: Dict[str, Any],
        hyde_debug: Dict[str, Any],
        routes: Dict[str, List[Dict[str, Any]]],
        raw_retrieval_top30: List[Dict[str, Any]],
        fused_results: List[Dict[str, Any]],
        reranked_results: List[Dict[str, Any]],
        final_results: List[Dict[str, Any]],
        rerank_debug: Dict[str, Any],
        final_context_top_k: int,
    ) -> Optional[Dict[str, str]]:
        if not self.trace_export_enabled:
            return None

        try:
            paper_id = self._sanitize_trace_slug(str(paper_context.get("arxiv_id", "") or collection_name or "query"))
            export_dir = self.trace_export_dir / paper_id
            export_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            short_id = uuid.uuid4().hex[:8]
            slug = self._sanitize_trace_slug(user_query)
            base_name = f"{stamp}_{collection_name}_{slug}_{short_id}"
            json_path = export_dir / f"{base_name}.json"
            md_path = export_dir / f"{base_name}.md"

            payload = {
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "arxiv_id": str(paper_context.get("arxiv_id", "") or ""),
                "collection_name": collection_name,
                "original_question": original_question,
                "user_query": user_query,
                "rerank_query": rerank_query,
                "intent_profile": self._normalize_trace_value(self._debug_intent_profile(intent_profile)),
                "paper_context": self._normalize_trace_value(paper_context),
                "options": self._normalize_trace_value(options),
                "query_profile": self._normalize_trace_value(self._debug_query_profile(query_profile)),
                "query_views": self._normalize_trace_value(query_views),
                "steps": [
                    {
                        "step": "query_profile",
                        "result": self._normalize_trace_value(self._debug_query_profile(query_profile)),
                    },
                    {
                        "step": "intent_profile",
                        "result": self._normalize_trace_value(self._debug_intent_profile(intent_profile)),
                    },
                    {
                        "step": "query_rewrite",
                        "result": self._normalize_trace_value(query_views.get("rewrite_debug", {})),
                    },
                    {
                        "step": "original_question",
                        "result": self._normalize_trace_value(original_question),
                    },
                    {
                        "step": "rerank_query",
                        "result": self._normalize_trace_value(rerank_query),
                    },
                    {
                        "step": "hyde",
                        "result": self._normalize_trace_value(hyde_debug),
                    },
                    {
                        "step": "routes",
                        "result": {
                            route_name: [self._normalize_trace_value(self._debug_chunk_item(item)) for item in route_results]
                            for route_name, route_results in routes.items()
                        },
                    },
                    {
                        "step": "raw_retrieval_top30",
                        "result": [self._normalize_trace_value(self._debug_chunk_item(item)) for item in raw_retrieval_top30],
                    },
                    {
                        "step": "fused_top30",
                        "result": [self._normalize_trace_value(self._debug_chunk_item(item)) for item in fused_results[:30]],
                    },
                    {
                        "step": "reranked_top30",
                        "result": [self._normalize_trace_value(self._debug_chunk_item(item)) for item in reranked_results[:30]],
                    },
                    {
                        "step": "final_context_top15",
                        "result": [self._normalize_trace_value(self._debug_chunk_item(item)) for item in final_results],
                    },
                ],
            }

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            with md_path.open("w", encoding="utf-8") as f:
                f.write(self._render_retrieval_trace_text(payload))

            return {"json": str(json_path), "md": str(md_path)}
        except Exception as exc:  # pragma: no cover - trace export should never break retrieval
            logger.warning("Failed to export retrieval trace: %s", exc)
            return None

    def _render_retrieval_trace_text(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("# Retrieval Trace")
        lines.append("")
        lines.append(f"- exported_at: {payload.get('exported_at', '')}")
        lines.append(f"- collection_name: {payload.get('collection_name', '')}")
        lines.append("")
        lines.append("## User Query")
        lines.append("")
        lines.append("```text")
        lines.append(self._normalize_trace_newlines(str(payload.get("user_query", ""))))
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
            lines.append(self._format_trace_block(step.get("result")))
            lines.append("```")

        return "\n".join(lines).rstrip() + "\n"

    def _format_trace_block(self, value: Any, indent: int = 0) -> str:
        pad = "  " * indent
        if isinstance(value, dict):
            if not value:
                return f"{pad}{{}}"
            lines: List[str] = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.append(self._format_trace_block(item, indent + 1))
                else:
                    lines.append(f"{pad}{key}: {self._normalize_trace_newlines(str(item))}")
            return "\n".join(lines)
        if isinstance(value, list):
            if not value:
                return f"{pad}[]"
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.append(self._format_trace_block(item, indent + 1))
                else:
                    lines.append(f"{pad}- {self._normalize_trace_newlines(str(item))}")
            return "\n".join(lines)
        return f"{pad}{self._normalize_trace_newlines(str(value))}"

    def _build_fusion_trace(
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

    def _normalize_trace_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._normalize_trace_newlines(value)
        if isinstance(value, dict):
            return {key: self._normalize_trace_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize_trace_value(item) for item in value]
        return value

    def _normalize_trace_newlines(self, text: str) -> str:
        if not text:
            return text
        return text.replace("\r\n", "\n").replace("\\r\\n", "\n").replace("\\n", "\n")

    def _sanitize_trace_slug(self, text: str, max_length: int = ENHANCED_RETRIEVAL_CONFIG["sanitize_trace_slug_max_length"]) -> str:
        slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        if not slug:
            slug = "query"
        return slug[:max_length]
