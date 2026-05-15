import math
import re
import logging
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import requests

from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from utils.config import RETRIEVAL_CONFIG
from utils.model_utils import get_huggingface_model_path

try:  # pragma: no cover - optional dependency import is environment dependent
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

if TYPE_CHECKING:
    from services.generation_service import GenerationService
else:
    try:
        from services.generation_service import GenerationService
    except Exception:  # pragma: no cover - optional dependency fallback
        GenerationService = Any  # type: ignore


QUERY_VIEW_LIMIT = 6

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


@dataclass
class RetrievalOptions:
    top_k: int = RETRIEVAL_CONFIG["default_top_k"]
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    debug: Optional[bool] = None


@dataclass
class QueryProfile:
    original_query: str
    normalized_query: str
    language: str
    tokens: List[str]
    keywords: List[str]
    intent_tags: List[str]
    ambiguity_score: float
    semantic_query: str
    evidence_query: str
    keyword_query: str
    section_preferences: List[str]


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
        self._llm_reranker = None
        self._llm_reranker_lock = threading.Lock()
        self._llm_reranker_path: Optional[str] = None
        self._llm_reranker_device: Optional[str] = None
        self._llm_reranker_error: Optional[str] = None

    def enhanced_retrieve(
        self,
        user_query: str,
        collection_name: str,
        options: Optional[RetrievalOptions] = None,
    ) -> Dict[str, Any]:
        options = options or RetrievalOptions()
        effective_top_k = max(1, options.top_k or RETRIEVAL_CONFIG["default_top_k"])
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
        candidate_k = max(effective_top_k, effective_top_k * self.candidate_multiplier)

        query_profile = self._build_query_profile(user_query)
        query_views = self._build_query_views(user_query, query_profile, enable_query_rewrite)

        hyde_text = ""
        hyde_debug: Dict[str, Any] = {
            "enabled": enable_hyde,
            "text": "",
            "source_queries": [],
            "focus_queries": [],
            "confidence": 0.0,
        }
        if enable_hyde:
            hyde_text = self._generate_hyde_document(user_query, query_profile, query_views["selected_queries"])
            hyde_debug = {
                "enabled": True,
                "text": hyde_text,
                "source_queries": [user_query, *query_views["selected_queries"]],
                "focus_queries": query_views["selected_queries"][:2] if query_views["selected_queries"] else [user_query],
                "confidence": self._route_confidence(
                    "vector_hyde",
                    query_profile,
                    hyde_text or user_query,
                    route_queries=query_views["selected_queries"] or [user_query],
                ),
            }

        normalized_collection_name = self._resolve_collection_name(collection_name)
        routes: Dict[str, List[Dict[str, Any]]] = {}
        routes["vector_original"] = self._vector_retrieve(
            collection_name=normalized_collection_name,
            query=user_query,
            top_k=candidate_k,
            route_name="vector_original",
            source_query=user_query,
            query_profile=query_profile,
            route_queries=[user_query],
        )

        if enable_query_rewrite and query_views["selected_queries"]:
            rewrite_hits: List[Dict[str, Any]] = []
            for query in query_views["selected_queries"]:
                rewrite_hits.extend(
                    self._vector_retrieve(
                        collection_name=normalized_collection_name,
                        query=query,
                        top_k=candidate_k,
                        route_name="vector_rewrite",
                        source_query=query,
                        query_profile=query_profile,
                        route_queries=query_views["selected_queries"],
                    )
                )
            routes["vector_rewrite"] = self._dedupe_preserve_order(rewrite_hits)
        else:
            routes["vector_rewrite"] = []

        if hyde_text:
            routes["vector_hyde"] = self._vector_retrieve(
                collection_name=normalized_collection_name,
                query=hyde_text,
                top_k=candidate_k,
                route_name="vector_hyde",
                source_query="hyde",
                query_profile=query_profile,
                route_queries=[hyde_text],
            )
        else:
            routes["vector_hyde"] = []

        if enable_keyword_search:
            keyword_queries = [user_query, *query_views["selected_queries"], query_profile.semantic_query, query_profile.evidence_query]
            routes["keyword"] = self._keyword_retrieve(
                collection_name=normalized_collection_name,
                queries=keyword_queries,
                top_k=candidate_k,
                query_profile=query_profile,
            )
        else:
            routes["keyword"] = []
            keyword_queries = []

        keyword_debug = {
            "enabled": enable_keyword_search,
            "queries": keyword_queries,
            "selected_rewrite_queries": query_views["selected_queries"],
            "query_details": self._build_query_term_details(keyword_queries),
            "keywords": self._build_query_keywords(keyword_queries),
        }

        fused_limit = candidate_k if enable_llm_rerank else effective_top_k
        fused_results = self._fuse_routes(routes, fused_limit, query_profile)
        rerank_debug: Dict[str, Any] = {
            "enabled": enable_llm_rerank,
            "applied": False,
            "mode": "passthrough",
            "reason": "disabled",
            "input_chunks": len(fused_results),
            "output_chunks": len(fused_results),
        }
        if enable_llm_rerank:
            rerank_result = self.llm_rerank(user_query, fused_results, effective_top_k, query_profile=query_profile)
            fused_results = rerank_result["chunks"]
            rerank_debug = rerank_result["debug"]

        result: Dict[str, Any] = {"chunks": fused_results}

        if debug_enabled:
            result["debug"] = {
                "original_query": user_query,
                "query_profile": self._debug_query_profile(query_profile),
                "query_views": query_views,
                "rewritten_queries": query_views["selected_queries"],
                "hyde_text": hyde_text,
                "query_rewrite": query_views["rewrite_debug"],
                "hyde": hyde_debug,
                "keyword_search": keyword_debug,
                "routes": {
                    route_name: [self._debug_chunk_item(item) for item in route_results]
                    for route_name, route_results in routes.items()
                },
                "final_chunks": [self._debug_chunk_item(item) for item in fused_results],
                "config": {
                    "top_k": effective_top_k,
                    "candidate_k": candidate_k,
                    "enable_query_rewrite": enable_query_rewrite,
                    "enable_hyde": enable_hyde,
                    "enable_keyword_search": enable_keyword_search,
                    "enable_llm_rerank": enable_llm_rerank,
                },
                "fusion": {
                    "algorithm": "weighted_rrf",
                    "rrf_k": self.rrf_k,
                    "route_weights": self.route_weights,
                    "route_confidence": {
                        route_name: self._route_confidence(
                            route_name,
                            query_profile,
                            (route_results[0]["source_query"] if route_results else user_query),
                            route_queries=self._collect_route_queries(route_results, query_views["selected_queries"], user_query),
                        )
                        for route_name, route_results in routes.items()
                    },
                },
                "llm_rerank": rerank_debug,
            }

        return result

    def llm_rerank(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        query_profile: Optional[QueryProfile] = None,
    ) -> Dict[str, Any]:
        rerank_limit = min(len(chunks), max(top_k, self.llm_rerank_candidate_limit))
        rerank_limit = max(0, rerank_limit)
        limited_chunks = chunks[: max(1, top_k)]

        if not chunks:
            return {
                "chunks": [],
                "debug": {
                    "enabled": True,
                    "applied": False,
                    "mode": "passthrough",
                    "reason": "no_candidates",
                    "model_path": None,
                    "input_chunks": 0,
                    "output_chunks": 0,
                    "candidate_limit": rerank_limit,
                    "query": user_query,
                    "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                },
            }

        candidate_chunks = [dict(chunk) for chunk in chunks[:rerank_limit]]
        rerank_documents = [
            self._build_rerank_document_text(chunk, query_profile)
            for chunk in candidate_chunks
        ]

        if self.llm_rerank_provider == "dashscope":
            remote_result = self._rerank_with_dashscope(
                user_query=user_query,
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
                        "query": user_query,
                        "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                    },
                }

        reranker = self._load_llm_reranker()
        if reranker is None:
            return {
                "chunks": limited_chunks,
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
                    "query": user_query,
                    "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                },
            }

        pairs = [(user_query, document_text) for document_text in rerank_documents]

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
        except Exception as exc:  # pragma: no cover - model/runtime failures are environment dependent
            logger.exception("Failed to rerank chunks with Qwen3-VL-Reranker")
            return {
                "chunks": limited_chunks,
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
                    "query": user_query,
                    "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                },
            }

        ranked_candidates: List[Dict[str, Any]] = []
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
            reranked_chunk["llm_rerank_document_preview"] = document_text[:220]
            reranked_chunk["score"] = rerank_score
            ranked_candidates.append(reranked_chunk)
            candidate_debug.append(
                {
                    "input_rank": idx,
                    "chunk_id": reranked_chunk.get("chunk_id"),
                    "page_number": reranked_chunk.get("page_number"),
                    "fusion_score": fused_score,
                    "rerank_score": rerank_score,
                    "section_tags": reranked_chunk.get("section_tags", []),
                    "source_query": reranked_chunk.get("source_query", ""),
                }
            )

        ranked_candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(ranked_candidates, start=1):
            item["llm_rerank_rank"] = rank

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
                "query": user_query,
                "query_profile": self._debug_query_profile(query_profile) if query_profile else None,
                "candidate_scores": candidate_debug[: min(10, len(candidate_debug))],
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
            rerank_score = float(score_by_index.get(idx - 1, score_by_index.get(idx, 0.0)))
            fused_score = float(chunk.get("score", 0.0) or 0.0)
            reranked_chunk = dict(chunk)
            reranked_chunk["fusion_score"] = fused_score
            reranked_chunk["fusion_rank"] = int(chunk.get("fusion_rank", chunk.get("route_rank", idx)) or idx)
            reranked_chunk["llm_rerank_score"] = rerank_score
            reranked_chunk["llm_rerank_model"] = self.llm_rerank_model_name
            reranked_chunk["llm_rerank_prompt"] = self.llm_rerank_prompt
            reranked_chunk["llm_rerank_input_rank"] = idx
            reranked_chunk["llm_rerank_document_preview"] = document_text[:220]
            reranked_chunk["score"] = rerank_score
            ranked_candidates.append(reranked_chunk)
            candidate_debug.append(
                {
                    "input_rank": idx,
                    "chunk_id": reranked_chunk.get("chunk_id"),
                    "page_number": reranked_chunk.get("page_number"),
                    "fusion_score": fused_score,
                    "rerank_score": rerank_score,
                    "section_tags": reranked_chunk.get("section_tags", []),
                    "source_query": reranked_chunk.get("source_query", ""),
                }
            )

        ranked_candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(ranked_candidates, start=1):
            item["llm_rerank_rank"] = rank

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
                "candidate_scores": candidate_debug[: min(10, len(candidate_debug))],
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

    def _build_rerank_document_text(self, chunk: Dict[str, Any], query_profile: Optional[QueryProfile]) -> str:
        metadata_bits: List[str] = []
        title = str(chunk.get("title", "") or "").strip()
        source = str(chunk.get("source", "") or "").strip()
        page_range = str(chunk.get("page_range", "") or "").strip()
        page_number = str(chunk.get("page_number", "") or "").strip()
        content_part_label = str(chunk.get("content_part_label", "") or "").strip()
        section_tags = chunk.get("section_tags", []) or []

        if title:
            metadata_bits.append(f"Title: {title}")
        if source:
            metadata_bits.append(f"Source: {source}")
        if page_range:
            metadata_bits.append(f"Pages: {page_range}")
        elif page_number:
            metadata_bits.append(f"Page: {page_number}")
        if content_part_label:
            metadata_bits.append(f"Chunk: {content_part_label}")
        if section_tags:
            metadata_bits.append(f"Section tags: {', '.join(str(tag) for tag in section_tags[:4])}")
        if query_profile and query_profile.intent_tags:
            metadata_bits.append(f"Query intent: {', '.join(query_profile.intent_tags[:3])}")

        content = self._limit_rerank_text(str(chunk.get("content", "") or ""), self.llm_rerank_max_doc_chars)
        if not content:
            content = str(chunk.get("text", "") or "")
        content = self._limit_rerank_text(content, self.llm_rerank_max_doc_chars)

        if metadata_bits:
            return "\n".join(metadata_bits + ["", content]).strip()
        return content.strip()

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

    def _build_query_profile(self, user_query: str) -> QueryProfile:
        normalized_query = self._normalize_query_text(user_query)
        tokens = self._tokenize_for_keyword_search(user_query)
        keywords = self._extract_query_keywords(tokens)
        language = self._detect_language(user_query, tokens)
        intent_tags = self._detect_intent_tags(normalized_query, tokens)
        ambiguity_score = self._estimate_ambiguity(keywords, intent_tags, language, user_query)
        semantic_query = self._build_semantic_query(user_query, keywords, intent_tags)
        evidence_query = self._build_evidence_query(keywords, intent_tags, language)
        keyword_query = self._build_keyword_query(keywords, intent_tags)
        section_preferences = self._preferred_section_tags(intent_tags)
        return QueryProfile(
            original_query=user_query,
            normalized_query=normalized_query,
            language=language,
            tokens=tokens,
            keywords=keywords,
            intent_tags=intent_tags,
            ambiguity_score=ambiguity_score,
            semantic_query=semantic_query,
            evidence_query=evidence_query,
            keyword_query=keyword_query,
            section_preferences=section_preferences,
        )

    def _build_query_views(
        self,
        user_query: str,
        query_profile: QueryProfile,
        enable_query_rewrite: bool,
    ) -> Dict[str, Any]:
        llm_rewrites: List[str] = []
        llm_error: Optional[str] = None
        if enable_query_rewrite and self.generation_service is not None:
            try:
                llm_rewrites = self.generation_service.rewrite_query_for_retrieval(user_query, max_queries=3)
            except Exception as exc:  # pragma: no cover - remote model failures are environment dependent
                llm_error = str(exc)

        heuristic_rewrites = self._heuristic_query_rewrites(query_profile)
        core_views = [
            query_profile.original_query,
            query_profile.semantic_query,
            query_profile.evidence_query,
            query_profile.keyword_query,
        ]
        candidate_rows: List[Dict[str, Any]] = []
        seen = set()
        selected_queries: List[str] = []

        def add_candidate(source: str, query: str, rank: int) -> None:
            stripped = query.strip()
            normalized = self._normalize_query_text(stripped)
            row = {
                "query": stripped,
                "source": source,
                "source_index": rank,
                "normalized": normalized,
                "selected": False,
                "reason": "kept",
            }
            if not normalized:
                row["reason"] = "empty"
                candidate_rows.append(row)
                return
            if normalized == query_profile.normalized_query:
                row["reason"] = "same_as_original"
                candidate_rows.append(row)
                return
            if normalized in seen:
                row["reason"] = "duplicate"
                candidate_rows.append(row)
                return
            seen.add(normalized)
            candidate_rows.append(row)
            if len(selected_queries) < QUERY_VIEW_LIMIT:
                row["selected"] = True
                selected_queries.append(stripped)
            else:
                row["reason"] = "trimmed_to_top_k"

        for idx, query in enumerate(core_views):
            add_candidate("core", query, idx)
        if enable_query_rewrite:
            for idx, query in enumerate(llm_rewrites, start=len(core_views)):
                add_candidate("llm", query, idx)
            for idx, query in enumerate(heuristic_rewrites, start=len(core_views) + len(llm_rewrites)):
                add_candidate("heuristic", query, idx)

        if not selected_queries:
            selected_queries.append(query_profile.semantic_query)

        rewrite_debug = {
            "enabled": enable_query_rewrite,
            "original_query": user_query,
            "model_queries": llm_rewrites,
            "heuristic_queries": heuristic_rewrites,
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
            "model_queries": llm_rewrites,
            "heuristic_queries": heuristic_rewrites,
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

    def _vector_retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        route_name: str,
        source_query: str,
        query_profile: QueryProfile,
        route_queries: List[str],
    ) -> List[Dict[str, Any]]:
        sample_chunks = self.vector_store_service.get_all_chunks(collection_name, limit=1)
        sample_metadata = sample_chunks[0].get("metadata", {}) if sample_chunks else {}
        collection_info = self.vector_store_service.get_collection_info("milvus", collection_name)
        vector_dimension = None
        schema = collection_info.get("schema", {}) if isinstance(collection_info, dict) else {}
        for field in schema.get("fields", []) if isinstance(schema, dict) else []:
            if field.get("name") == "vector":
                vector_dimension = field.get("dim")
                if vector_dimension is None:
                    params = field.get("params", {})
                    if isinstance(params, dict):
                        vector_dimension = params.get("dim")
                break
        embedding_provider = sample_metadata.get("embedding_provider") or self.embedding_service.get_default_embedding_config().provider
        embedding_model = sample_metadata.get("embedding_model") or self.embedding_service.get_default_embedding_config().model_name
        embedding = self.embedding_service.create_single_embedding(
            query,
            provider=str(embedding_provider),
            model=str(embedding_model),
            dimension=int(vector_dimension) if vector_dimension else None,
        )
        results = self.vector_store_service.search_similar_vectors(
            collection_name=collection_name,
            query_vector=embedding,
            top_k=top_k,
        )
        route_confidence = self._route_confidence(
            route_name,
            query_profile,
            source_query,
            route_queries=route_queries,
        )
        return self._normalize_route_results(results, route_name, source_query, route_confidence, query_profile)

    def _keyword_retrieve(
        self,
        collection_name: str,
        queries: List[str],
        top_k: int,
        query_profile: QueryProfile,
    ) -> List[Dict[str, Any]]:
        chunks = [self._normalize_chunk(chunk) for chunk in self.vector_store_service.get_all_chunks(collection_name)]
        if not chunks:
            return []

        doc_tokens = [self._tokenize_for_keyword_search(chunk.get("content", "")) for chunk in chunks]
        avgdl = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
        document_frequencies = defaultdict(int)
        for tokens in doc_tokens:
            for token in set(tokens):
                document_frequencies[token] += 1

        per_chunk_scores = [0.0 for _ in chunks]
        query_signatures: List[str] = []
        for query in queries:
            tokens = self._tokenize_for_keyword_search(query)
            if not tokens:
                continue
            query_signatures.append(query)
            token_counter = Counter(tokens)
            for idx, chunk in enumerate(chunks):
                per_chunk_scores[idx] += self._bm25_score(
                    token_counter=token_counter,
                    doc_tokens=doc_tokens[idx],
                    doc_freqs=document_frequencies,
                    total_docs=len(chunks),
                    avg_doc_length=avgdl,
                    content=chunk.get("content", ""),
                )

        ranked: List[Tuple[int, float]] = [(idx, score) for idx, score in enumerate(per_chunk_scores) if score > 0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return []

        raw_scores = [score for _, score in ranked]
        route_confidence = self._route_confidence(
            "keyword",
            query_profile,
            " | ".join(query_signatures) if query_signatures else query_profile.original_query,
            route_queries=query_signatures or [query_profile.keyword_query],
        )
        normalized_scores = self._normalize_scores(raw_scores)

        results: List[Dict[str, Any]] = []
        for rank, ((idx, score), normalized_score) in enumerate(zip(ranked[:top_k], normalized_scores[:top_k])):
            chunk = dict(chunks[idx])
            chunk["retrieval_route"] = "keyword"
            chunk["source_query"] = " | ".join(query_signatures)
            chunk["route_rank"] = rank + 1
            chunk["route_score"] = float(score)
            chunk["normalized_route_score"] = float(normalized_score)
            chunk["route_confidence"] = float(route_confidence)
            chunk["structural_bonus"] = float(self._compute_structural_bonus(chunk, query_profile))
            results.append(chunk)
        return results

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
        for route_name, route_results in routes.items():
            weight = self.route_weights.get(route_name, 1.0)
            for rank, item in enumerate(route_results):
                chunk_key = self._chunk_unique_key(item)
                route_confidence = float(item.get("route_confidence", 1.0) or 1.0)
                normalized_route_score = float(item.get("normalized_route_score", 0.0) or 0.0)
                structural_bonus = float(item.get("structural_bonus", 0.0) or 0.0)
                entry = aggregated.setdefault(
                    chunk_key,
                    {
                        **item,
                        "score": 0.0,
                        "matched_routes": [],
                        "route_scores": {},
                        "normalized_route_scores": {},
                        "source_queries": [],
                    },
                )
                vote = weight * route_confidence * (
                    (1.0 / (self.rrf_k + rank + 1))
                    + (0.12 * normalized_route_score)
                    + structural_bonus
                )
                entry["score"] += vote
                entry["matched_routes"].append(route_name)
                entry["route_scores"][route_name] = item.get("route_score")
                entry["normalized_route_scores"][route_name] = normalized_route_score
                entry["route_confidences"] = entry.get("route_confidences", {})
                entry["route_confidences"][route_name] = route_confidence
                if item.get("source_query") and item["source_query"] not in entry["source_queries"]:
                    entry["source_queries"].append(item["source_query"])

        fused = sorted(
            aggregated.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                max(item.get("normalized_route_scores", {}).values() or [0.0]),
                max(item.get("route_scores", {}).values() or [0.0]),
            ),
            reverse=True,
        )

        fused = fused[:top_k]
        if query_profile.section_preferences:
            for chunk in fused:
                chunk["structural_bonus"] = float(self._compute_structural_bonus(chunk, query_profile))
        return fused

    def _normalize_chunk(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = item.get("metadata", {})
        chunk = dict(item)
        chunk["content"] = item.get("content") or item.get("text") or ""
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

        for intent in query_profile.intent_tags:
            if intent == "summary":
                rewrites.extend(
                    [
                        "main contributions key findings paper summary",
                        "abstract introduction conclusion contributions",
                    ]
                )
            elif intent == "method":
                rewrites.extend(
                    [
                        "proposed method approach architecture implementation details",
                        "model framework training objective algorithm design",
                    ]
                )
            elif intent == "experiment":
                rewrites.extend(
                    [
                        "experimental results evaluation benchmarks ablation",
                        "results comparison metrics dataset setup",
                    ]
                )
            elif intent == "comparison":
                rewrites.extend(
                    [
                        "baseline comparison ablation study competing methods",
                        "comparison with previous methods experimental comparison",
                    ]
                )
            elif intent == "limitation":
                rewrites.extend(
                    [
                        "limitations future work failure cases",
                        "discussion limitations assumptions",
                    ]
                )
            elif intent == "definition":
                rewrites.extend(
                    [
                        "problem formulation definition task setup",
                        "what is the task definition and setup",
                    ]
                )
            elif intent == "dataset":
                rewrites.extend(
                    [
                        "datasets corpus data splits evaluation setup",
                        "training data benchmark datasets",
                    ]
                )

        return rewrites[: QUERY_VIEW_LIMIT]

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

    def _build_semantic_query(self, user_query: str, keywords: List[str], intent_tags: List[str]) -> str:
        parts: List[str] = []
        if intent_tags:
            parts.extend(self._intent_to_terms(intent_tags))
        parts.extend(keywords[:6])
        if not parts:
            parts.extend(self._tokenize_for_keyword_search(user_query)[:6])
        return self._dedupe_terms(parts) or self._normalize_query_text(user_query)

    def _build_evidence_query(self, keywords: List[str], intent_tags: List[str], language: str) -> str:
        parts = self._intent_to_evidence_terms(intent_tags)
        parts.extend(keywords[:4])
        if language == "zh":
            parts.extend(["论文", "证据", "段落"])
        else:
            parts.extend(["paper", "evidence", "passage"])
        return self._dedupe_terms(parts)

    def _build_keyword_query(self, keywords: List[str], intent_tags: List[str]) -> str:
        parts = keywords[:8] + self._intent_to_terms(intent_tags)
        if not parts:
            parts = keywords[:8]
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
        content_weight = min(1.0, len(keywords) / 8.0)
        intent_weight = min(1.0, len(intent_tags) / 3.0)
        length_weight = min(1.0, len(user_query.strip()) / 50.0)
        language_weight = 0.1 if language in {"zh", "mixed"} else 0.0
        specificity = min(1.0, 0.45 * content_weight + 0.3 * intent_weight + 0.15 * length_weight + language_weight)
        return max(0.1, min(1.0, 1.0 - specificity))

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

    def _extract_query_keywords(self, tokens: List[str], limit: int = 8) -> List[str]:
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

    def _build_query_keywords(self, queries: List[str], limit: int = 12) -> List[str]:
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

    def _route_confidence(
        self,
        route_name: str,
        query_profile: QueryProfile,
        source_query: str,
        route_queries: Optional[List[str]] = None,
    ) -> float:
        route_queries = route_queries or [source_query]
        source_text = " ".join(route_queries) if route_queries else source_query
        similarity = self._query_similarity(query_profile.normalized_query, source_text)
        ambiguity = query_profile.ambiguity_score
        if route_name == "vector_original":
            base = 1.0
        elif route_name == "vector_rewrite":
            base = 0.62 + 0.22 * ambiguity
        elif route_name == "vector_hyde":
            base = 0.45 + 0.25 * ambiguity
        elif route_name == "keyword":
            base = 0.56 + 0.16 * min(1.0, len(query_profile.keywords) / 8.0)
        else:
            base = 0.5
        return max(0.2, min(1.0, base * (0.65 + 0.35 * similarity)))

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
            bonus += 0.045 * len(section_tags & preferred)
        if section_tags & NOISY_SECTION_TAGS:
            bonus -= 0.02 * len(section_tags & NOISY_SECTION_TAGS)
        if "abstract" in section_tags and "summary" in query_profile.intent_tags:
            bonus += 0.03
        if "conclusion" in section_tags and "summary" in query_profile.intent_tags:
            bonus += 0.02
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
        k1 = 1.5
        b = 0.75
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
                    score += 0.2

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
                str(item.get("source", "")),
                str(item.get("original_chunk_id", item.get("parent_chunk_id", item.get("chunk_id", 0)))),
                str(item.get("content_part_label", "")),
                str(item.get("page_range", "")),
            ]
        )

    def _debug_query_profile(self, query_profile: Optional[QueryProfile]) -> Optional[Dict[str, Any]]:
        if query_profile is None:
            return None
        return {
            "original_query": query_profile.original_query,
            "normalized_query": query_profile.normalized_query,
            "language": query_profile.language,
            "tokens": query_profile.tokens,
            "keywords": query_profile.keywords,
            "intent_tags": query_profile.intent_tags,
            "ambiguity_score": query_profile.ambiguity_score,
            "semantic_query": query_profile.semantic_query,
            "evidence_query": query_profile.evidence_query,
            "keyword_query": query_profile.keyword_query,
            "section_preferences": query_profile.section_preferences,
        }

    def _debug_chunk_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        preview = item.get("content", "")[:220].replace("\n", " ").strip()
        return {
            "chunk_id": item.get("chunk_id"),
            "original_chunk_id": item.get("original_chunk_id"),
            "page_number": item.get("page_number"),
            "page_range": item.get("page_range"),
            "score": item.get("score"),
            "route_score": item.get("route_score"),
            "normalized_route_score": item.get("normalized_route_score"),
            "route_confidence": item.get("route_confidence"),
            "structural_bonus": item.get("structural_bonus"),
            "retrieval_route": item.get("retrieval_route"),
            "matched_routes": item.get("matched_routes", []),
            "route_scores": item.get("route_scores", {}),
            "source_query": item.get("source_query"),
            "source_queries": item.get("source_queries", []),
            "subchunk_label": item.get("subchunk_label"),
            "section_tags": item.get("section_tags", []),
            "preview": preview,
        }
