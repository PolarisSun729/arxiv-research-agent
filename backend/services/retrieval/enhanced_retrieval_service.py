from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from services.embedding.embedding_service import EmbeddingService
from services.intent.intent_service import IntentService
from services.retrieval.collection_profile import CollectionRetrievalProfileProvider
from services.retrieval.contracts import RetrievalOptions
from services.retrieval.query_planner import QueryPlanner
from services.retrieval.rerank_service import RerankService
from services.retrieval.result_fusion_service import ResultFusionService
from services.retrieval.retrieval_index import CollectionRetrievalIndexProvider
from services.retrieval.retrieval_pipeline import RetrievalPipeline
from services.retrieval.retrieval_rules import RetrievalRules
from services.retrieval.route_retriever import RouteRetriever
from services.retrieval.table_structured_retriever import TableStructuredRetriever
from services.retrieval.trace_builder import RetrievalTraceBuilder
from services.storage.vector_store_service import VectorStoreService
from utils.config import RETRIEVAL_CONFIG, get_enhanced_retrieval_runtime_config, get_memory_runtime_config

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


class EnhancedRetrievalService:
    """兼容入口只负责依赖装配、稳定对外入口和 pipeline 委托。"""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        generation_service: Optional[GenerationService] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.intent_service = IntentService(generation_service=self.generation_service)
        self.retrieval_rules = RetrievalRules(config=ENHANCED_RETRIEVAL_CONFIG)

        self.rrf_k = RETRIEVAL_CONFIG["rrf_k"]
        self.route_weights = RETRIEVAL_CONFIG["route_weights"]
        self.candidate_multiplier = RETRIEVAL_CONFIG["candidate_multiplier"]

        # rerank 配置继续保留在兼容壳上，避免现有测试和运行时热改配置失效。
        self.llm_rerank_model_name_or_path = RETRIEVAL_CONFIG.get(
            "llm_rerank_model_name_or_path",
            "00-models/Qwen3-VL-Reranker-2B",
        )
        self.llm_rerank_prompt = RETRIEVAL_CONFIG.get(
            "llm_rerank_prompt",
            "Retrieve text relevant to the user's query.",
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
        self.llm_rerank_model_name = str(RETRIEVAL_CONFIG.get("llm_rerank_model_name_or_path", "qwen3-rerank")).strip()
        self.llm_rerank_local_model_name_or_path = str(
            RETRIEVAL_CONFIG.get("llm_rerank_local_model_name_or_path", "00-models/Qwen3-VL-Reranker-2B")
        ).strip()
        self.llm_rerank_batch_size = int(RETRIEVAL_CONFIG.get("llm_rerank_batch_size", 8))
        self.llm_rerank_candidate_limit = int(RETRIEVAL_CONFIG.get("llm_rerank_candidate_limit", 24))
        self.llm_rerank_max_doc_chars = int(RETRIEVAL_CONFIG.get("llm_rerank_max_doc_chars", 4096))
        self._llm_reranker_path: Optional[str] = None
        self._llm_reranker_device: Optional[str] = None
        self._llm_reranker_error: Optional[str] = None

        self.trace_export_enabled = bool(RETRIEVAL_CONFIG.get("trace_export_enabled", True))
        self.trace_export_dir = Path(str(RETRIEVAL_CONFIG.get("trace_export_dir", "temp/retrieval-traces")))
        self.memory_runtime_config = get_memory_runtime_config()

        # 这些子服务组成真实检索链路；EnhancedRetrievalService 只负责把共享规则和配置接成依赖图。
        self.collection_profile_provider = CollectionRetrievalProfileProvider(
            vector_store_service=self.vector_store_service,
            embedding_service=self.embedding_service,
        )
        self.collection_retrieval_index_provider = CollectionRetrievalIndexProvider(
            vector_store_service=self.vector_store_service,
            chunk_normalizer=self.retrieval_rules.normalize_chunk,
            tokenizer=self.retrieval_rules.tokenize_for_keyword_search,
        )
        self.fusion_service = ResultFusionService(
            rrf_k=self.rrf_k,
            route_weights=self.route_weights,
        )
        self.trace_builder = RetrievalTraceBuilder(
            trace_export_enabled=self.trace_export_enabled,
            trace_export_dir=self.trace_export_dir,
            rrf_k=self.rrf_k,
            route_weights=self.route_weights,
            route_confidence_builder=self.retrieval_rules.route_confidence,
            route_weights_builder=self.fusion_service.route_weights_for_intent,
        )
        self.rerank_service = RerankService(
            generation_service=self.generation_service,
            config_owner=self,
            query_normalizer=self.retrieval_rules.normalize_query_text,
            query_profile_debugger=self.trace_builder.debug_query_profile,
            trace_builder=self.trace_builder,
        )
        self.query_planner = QueryPlanner(
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
            intent_service=self.intent_service,
            rerank_service=self.rerank_service,
            enhanced_config=ENHANCED_RETRIEVAL_CONFIG,
            retrieval_rules=self.retrieval_rules,
        )
        self.route_retriever = RouteRetriever(
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
            query_tools=self.query_planner,
            fusion_service=self.fusion_service,
            table_structured_retriever=TableStructuredRetriever(
                query_tools=self.query_planner,
                route_confidence_builder=self.retrieval_rules.route_confidence,
                structural_bonus_builder=self.retrieval_rules.compute_structural_bonus,
                config=ENHANCED_RETRIEVAL_CONFIG,
            ),
            route_confidence_builder=self.retrieval_rules.route_confidence,
            structural_bonus_builder=self.retrieval_rules.compute_structural_bonus,
            chunk_normalizer=self.retrieval_rules.normalize_chunk,
            memory_flag_reader=self._memory_flag,
            collection_profile_provider=self.collection_profile_provider,
            collection_retrieval_index_provider=self.collection_retrieval_index_provider,
        )
        self.retrieval_pipeline = RetrievalPipeline(
            query_planner=self.query_planner,
            route_retriever=self.route_retriever,
            fusion_service=self.fusion_service,
            rerank_service=self.rerank_service,
            trace_builder=self.trace_builder,
            collection_resolver=self._resolve_collection_name,
            collection_profile_provider=self.collection_profile_provider,
            collection_retrieval_index_provider=self.collection_retrieval_index_provider,
            option_resolver=self._resolve_option,
            memory_flag_reader=self._memory_flag,
            retrieval_config=RETRIEVAL_CONFIG,
            enhanced_config=ENHANCED_RETRIEVAL_CONFIG,
        )

    def enhanced_retrieve(
        self,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        options: Optional[RetrievalOptions] = None,
    ) -> Dict[str, Any]:
        # 对外兼容入口只负责委托，真正的 query planning / route / rerank / trace 编排都收口到 pipeline。
        return self.retrieval_pipeline.retrieve(
            user_query=user_query,
            collection_name=collection_name,
            paper_context=paper_context,
            options=options,
        )

    # 以下私有方法是 facade 适配点：它们处理运行时状态、参数兜底和可选依赖能力，
    # 不承担 query planning / route / rerank / fusion 等检索编排职责。
    def _memory_flag(self, key: str, default: Any = None) -> Any:
        return self.memory_runtime_config.get(key, default)

    def _resolve_option(self, runtime_value: Optional[bool], default_value: bool) -> bool:
        return default_value if runtime_value is None else bool(runtime_value)

    def _resolve_collection_name(self, collection_name: str) -> str:
        if hasattr(self.vector_store_service, "resolve_collection_name"):
            return self.vector_store_service.resolve_collection_name(collection_name)
        return collection_name
