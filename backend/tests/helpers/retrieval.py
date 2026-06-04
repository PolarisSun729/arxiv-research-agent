"""Helpers for retrieval / RAG unittest suites."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

from tests.helpers.fake_embedding_service import FakeEmbeddingService
from tests.helpers.fake_generation_service import FakeGenerationService
from tests.helpers.fake_vector_store_service import FakeVectorStoreService


def _ensure_retrieval_import_stubs() -> None:
    if "services.embedding.embedding_service" not in sys.modules:
        module = types.ModuleType("services.embedding.embedding_service")

        class EmbeddingConfig:
            def __init__(self, provider: str = "fake", model_name: str = "fake-embedding-model", dimension: int = 8):
                self.provider = provider
                self.model_name = model_name
                self.dimension = dimension

        class EmbeddingService:
            pass

        module.EmbeddingConfig = EmbeddingConfig
        module.EmbeddingService = EmbeddingService
        sys.modules[module.__name__] = module

    if "services.storage.vector_store_service" not in sys.modules:
        module = types.ModuleType("services.storage.vector_store_service")

        class VectorStoreService:
            pass

        module.VectorStoreService = VectorStoreService
        sys.modules[module.__name__] = module

    if "services.llm.generation_service" not in sys.modules:
        module = types.ModuleType("services.llm.generation_service")

        class GenerationService:
            pass

        module.GenerationService = GenerationService
        sys.modules[module.__name__] = module


def load_retrieval_modules() -> Dict[str, Any]:
    _ensure_retrieval_import_stubs()

    backend_dir = Path(__file__).resolve().parents[2]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    enhanced_module = importlib.import_module("services.retrieval.enhanced_retrieval_service")
    query_planner_module = importlib.import_module("services.retrieval.query_planner")
    route_retriever_module = importlib.import_module("services.retrieval.route_retriever")
    rerank_service_module = importlib.import_module("services.retrieval.rerank_service")

    enhanced_module.ENHANCED_RETRIEVAL_CONFIG.update(
        {
            "final_context_top_k": 3,
            "max_final_context_top_k": 4,
            "recall_candidate_limit": 8,
            "rrf_candidate_limit": 6,
            "rerank_candidate_limit": 4,
            "sample_limit": 6,
            "source_sample_limit": 3,
            "source_sample_primary_limit": 120,
            "source_sample_secondary_limit": 80,
            "merge_candidate_terms_limit": 12,
            "extract_paper_terms_limit": 12,
            "keyword_parts_limit": 6,
            "build_query_keywords_limit": 12,
            "preview_text_limit": 120,
            "rerank_document_preview_limit": 160,
            "short_text_preview_limit": 80,
            "candidate_debug_limit": 8,
        }
    )
    enhanced_module.RETRIEVAL_CONFIG.update(
        {
            "enable_query_rewrite": True,
            "enable_hyde": True,
            "enable_keyword_search": True,
            "enable_llm_rerank": False,
            "debug": True,
            "rrf_k": 20,
            "candidate_multiplier": 2,
            "route_weights": {
                "vector_original": 1.0,
                "vector_rewrite": 0.9,
                "vector_hyde": 0.75,
                "keyword": 0.8,
                "memory_context": 0.7,
            },
        }
    )
    enhanced_module.MEMORY_RUNTIME_CONFIG.update({"enable_memory_aware_retrieval": False})
    enhanced_module.QUERY_VIEW_LIMIT = 3
    enhanced_module.QUERY_PLAN_LIMIT = 3

    return {
        "enhanced": enhanced_module,
        "query_planner": query_planner_module,
        "route_retriever": route_retriever_module,
        "rerank_service": rerank_service_module,
    }


class ControlledFakeEmbeddingService(FakeEmbeddingService):
    def __init__(self) -> None:
        super().__init__(dimension=3)

    def _vector_for_text(self, text: str) -> list[float]:
        lowered = str(text or "").lower()
        if any(token in lowered for token in ["method", "approach", "framework", "pipeline", "architecture", "方法"]):
            return [1.0, 0.0, 0.0]
        if any(token in lowered for token in ["experiment", "evaluation", "baseline", "metric", "实验", "评估"]):
            return [0.0, 1.0, 0.0]
        if any(token in lowered for token in ["result", "ablation", "performance", "结果", "对比"]):
            return [0.0, 0.0, 1.0]
        if any(token in lowered for token in ["limitation", "future work", "weakness", "局限", "不足"]):
            return [0.7, 0.7, 0.0]
        if any(token in lowered for token in ["dataset", "benchmark", "corpus", "数据集", "语料"]):
            return [0.8, 0.2, 0.0]
        if any(token in lowered for token in ["figure", "table", "fig.", "图", "表"]):
            return [0.2, 0.8, 0.2]
        return [0.3, 0.3, 0.3]


class ControlledFakeVectorStoreService(FakeVectorStoreService):
    def resolve_collection_name(self, collection_name: str) -> str:
        return self._normalize_collection_name(collection_name)

    def get_collection_info(self, _provider: str, collection_name: str) -> Dict[str, Any]:
        info = super().get_collection_info(_provider, collection_name)
        info["schema"] = {"fields": [{"name": "vector", "dim": 3}]}
        return info


def build_sample_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "content": "Method section: the framework uses a retrieval pipeline with two encoder stages.",
            "metadata": {
                "chunk_id": "chunk-method",
                "parent_chunk_id": "parent-method",
                "page_number": 2,
                "page_range": "2",
                "section_title": "Method",
                "section_path": "2 Method",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 1,
            },
        },
        {
            "content": "Experiment section: evaluation uses the LongBench dataset, strong baselines, and exact match metrics.",
            "metadata": {
                "chunk_id": "chunk-experiment",
                "parent_chunk_id": "parent-experiment",
                "page_number": 4,
                "page_range": "4",
                "section_title": "Experiments",
                "section_path": "4 Experiments",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 2,
            },
        },
        {
            "content": "Results section: ablation results show better performance and stronger comparison against baselines.",
            "metadata": {
                "chunk_id": "chunk-results",
                "parent_chunk_id": "parent-results",
                "page_number": 5,
                "page_range": "5",
                "section_title": "Results",
                "section_path": "5 Results",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 3,
            },
        },
        {
            "content": "Limitation discussion: the method struggles on noisy prompts and future work is needed.",
            "metadata": {
                "chunk_id": "chunk-limitation",
                "parent_chunk_id": "parent-limitation",
                "page_number": 8,
                "page_range": "8",
                "section_title": "Limitations",
                "section_path": "8 Limitations",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 4,
            },
        },
        {
            "content": "Dataset details: the benchmark corpus includes training, dev, and test splits.",
            "metadata": {
                "chunk_id": "chunk-dataset",
                "parent_chunk_id": "parent-dataset",
                "page_number": 3,
                "page_range": "3",
                "section_title": "Dataset",
                "section_path": "3 Dataset",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 5,
            },
        },
        {
            "content": "Figure 2 and Table 3 summarize the main experimental trend and the best score.",
            "metadata": {
                "chunk_id": "chunk-figure-table",
                "parent_chunk_id": "parent-figure-table",
                "page_number": 6,
                "page_range": "6",
                "section_title": "Results Figure",
                "section_path": "6 Results/Figure 2",
                "source": "paper.pdf",
                "chunk_type": "figure",
                "asset_kind": "image",
                "asset_path": "figure-2.png",
                "asset_summary": "Figure 2 compares model performance; Table 3 reports exact numbers.",
                "asset_preview_text": "Model A 82.1, Model B 84.7",
                "order_index": 6,
            },
        },
        {
            "content": "Appendix prompt template: the prompt text is long and mostly implementation detail.",
            "metadata": {
                "chunk_id": "chunk-appendix",
                "parent_chunk_id": "parent-appendix",
                "page_number": 10,
                "page_range": "10",
                "section_title": "Appendix",
                "section_path": "Appendix A",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 7,
            },
        },
        {
            "content": "References bibliography listing prior work citations only.",
            "metadata": {
                "chunk_id": "chunk-references",
                "parent_chunk_id": "parent-references",
                "page_number": 11,
                "page_range": "11",
                "section_title": "References",
                "section_path": "References",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 8,
            },
        },
    ]


def seed_collection(
    vector_store_service: ControlledFakeVectorStoreService,
    collection_name: str,
    embedding_service: ControlledFakeEmbeddingService,
    chunks: List[Dict[str, Any]],
) -> str:
    resolved = vector_store_service.resolve_collection_name(collection_name)
    vector_store_service.collections[resolved] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.get("metadata", {}) or {})
        text = str(chunk.get("content", "") or metadata.get("asset_summary", ""))
        vector_store_service.collections[resolved].append(
            {
                "id": index,
                "content": text,
                "embedding": embedding_service.create_single_embedding(text),
                "metadata": metadata,
            }
        )
    return resolved


def build_retrieval_service(chunks: List[Dict[str, Any]] | None = None):
    modules = load_retrieval_modules()
    EnhancedRetrievalService = modules["enhanced"].EnhancedRetrievalService
    embedding_service = ControlledFakeEmbeddingService()
    vector_store_service = ControlledFakeVectorStoreService()
    generation_service = FakeGenerationService(response_text="fake generation response")
    collection_name = seed_collection(
        vector_store_service,
        "paper_qa_test",
        embedding_service,
        chunks or build_sample_chunks(),
    )
    service = EnhancedRetrievalService(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        generation_service=generation_service,
    )
    service.memory_runtime_config = {"enable_memory_aware_retrieval": False}
    return service, collection_name, embedding_service, vector_store_service, generation_service
