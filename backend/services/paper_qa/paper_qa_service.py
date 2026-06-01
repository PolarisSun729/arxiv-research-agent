from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from services.arxiv.arxiv_search_service import ArxivSearchService
from services.document.chunking_service import ChunkingService
from services.storage.database_service import DatabaseService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService, RetrievalOptions
from services.llm.generation_service import GenerationService
from services.document.loading_service import LoadingService
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.storage.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class PaperQAService:
    def __init__(
        self,
        *,
        db_service: Optional[DatabaseService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        generation_service: Optional[GenerationService] = None,
        enhanced_retrieval_service: Optional[EnhancedRetrievalService] = None,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        get_embedding_config: Optional[Callable[[], EmbeddingConfig]] = None,
        loading_service_factory: Optional[Callable[[], LoadingService]] = None,
        chunking_service_factory: Optional[Callable[[], ChunkingService]] = None,
        qa_index_builder: Optional[PaperQAIndexBuilder] = None,
    ):
        self.db_service = db_service or DatabaseService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.generation_service = generation_service or GenerationService()
        self.enhanced_retrieval_service = enhanced_retrieval_service or EnhancedRetrievalService(
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
        )
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
        self.get_embedding_config = get_embedding_config or self.embedding_service.get_default_embedding_config
        self.loading_service_factory = loading_service_factory or LoadingService
        self.chunking_service_factory = chunking_service_factory or ChunkingService
        self.qa_index_builder = qa_index_builder or PaperQAIndexBuilder(
            db_service=self.db_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
            arxiv_service_factory=self.arxiv_service_factory,
            get_embedding_config=self.get_embedding_config,
            loading_service_factory=self.loading_service_factory,
            chunking_service_factory=self.chunking_service_factory,
        )

    @staticmethod
    def _payload_get(payload: Any, key: str, default: Any = None) -> Any:
        if payload is None:
            return default
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)

    def get_qa_status(self, arxiv_id: str) -> Dict[str, Any]:
        qa_index = self.db_service.get_paper_qa_index(arxiv_id)
        if qa_index:
            return {
                "arxiv_id": arxiv_id,
                "has_index": qa_index["status"] == "indexed",
                "status": qa_index["status"],
                "collection_name": qa_index["collection_name"],
                "chunk_count": qa_index["chunk_count"],
                "embedding_model": qa_index["embedding_model"],
            }
        return {
            "arxiv_id": arxiv_id,
            "has_index": False,
            "status": "not_indexed",
        }

    def build_qa_index(self, arxiv_id: str, loading_method: str = "docling") -> Dict[str, Any]:
        return self.qa_index_builder.build_qa_index(arxiv_id, loading_method=loading_method)

    def build_generation_context(self, search_results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        text_parts: List[str] = []
        image_inputs: List[Dict[str, Any]] = []
        asset_metadata: List[Dict[str, Any]] = []

        for index, result in enumerate(search_results, start=1):
            chunk_type = str(result.get("chunk_type", "text") or "text").strip().lower()
            asset_info = {
                "chunk_type": chunk_type,
                "asset_kind": result.get("asset_kind", ""),
                "asset_path": result.get("asset_path", ""),
                "asset_abs_path": result.get("asset_abs_path", ""),
                "asset_summary": result.get("asset_summary", ""),
                "asset_preview_text": result.get("asset_preview_text", ""),
                "page_number": result.get("page_number", ""),
                "page_range": result.get("page_range", ""),
                "section_path": result.get("section_path", ""),
                "section_title": result.get("section_title", ""),
                "source": result.get("source", ""),
            }

            if chunk_type == "figure" and asset_info["asset_abs_path"]:
                image_inputs.append(
                    {
                        "image_path": asset_info["asset_abs_path"],
                        "page_number": asset_info["page_number"],
                        "asset_summary": asset_info["asset_summary"],
                        "section_path": asset_info["section_path"],
                    }
                )
                asset_metadata.append(asset_info)
                continue

            if chunk_type == "table":
                table_text = "\n".join(
                    part for part in [
                        f"[Table {index}]",
                        str(result.get("asset_summary", "") or "").strip(),
                        str(result.get("asset_preview_text", "") or "").strip(),
                    ] if part
                ).strip()
                if table_text:
                    text_parts.append(table_text)
                asset_metadata.append(asset_info)
                continue

            content = str(result.get("content", "") or "").strip()
            if content:
                text_parts.append(content)

        return "\n\n".join(text_parts), image_inputs, asset_metadata

    def build_source_payload(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "content": r.get("content", ""),
                "page_number": r.get("page_number", ""),
                "source": r.get("source", ""),
                "subchunk_label": r.get("subchunk_label", ""),
                "chunk_label": r.get("subchunk_label", ""),
                "section_path": r.get("section_path", ""),
                "parent_chunk_id": r.get("parent_chunk_id", r.get("chunk_id", 0)),
                "chunk_type": r.get("chunk_type", "text"),
                "asset_kind": r.get("asset_kind", ""),
                "asset_path": r.get("asset_path", ""),
                "asset_summary": r.get("asset_summary", ""),
                "asset_preview_text": r.get("asset_preview_text", ""),
            }
            for r in search_results
        ]

    def build_qa_context(self, arxiv_id: str, payload: Any):
        qa_index = self.db_service.get_paper_qa_index(arxiv_id)
        if not qa_index or qa_index["status"] != "indexed":
            raise HTTPException(status_code=400, detail="Paper does not have QA index. Please create index first.")

        question = str(self._payload_get(payload, "question", "") or "").strip()
        collection_name = qa_index["collection_name"]
        paper = self.db_service.get_paper(arxiv_id) or {}
        retrieval_result = self.enhanced_retrieval_service.enhanced_retrieve(
            collection_name=collection_name,
            user_query=question,
            paper_context={
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", ""),
                "categories": paper.get("categories", ""),
                "published_date": paper.get("published_date", ""),
                "url": paper.get("url", ""),
            },
            options=RetrievalOptions(
                top_k=self._payload_get(payload, "top_k", None) or 15,
                enable_query_rewrite=self._payload_get(payload, "enable_query_rewrite", None),
                enable_hyde=self._payload_get(payload, "enable_hyde", None),
                enable_keyword_search=self._payload_get(payload, "enable_keyword_search", None),
                enable_llm_rerank=self._payload_get(payload, "enable_llm_rerank", None),
                debug=self._payload_get(payload, "debug", None),
            ),
        )

        final_context_results = retrieval_result["chunks"]
        search_results = final_context_results
        if not search_results:
            raise HTTPException(status_code=400, detail="No relevant chunks found")

        text_context, image_inputs, asset_metadata = self.build_generation_context(search_results)
        return qa_index, search_results, {
            "text_context": text_context,
            "image_inputs": image_inputs,
            "asset_metadata": asset_metadata,
        }, retrieval_result.get("debug")

    def answer_question(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        question = str(self._payload_get(payload, "question", "") or "").strip()
        logger.info("QA request for paper: %s, question: %s", arxiv_id, question)
        _, search_results, qa_context, retrieval_debug = self.build_qa_context(arxiv_id, payload)
        source_payload = self.build_source_payload(search_results)

        logger.info("Generating answer...")
        try:
            qwen_search_results = [
                {
                    "text": result.get("content", ""),
                    "page_number": result.get("page_number", ""),
                    "source": result.get("source", ""),
                    "subchunk_label": result.get("subchunk_label", ""),
                    "chunk_label": result.get("subchunk_label", ""),
                    "section_path": result.get("section_path", ""),
                    "chunk_type": result.get("chunk_type", "text"),
                    "asset_kind": result.get("asset_kind", ""),
                    "asset_summary": result.get("asset_summary", ""),
                    "asset_preview_text": result.get("asset_preview_text", ""),
                }
                for result in search_results
            ]
            generation_result = self.generation_service.generate(
                provider="qwen",
                # 最终答案生成保留大模型，以保证长上下文综合与表述质量。
                task_type="paper_qa_final_answer",
                query=question,
                search_results=qwen_search_results,
                image_inputs=qa_context["image_inputs"],
                asset_metadata=[item for item in qa_context["asset_metadata"] if item.get("chunk_type") == "figure"],
            )
            answer = generation_result["response"]
        except Exception as exc:
            logger.warning("Qwen generation failed, using fallback: %s", exc)
            answer = f'根据论文内容，关于您的问题 "{question}" 的相关信息如下：\n\n{qa_context["text_context"][:1000]}...'

        return {
            "status": "success",
            "arxiv_id": arxiv_id,
            "question": question,
            "answer": answer,
            "sources": source_payload,
            "image_inputs": qa_context["image_inputs"],
            "asset_metadata": qa_context["asset_metadata"],
            "retrieval_debug": retrieval_debug,
        }

    def _sanitize_trace_slug(self, text: str, max_length: int = 40) -> str:
        import re

        slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        if not slug:
            slug = "query"
        return slug[:max_length]

    def _get_latest_retrieval_trace(self, arxiv_id: str, format_name: str = "md") -> Optional[Path]:
        trace_root = Path(str(self.enhanced_retrieval_service.trace_export_dir))
        paper_dir = trace_root / self._sanitize_trace_slug(arxiv_id)
        if not paper_dir.exists() or not paper_dir.is_dir():
            return None

        suffix = ".json" if format_name == "json" else ".md"
        trace_files = sorted(
            paper_dir.glob(f"*{suffix}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return trace_files[0] if trace_files else None
