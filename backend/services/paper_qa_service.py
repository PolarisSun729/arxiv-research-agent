from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from services.arxiv_search_service import ArxivSearchService
from services.chunking_service import ChunkingService
from services.database_service import DatabaseService
from services.embedding_service import EmbeddingConfig, EmbeddingService
from services.enhanced_retrieval_service import EnhancedRetrievalService, RetrievalOptions
from services.generation_service import GenerationService, RERANK_QWEN_MODEL_NAME
from services.loading_service import LoadingService
from services.vector_store_service import VectorDBConfig, VectorStoreService

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
        logger.info("Creating QA index for paper: %s", arxiv_id)
        try:
            loading_method = str(loading_method or "pymupdf").strip().lower()
            if loading_method not in {"pymupdf", "docling"}:
                raise HTTPException(status_code=400, detail="loading_method must be either pymupdf or docling")

            self.db_service.insert_paper_qa_index(arxiv_id, status="processing")

            arxiv_service = self.arxiv_service_factory()
            paper = self.db_service.get_paper(arxiv_id)
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found in database")

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            logger.info("Downloading PDF from: %s", pdf_url)
            pdf_path = arxiv_service.download_pdf(pdf_url, arxiv_id)
            logger.info("PDF downloaded to: %s", pdf_path)

            loading_service = self.loading_service_factory()
            logger.info("Loading PDF content...")
            document = loading_service.load_pdf(pdf_path, method=loading_method)

            page_map = loading_service.get_page_map()
            logger.info("Loaded %s pages from PDF", len(page_map))

            chunking_service = self.chunking_service_factory()
            logger.info("Chunking text...")
            metadata = {"filename": f"{arxiv_id}.pdf", "loading_method": loading_method, "source": f"{arxiv_id}.pdf"}
            if loading_method == "docling":
                chunked_data = chunking_service.chunk_docling(document, metadata=metadata, page_map=page_map)
                chunking_strategy = "docling_sections"
            else:
                chunked_data = chunking_service.chunk_pymupdf(document, method="by_titles", metadata=metadata, page_map=page_map)
                chunking_strategy = "pymupdf_by_titles"

            chunks = chunked_data["chunks"]
            logger.info("Created %s chunks", len(chunks))
            logger.info(
                "Chunk composition: text=%d figure=%d table=%d",
                sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "text")) == "text"),
                sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "figure"),
                sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "table"),
            )

            chunk_file = loading_service.save_document(
                filename=f"{arxiv_id}.pdf",
                chunks=chunks,
                metadata={"total_pages": len(page_map)},
                loading_method=loading_method,
                chunking_strategy=chunking_strategy,
                document_data=document,
            )
            logger.info("Chunked document saved to: %s", chunk_file)

            logger.info("Compressing chunk text for rerank with Qwen...")
            chunks = self.generation_service.compress_chunks_for_rerank(
                chunks=chunks,
                model_name=RERANK_QWEN_MODEL_NAME,
            )
            logger.info("Generated rerank_text for %d chunks", len(chunks))

            embedding_config = self.get_embedding_config()
            logger.info(
                "Creating embeddings with %s / %s...",
                embedding_config.provider,
                embedding_config.model_name,
            )

            input_data = {
                "chunks": chunks,
                "metadata": {"filename": f"{arxiv_id}.pdf"},
            }
            embeddings, _ = self.embedding_service.create_embeddings(input_data, embedding_config)

            logger.info("Created %d embeddings", len(embeddings))
            logger.info(
                "Embedding composition: text=%d figure=%d table=%d",
                sum(1 for item in embeddings if str((item.get("metadata", {}) or {}).get("chunk_type", "text")) == "text"),
                sum(1 for item in embeddings if str((item.get("metadata", {}) or {}).get("chunk_type", "")) == "figure"),
                sum(1 for item in embeddings if str((item.get("metadata", {}) or {}).get("chunk_type", "")) == "table"),
            )

            embedding_file = self.embedding_service.save_embeddings(f"{arxiv_id}.pdf", embeddings)
            logger.info("Embeddings saved to: %s", embedding_file)

            vector_db_config = VectorDBConfig(provider="milvus", index_mode="default")
            index_result = self.vector_store_service.index_embeddings(embedding_file, vector_db_config)
            collection_name = index_result.get("collection_name", "")
            logger.info("Index created in collection: %s", collection_name)

            self.db_service.update_paper_qa_index(
                arxiv_id,
                collection_name=collection_name,
                status="indexed",
                chunk_count=len(chunks),
                embedding_model=embedding_config.model_name,
                pdf_path=pdf_path,
            )

            return {
                "status": "success",
                "message": "QA index created successfully",
                "arxiv_id": arxiv_id,
                "loading_method": loading_method,
                "pdf_path": pdf_path,
                "collection_name": collection_name,
                "chunk_count": len(chunks),
                "embedding_model": embedding_config.model_name,
                "chunk_file": chunk_file,
            }
        except HTTPException:
            self.db_service.update_paper_qa_index(arxiv_id, status="failed")
            raise
        except Exception:
            self.db_service.update_paper_qa_index(arxiv_id, status="failed")
            raise

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
                model_name="qwen3.6-plus",
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
