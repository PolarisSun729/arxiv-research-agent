from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from services.document.chunking_service import ChunkingService
from services.storage.database_service import DatabaseService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.llm.generation_service import GenerationService, QWEN_RERANK_COMPRESS_MODEL_NAME
from services.document.loading_service import LoadingService
from services.storage.vector_store_service import VectorDBConfig, VectorStoreService

logger = logging.getLogger(__name__)


class PaperQAIndexBuilder:
    def __init__(
        self,
        *,
        db_service: DatabaseService,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        generation_service: GenerationService,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        get_embedding_config: Optional[Callable[[], EmbeddingConfig]] = None,
        loading_service_factory: Optional[Callable[[], LoadingService]] = None,
        chunking_service_factory: Optional[Callable[[], ChunkingService]] = None,
    ):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.arxiv_service_factory = arxiv_service_factory
        self.get_embedding_config = get_embedding_config or self.embedding_service.get_default_embedding_config
        self.loading_service_factory = loading_service_factory
        self.chunking_service_factory = chunking_service_factory

    @staticmethod
    def _chunk_type_counts(chunks: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        text_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "text")) == "text")
        figure_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "figure")
        table_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "table")
        return text_count, figure_count, table_count

    def validate_loading_method(self, loading_method: str) -> str:
        normalized_method = str(loading_method or "pymupdf").strip().lower()
        if normalized_method not in {"pymupdf", "docling"}:
            raise HTTPException(status_code=400, detail="loading_method must be either pymupdf or docling")
        return normalized_method

    def mark_index_processing(self, arxiv_id: str) -> None:
        self.db_service.insert_paper_qa_index(arxiv_id, status="processing")

    def load_paper_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        paper = self.db_service.get_paper(arxiv_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found in database")
        return paper

    def download_pdf(self, arxiv_id: str) -> str:
        if self.arxiv_service_factory is None:
            raise RuntimeError("arxiv_service_factory is required")
        arxiv_service = self.arxiv_service_factory()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        logger.info("Downloading PDF from: %s", pdf_url)
        pdf_path = arxiv_service.download_pdf(pdf_url, arxiv_id)
        logger.info("PDF downloaded to: %s", pdf_path)
        return pdf_path

    def load_pdf_document(self, pdf_path: str, loading_method: str) -> Tuple[LoadingService, Dict[str, Any], List[Dict[str, Any]]]:
        if self.loading_service_factory is None:
            raise RuntimeError("loading_service_factory is required")
        loading_service = self.loading_service_factory()
        logger.info("Loading PDF content...")
        document = loading_service.load_pdf(pdf_path, method=loading_method)
        page_map = loading_service.get_page_map()
        logger.info("Loaded %s pages from PDF", len(page_map))
        return loading_service, document, page_map

    def chunk_document(
        self,
        arxiv_id: str,
        loading_method: str,
        document: Dict[str, Any],
        page_map: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        if self.chunking_service_factory is None:
            raise RuntimeError("chunking_service_factory is required")
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
        text_count, figure_count, table_count = self._chunk_type_counts(chunks)
        logger.info(
            "Chunk composition: text=%d figure=%d table=%d",
            text_count,
            figure_count,
            table_count,
        )
        return chunked_data, chunking_strategy

    def save_chunk_file(
        self,
        loading_service: LoadingService,
        arxiv_id: str,
        loading_method: str,
        chunks: List[Dict[str, Any]],
        page_map: List[Dict[str, Any]],
        document: Dict[str, Any],
        chunking_strategy: str,
    ) -> str:
        chunk_file = loading_service.save_document(
            filename=f"{arxiv_id}.pdf",
            chunks=chunks,
            metadata={"total_pages": len(page_map)},
            loading_method=loading_method,
            chunking_strategy=chunking_strategy,
            document_data=document,
        )
        logger.info("Chunked document saved to: %s", chunk_file)
        return chunk_file

    def compress_chunks_for_rerank(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Compressing chunk text for rerank with Qwen...")
        compressed_chunks = self.generation_service.compress_chunks_for_rerank(
            chunks=chunks,
            model_name=QWEN_RERANK_COMPRESS_MODEL_NAME,
        )
        logger.info("Generated rerank_text for %d chunks", len(compressed_chunks))
        return compressed_chunks

    def create_chunk_embeddings(
        self,
        arxiv_id: str,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], EmbeddingConfig]:
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
        text_count, figure_count, table_count = self._chunk_type_counts(embeddings)
        logger.info(
            "Embedding composition: text=%d figure=%d table=%d",
            text_count,
            figure_count,
            table_count,
        )
        return embeddings, embedding_config

    def save_embeddings(self, arxiv_id: str, embeddings: List[Dict[str, Any]]) -> str:
        embedding_file = self.embedding_service.save_embeddings(f"{arxiv_id}.pdf", embeddings)
        logger.info("Embeddings saved to: %s", embedding_file)
        return embedding_file

    def index_embeddings_to_vector_store(self, embedding_file: str) -> Dict[str, Any]:
        vector_db_config = VectorDBConfig(provider="milvus", index_mode="default")
        index_result = self.vector_store_service.index_embeddings(embedding_file, vector_db_config)
        collection_name = index_result.get("collection_name", "")
        logger.info("Index created in collection: %s", collection_name)
        return index_result

    def mark_index_success(
        self,
        arxiv_id: str,
        *,
        collection_name: str,
        chunk_count: int,
        embedding_model: str,
        pdf_path: str,
    ) -> bool:
        return self.db_service.update_paper_qa_index(
            arxiv_id,
            collection_name=collection_name,
            status="indexed",
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            pdf_path=pdf_path,
        )

    def mark_index_failed(self, arxiv_id: str) -> bool:
        return self.db_service.update_paper_qa_index(arxiv_id, status="failed")

    def build_qa_index(self, arxiv_id: str, loading_method: str = "docling") -> Dict[str, Any]:
        logger.info("Creating QA index for paper: %s", arxiv_id)
        try:
            loading_method = self.validate_loading_method(loading_method)
            self.mark_index_processing(arxiv_id)

            self.load_paper_metadata(arxiv_id)
            pdf_path = self.download_pdf(arxiv_id)
            loading_service, document, page_map = self.load_pdf_document(pdf_path, loading_method)
            chunked_data, chunking_strategy = self.chunk_document(arxiv_id, loading_method, document, page_map)
            chunks = chunked_data["chunks"]
            chunk_file = self.save_chunk_file(
                loading_service=loading_service,
                arxiv_id=arxiv_id,
                loading_method=loading_method,
                chunks=chunks,
                page_map=page_map,
                document=document,
                chunking_strategy=chunking_strategy,
            )

            chunks = self.compress_chunks_for_rerank(chunks)
            embeddings, embedding_config = self.create_chunk_embeddings(arxiv_id, chunks)
            embedding_file = self.save_embeddings(arxiv_id, embeddings)
            index_result = self.index_embeddings_to_vector_store(embedding_file)
            collection_name = index_result.get("collection_name", "")

            self.mark_index_success(
                arxiv_id,
                collection_name=collection_name,
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
            self.mark_index_failed(arxiv_id)
            raise
        except Exception:
            self.mark_index_failed(arxiv_id)
            raise
