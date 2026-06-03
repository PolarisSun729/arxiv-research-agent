from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from services.arxiv.arxiv_search_service import ArxivSearchService
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

    @staticmethod
    def _exception_detail(exc: Exception) -> str:
        detail = getattr(exc, "detail", None)
        if isinstance(detail, dict):
            message = str(detail.get("message") or detail.get("detail") or detail.get("error") or "").strip()
            if message:
                return message
            return json.dumps(detail, ensure_ascii=False)
        if detail is not None:
            text = str(detail).strip()
            if text:
                return text
        text = str(exc).strip()
        return text or exc.__class__.__name__

    def _log_stage(
        self,
        stage: str,
        arxiv_id: str,
        loading_method: str,
        message: str,
        **extra: Any,
    ) -> None:
        # 阶段日志只记录摘要信息，方便定位卡点，同时避免把论文正文或 chunk 内容打进日志。
        extra_parts = ", ".join(
            f"{key}={value}"
            for key, value in extra.items()
            if value not in (None, "", [], {})
        )
        if extra_parts:
            logger.info(
                "QA index stage=%s arxiv_id=%s loading_method=%s %s | %s",
                stage,
                arxiv_id,
                loading_method,
                message,
                extra_parts,
            )
        else:
            logger.info(
                "QA index stage=%s arxiv_id=%s loading_method=%s %s",
                stage,
                arxiv_id,
                loading_method,
                message,
            )

    def _tag_exception(
        self,
        exc: Exception,
        *,
        stage: str,
        arxiv_id: str,
        loading_method: str,
    ) -> str:
        # 把失败阶段挂到异常对象上，后续 agent 层可以直接把它写回结果结构里。
        detail = self._exception_detail(exc)
        setattr(exc, "error_stage", stage)
        setattr(exc, "error_detail", detail)
        setattr(exc, "error_arxiv_id", arxiv_id)
        setattr(exc, "error_loading_method", loading_method)
        return detail

    def _notify_progress(
        self,
        progress_callback: Optional[Callable[..., Any]],
        *,
        current_stage: str,
        progress: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                current_stage=current_stage,
                progress=progress,
                message=message,
            )
        except Exception as exc:
            logger.warning(
                "Progress callback failed at stage=%s progress=%s: %s",
                current_stage,
                progress,
                exc,
            )

    def load_paper_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        paper = self.db_service.get_paper(arxiv_id)
        if not paper:
            # 索引链路依赖论文元数据；如果本地库里还没有，就先回源 arXiv 补齐，再继续后续解析。
            logger.info("Paper metadata missing in database, trying arXiv lookup: %s", arxiv_id)
            paper = self._fetch_and_store_paper_metadata(arxiv_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found in database")
        return paper

    def _fetch_and_store_paper_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            search_service = ArxivSearchService()
            search_result = search_service.search(id_list=[arxiv_id], max_results=1)
            papers = list(search_result.get("papers") or [])
            paper = papers[0] if papers else None
            if not isinstance(paper, dict):
                logger.warning("ArXiv lookup returned no paper metadata: %s", arxiv_id)
                return None

            normalized_paper = {
                "arxiv_id": str(paper.get("arxiv_id") or arxiv_id).strip(),
                "title": str(paper.get("title") or "").strip(),
                "authors": paper.get("authors") or [],
                # arXiv API 返回的是 summary，这里写入 abstract 供后续建索引和问答复用。
                "abstract": str(paper.get("summary") or "").strip(),
                "categories": paper.get("categories") or [],
                "published_date": str(paper.get("published") or "").strip(),
                "url": str(paper.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}").strip(),
                "embedding_id": "",
                "embedding_model": "",
            }
            if not normalized_paper["title"] or not normalized_paper["abstract"]:
                logger.warning(
                    "ArXiv lookup returned incomplete metadata for %s: title=%s abstract=%s",
                    arxiv_id,
                    bool(normalized_paper["title"]),
                    bool(normalized_paper["abstract"]),
                )
                return None

            if not self.db_service.add_paper(normalized_paper):
                logger.error("Failed to persist arXiv metadata into database: %s", arxiv_id)
                return None

            logger.info("ArXiv metadata fetched and stored for: %s", arxiv_id)
            return self.db_service.get_paper(arxiv_id) or normalized_paper
        except Exception as exc:
            logger.exception("Failed to fetch arXiv metadata for %s: %s", arxiv_id, exc)
            return None

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

    def build_qa_index(
        self,
        arxiv_id: str,
        loading_method: str = "docling",
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        logger.info("Creating QA index for paper: %s", arxiv_id)
        requested_loading_method = str(loading_method or "docling").strip().lower()
        current_stage = "validate_loading_method"
        try:
            self._notify_progress(
                progress_callback,
                current_stage="validate_loading_method",
                progress=5,
                message="Validating loading method",
            )
            loading_method = self.validate_loading_method(requested_loading_method)
            self._log_stage("validate_loading_method", arxiv_id, loading_method, "loading method validated")

            current_stage = "mark_index_processing"
            self._notify_progress(
                progress_callback,
                current_stage="mark_index_processing",
                progress=10,
                message="Marking QA index as processing",
            )
            self.mark_index_processing(arxiv_id)
            self._log_stage("mark_index_processing", arxiv_id, loading_method, "paper QA index marked as processing")

            current_stage = "load_paper_metadata"
            self._notify_progress(
                progress_callback,
                current_stage="load_paper_metadata",
                progress=15,
                message="Loading paper metadata",
            )
            paper = self.load_paper_metadata(arxiv_id)
            self._log_stage(
                "load_paper_metadata",
                arxiv_id,
                loading_method,
                "paper metadata loaded",
                title=paper.get("title", ""),
                has_abstract=bool(paper.get("abstract")),
                has_url=bool(paper.get("url")),
            )

            current_stage = "download_pdf"
            self._notify_progress(
                progress_callback,
                current_stage="download_pdf",
                progress=25,
                message="Downloading PDF",
            )
            pdf_path = self.download_pdf(arxiv_id)
            self._log_stage("download_pdf", arxiv_id, loading_method, "pdf downloaded", pdf_path=pdf_path)

            current_stage = "load_pdf_document"
            self._notify_progress(
                progress_callback,
                current_stage="load_pdf_document",
                progress=35,
                message="Loading PDF document",
            )
            loading_service, document, page_map = self.load_pdf_document(pdf_path, loading_method)
            self._log_stage(
                "load_pdf_document",
                arxiv_id,
                loading_method,
                "pdf content loaded",
                page_count=len(page_map),
            )

            current_stage = "chunk_document"
            self._notify_progress(
                progress_callback,
                current_stage="chunk_document",
                progress=45,
                message="Chunking document",
            )
            chunked_data, chunking_strategy = self.chunk_document(arxiv_id, loading_method, document, page_map)
            chunks = chunked_data["chunks"]
            self._log_stage(
                "chunk_document",
                arxiv_id,
                loading_method,
                "document chunked",
                chunk_count=len(chunks),
                chunking_strategy=chunking_strategy,
            )

            current_stage = "save_chunk_file"
            self._notify_progress(
                progress_callback,
                current_stage="save_chunk_file",
                progress=55,
                message="Saving chunk file",
            )
            chunk_file = self.save_chunk_file(
                loading_service=loading_service,
                arxiv_id=arxiv_id,
                loading_method=loading_method,
                chunks=chunks,
                page_map=page_map,
                document=document,
                chunking_strategy=chunking_strategy,
            )
            self._log_stage("save_chunk_file", arxiv_id, loading_method, "chunk file saved", chunk_file=chunk_file)

            current_stage = "compress_chunks_for_rerank"
            self._notify_progress(
                progress_callback,
                current_stage="compress_chunks_for_rerank",
                progress=65,
                message="Compressing chunk text for rerank",
            )
            chunks = self.compress_chunks_for_rerank(chunks)
            self._log_stage("compress_chunks_for_rerank", arxiv_id, loading_method, "chunk text compressed", chunk_count=len(chunks))

            current_stage = "create_chunk_embeddings"
            self._notify_progress(
                progress_callback,
                current_stage="create_chunk_embeddings",
                progress=78,
                message="Creating chunk embeddings",
            )
            embeddings, embedding_config = self.create_chunk_embeddings(arxiv_id, chunks)
            self._log_stage(
                "create_chunk_embeddings",
                arxiv_id,
                loading_method,
                "embeddings created",
                embedding_provider=embedding_config.provider,
                embedding_model=embedding_config.model_name,
                embedding_count=len(embeddings),
            )

            current_stage = "save_embeddings"
            self._notify_progress(
                progress_callback,
                current_stage="save_embeddings",
                progress=88,
                message="Saving embeddings",
            )
            embedding_file = self.save_embeddings(arxiv_id, embeddings)
            self._log_stage("save_embeddings", arxiv_id, loading_method, "embedding file saved", embedding_file=embedding_file)

            current_stage = "index_embeddings_to_vector_store"
            self._notify_progress(
                progress_callback,
                current_stage="index_embeddings_to_vector_store",
                progress=95,
                message="Indexing embeddings to vector store",
            )
            index_result = self.index_embeddings_to_vector_store(embedding_file)
            collection_name = index_result.get("collection_name", "")
            self._log_stage(
                "index_embeddings_to_vector_store",
                arxiv_id,
                loading_method,
                "embeddings indexed to vector store",
                collection_name=collection_name,
            )

            current_stage = "mark_index_success"
            self._notify_progress(
                progress_callback,
                current_stage="mark_index_success",
                progress=100,
                message="Marking QA index as success",
            )
            self.mark_index_success(
                arxiv_id,
                collection_name=collection_name,
                chunk_count=len(chunks),
                embedding_model=embedding_config.model_name,
                pdf_path=pdf_path,
            )
            self._log_stage(
                "mark_index_success",
                arxiv_id,
                loading_method,
                "paper QA index marked as indexed",
                collection_name=collection_name,
                chunk_count=len(chunks),
                embedding_model=embedding_config.model_name,
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
        except HTTPException as exc:
            detail = self._tag_exception(
                exc,
                stage=current_stage,
                arxiv_id=arxiv_id,
                loading_method=requested_loading_method,
            )
            logger.exception(
                "QA index build failed at stage=%s arxiv_id=%s loading_method=%s error_type=%s detail=%s",
                current_stage,
                arxiv_id,
                requested_loading_method,
                type(exc).__name__,
                detail,
            )
            self.mark_index_failed(arxiv_id)
            raise
        except Exception as exc:
            detail = self._tag_exception(
                exc,
                stage=current_stage,
                arxiv_id=arxiv_id,
                loading_method=requested_loading_method,
            )
            logger.exception(
                "QA index build failed at stage=%s arxiv_id=%s loading_method=%s error_type=%s detail=%s",
                current_stage,
                arxiv_id,
                requested_loading_method,
                type(exc).__name__,
                detail,
            )
            self.mark_index_failed(arxiv_id)
            raise
