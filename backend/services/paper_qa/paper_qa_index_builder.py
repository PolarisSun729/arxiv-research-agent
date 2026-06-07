from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from core.errors import AppError, ErrorCode
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.document.chunking_service import ChunkingService
from services.storage.database_service import DatabaseService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.llm.generation_service import GenerationService, QWEN_RERANK_COMPRESS_MODEL_NAME
from services.document.loading_service import LoadingService
from services.storage.vector_store_service import VectorDBConfig, VectorStoreService

logger = logging.getLogger(__name__)

VECTOR_STORE_STAGES = {"index_embeddings_to_vector_store", "cleanup_old_artifacts"}
DATABASE_WRITE_STAGES = {
    "mark_index_processing",
    "record_index_stage",
    "mark_index_success",
}

QA_INDEX_ARTIFACT_FIELDS = {
    "pdf_path",
    "chunk_file",
    "embedding_file",
    "collection_name",
    "loading_method",
    "chunking_strategy",
    "chunk_count",
    "embedding_model",
}


class PaperQAIndexBuilder:
    def __init__(
        self,
        *,
        db_service: DatabaseService,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        generation_service: GenerationService,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        oai_db_service: Optional[ArxivOaiDatabaseService] = None,
        get_embedding_config: Optional[Callable[[], EmbeddingConfig]] = None,
        loading_service_factory: Optional[Callable[[], LoadingService]] = None,
        chunking_service_factory: Optional[Callable[[], ChunkingService]] = None,
    ):
        """初始化论文问答索引构建器，并注入建索引链路所需依赖。"""
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.arxiv_service_factory = arxiv_service_factory
        self.oai_db_service = oai_db_service
        self.get_embedding_config = get_embedding_config or self.embedding_service.get_default_embedding_config
        self.loading_service_factory = loading_service_factory
        self.chunking_service_factory = chunking_service_factory

    @staticmethod
    def _chunk_type_counts(chunks: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """统计 chunk 中的文本、图片和表格数量，便于记录索引摘要。"""
        text_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "text")) == "text")
        figure_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "figure")
        table_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "table")
        return text_count, figure_count, table_count

    def validate_loading_method(self, loading_method: str) -> str:
        """校验并规范化 PDF 加载方式，只允许当前支持的方法。"""
        normalized_method = str(loading_method or "pymupdf").strip().lower()
        if normalized_method not in {"pymupdf", "docling"}:
            raise HTTPException(status_code=400, detail="loading_method must be either pymupdf or docling")
        return normalized_method

    def mark_index_processing(self, arxiv_id: str, *, loading_method: str) -> None:
        """把论文索引状态标记为处理中，供外部轮询和后台任务联动使用。"""
        created = self.db_service.insert_paper_qa_index(
            arxiv_id,
            collection_name="",
            status="processing",
            chunk_count=0,
            embedding_model="",
            pdf_path="",
            chunk_file="",
            embedding_file="",
            loading_method=loading_method,
            chunking_strategy="",
            current_stage="mark_index_processing",
            failed_stage="",
            error_message="",
            artifact_status="active",
            indexed_at=None,
        )
        if not created:
            raise RuntimeError("Failed to mark QA index as processing")

    def _workspace_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _safe_delete_file(self, file_path: Any) -> bool:
        path_text = str(file_path or "").strip()
        if not path_text:
            return False
        path = Path(path_text)
        if not path.is_absolute():
            path = self._workspace_root() / path
        try:
            resolved_path = path.resolve()
            workspace_root = self._workspace_root().resolve()
            if workspace_root not in resolved_path.parents and resolved_path != workspace_root:
                logger.warning("Skip deleting QA artifact outside workspace: %s", resolved_path)
                return False
            if not resolved_path.exists():
                return False
            if not resolved_path.is_file():
                logger.warning("Skip deleting non-file QA artifact: %s", resolved_path)
                return False
            resolved_path.unlink()
            return True
        except Exception:
            logger.exception("Failed to delete QA artifact file: %s", path_text)
            raise

    def cleanup_qa_index_artifacts(self, arxiv_id: str, qa_index: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """清理指定论文上一轮 QA 索引留下的文件和 Milvus collection，用于重建或删除前的补偿。"""
        existing = qa_index if qa_index is not None else self.db_service.get_paper_qa_index(arxiv_id)
        result = {"collection_deleted": False, "files_deleted": [], "files_missing": []}
        if not existing:
            return result

        collection_name = str(existing.get("collection_name") or "").strip()
        if collection_name:
            # 重建前必须先移除旧 collection，否则检索链路可能继续命中上一轮 chunk。
            result["collection_deleted"] = self.vector_store_service.delete_collection("milvus", collection_name)

        for field_name in ("pdf_path", "chunk_file", "embedding_file"):
            artifact_path = str(existing.get(field_name) or "").strip()
            if not artifact_path:
                continue
            deleted = self._safe_delete_file(artifact_path)
            if deleted:
                result["files_deleted"].append({"field": field_name, "path": artifact_path})
            else:
                result["files_missing"].append({"field": field_name, "path": artifact_path})
        return result

    def prepare_rebuild(self, arxiv_id: str, *, loading_method: str) -> None:
        """重建前清理旧 artifact，避免本轮构建和上一轮残留数据混用。"""
        existing = self.db_service.get_paper_qa_index(arxiv_id)
        if existing:
            self.db_service.update_paper_qa_index(
                arxiv_id,
                status="processing",
                loading_method=loading_method,
                current_stage="cleanup_old_artifacts",
                artifact_status="cleanup_in_progress",
                error_message="",
                failed_stage="",
            )
            cleanup_result = self.cleanup_qa_index_artifacts(arxiv_id, existing)
            self._log_stage(
                "cleanup_old_artifacts",
                arxiv_id,
                loading_method,
                "old QA artifacts cleaned before rebuild",
                collection_deleted=cleanup_result.get("collection_deleted"),
                files_deleted=len(cleanup_result.get("files_deleted") or []),
                files_missing=len(cleanup_result.get("files_missing") or []),
            )

    def record_index_stage(
        self,
        arxiv_id: str,
        *,
        current_stage: str,
        loading_method: str,
        status: str = "processing",
        **artifacts: Any,
    ) -> None:
        # 每个关键阶段都同步数据库状态，避免进程在文件/向量库写入后崩溃却没有可追踪记录。
        payload = {
            "status": status,
            "current_stage": current_stage,
            "loading_method": loading_method,
            "artifact_status": "active",
        }
        payload.update({key: value for key, value in artifacts.items() if key in QA_INDEX_ARTIFACT_FIELDS})
        updated = self.db_service.update_paper_qa_index(arxiv_id, **payload)
        if not updated:
            inserted = self.db_service.insert_paper_qa_index(arxiv_id, **payload)
            if not inserted:
                raise AppError(
                    ErrorCode.DATABASE_WRITE_FAILED,
                    detail=f"Failed to record QA index stage: {current_stage}",
                    context={"arxiv_id": arxiv_id, "stage": current_stage},
                )

    @staticmethod
    def _exception_detail(exc: Exception) -> str:
        """从异常对象中提取较稳定的错误详情，便于写回任务结果。"""
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
            logger.debug(
                "QA index stage=%s arxiv_id=%s loading_method=%s %s | %s",
                stage,
                arxiv_id,
                loading_method,
                message,
                extra_parts,
            )
        else:
            logger.debug(
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

    @staticmethod
    def _error_code_for_stage(stage: str) -> str:
        """按构建阶段映射错误码，保证索引失败不是一律落成 unknown。"""
        if stage in VECTOR_STORE_STAGES:
            return ErrorCode.VECTOR_STORE_ERROR
        if stage in DATABASE_WRITE_STAGES:
            return ErrorCode.DATABASE_WRITE_FAILED
        return ErrorCode.QA_INDEX_BUILD_FAILED

    def _notify_progress(
        self,
        progress_callback: Optional[Callable[..., Any]],
        *,
        current_stage: str,
        progress: int,
        message: str,
    ) -> None:
        """安全触发进度回调，避免回调异常打断主建索引流程。"""
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
        """加载论文元数据，优先查本地库，缺失时再逐级回源补齐。"""
        paper = self.db_service.get_paper(arxiv_id)
        if not paper:
            logger.debug("Paper metadata missing in primary database, trying local OAI database: %s", arxiv_id)
            paper = self._fetch_and_store_paper_metadata_from_oai(arxiv_id)
        if not paper:
            # 索引链路依赖论文元数据；本地两个库都没有时，才回源 arXiv 补齐。
            logger.debug("Paper metadata missing in local databases, trying arXiv lookup: %s", arxiv_id)
            paper = self._fetch_and_store_paper_metadata(arxiv_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found in database")
        return paper

    @staticmethod
    def _normalize_oai_paper(arxiv_id: str, paper: Dict[str, Any]) -> Dict[str, Any]:
        """把 OAI 数据源的论文结构规范化为系统内部统一字段格式。"""
        return {
            "arxiv_id": str(paper.get("arxiv_id") or arxiv_id).strip(),
            "title": str(paper.get("title") or "").strip(),
            "authors": paper.get("authors") or [],
            "abstract": str(paper.get("abstract") or "").strip(),
            "categories": paper.get("categories") or [],
            "published_date": str(
                paper.get("created")
                or paper.get("updated")
                or paper.get("oai_datestamp")
                or ""
            ).strip(),
            "url": str(paper.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}").strip(),
            "embedding_id": "",
            "embedding_model": "",
        }

    def _persist_normalized_paper_metadata(self, arxiv_id: str, paper: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """持久化规范化后的论文元数据，并返回数据库中的最终记录。"""
        if not paper.get("title") or not paper.get("abstract"):
            logger.warning(
                "%s lookup returned incomplete metadata for %s: title=%s abstract=%s",
                source,
                arxiv_id,
                bool(paper.get("title")),
                bool(paper.get("abstract")),
            )
            return None

        if not self.db_service.add_paper(paper):
            logger.error("Failed to persist %s metadata into database: %s", source, arxiv_id)
            return None

        logger.debug("%s metadata stored for: %s", source, arxiv_id)
        return self.db_service.get_paper(arxiv_id) or paper

    def _fetch_and_store_paper_metadata_from_oai(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """从本地 OAI 数据库获取论文元数据，并在成功时写回主数据库。"""
        if self.oai_db_service is None:
            return None

        try:
            paper = self.oai_db_service.get_paper(arxiv_id)
            if not isinstance(paper, dict):
                logger.debug("Local OAI database returned no paper metadata: %s", arxiv_id)
                return None

            normalized_paper = self._normalize_oai_paper(arxiv_id, paper)
            return self._persist_normalized_paper_metadata(arxiv_id, normalized_paper, source="Local OAI")
        except Exception as exc:
            logger.exception("Failed to fetch local OAI metadata for %s: %s", arxiv_id, exc)
            return None

    def _fetch_and_store_paper_metadata(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """从 arXiv 接口回源获取论文元数据，并在成功时写回主数据库。"""
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
            return self._persist_normalized_paper_metadata(arxiv_id, normalized_paper, source="ArXiv")
        except Exception as exc:
            logger.exception("Failed to fetch arXiv metadata for %s: %s", arxiv_id, exc)
            return None

    def download_pdf(self, arxiv_id: str) -> str:
        """下载指定论文的 PDF 文件，并返回本地保存路径。"""
        if self.arxiv_service_factory is None:
            raise RuntimeError("arxiv_service_factory is required")
        arxiv_service = self.arxiv_service_factory()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        logger.info("Downloading PDF from: %s", pdf_url)
        pdf_path = arxiv_service.download_pdf(pdf_url, arxiv_id)
        logger.info("PDF downloaded to: %s", pdf_path)
        return pdf_path

    def load_pdf_document(self, pdf_path: str, loading_method: str) -> Tuple[LoadingService, Dict[str, Any], List[Dict[str, Any]]]:
        """加载 PDF 文档内容与页码映射，供后续切块与索引构建使用。"""
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
        """按指定加载方式对论文内容切块，并返回切块结果与切块策略名。"""
        if self.chunking_service_factory is None:
            raise RuntimeError("chunking_service_factory is required")
        chunking_service = self.chunking_service_factory()
        logger.info("Chunking text...")
        metadata = {"filename": f"{arxiv_id}.pdf", "loading_method": loading_method, "source": f"{arxiv_id}.pdf"}
        if loading_method == "docling":
            # Docling 结构更强，优先按章节语义切块，保留表格/图片等结构信息。
            chunked_data = chunking_service.chunk_docling(document, metadata=metadata, page_map=page_map)
            chunking_strategy = "docling_sections"
        else:
            # PyMuPDF 主要依赖标题层级切块，以保证普通 PDF 也能稳定建索引。
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
        chunk_file: str,
        embedding_file: str,
        loading_method: str,
        chunking_strategy: str,
    ) -> bool:
        return self.db_service.update_paper_qa_index(
            arxiv_id,
            collection_name=collection_name,
            status="indexed",
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            pdf_path=pdf_path,
            chunk_file=chunk_file,
            embedding_file=embedding_file,
            loading_method=loading_method,
            chunking_strategy=chunking_strategy,
            current_stage="mark_index_success",
            failed_stage="",
            error_message="",
            artifact_status="active",
            indexed_at=datetime.now().isoformat(timespec="seconds"),
        )

    def mark_index_failed(
        self,
        arxiv_id: str,
        *,
        failed_stage: str,
        error_message: str,
        loading_method: str,
        **artifacts: Any,
    ) -> bool:
        # 失败记录承担补偿线索职责：即使流程没完成，也要知道哪些文件或 collection 已经生成。
        payload = {
            "status": "failed",
            "current_stage": failed_stage,
            "failed_stage": failed_stage,
            "error_message": str(error_message or "")[:2000],
            "loading_method": loading_method,
            "artifact_status": "active",
        }
        payload.update({key: value for key, value in artifacts.items() if key in QA_INDEX_ARTIFACT_FIELDS})
        updated = self.db_service.update_paper_qa_index(arxiv_id, **payload)
        if not updated:
            updated = self.db_service.insert_paper_qa_index(arxiv_id, **payload)
        if not updated:
            logger.error(
                "Failed to persist QA index failure record: arxiv_id=%s stage=%s collection_name=%s",
                arxiv_id,
                failed_stage,
                artifacts.get("collection_name"),
            )
        return updated

    def build_qa_index(
        self,
        arxiv_id: str,
        loading_method: str = "docling",
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        logger.info("Creating QA index for paper: %s", arxiv_id)
        requested_loading_method = str(loading_method or "docling").strip().lower()
        current_stage = "validate_loading_method"
        artifact_state: Dict[str, Any] = {}
        effective_loading_method = requested_loading_method
        try:
            self._notify_progress(
                progress_callback,
                current_stage="validate_loading_method",
                progress=5,
                message="Validating loading method",
            )
            loading_method = self.validate_loading_method(requested_loading_method)
            effective_loading_method = loading_method
            self._log_stage("validate_loading_method", arxiv_id, loading_method, "loading method validated")

            current_stage = "cleanup_old_artifacts"
            self._notify_progress(
                progress_callback,
                current_stage="cleanup_old_artifacts",
                progress=8,
                message="Cleaning old QA index artifacts",
            )
            self.prepare_rebuild(arxiv_id, loading_method=loading_method)

            current_stage = "mark_index_processing"
            self._notify_progress(
                progress_callback,
                current_stage="mark_index_processing",
                progress=10,
                message="Marking QA index as processing",
            )
            self.mark_index_processing(arxiv_id, loading_method=loading_method)
            self._log_stage("mark_index_processing", arxiv_id, loading_method, "paper QA index marked as processing")

            current_stage = "load_paper_metadata"
            self._notify_progress(
                progress_callback,
                current_stage="load_paper_metadata",
                progress=15,
                message="Loading paper metadata",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
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
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            pdf_path = self.download_pdf(arxiv_id)
            artifact_state["pdf_path"] = pdf_path
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            self._log_stage("download_pdf", arxiv_id, loading_method, "pdf downloaded", pdf_path=pdf_path)

            current_stage = "load_pdf_document"
            self._notify_progress(
                progress_callback,
                current_stage="load_pdf_document",
                progress=35,
                message="Loading PDF document",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
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
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            chunked_data, chunking_strategy = self.chunk_document(arxiv_id, loading_method, document, page_map)
            chunks = chunked_data["chunks"]
            artifact_state["chunking_strategy"] = chunking_strategy
            artifact_state["chunk_count"] = len(chunks)
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
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
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            chunk_file = self.save_chunk_file(
                loading_service=loading_service,
                arxiv_id=arxiv_id,
                loading_method=loading_method,
                chunks=chunks,
                page_map=page_map,
                document=document,
                chunking_strategy=chunking_strategy,
            )
            artifact_state["chunk_file"] = chunk_file
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            self._log_stage("save_chunk_file", arxiv_id, loading_method, "chunk file saved", chunk_file=chunk_file)

            current_stage = "compress_chunks_for_rerank"
            self._notify_progress(
                progress_callback,
                current_stage="compress_chunks_for_rerank",
                progress=65,
                message="Compressing chunk text for rerank",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            chunks = self.compress_chunks_for_rerank(chunks)
            self._log_stage("compress_chunks_for_rerank", arxiv_id, loading_method, "chunk text compressed", chunk_count=len(chunks))

            current_stage = "create_chunk_embeddings"
            self._notify_progress(
                progress_callback,
                current_stage="create_chunk_embeddings",
                progress=78,
                message="Creating chunk embeddings",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            embeddings, embedding_config = self.create_chunk_embeddings(arxiv_id, chunks)
            artifact_state["embedding_model"] = embedding_config.model_name
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
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
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            embedding_file = self.save_embeddings(arxiv_id, embeddings)
            artifact_state["embedding_file"] = embedding_file
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            self._log_stage("save_embeddings", arxiv_id, loading_method, "embedding file saved", embedding_file=embedding_file)

            current_stage = "index_embeddings_to_vector_store"
            self._notify_progress(
                progress_callback,
                current_stage="index_embeddings_to_vector_store",
                progress=95,
                message="Indexing embeddings to vector store",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            index_result = self.index_embeddings_to_vector_store(embedding_file)
            collection_name = index_result.get("collection_name", "")
            artifact_state["collection_name"] = collection_name
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
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
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, **artifact_state)
            success_marked = self.mark_index_success(
                arxiv_id,
                collection_name=collection_name,
                chunk_count=len(chunks),
                embedding_model=embedding_config.model_name,
                pdf_path=pdf_path,
                chunk_file=chunk_file,
                embedding_file=embedding_file,
                loading_method=loading_method,
                chunking_strategy=chunking_strategy,
            )
            if not success_marked:
                # Milvus 已经写入时，最终成功状态必须落库；否则下一次重试需要依赖 collection_name 做补偿清理。
                raise RuntimeError("Milvus collection was created, but SQLite failed to mark QA index as indexed")
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
        except AppError as exc:
            self._tag_exception(
                exc,
                stage=str(exc.context.get("stage") or current_stage),
                arxiv_id=arxiv_id,
                loading_method=effective_loading_method,
            )
            logger.exception(
                "QA index build failed with code=%s at stage=%s arxiv_id=%s loading_method=%s detail=%s",
                exc.code,
                exc.context.get("stage") or current_stage,
                arxiv_id,
                effective_loading_method,
                exc.detail,
            )
            self.mark_index_failed(
                arxiv_id,
                failed_stage=str(exc.context.get("stage") or current_stage),
                error_message=f"{exc.code}: {exc.detail or exc.message}",
                loading_method=effective_loading_method,
                **artifact_state,
            )
            raise
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
            self.mark_index_failed(
                arxiv_id,
                failed_stage=current_stage,
                error_message=detail,
                loading_method=effective_loading_method,
                **artifact_state,
            )
            error_code = self._error_code_for_stage(current_stage)
            raise AppError(
                error_code,
                detail={"stage": current_stage, "error": detail, "arxiv_id": arxiv_id},
                context={"arxiv_id": arxiv_id, "stage": current_stage, "loading_method": effective_loading_method},
            ) from exc
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
            self.mark_index_failed(
                arxiv_id,
                failed_stage=current_stage,
                error_message=detail,
                loading_method=effective_loading_method,
                **artifact_state,
            )
            error_code = self._error_code_for_stage(current_stage)
            raise AppError(
                error_code,
                detail={"stage": current_stage, "error": detail, "arxiv_id": arxiv_id},
                context={"arxiv_id": arxiv_id, "stage": current_stage, "loading_method": effective_loading_method},
            ) from exc
