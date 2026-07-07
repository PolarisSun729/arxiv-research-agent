from __future__ import annotations

import json
import hashlib
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from core.errors import AppError, ErrorCode
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.document.chunking_service import ChunkingService
from services.document.table_structure_service import TableStructureService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.llm.generation_service import GenerationService, QWEN_RERANK_COMPRESS_MODEL_NAME
from services.paper_qa.build_cache import (
    LLM_RERANK_TEXT_PROMPT_VERSION,
    get_paper_qa_build_cache,
)
from services.document.loading_service import LoadingService
from services.retrieval.retrieval_index import (
    RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION,
    SPARSE_INDEX_ARTIFACT_SCHEMA_VERSION,
    CollectionRetrievalIndexProvider,
    DEFAULT_RETRIEVAL_INDEX_MAX_QUESTIONS_PER_CHUNK,
    build_retrieval_index_payload,
    build_retrieval_indexes,
    compute_artifact_file_hash,
    load_sparse_index_artifact_files,
    normalize_sparse_source_type,
    save_retrieval_index_artifact as persist_retrieval_index_artifact,
    save_sparse_index_artifact as persist_sparse_index_artifact,
    summarize_retrieval_indexes,
)
from services.retrieval.retrieval_rules import RetrievalRules
from services.storage.sqlite.stores.paper_catalog import PaperCatalogStore
from services.storage.sqlite.stores.paper_qa_index import PaperQAIndexStore
from services.storage.vector_store_service import VectorDBConfig, VectorStoreService
from utils.config import get_enhanced_retrieval_runtime_config
from utils.logging_utils import info_event

logger = logging.getLogger(__name__)

VECTOR_STORE_STAGES = {"index_embeddings_to_vector_store", "validate_new_collection", "cleanup_old_artifacts"}
DATABASE_WRITE_STAGES = {
    "create_build_version",
    "mark_index_processing",
    "record_index_stage",
    "activate_index",
}

QA_INDEX_ARTIFACT_FIELDS = {
    "pdf_path",
    "chunk_file",
    "retrieval_index_file",
    "retrieval_index_count",
    "retrieval_index_types",
    "retrieval_index_version",
    "sparse_index_dir",
    "sparse_index_manifest_file",
    "sparse_index_document_count",
    "sparse_index_token_count",
    "sparse_index_backend",
    "sparse_index_schema_version",
    "sparse_index_source_file",
    "sparse_index_source_hash",
    "sparse_index_avgdl",
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
        paper_qa_index_store: PaperQAIndexStore,
        paper_catalog_store: PaperCatalogStore,
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
        self.paper_qa_index_store = paper_qa_index_store
        self.paper_catalog_store = paper_catalog_store
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.arxiv_service_factory = arxiv_service_factory
        self.oai_db_service = oai_db_service
        self.get_embedding_config = get_embedding_config or self.embedding_service.get_default_embedding_config
        self.loading_service_factory = loading_service_factory
        self.chunking_service_factory = chunking_service_factory
        self.retrieval_runtime_config = get_enhanced_retrieval_runtime_config()
        self.build_cache = get_paper_qa_build_cache()
        cache_config = getattr(self.build_cache, "config", {}) or {}
        self.llm_max_workers = max(1, int(cache_config.get("llm_max_workers") or 1))

    @staticmethod
    def _chunk_type_counts(chunks: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """统计 chunk 中的文本、图片和表格数量，便于记录索引摘要。"""
        text_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "text")) == "text")
        figure_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "figure")
        table_count = sum(1 for chunk in chunks if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")) == "table")
        return text_count, figure_count, table_count

    @staticmethod
    def _retrieval_index_type_list(retrieval_index_debug: Dict[str, Any]) -> List[str]:
        """把类型分布压成稳定列表，便于数据库记录和前端状态展示复用。"""
        type_counts = dict((retrieval_index_debug or {}).get("index_type_counts") or {})
        return sorted(str(index_type) for index_type in type_counts if str(index_type).strip())

    def validate_loading_method(self, loading_method: str) -> str:
        """校验并规范化 PDF 加载方式，只允许当前支持的方法。"""
        normalized_method = str(loading_method or "pymupdf").strip().lower()
        if normalized_method not in {"pymupdf", "docling"}:
            raise HTTPException(status_code=400, detail="loading_method must be either pymupdf or docling")
        return normalized_method

    def mark_index_processing(self, arxiv_id: str, *, loading_method: str) -> None:
        """把论文索引状态标记为处理中，供外部轮询和后台任务联动使用。"""
        created = self.paper_qa_index_store.insert_paper_qa_index(
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

    def mark_build_processing(self, build_id: str, *, loading_method: str) -> None:
        """把本次版本化构建标记为 building；active 指针保持不变，旧索引仍可服务问答。"""
        updated = self.paper_qa_index_store.update_paper_qa_index_build(
            build_id,
            status="building",
            loading_method=loading_method,
            current_stage="mark_index_processing",
            failed_stage="",
            error_message="",
            artifact_status="active",
        )
        if not updated:
            raise RuntimeError("Failed to mark QA index build as processing")

    def _workspace_root(self) -> Path:
        # QA artifact 目录（01-loaded-docs/02-retrieval-indexes 等）都落在 backend 下；清理相对路径时必须以 backend 为安全边界。
        return Path(__file__).resolve().parents[2]

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

    def _safe_delete_directory(self, dir_path: Any) -> bool:
        path_text = str(dir_path or "").strip()
        if not path_text:
            return False
        path = Path(path_text)
        if not path.is_absolute():
            path = self._workspace_root() / path
        try:
            resolved_path = path.resolve()
            workspace_root = self._workspace_root().resolve()
            if workspace_root not in resolved_path.parents and resolved_path != workspace_root:
                logger.warning("Skip deleting QA artifact directory outside workspace: %s", resolved_path)
                return False
            if not resolved_path.exists():
                return False
            if not resolved_path.is_dir():
                logger.warning("Skip deleting non-directory QA artifact: %s", resolved_path)
                return False
            # sparse index 是目录型 artifact；递归删除前只允许工作区内已解析路径，避免误删外部文件。
            shutil.rmtree(resolved_path)
            return True
        except Exception:
            logger.exception("Failed to delete QA artifact directory: %s", path_text)
            raise

    def _resolve_cleanup_path(self, path_text: Any) -> Optional[Path]:
        raw_text = str(path_text or "").strip()
        if not raw_text:
            return None
        path = Path(raw_text)
        if not path.is_absolute():
            path = self._workspace_root() / path
        try:
            return path.resolve()
        except OSError:
            return None

    def _active_artifact_guard(self, arxiv_id: str, *, allow_active: bool) -> Dict[str, Any]:
        if allow_active:
            return {"build_id": "", "collection_name": "", "paths": []}
        active = self.paper_qa_index_store.get_active_paper_qa_index_build(arxiv_id) or {}
        protected_paths = []
        for field_name in ("pdf_path", "chunk_file", "retrieval_index_file", "embedding_file", "sparse_index_dir", "sparse_index_manifest_file"):
            resolved = self._resolve_cleanup_path(active.get(field_name))
            if resolved is not None:
                protected_paths.append(resolved)
        return {
            "build_id": str(active.get("build_id") or ""),
            "collection_name": str(active.get("collection_name") or "").strip(),
            "paths": protected_paths,
        }

    def _is_active_artifact_path(self, path_text: Any, active_guard: Dict[str, Any]) -> bool:
        candidate = self._resolve_cleanup_path(path_text)
        if candidate is None:
            return False
        for protected in active_guard.get("paths") or []:
            # 清理目录时要防止删到 active 文件的父目录；清理文件时也不能删 active 目录下的成员。
            if candidate == protected or candidate in protected.parents or protected in candidate.parents:
                return True
        return False

    def cleanup_qa_index_artifacts(
        self,
        arxiv_id: str,
        qa_index: Optional[Dict[str, Any]] = None,
        *,
        allow_active: bool = False,
    ) -> Dict[str, Any]:
        """清理指定 QA 索引版本留下的文件和 Milvus collection。

        默认用于 cleanup_pending 旧版本清理，并防御性保护当前 active build；
        只有用户显式删除 QA 索引时，调用方才传入 allow_active=True。
        """
        existing = qa_index if qa_index is not None else self.paper_qa_index_store.get_paper_qa_index(arxiv_id)
        result = {"collection_deleted": False, "collection_skipped_active": False, "files_deleted": [], "files_missing": [], "files_skipped_active": [], "skipped_active_build": False}
        if not existing:
            return result
        active_guard = self._active_artifact_guard(arxiv_id, allow_active=allow_active)
        existing_build_id = str(existing.get("build_id") or existing.get("active_build_id") or "").strip()
        if not allow_active and existing_build_id and existing_build_id == active_guard.get("build_id"):
            # cleanup_pending 流程只清理旧版本；若输入误指向 active build，直接跳过整个版本。
            result["skipped_active_build"] = True
            return result

        collection_name = str(existing.get("collection_name") or "").strip()
        if collection_name:
            if not allow_active and collection_name == active_guard.get("collection_name"):
                result["collection_skipped_active"] = True
            else:
                # 只删除非 active collection，避免旧版本清理误切断当前问答路径。
                result["collection_deleted"] = self.vector_store_service.delete_collection("milvus", collection_name)

        for field_name in ("pdf_path", "chunk_file", "retrieval_index_file", "embedding_file"):
            artifact_path = str(existing.get(field_name) or "").strip()
            if not artifact_path:
                continue
            if self._is_active_artifact_path(artifact_path, active_guard):
                result["files_skipped_active"].append({"field": field_name, "path": artifact_path})
                continue
            deleted = self._safe_delete_file(artifact_path)
            if deleted:
                result["files_deleted"].append({"field": field_name, "path": artifact_path})
            else:
                result["files_missing"].append({"field": field_name, "path": artifact_path})
        sparse_dir = str(existing.get("sparse_index_dir") or "").strip()
        if sparse_dir:
            if self._is_active_artifact_path(sparse_dir, active_guard):
                result["files_skipped_active"].append({"field": "sparse_index_dir", "path": sparse_dir})
                return result
            deleted = self._safe_delete_directory(sparse_dir)
            if deleted:
                result["files_deleted"].append({"field": "sparse_index_dir", "path": sparse_dir})
            else:
                result["files_missing"].append({"field": "sparse_index_dir", "path": sparse_dir})
        elif str(existing.get("sparse_index_manifest_file") or "").strip():
            manifest_path = str(existing.get("sparse_index_manifest_file") or "").strip()
            if self._is_active_artifact_path(manifest_path, active_guard):
                result["files_skipped_active"].append({"field": "sparse_index_manifest_file", "path": manifest_path})
                return result
            deleted = self._safe_delete_file(manifest_path)
            if deleted:
                result["files_deleted"].append({"field": "sparse_index_manifest_file", "path": manifest_path})
            else:
                result["files_missing"].append({"field": "sparse_index_manifest_file", "path": manifest_path})
        return result

    def cleanup_pending_index_builds(self, arxiv_id: str, *, limit: int = 5) -> Dict[str, Any]:
        """延迟清理 cleanup_pending 版本；清理失败只进入结果，不影响 active 索引问答。"""
        active = self.paper_qa_index_store.get_active_paper_qa_index_build(arxiv_id)
        active_build_id = active.get("build_id") if active else None
        candidates = self.paper_qa_index_store.list_paper_qa_index_builds(
            arxiv_id,
            statuses=["cleanup_pending", "orphaned", "build_failed"],
            limit=limit,
        )
        result = {"cleaned": [], "failed": [], "skipped_active": []}
        for build in candidates:
            build_id = build.get("build_id")
            if build_id == active_build_id:
                # 防御性跳过：清理流程绝不能误删当前线上 collection。
                result["skipped_active"].append(build_id)
                continue
            try:
                cleanup_result = self.cleanup_qa_index_artifacts(arxiv_id, build)
                marked = self.paper_qa_index_store.mark_paper_qa_index_build_deleted(build_id)
                result["cleaned"].append({"build_id": build_id, "cleanup": cleanup_result, "marked_deleted": marked})
            except Exception as exc:
                logger.exception("Failed to cleanup QA index build: arxiv_id=%s build_id=%s", arxiv_id, build_id)
                result["failed"].append({"build_id": build_id, "error": str(exc)})
        return result

    def prepare_rebuild(self, arxiv_id: str, *, loading_method: str) -> None:
        """兼容旧调用的重建准备步骤。

        主构建链路不再在这里删除旧索引；旧 active 必须保留到新版本激活成功之后，后续由清理流程处理。
        """
        existing = self.paper_qa_index_store.get_paper_qa_index(arxiv_id)
        if existing:
            self.paper_qa_index_store.update_paper_qa_index(
                arxiv_id,
                loading_method=loading_method,
                current_stage="prepare_versioned_rebuild",
                error_message="",
                failed_stage="",
            )
            self._log_stage(
                "prepare_versioned_rebuild",
                arxiv_id,
                loading_method,
                "old QA artifacts kept until new version is activated",
                active_collection=existing.get("collection_name"),
            )

    def record_index_stage(
        self,
        arxiv_id: str,
        *,
        current_stage: str,
        loading_method: str,
        status: str = "processing",
        build_id: Optional[str] = None,
        **artifacts: Any,
    ) -> None:
        # 每个关键阶段都同步数据库状态，避免进程在文件/向量库写入后崩溃却没有可追踪记录。
        payload = {
            "status": "building" if status == "processing" else status,
            "current_stage": current_stage,
            "loading_method": loading_method,
            "artifact_status": "active",
        }
        payload.update({key: value for key, value in artifacts.items() if key in QA_INDEX_ARTIFACT_FIELDS})
        if build_id:
            updated = self.paper_qa_index_store.update_paper_qa_index_build(build_id, **payload)
            if not updated:
                raise AppError(
                    ErrorCode.DATABASE_WRITE_FAILED,
                    detail=f"Failed to record QA index build stage: {current_stage}",
                    context={"arxiv_id": arxiv_id, "stage": current_stage, "build_id": build_id},
                )
            return
        updated = self.paper_qa_index_store.update_paper_qa_index(arxiv_id, **payload)
        if not updated:
            inserted = self.paper_qa_index_store.insert_paper_qa_index(arxiv_id, **payload)
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
        info_event(
            logger,
            "qa.index_stage_done",
            stage=stage,
            arxiv_id=arxiv_id,
            loading_method=loading_method,
            message=message,
            **extra,
        )
        if extra_parts:
            logger.debug("QA index stage=%s arxiv_id=%s loading_method=%s %s | %s", stage, arxiv_id, loading_method, message, extra_parts)
        else:
            logger.debug("QA index stage=%s arxiv_id=%s loading_method=%s %s", stage, arxiv_id, loading_method, message)

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
        paper = self.paper_catalog_store.get_paper(arxiv_id)
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

        if not self.paper_catalog_store.add_paper(paper):
            logger.error("Failed to persist %s metadata into database: %s", source, arxiv_id)
            return None

        logger.debug("%s metadata stored for: %s", source, arxiv_id)
        return self.paper_catalog_store.get_paper(arxiv_id) or paper

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
        logger.debug("Downloading PDF from: %s", pdf_url)
        pdf_path = arxiv_service.download_pdf(pdf_url, arxiv_id)
        logger.debug("PDF downloaded to: %s", pdf_path)
        return pdf_path

    def load_pdf_document(self, pdf_path: str, loading_method: str) -> Tuple[LoadingService, Dict[str, Any], List[Dict[str, Any]]]:
        """加载 PDF 文档内容与页码映射，供后续切块与索引构建使用。"""
        if self.loading_service_factory is None:
            raise RuntimeError("loading_service_factory is required")
        loading_service = self.loading_service_factory()
        logger.debug("Loading PDF content...")
        document = loading_service.load_pdf(pdf_path, method=loading_method)
        page_map = loading_service.get_page_map()
        logger.debug("Loaded %s pages from PDF", len(page_map))
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
        logger.debug("Chunking text...")
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
        logger.debug("Created %s chunks", len(chunks))
        text_count, figure_count, table_count = self._chunk_type_counts(chunks)
        logger.debug(
            "Chunk composition: text=%d figure=%d table=%d",
            text_count,
            figure_count,
            table_count,
        )
        return chunked_data, chunking_strategy

    def structure_table_chunks(
        self,
        arxiv_id: str,
        loading_method: str,
        document: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """为 Docling table chunk 生成结构化表格对象，并把结果挂回文档产物。"""
        if loading_method != "docling":
            debug = {
                "table_count": self._chunk_type_counts(chunks)[2],
                "structured_table_count": 0,
                "failed_table_parse_count": 0,
                "docling_table_item_count": 0,
            }
            document["structured_tables"] = []
            document["table_structure_debug"] = debug
            return {"structured_tables": [], "debug": debug}

        # 结构化表格是索引构建阶段的旁路证据：失败只影响 table index 调试信息，
        # 不允许阻断 chunk 保存、embedding 和原有文本/图片/表格召回链路。
        try:
            service = TableStructureService(workspace_root=str(self._workspace_root()))
            result = service.build_structured_tables(arxiv_id=arxiv_id, document=document, chunks=chunks)
        except Exception as exc:
            table_count = self._chunk_type_counts(chunks)[2]
            logger.exception("Failed to build structured table index for %s: %s", arxiv_id, exc)
            result = {
                "structured_tables": [],
                "debug": {
                    "table_count": table_count,
                    "structured_table_count": 0,
                    "failed_table_parse_count": table_count,
                    "docling_table_item_count": 0,
                    "error": str(exc),
                },
            }

        document["structured_tables"] = list(result.get("structured_tables") or [])
        document["table_structure_debug"] = dict(result.get("debug") or {})
        return result

    def save_chunk_file(
        self,
        loading_service: LoadingService,
        arxiv_id: str,
        loading_method: str,
        chunks: List[Dict[str, Any]],
        page_map: List[Dict[str, Any]],
        document: Dict[str, Any],
        chunking_strategy: str,
        index_version: Optional[str] = None,
    ) -> str:
        filename = f"{arxiv_id}_{index_version}.pdf" if index_version else f"{arxiv_id}.pdf"
        retrieval_indexes = build_retrieval_indexes(chunks)
        document_for_save = dict(document or {})
        # retrieval index 是 chunk 之上的派生层，写入调试产物时不改变原 chunks 列表本身。
        document_for_save["retrieval_indexes"] = retrieval_indexes
        document_for_save["retrieval_index_debug"] = summarize_retrieval_indexes(retrieval_indexes)
        chunk_file = loading_service.save_document(
            filename=filename,
            chunks=chunks,
            metadata={"total_pages": len(page_map)},
            loading_method=loading_method,
            chunking_strategy=chunking_strategy,
            document_data=document_for_save,
        )
        logger.debug("Chunked document saved to: %s", chunk_file)
        return chunk_file

    def compress_chunks_for_rerank(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.debug(
            "Compressing chunk text for rerank with Qwen: chunk_count=%s max_workers=%s cache_enabled=%s",
            len(chunks),
            self.llm_max_workers,
            getattr(self.build_cache, "enabled", False),
        )

        def compress_with_service(chunk: Dict[str, Any], raw_content: str, metadata: Dict[str, Any]) -> str:
            single_compressor = getattr(self.generation_service, "compress_chunk_for_rerank", None)
            if callable(single_compressor):
                return single_compressor(
                    chunk_text=raw_content,
                    chunk_metadata=metadata,
                    model_name=QWEN_RERANK_COMPRESS_MODEL_NAME,
                )
            batch_compressor = getattr(self.generation_service, "compress_chunks_for_rerank", None)
            if callable(batch_compressor):
                # 测试替身和旧调用方可能只实现批量接口；这里用单元素批量调用保留兼容性。
                compressed = batch_compressor(chunks=[chunk], model_name=QWEN_RERANK_COMPRESS_MODEL_NAME)
                if compressed:
                    compressed_chunk = dict(compressed[0] or {})
                    compressed_metadata = dict(compressed_chunk.get("metadata", {}) or {})
                    return str(compressed_chunk.get("rerank_text") or compressed_metadata.get("rerank_text") or "").strip()
            raise AttributeError("generation_service must provide compress_chunk_for_rerank or compress_chunks_for_rerank")

        def compress_one(position: int, chunk: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            updated_chunk = dict(chunk)
            metadata = dict(updated_chunk.get("metadata", {}) or {})
            raw_content = str(updated_chunk.get("content", "") or "")
            cache_kwargs = {
                "kind": "rerank_text",
                "model_name": QWEN_RERANK_COMPRESS_MODEL_NAME,
                "prompt_version": LLM_RERANK_TEXT_PROMPT_VERSION,
                "chunk_text": raw_content,
                "metadata": metadata,
            }
            cached = self.build_cache.get_llm_result(**cache_kwargs)
            if isinstance(cached, str) and cached.strip():
                rerank_text = cached.strip()
                cache_hit = True
            else:
                # 每个 chunk 的压缩互不依赖，未命中缓存时才进入受限并发，避免重复重建反复消耗 LLM 调用。
                rerank_text = compress_with_service(updated_chunk, raw_content, metadata)
                self.build_cache.set_llm_result(rerank_text, **cache_kwargs)
                cache_hit = False

            metadata["rerank_text"] = rerank_text
            metadata["rerank_text_model"] = QWEN_RERANK_COMPRESS_MODEL_NAME
            metadata["rerank_text_generated_at"] = datetime.now().isoformat()
            metadata["rerank_text_cache_hit"] = cache_hit
            updated_chunk["metadata"] = metadata
            updated_chunk["rerank_text"] = rerank_text
            return position, updated_chunk

        if self.llm_max_workers <= 1 or len(chunks) <= 1:
            compressed_pairs = [compress_one(index, chunk) for index, chunk in enumerate(chunks)]
        else:
            with ThreadPoolExecutor(max_workers=self.llm_max_workers, thread_name_prefix="qa-rerank-compress") as executor:
                compressed_pairs = list(executor.map(lambda item: compress_one(item[0], item[1]), enumerate(chunks)))

        compressed_chunks = [chunk for _, chunk in sorted(compressed_pairs, key=lambda item: item[0])]
        logger.debug("Generated rerank_text for %d chunks", len(compressed_chunks))
        return compressed_chunks

    def build_retrieval_indexes(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """显式生成 chunk 之上的 retrieval index 层，失败时保留规则索引继续建库。"""
        enable_generated_question_index = bool(self.retrieval_runtime_config.get("enable_generated_question_index", True))
        max_questions_per_chunk = (
            DEFAULT_RETRIEVAL_INDEX_MAX_QUESTIONS_PER_CHUNK if enable_generated_question_index else 0
        )
        generation_model_name = self._retrieval_index_generation_model_name()
        # 生成式 summary/question 只是召回增强；失败时 builder 会回落到规则索引，不能阻断整篇论文建库。
        retrieval_indexes, retrieval_index_debug = build_retrieval_index_payload(
            chunks,
            generation_service=self.generation_service,
            enable_generative_indexes=True,
            max_questions_per_chunk=max_questions_per_chunk,
            max_workers=self.llm_max_workers,
            generation_cache=self.build_cache,
            generation_model_name=generation_model_name,
        )
        retrieval_index_debug["generated_question_index_enabled"] = enable_generated_question_index
        logger.debug(
            "Built retrieval indexes: count=%s type_counts=%s generation_errors=%s",
            retrieval_index_debug.get("index_count"),
            retrieval_index_debug.get("index_type_counts"),
            retrieval_index_debug.get("generation_error_count"),
        )
        return retrieval_indexes, retrieval_index_debug

    def _retrieval_index_generation_model_name(self) -> str:
        resolver = getattr(self.generation_service, "_resolve_qwen_model_selection", None)
        if not callable(resolver):
            return "retrieval_index_generation"
        try:
            selection = resolver(task_type="retrieval_index_generation", default_role="large")
            selected_model = str((selection or {}).get("selected_model") or "").strip()
            return selected_model or "retrieval_index_generation"
        except Exception as exc:
            # 模型名只影响缓存隔离；解析失败时降级到任务名，不能阻断建库主流程。
            logger.warning("Resolve retrieval index generation model for cache key failed: %s", exc)
            return "retrieval_index_generation"

    def save_retrieval_index_artifact(
        self,
        arxiv_id: str,
        chunks: List[Dict[str, Any]],
        retrieval_indexes: List[Dict[str, Any]],
        retrieval_index_debug: Dict[str, Any],
        *,
        index_version: Optional[str] = None,
    ) -> str:
        """把 RetrievalIndex 独立落盘，供后续重建 embedding、BM25 和 debug trace 使用。"""
        artifact_file = persist_retrieval_index_artifact(
            paper_id=arxiv_id,
            retrieval_indexes=retrieval_indexes,
            retrieval_index_debug=retrieval_index_debug,
            chunks=chunks,
            index_version=index_version,
        )
        logger.debug("Retrieval index artifact saved to: %s", artifact_file)
        return artifact_file

    def save_sparse_index_artifact(
        self,
        arxiv_id: str,
        *,
        build_id: str,
        chunks: List[Dict[str, Any]],
        chunk_file: str,
        chunk_count: int,
        index_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """基于 chunk file 生成 chunk-level sparse index，作为第一版低风险 BM25 构建产物。"""
        sparse_rows = self._load_sparse_source_chunks(chunk_file, fallback_chunks=chunks)

        class _SparseBuildVectorStore:
            def __init__(self, rows: List[Dict[str, Any]]) -> None:
                self.rows = [dict(row) for row in rows]

            def get_all_chunks(self, _collection_name: str):
                return [dict(row) for row in self.rows]

        retrieval_rules = RetrievalRules(config=self.retrieval_runtime_config)
        provider = CollectionRetrievalIndexProvider(
            vector_store_service=_SparseBuildVectorStore(sparse_rows),
            chunk_normalizer=retrieval_rules.normalize_chunk,
            tokenizer=retrieval_rules.tokenize_for_keyword_search,
        )
        version = str(index_version or RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION)
        source_hash = self._compute_sparse_source_hash(chunk_file, sparse_rows)
        index_record = {
            "arxiv_id": arxiv_id,
            "chunk_count": int(chunk_count or 0),
            "active_index_version": version,
            "active_build_id": build_id,
        }
        collection_name = f"sparse_artifact_{str(arxiv_id or 'paper').replace('.', '_').replace('/', '_')}_{version}"
        sparse_index = provider.get_index(
            collection_name,
            index_record=index_record,
            force_refresh=True,
        )
        if not sparse_index.documents:
            # sparse artifact 必须有可查询文档；为空说明 chunk 文件无法支撑 keyword route，不能激活新版本。
            raise RuntimeError(
                "Sparse index artifact build failed: source="
                f"{sparse_index.build_source} documents={len(sparse_index.documents)}"
            )
        # 第一版 sparse artifact 明确以 chunk 文件为源，降低 retrieval-index-level 迁移风险。
        sparse_index.build_source = "chunk"
        sparse_index.fallback_reason = ""
        sparse_index.retrieval_index_artifact_debug = {
            "source": "chunk",
            "chunk_file": chunk_file,
            "row_count": len(sparse_rows),
            "document_count": len(sparse_index.documents),
            "reason": "",
        }
        artifact = persist_sparse_index_artifact(
            index=sparse_index,
            paper_id=arxiv_id,
            build_id=build_id,
            index_version=version,
            source_type="chunk",
            source_file=chunk_file,
            source_hash=source_hash,
            backend="internal_bm25",
        )
        logger.debug(
            "Chunk-level sparse index artifact saved to: %s document_count=%s token_count=%s source_hash=%s",
            artifact.get("manifest_file"),
            artifact.get("document_count"),
            artifact.get("token_count"),
            artifact.get("source_hash"),
        )
        return artifact

    @staticmethod
    def _compute_sparse_source_hash(chunk_file: str, sparse_rows: List[Dict[str, Any]]) -> str:
        resolved = Path(str(chunk_file or "").strip())
        candidates = [resolved]
        if not resolved.is_absolute():
            workspace_root = Path(__file__).resolve().parents[3]
            candidates.extend([Path.cwd() / resolved, workspace_root / resolved])
        for candidate in candidates:
            try:
                path = candidate.resolve()
            except OSError:
                continue
            if path.is_file():
                return compute_artifact_file_hash(str(path))
        # 生产路径必须以真实 chunk file 为 source；测试 stub 可能只返回文件名不落盘，
        # 此时用同一批 sparse 输入生成稳定 hash，避免把轻量测试误判为构建失败。
        payload = json.dumps({"chunks": sparse_rows}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_sparse_source_chunks(chunk_file: str, *, fallback_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从 chunk file 恢复 sparse 输入；测试 stub 未落盘时才使用内存 chunk 兜底。"""
        resolved = Path(str(chunk_file or "").strip())
        candidates = [resolved]
        if not resolved.is_absolute():
            workspace_root = Path(__file__).resolve().parents[3]
            candidates.extend([Path.cwd() / resolved, workspace_root / resolved])
        for candidate in candidates:
            try:
                path = candidate.resolve()
            except OSError:
                continue
            if not path.is_file():
                continue
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            chunks = payload.get("chunks") if isinstance(payload, dict) else []
            if not isinstance(chunks, list):
                raise RuntimeError(f"Chunk file missing chunks list: {chunk_file}")
            return [dict(chunk) for chunk in chunks if isinstance(chunk, dict)]
        if fallback_chunks:
            # 兼容旧测试和轻量 stub：生产环境应以真实 chunk file 为准，兜底只保证构建链路可验证。
            return [dict(chunk) for chunk in fallback_chunks if isinstance(chunk, dict)]
        raise FileNotFoundError(str(chunk_file or ""))

    def create_chunk_embeddings(
        self,
        arxiv_id: str,
        chunks: List[Dict[str, Any]],
        retrieval_indexes: Optional[List[Dict[str, Any]]] = None,
        retrieval_index_debug: Optional[Dict[str, Any]] = None,
        index_version: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], EmbeddingConfig]:
        embedding_config = self.get_embedding_config()
        logger.debug(
            "Creating embeddings with %s / %s...",
            embedding_config.provider,
            embedding_config.model_name,
        )

        filename = f"{arxiv_id}_{index_version}.pdf" if index_version else f"{arxiv_id}.pdf"
        if retrieval_indexes is None:
            # 兼容旧调用入口：没有显式阶段产物时仍能基于 chunk 派生规则索引。
            retrieval_indexes, retrieval_index_debug = build_retrieval_index_payload(chunks)
        elif retrieval_index_debug is None:
            retrieval_index_debug = summarize_retrieval_indexes(retrieval_indexes)
        input_data = {
            "chunks": chunks,
            "retrieval_indexes": retrieval_indexes,
            "metadata": {
                "filename": filename,
                "embedding_mode": (
                    "retrieval_index"
                    if self.retrieval_runtime_config.get("enable_multi_index_embedding", True)
                    else "legacy_chunk_level"
                ),
                "enable_multi_index_embedding": bool(
                    self.retrieval_runtime_config.get("enable_multi_index_embedding", True)
                ),
                "enable_chunk_level_retrieval_fallback": bool(
                    self.retrieval_runtime_config.get("enable_chunk_level_retrieval_fallback", True)
                ),
            },
        }
        embeddings, _ = self.embedding_service.create_embeddings(input_data, embedding_config)
        if not embeddings:
            raise RuntimeError("No valid retrieval indexes produced embeddings")

        # 一个 PaperChunk 会派生多个 RetrievalIndex，因此 embedding 数量现在表示可检索入口数。
        logger.debug(
            "Created %d retrieval-index embeddings: %s",
            len(embeddings),
            retrieval_index_debug,
        )
        text_count, figure_count, table_count = self._chunk_type_counts(embeddings)
        logger.debug(
            "Embedding composition: text=%d figure=%d table=%d",
            text_count,
            figure_count,
            table_count,
        )
        return embeddings, embedding_config

    def save_embeddings(self, arxiv_id: str, embeddings: List[Dict[str, Any]], index_version: Optional[str] = None) -> str:
        filename = f"{arxiv_id}_{index_version}.pdf" if index_version else f"{arxiv_id}.pdf"
        embedding_file = self.embedding_service.save_embeddings(filename, embeddings)
        logger.debug("Embeddings saved to: %s", embedding_file)
        return embedding_file

    def index_embeddings_to_vector_store(self, embedding_file: str) -> Dict[str, Any]:
        vector_db_config = VectorDBConfig(provider="milvus", index_mode="default")
        index_result = self.vector_store_service.index_embeddings(embedding_file, vector_db_config)
        collection_name = index_result.get("collection_name", "")
        logger.debug("Index created in collection: %s", collection_name)
        return index_result

    def validate_new_collection(self, collection_name: str, expected_count: int) -> None:
        """激活前校验新 collection 至少存在；校验失败时旧 active 仍不会被切走。"""
        normalized_collection = str(collection_name or "").strip()
        if not normalized_collection:
            raise RuntimeError("Vector store returned empty collection_name")
        checker = getattr(self.vector_store_service, "collection_exists", None)
        if callable(checker) and not checker("milvus", normalized_collection):
            raise RuntimeError(f"New QA index collection does not exist: {normalized_collection}")
        if expected_count <= 0:
            raise RuntimeError("New QA index has no chunks to activate")

    def validate_sparse_index_artifact(
        self,
        *,
        build_id: str,
        index_version: str,
        sparse_index_manifest_file: str,
        sparse_index_source_hash: str,
        sparse_index_document_count: int,
        sparse_index_token_count: int,
        sparse_index_backend: str,
    ) -> Dict[str, Any]:
        """激活前校验 sparse artifact 与当前 build 一致，避免 dense/sparse 指向不同版本。"""
        manifest_file = str(sparse_index_manifest_file or "").strip()
        if not manifest_file:
            raise RuntimeError("Sparse index manifest is missing")
        payload = load_sparse_index_artifact_files(manifest_file)
        manifest = dict(payload.get("manifest") or {})
        source_type = normalize_sparse_source_type(manifest.get("source_type"))
        if source_type not in {"chunk", "retrieval_index"}:
            raise RuntimeError(f"Unsupported sparse index source_type: {manifest.get('source_type')}")
        checks = {
            "build_id": (str(build_id or ""), str(manifest.get("build_id") or "")),
            "index_version": (str(index_version or ""), str(manifest.get("index_version") or "")),
            "source_hash": (str(sparse_index_source_hash or ""), str(manifest.get("source_hash") or "")),
            "backend": (str(sparse_index_backend or ""), str(manifest.get("backend") or "")),
        }
        for field_name, (expected, actual) in checks.items():
            if expected and actual and expected != actual:
                raise RuntimeError(f"Sparse index {field_name} mismatch: expected={expected} actual={actual}")
            if expected and not actual:
                raise RuntimeError(f"Sparse index {field_name} missing in manifest")
        documents = list(payload.get("documents") or [])
        manifest_document_count = int(manifest.get("document_count", 0) or 0)
        if manifest_document_count != len(documents):
            raise RuntimeError("Sparse index document_count mismatch between manifest and documents")
        if int(sparse_index_document_count or 0) <= 0 or int(sparse_index_document_count or 0) != manifest_document_count:
            raise RuntimeError("Sparse index document_count does not match build record")
        manifest_token_count = int(manifest.get("token_count", 0) or 0)
        if int(sparse_index_token_count or 0) > 0 and int(sparse_index_token_count or 0) != manifest_token_count:
            raise RuntimeError("Sparse index token_count does not match build record")
        return {
            "source_type": source_type,
            "document_count": manifest_document_count,
            "token_count": manifest_token_count,
            "backend": str(manifest.get("backend") or ""),
        }

    def mark_index_success(
        self,
        arxiv_id: str,
        *,
        build_id: Optional[str] = None,
        collection_name: str,
        chunk_count: int,
        embedding_model: str,
        pdf_path: str,
        chunk_file: str,
        retrieval_index_file: str,
        retrieval_index_count: int,
        retrieval_index_types: str,
        retrieval_index_version: str,
        embedding_file: str,
        loading_method: str,
        chunking_strategy: str,
        sparse_index_dir: str = "",
        sparse_index_manifest_file: str = "",
        sparse_index_document_count: int = 0,
        sparse_index_token_count: int = 0,
        sparse_index_backend: str = "",
        sparse_index_schema_version: str = "",
        sparse_index_source_file: str = "",
        sparse_index_source_hash: str = "",
        sparse_index_avgdl: float = 0.0,
    ) -> bool:
        if build_id and (not str(sparse_index_manifest_file or "").strip() or int(sparse_index_document_count or 0) <= 0):
            # 新版 QA index 必须 dense/sparse 同时完成；缺 sparse 时不能进入 active 事务。
            raise RuntimeError("Sparse index artifact is required before activating QA index build")
        if build_id:
            indexed_at = datetime.now().isoformat(timespec="seconds")
            updated = self.paper_qa_index_store.update_paper_qa_index_build(
                build_id,
                collection_name=collection_name,
                status="build_success",
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                pdf_path=pdf_path,
                chunk_file=chunk_file,
                retrieval_index_file=retrieval_index_file,
                retrieval_index_count=retrieval_index_count,
                retrieval_index_types=retrieval_index_types,
                retrieval_index_version=retrieval_index_version,
                sparse_index_dir=sparse_index_dir,
                sparse_index_manifest_file=sparse_index_manifest_file,
                sparse_index_document_count=sparse_index_document_count,
                sparse_index_token_count=sparse_index_token_count,
                sparse_index_backend=sparse_index_backend,
                sparse_index_schema_version=sparse_index_schema_version,
                sparse_index_source_file=sparse_index_source_file,
                sparse_index_source_hash=sparse_index_source_hash,
                sparse_index_avgdl=sparse_index_avgdl,
                embedding_file=embedding_file,
                loading_method=loading_method,
                chunking_strategy=chunking_strategy,
                current_stage="activate_index",
                failed_stage="",
                error_message="",
                artifact_status="active",
                indexed_at=indexed_at,
            )
            if not updated:
                return False
            return self.paper_qa_index_store.activate_paper_qa_index_build(build_id)
        return self.paper_qa_index_store.update_paper_qa_index(
            arxiv_id,
            collection_name=collection_name,
            status="indexed",
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            pdf_path=pdf_path,
            chunk_file=chunk_file,
            retrieval_index_file=retrieval_index_file,
            retrieval_index_count=retrieval_index_count,
            retrieval_index_types=retrieval_index_types,
            retrieval_index_version=retrieval_index_version,
            sparse_index_dir=sparse_index_dir,
            sparse_index_manifest_file=sparse_index_manifest_file,
            sparse_index_document_count=sparse_index_document_count,
            sparse_index_token_count=sparse_index_token_count,
            sparse_index_backend=sparse_index_backend,
            sparse_index_schema_version=sparse_index_schema_version,
            sparse_index_source_file=sparse_index_source_file,
            sparse_index_source_hash=sparse_index_source_hash,
            sparse_index_avgdl=sparse_index_avgdl,
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
        build_id: Optional[str] = None,
        **artifacts: Any,
    ) -> bool:
        # 失败记录承担补偿线索职责：即使流程没完成，也要知道哪些文件或 collection 已经生成。
        payload = {
            "status": "build_failed" if build_id else "failed",
            "current_stage": failed_stage,
            "failed_stage": failed_stage,
            "error_message": str(error_message or "")[:2000],
            "loading_method": loading_method,
            # 构建失败产生的新 artifact 不能成为 active，只能等待后续清理。
            "artifact_status": (
                "cleanup_pending"
                if any(
                    artifacts.get(field_name)
                    for field_name in (
                        "collection_name",
                        "pdf_path",
                        "chunk_file",
                        "retrieval_index_file",
                        "sparse_index_dir",
                        "sparse_index_manifest_file",
                        "embedding_file",
                    )
                )
                else "active"
            ),
        }
        payload.update({key: value for key, value in artifacts.items() if key in QA_INDEX_ARTIFACT_FIELDS})
        if build_id:
            updated = self.paper_qa_index_store.update_paper_qa_index_build(build_id, **payload)
            if not updated:
                logger.error(
                    "Failed to persist QA index build failure record: arxiv_id=%s build_id=%s stage=%s collection_name=%s",
                    arxiv_id,
                    build_id,
                    failed_stage,
                    artifacts.get("collection_name"),
                )
            return updated
        updated = self.paper_qa_index_store.update_paper_qa_index(arxiv_id, **payload)
        if not updated:
            updated = self.paper_qa_index_store.insert_paper_qa_index(arxiv_id, **payload)
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
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        info_event(logger, "qa.index_start", run_id=run_id, arxiv_id=arxiv_id, loading_method=loading_method)
        requested_loading_method = str(loading_method or "docling").strip().lower()
        current_stage = "validate_loading_method"
        artifact_state: Dict[str, Any] = {}
        effective_loading_method = requested_loading_method
        build_id: Optional[str] = None
        index_version: Optional[str] = None
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

            current_stage = "create_build_version"
            self._notify_progress(
                progress_callback,
                current_stage="create_build_version",
                progress=8,
                message="Creating QA index build version",
            )
            build = self.paper_qa_index_store.create_paper_qa_index_build(arxiv_id, loading_method)
            if not build:
                raise RuntimeError("Failed to create QA index build version")
            build_id = str(build.get("build_id") or "")
            index_version = str(build.get("index_version") or "")
            self._log_stage(
                "create_build_version",
                arxiv_id,
                loading_method,
                "new QA index build version created",
                build_id=build_id,
                index_version=index_version,
            )

            current_stage = "mark_index_processing"
            self._notify_progress(
                progress_callback,
                current_stage="mark_index_processing",
                progress=10,
                message="Marking QA index as processing",
            )
            self.mark_build_processing(build_id, loading_method=loading_method)
            self._log_stage("mark_index_processing", arxiv_id, loading_method, "paper QA index build marked as processing")

            current_stage = "load_paper_metadata"
            self._notify_progress(
                progress_callback,
                current_stage="load_paper_metadata",
                progress=15,
                message="Loading paper metadata",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
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
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            pdf_path = self.download_pdf(arxiv_id)
            artifact_state["pdf_path"] = pdf_path
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage("download_pdf", arxiv_id, loading_method, "pdf downloaded", pdf_path=pdf_path)

            current_stage = "load_pdf_document"
            self._notify_progress(
                progress_callback,
                current_stage="load_pdf_document",
                progress=35,
                message="Loading PDF document",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
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
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            chunked_data, chunking_strategy = self.chunk_document(arxiv_id, loading_method, document, page_map)
            chunks = chunked_data["chunks"]
            artifact_state["chunking_strategy"] = chunking_strategy
            artifact_state["chunk_count"] = len(chunks)
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage(
                "chunk_document",
                arxiv_id,
                loading_method,
                "document chunked",
                chunk_count=len(chunks),
                chunking_strategy=chunking_strategy,
            )

            current_stage = "structure_table_chunks"
            self._notify_progress(
                progress_callback,
                current_stage=current_stage,
                progress=50,
                message="Structuring table chunks",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            table_structure_result = self.structure_table_chunks(arxiv_id, loading_method, document, chunks)
            table_debug = dict(table_structure_result.get("debug") or {})
            self._log_stage(
                "structure_table_chunks",
                arxiv_id,
                loading_method,
                "table chunks structured",
                table_count=table_debug.get("table_count"),
                structured_table_count=table_debug.get("structured_table_count"),
                failed_table_parse_count=table_debug.get("failed_table_parse_count"),
            )

            current_stage = "save_chunk_file"
            self._notify_progress(
                progress_callback,
                current_stage="save_chunk_file",
                progress=55,
                message="Saving chunk file",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            chunk_file = self.save_chunk_file(
                loading_service=loading_service,
                arxiv_id=arxiv_id,
                loading_method=loading_method,
                chunks=chunks,
                page_map=page_map,
                document=document,
                chunking_strategy=chunking_strategy,
                index_version=index_version,
            )
            artifact_state["chunk_file"] = chunk_file
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage("save_chunk_file", arxiv_id, loading_method, "chunk file saved", chunk_file=chunk_file)

            current_stage = "save_sparse_index_artifact"
            self._notify_progress(
                progress_callback,
                current_stage=current_stage,
                progress=62,
                message="Saving chunk-level sparse keyword index artifact",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            sparse_artifact = self.save_sparse_index_artifact(
                arxiv_id,
                build_id=build_id,
                chunks=chunks,
                chunk_file=chunk_file,
                chunk_count=len(chunks),
                index_version=index_version,
            )
            artifact_state["sparse_index_dir"] = sparse_artifact.get("artifact_dir", "")
            artifact_state["sparse_index_manifest_file"] = sparse_artifact.get("manifest_file", "")
            artifact_state["sparse_index_document_count"] = int(sparse_artifact.get("document_count", 0) or 0)
            artifact_state["sparse_index_token_count"] = int(sparse_artifact.get("token_count", 0) or 0)
            artifact_state["sparse_index_backend"] = str(sparse_artifact.get("backend", "") or "")
            artifact_state["sparse_index_schema_version"] = str(sparse_artifact.get("schema_version", "") or SPARSE_INDEX_ARTIFACT_SCHEMA_VERSION)
            artifact_state["sparse_index_source_file"] = str(sparse_artifact.get("source_file", "") or "")
            artifact_state["sparse_index_source_hash"] = str(sparse_artifact.get("source_hash", "") or "")
            artifact_state["sparse_index_avgdl"] = float(sparse_artifact.get("avgdl", 0.0) or 0.0)
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage(
                "save_sparse_index_artifact",
                arxiv_id,
                loading_method,
                "chunk-level sparse index artifact saved",
                sparse_index_manifest_file=artifact_state["sparse_index_manifest_file"],
                sparse_index_document_count=artifact_state["sparse_index_document_count"],
                sparse_index_token_count=artifact_state["sparse_index_token_count"],
                sparse_index_source_hash=artifact_state["sparse_index_source_hash"],
            )

            current_stage = "compress_chunks_for_rerank"
            self._notify_progress(
                progress_callback,
                current_stage="compress_chunks_for_rerank",
                progress=65,
                message="Compressing chunk text for rerank",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            chunks = self.compress_chunks_for_rerank(chunks)
            self._log_stage("compress_chunks_for_rerank", arxiv_id, loading_method, "chunk text compressed", chunk_count=len(chunks))

            current_stage = "build_retrieval_indexes"
            self._notify_progress(
                progress_callback,
                current_stage=current_stage,
                progress=72,
                message="Building retrieval indexes",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            # retrieval index 必须在 embedding 前冻结：向量命中 index，但后续 rerank/生成仍回填原始 chunk。
            retrieval_indexes, retrieval_index_debug = self.build_retrieval_indexes(chunks)
            self._log_stage(
                "build_retrieval_indexes",
                arxiv_id,
                loading_method,
                "retrieval indexes built",
                retrieval_index_count=retrieval_index_debug.get("index_count"),
                retrieval_index_type_counts=retrieval_index_debug.get("index_type_counts"),
                retrieval_index_generation_error_count=retrieval_index_debug.get("generation_error_count"),
                empty_index_text_count=retrieval_index_debug.get("empty_index_text_count"),
            )

            current_stage = "save_retrieval_index_artifact"
            self._notify_progress(
                progress_callback,
                current_stage=current_stage,
                progress=76,
                message="Saving retrieval index artifact",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            retrieval_index_file = self.save_retrieval_index_artifact(
                arxiv_id,
                chunks,
                retrieval_indexes,
                retrieval_index_debug,
                index_version=index_version,
            )
            retrieval_index_types = self._retrieval_index_type_list(retrieval_index_debug)
            artifact_state["retrieval_index_file"] = retrieval_index_file
            artifact_state["retrieval_index_count"] = int(retrieval_index_debug.get("index_count", 0) or 0)
            artifact_state["retrieval_index_types"] = json.dumps(retrieval_index_types, ensure_ascii=False)
            artifact_state["retrieval_index_version"] = index_version or RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage(
                "save_retrieval_index_artifact",
                arxiv_id,
                loading_method,
                "retrieval index artifact saved",
                retrieval_index_file=retrieval_index_file,
                retrieval_index_count=artifact_state["retrieval_index_count"],
                retrieval_index_types=retrieval_index_types,
            )

            current_stage = "create_chunk_embeddings"
            self._notify_progress(
                progress_callback,
                current_stage=current_stage,
                progress=80,
                message="Creating retrieval index embeddings",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            embeddings, embedding_config = self.create_chunk_embeddings(
                arxiv_id,
                chunks,
                retrieval_indexes=retrieval_indexes,
                retrieval_index_debug=retrieval_index_debug,
                index_version=index_version,
            )
            artifact_state["embedding_model"] = embedding_config.model_name
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage(
                "create_chunk_embeddings",
                arxiv_id,
                loading_method,
                "embeddings created",
                embedding_provider=embedding_config.provider,
                embedding_model=embedding_config.model_name,
                embedding_count=len(embeddings),
                retrieval_index_count=retrieval_index_debug.get("index_count"),
                retrieval_index_type_counts=retrieval_index_debug.get("index_type_counts"),
            )

            current_stage = "save_embeddings"
            self._notify_progress(
                progress_callback,
                current_stage="save_embeddings",
                progress=88,
                message="Saving embeddings",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            embedding_file = self.save_embeddings(arxiv_id, embeddings, index_version=index_version)
            artifact_state["embedding_file"] = embedding_file
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage("save_embeddings", arxiv_id, loading_method, "embedding file saved", embedding_file=embedding_file)

            current_stage = "index_embeddings_to_vector_store"
            self._notify_progress(
                progress_callback,
                current_stage="index_embeddings_to_vector_store",
                progress=95,
                message="Indexing embeddings to vector store",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            index_result = self.index_embeddings_to_vector_store(embedding_file)
            collection_name = index_result.get("collection_name", "")
            artifact_state["collection_name"] = collection_name
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            self._log_stage(
                "index_embeddings_to_vector_store",
                arxiv_id,
                loading_method,
                "embeddings indexed to vector store",
                collection_name=collection_name,
            )

            current_stage = "validate_new_collection"
            self._notify_progress(
                progress_callback,
                current_stage="validate_new_collection",
                progress=98,
                message="Validating new QA index collection",
            )
            self.validate_new_collection(collection_name, len(chunks))
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)

            current_stage = "validate_sparse_index_artifact"
            self._notify_progress(
                progress_callback,
                current_stage=current_stage,
                progress=99,
                message="Validating sparse index artifact",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            sparse_validation = self.validate_sparse_index_artifact(
                build_id=build_id,
                index_version=index_version or RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION,
                sparse_index_manifest_file=str(artifact_state.get("sparse_index_manifest_file", "") or ""),
                sparse_index_source_hash=str(artifact_state.get("sparse_index_source_hash", "") or ""),
                sparse_index_document_count=int(artifact_state.get("sparse_index_document_count", 0) or 0),
                sparse_index_token_count=int(artifact_state.get("sparse_index_token_count", 0) or 0),
                sparse_index_backend=str(artifact_state.get("sparse_index_backend", "") or ""),
            )
            self._log_stage(
                "validate_sparse_index_artifact",
                arxiv_id,
                loading_method,
                "sparse index artifact validated",
                sparse_source_type=sparse_validation.get("source_type"),
                sparse_document_count=sparse_validation.get("document_count"),
                sparse_token_count=sparse_validation.get("token_count"),
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)

            current_stage = "activate_index"
            self._notify_progress(
                progress_callback,
                current_stage="activate_index",
                progress=100,
                message="Activating QA index version",
            )
            self.record_index_stage(arxiv_id, current_stage=current_stage, loading_method=loading_method, build_id=build_id, **artifact_state)
            success_marked = self.mark_index_success(
                arxiv_id,
                build_id=build_id,
                collection_name=collection_name,
                chunk_count=len(chunks),
                embedding_model=embedding_config.model_name,
                pdf_path=pdf_path,
                chunk_file=chunk_file,
                retrieval_index_file=retrieval_index_file,
                retrieval_index_count=int(retrieval_index_debug.get("index_count", 0) or 0),
                retrieval_index_types=json.dumps(self._retrieval_index_type_list(retrieval_index_debug), ensure_ascii=False),
                retrieval_index_version=index_version or RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION,
                embedding_file=embedding_file,
                loading_method=loading_method,
                chunking_strategy=chunking_strategy,
                sparse_index_dir=str(artifact_state.get("sparse_index_dir", "") or ""),
                sparse_index_manifest_file=str(artifact_state.get("sparse_index_manifest_file", "") or ""),
                sparse_index_document_count=int(artifact_state.get("sparse_index_document_count", 0) or 0),
                sparse_index_token_count=int(artifact_state.get("sparse_index_token_count", 0) or 0),
                sparse_index_backend=str(artifact_state.get("sparse_index_backend", "") or ""),
                sparse_index_schema_version=str(artifact_state.get("sparse_index_schema_version", "") or ""),
                sparse_index_source_file=str(artifact_state.get("sparse_index_source_file", "") or ""),
                sparse_index_source_hash=str(artifact_state.get("sparse_index_source_hash", "") or ""),
                sparse_index_avgdl=float(artifact_state.get("sparse_index_avgdl", 0.0) or 0.0),
            )
            if not success_marked:
                # 新 collection 已经生成但未能激活时，不能覆盖旧 active，只把新版本留给后续清理。
                self.paper_qa_index_store.update_paper_qa_index_build(
                    build_id,
                    status="orphaned",
                    current_stage="activate_index",
                    failed_stage="activate_index",
                    error_message="SQLite failed to atomically activate QA index build",
                    artifact_status="cleanup_pending",
                )
                raise RuntimeError("Milvus collection was created, but SQLite failed to activate QA index build")
            self._log_stage(
                "mark_index_success",
                arxiv_id,
                loading_method,
                "paper QA index marked as indexed",
                collection_name=collection_name,
                chunk_count=len(chunks),
                embedding_model=embedding_config.model_name,
            )

            info_event(
                logger,
                "qa.index_done",
                run_id=run_id,
                arxiv_id=arxiv_id,
                build_id=build_id,
                status="success",
                loading_method=loading_method,
                collection_name=collection_name,
                chunk_count=len(chunks),
                embedding_model=embedding_config.model_name,
                retrieval_index_count=retrieval_index_debug.get("index_count"),
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
                "index_version": index_version,
                "build_id": build_id,
                "chunk_file": chunk_file,
                "retrieval_index_file": retrieval_index_file,
                "retrieval_index_count": retrieval_index_debug.get("index_count"),
                "retrieval_index_type_counts": retrieval_index_debug.get("index_type_counts"),
                "retrieval_index_types": retrieval_index_types,
                "retrieval_index_version": index_version or RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION,
                "sparse_index_dir": artifact_state.get("sparse_index_dir", ""),
                "sparse_index_manifest_file": artifact_state.get("sparse_index_manifest_file", ""),
                "sparse_index_document_count": artifact_state.get("sparse_index_document_count", 0),
                "sparse_index_token_count": artifact_state.get("sparse_index_token_count", 0),
                "sparse_index_backend": artifact_state.get("sparse_index_backend", ""),
                "sparse_index_schema_version": artifact_state.get("sparse_index_schema_version", ""),
                "sparse_index_source_file": artifact_state.get("sparse_index_source_file", ""),
                "sparse_index_source_hash": artifact_state.get("sparse_index_source_hash", ""),
                "sparse_index_avgdl": artifact_state.get("sparse_index_avgdl", 0.0),
                "retrieval_index_generation_error_count": retrieval_index_debug.get("generation_error_count"),
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
            info_event(
                logger,
                "qa.index_done",
                run_id=run_id,
                arxiv_id=arxiv_id,
                build_id=build_id,
                status="error",
                code=exc.code,
                failed_stage=str(exc.context.get("stage") or current_stage),
                loading_method=effective_loading_method,
                message=exc.message,
            )
            self.mark_index_failed(
                arxiv_id,
                failed_stage=str(exc.context.get("stage") or current_stage),
                error_message=f"{exc.code}: {exc.detail or exc.message}",
                loading_method=effective_loading_method,
                build_id=build_id,
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
            info_event(
                logger,
                "qa.index_done",
                run_id=run_id,
                arxiv_id=arxiv_id,
                build_id=build_id,
                status="error",
                failed_stage=current_stage,
                loading_method=effective_loading_method,
                error_type=type(exc).__name__,
                message=detail,
            )
            self.mark_index_failed(
                arxiv_id,
                failed_stage=current_stage,
                error_message=detail,
                loading_method=effective_loading_method,
                build_id=build_id,
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
            info_event(
                logger,
                "qa.index_done",
                run_id=run_id,
                arxiv_id=arxiv_id,
                build_id=build_id,
                status="error",
                failed_stage=current_stage,
                loading_method=effective_loading_method,
                error_type=type(exc).__name__,
                message=detail,
            )
            self.mark_index_failed(
                arxiv_id,
                failed_stage=current_stage,
                error_message=detail,
                loading_method=effective_loading_method,
                build_id=build_id,
                **artifact_state,
            )
            error_code = self._error_code_for_stage(current_stage)
            raise AppError(
                error_code,
                detail={"stage": current_stage, "error": detail, "arxiv_id": arxiv_id},
                context={"arxiv_id": arxiv_id, "stage": current_stage, "loading_method": effective_loading_method},
            ) from exc
