from __future__ import annotations

"""论文基础信息管理路由。

该模块负责论文元数据的新增、查询、删除以及按分类筛选。
其中新增论文时还会同步创建摘要向量并写入向量库，
因此它是“论文入库”和“论文可检索化”的关键入口之一。
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from auth.context import current_auth
from core.errors import AppError, ErrorCode, error_response
from dependencies import (
    get_embedding_service,
    get_current_embedding_config,
    get_oai_database_service,
    get_paper_catalog_store,
    get_paper_qa_service,
    get_recommendation_service,
    get_user_preference_store,
    get_vector_store_service,
)
from utils.config import get_default_user_id
from utils.storage_paths import ARXIV_OAI_SYNC_META_FILE, ARXIV_OAI_SYNC_STATE_FILE

logger = logging.getLogger(__name__)

# 运行状态不随 release 版本切换；backend/data 在生产中由部署器接到 shared/backend/data。
SYNC_STATE_FILE = ARXIV_OAI_SYNC_STATE_FILE
SYNC_META_FILE = ARXIV_OAI_SYNC_META_FILE

router = APIRouter(tags=["paper"])


def _read_text_file(path: Path) -> Optional[str]:
    try:
        if not path.exists() or not path.is_file():
            return None
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except (OSError, UnicodeError) as exc:
        # 状态文件是可选诊断数据；权限或编码异常不能阻断首页看板。
        logger.warning("Unable to read sync state file %s: %s", path, exc)
        return None


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists() or not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        # 损坏或短暂不可读的摘要只影响展示，统一回退为空状态。
        logger.warning("Unable to read sync metadata file %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    """把外部状态中的计数安全转换为非负整数，防止损坏文件触发 500。"""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _get_sync_status_payload() -> Dict[str, Any]:
    meta = _read_json_file(SYNC_META_FILE)
    state_date = _read_text_file(SYNC_STATE_FILE)
    summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
    meta_last_successful_until = meta.get("last_successful_until")

    return {
        "status": meta.get("status") or "unknown",
        "mode": meta.get("mode") or "sync",
        "from_date": meta.get("from_date"),
        "until_date": meta.get("until_date"),
        "lastSyncRunAt": meta.get("finished_at"),
        "lastSyncedDate": meta_last_successful_until or state_date,
        "latestSyncNewPapers": _safe_int(meta.get("records_written", summary.get("records_written", 0))),
        "latestSyncMatchedPapers": _safe_int(meta.get("records_matched", summary.get("records_matched", 0))),
        "syncErrors": _safe_int(meta.get("errors", summary.get("errors", 0))),
        "syncErrorMessage": meta.get("error_message"),
    }


def _paper_has_display_metadata(paper: Dict[str, Any]) -> bool:
    """判断本地论文记录是否足以支撑列表卡片展示，缺字段时允许详情接口回源修复。"""
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or paper.get("summary") or "").strip()
    return bool(title and abstract)


@router.get("/stats")
async def get_dashboard_stats(
    user_id: str = Query(default_factory=get_default_user_id),
    user_preference_store=Depends(get_user_preference_store),
    oai_db_service=Depends(get_oai_database_service),
):
    """返回首页看板所需的聚合统计数据。"""
    try:
        sync_status = _get_sync_status_payload()
        identity = current_auth.get()
        sync_error = sync_status["syncErrorMessage"]
        if sync_error and (identity is None or identity.session.user.role != "admin"):
            # 看板对所有账号开放；保留字段和失败状态，内部路径及诊断仅向已认证管理员展示。
            sync_error = "最近一次同步失败，请联系管理员查看运行日志。"
        return {
            "totalPapers": oai_db_service.get_total_paper_count(),
            "labeledPapers": user_preference_store.get_user_labeled_paper_count(user_id=user_id),
            "todayNewPapers": sync_status["latestSyncNewPapers"],
            "latestSyncNewPapers": sync_status["latestSyncNewPapers"],
            "lastSyncedDate": sync_status["lastSyncedDate"],
            "lastSyncRunAt": sync_status["lastSyncRunAt"],
            "lastSyncStatus": sync_status["status"],
            "lastSyncMode": sync_status["mode"],
            "latestSyncMatchedPapers": sync_status["latestSyncMatchedPapers"],
            "syncErrors": sync_status["syncErrors"],
            "syncErrorMessage": sync_error,
        }
    except Exception as exc:
        logger.error("Error getting dashboard stats: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sync-status")
async def get_sync_status():
    """返回最近一次增量同步的状态信息，供前端看板使用。"""
    try:
        return _get_sync_status_payload()
    except Exception as exc:
        logger.error("Error getting sync status: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/paper")
async def add_paper(
    arxiv_id: str = Body(...),
    title: str = Body(...),
    authors: str = Body(...),
    abstract: str = Body(...),
    categories: str = Body(...),
    published_date: str = Body(...),
    url: str = Body(...),
    collection_name: str = Body("arxiv_abstracts"),
    paper_catalog_store=Depends(get_paper_catalog_store),
    embedding_service=Depends(get_embedding_service),
):
    """新增论文，并同步为其摘要生成 embedding。

    该流程会把论文摘要写入向量库，随后再把论文主记录写入数据库，
    从而让这篇论文既能被结构化查询，也能参与向量检索与推荐。
    """
    try:
        logger.debug("Adding paper with embedding: %s", arxiv_id)

        # 读取当前生效的 embedding 配置，确保向量生成与系统其他模块使用同一套模型参数。
        embedding_config = get_current_embedding_config()
        logger.debug(
            "Creating embedding for abstract using %s / %s",
            embedding_config.provider,
            embedding_config.model_name,
        )
        # 通常不会只拿 abstract 做向量，标题和摘要组合后的语义表示往往更稳定。
        text_to_embed = embedding_service.build_paper_embedding_text(title, abstract)
        embedding = embedding_service.create_single_embedding(
            text_to_embed,
            provider=embedding_config.provider,
            model=embedding_config.model_name,
            api_key=embedding_config.api_key,
            base_url=embedding_config.base_url,
            dimension=embedding_config.dimension,
        )

        logger.debug("Embedding created, dimension: %s", len(embedding))

        # 这份 metadata 会作为向量记录的附加信息，便于后续检索结果回显和过滤。
        metadata = {
            "content": abstract,
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "published_date": published_date,
            "url": url,
            "embedding_model": embedding_config.model_name,
        }

        logger.debug("Inserting embedding to collection: %s", collection_name)
        # 向量写入是论文可检索化的关键步骤，失败时要暴露 vector_store_error 而不是泛化成入库失败。
        vector_store_service = get_vector_store_service()
        try:
            embedding_id = vector_store_service.insert_single_embedding(collection_name, embedding, metadata)
        except Exception as exc:
            logger.exception(
                "Paper vector write failed: code=%s arxiv_id=%s collection_name=%s stage=%s",
                ErrorCode.VECTOR_STORE_ERROR,
                arxiv_id,
                collection_name,
                "insert_single_embedding",
            )
            return error_response(
                AppError(
                    ErrorCode.VECTOR_STORE_ERROR,
                    detail=exc,
                    context={"arxiv_id": arxiv_id, "stage": "insert_single_embedding", "collection_name": collection_name},
                )
            )

        logger.debug("Embedding inserted with ID: %s", embedding_id)
        logger.debug("Adding paper to database: %s, embedding_id: %s", metadata, embedding_id)

        # 数据库中保留 embedding_id / embedding_model，方便后续追踪向量来源与重建。
        success = paper_catalog_store.add_paper(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "categories": categories,
                "published_date": published_date,
                "url": url,
                "embedding_id": str(embedding_id),
                "embedding_model": embedding_config.model_name,
            }
        )

        if success:
            logger.info("Paper %s added successfully with embedding", arxiv_id)
            return {
                "status": "success",
                "message": "Paper added with embedding",
                "arxiv_id": arxiv_id,
                "embedding_id": embedding_id,
                "embedding_model": embedding_config.model_name,
                "vector_dimension": len(embedding),
                "collection_name": collection_name,
            }
        logger.error(
            "Paper database write failed: code=%s arxiv_id=%s stage=%s",
            ErrorCode.DATABASE_WRITE_FAILED,
            arxiv_id,
            "add_paper",
        )
        return error_response(
            AppError(
                ErrorCode.DATABASE_WRITE_FAILED,
                detail="paper_catalog_store.add_paper returned False",
                context={"arxiv_id": arxiv_id, "stage": "add_paper"},
            )
        )
    except Exception as exc:
        logger.exception("Error adding paper: arxiv_id=%s code=%s", arxiv_id, ErrorCode.UNKNOWN_ERROR)
        return error_response(AppError(ErrorCode.UNKNOWN_ERROR, detail=exc, context={"arxiv_id": arxiv_id, "stage": "add_paper"}))


@router.get("/paper/{arxiv_id}")
async def get_paper(
    arxiv_id: str,
    paper_catalog_store=Depends(get_paper_catalog_store),
    recommendation_service=Depends(get_recommendation_service),
):
    """获取单篇论文详情。

    读取顺序是：先查本地数据库；若本地没有，再尝试回源 arXiv，
    并将回源结果物化成系统内可用的论文对象后返回。
    """
    try:
        paper = paper_catalog_store.get_paper(arxiv_id)
        if paper and _paper_has_display_metadata(paper):
            return paper

        # 历史数据可能只落了 ID/向量；缺少标题或摘要时回源并修复本地记录，避免已标记页出现空卡片。
        try:
            source_paper = recommendation_service._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
        except Exception as exc:
            logger.warning("Failed to backfill incomplete paper %s: %s", arxiv_id, exc)
            if paper:
                return paper
            raise
        if source_paper:
            try:
                # 优先复用已有 embedding，只补齐标题/摘要等展示元数据，避免回源一次就重复生成向量。
                ensure_materialized = getattr(recommendation_service, "_ensure_paper_materialized", None)
                if callable(ensure_materialized):
                    return ensure_materialized(arxiv_id, paper_payload=source_paper)
                # 兼容尚未提供增量物化能力的旧实现。
                return recommendation_service._materialize_paper_from_source(source_paper, arxiv_id)
            except Exception as exc:
                logger.warning("Failed to persist backfilled paper %s: %s", arxiv_id, exc)
                if paper:
                    return paper
                raise
        if paper:
            return paper
        raise HTTPException(status_code=404, detail="Paper not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/paper/{arxiv_id}")
async def delete_paper(
    arxiv_id: str,
    paper_catalog_store=Depends(get_paper_catalog_store),
    paper_qa_service=Depends(get_paper_qa_service),
):
    """删除指定论文，并同步清理该论文的 QA 索引 artifact。"""
    try:
        # 论文删除前先让 QA 索引不可检索，避免 SQLite 元数据消失后 Milvus 仍留下可命中的旧 chunk。
        qa_cleanup = paper_qa_service.delete_qa_index(arxiv_id)
        success = paper_catalog_store.delete_paper(arxiv_id)
        if success:
            return {"status": "success", "message": "Paper deleted", "qa_cleanup": qa_cleanup}
        raise HTTPException(status_code=404, detail="Paper not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error deleting paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/papers")
async def get_all_papers(paper_catalog_store=Depends(get_paper_catalog_store)):
    """获取当前数据库中的全部论文列表。"""
    try:
        papers = paper_catalog_store.get_all_papers()
        return {"papers": papers}
    except Exception as exc:
        logger.error("Error getting all papers: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/papers/category/{category}")
async def search_papers_by_category(category: str, paper_catalog_store=Depends(get_paper_catalog_store)):
    """按 arXiv 分类查询论文。"""
    try:
        papers = paper_catalog_store.search_papers_by_category(category)
        return {"papers": papers}
    except Exception as exc:
        logger.error("Error searching papers by category: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
