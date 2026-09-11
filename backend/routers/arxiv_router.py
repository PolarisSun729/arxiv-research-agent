from __future__ import annotations

"""arXiv 论文检索与下载路由。

该模块只负责 HTTP 入参、搜索请求归一化和错误转换；真正的搜索能力由
依赖注入层选择的统一后端协议提供，避免路由里重新长出本地/远程分支。
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from dependencies import get_arxiv_api_service, get_arxiv_search_backend
from services.arxiv.arxiv_query_builder import prepare_arxiv_search_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arxiv", tags=["arxiv"])


class ArxivSearchRequest(BaseModel):
    """HTTP 搜索请求体。

    字段名保持原 `/arxiv/search` 合约不变；内部会统一映射到 query_builder
    的标准参数，避免调用方感知后端搜索协议的收紧。
    """

    search_query: Optional[str] = Field(None, description="搜索查询字符串，支持字段前缀语法如 ti:deep learning")
    id_list: Optional[List[str]] = Field(None, description="arXiv 论文 ID 列表，用于精确匹配")
    title: Optional[str] = Field(None, description="标题关键词")
    author: Optional[str] = Field(None, description="作者姓名")
    abstract: Optional[str] = Field(None, description="摘要关键词")
    category: Optional[str] = Field(None, description="学科分类代码，如 cs.AI")
    categories: Optional[List[str]] = Field(None, description="学科分类代码列表")
    comment: Optional[str] = Field(None, description="评论关键词")
    journal_ref: Optional[str] = Field(None, description="期刊引用关键词")
    report_number: Optional[str] = Field(None, description="报告编号关键词")
    operator: Optional[str] = Field("AND", description="结构化字段之间的逻辑操作符：AND、OR 或 ANDNOT")
    category_operator: Optional[str] = Field("OR", description="多个分类之间的逻辑操作符：AND 或 OR")
    max_results: int = Field(10, description="最大返回条数")
    start: int = Field(0, description="分页起始偏移量")
    sort_by: str = Field("relevance", description="排序字段")
    sort_order: str = Field("descending", description="排序方向")
    submitted_days_ago: Optional[int] = Field(None, description="搜索提交日期在多少天内的文章")

def _prepare_search_request(request: ArxivSearchRequest, *, append_date_when_query_missing: bool) -> Dict[str, Any]:
    """把 HTTP 请求模型转换成统一搜索后端能执行的参数。"""
    return prepare_arxiv_search_request(
        search_query=request.search_query,
        id_list=request.id_list,
        title_query=request.title,
        author_query=request.author,
        abstract_query=request.abstract,
        categories=request.categories,
        category=request.category,
        comment_query=request.comment,
        journal_ref_query=request.journal_ref,
        report_number_query=request.report_number,
        field_operator=request.operator or "AND",
        category_operator=request.category_operator or "OR",
        submitted_days_ago=request.submitted_days_ago,
        max_results=request.max_results,
        start=request.start,
        sort_by=request.sort_by,
        sort_order=request.sort_order,
        append_date_when_query_missing=append_date_when_query_missing,
        # submitted_days_ago 属于输入层语义，负数应在这里失败，不能泄漏到底层后端。
        strict_submitted_days_ago=True,
    )


def _raise_arxiv_http_error(exc: Exception, *, operation: str, default_status_code: int = 500) -> None:
    """把搜索边界的结构化异常转换成 FastAPI HTTPException。"""
    if isinstance(exc, HTTPException):
        # 已经是 HTTP 边界异常时保持原状态码，避免二次包装成 500 掩盖真实输入错误。
        raise exc
    logger.error("arXiv %s failed: %s", operation, str(exc))
    detail_factory = getattr(exc, "to_error_detail", None)
    error_code = str(getattr(exc, "code", "") or "")
    if callable(detail_factory) or error_code:
        detail = detail_factory() if callable(detail_factory) else {"code": error_code, "message": str(exc)}
        raise HTTPException(status_code=int(getattr(exc, "status_code", 400)), detail=detail)
    raise HTTPException(status_code=default_status_code, detail=str(exc))


@router.post("/search")
async def arxiv_search(
    request: Optional[ArxivSearchRequest] = Body(None),
    arxiv_backend=Depends(get_arxiv_search_backend),
):
    """执行 arXiv 检索。"""
    try:
        # 空请求体仍走统一校验，最终会得到明确的 400，而不是路由层 422。
        prepared = _prepare_search_request(request or ArxivSearchRequest(), append_date_when_query_missing=False)
        # 搜索后端包含同步 HTTP/SQLite I/O；交给有界线程池，避免占用事件循环并保留请求身份上下文。
        return await run_in_threadpool(
            arxiv_backend.search,
            search_query=prepared["final_search_query"],
            id_list=prepared["id_list"],
            max_results=prepared["max_results"],
            start=prepared["start"],
            sort_by=prepared["sort_by"],
            sort_order=prepared["sort_order"],
        )
    except Exception as exc:
        _raise_arxiv_http_error(exc, operation="search")


@router.get("/fields")
async def arxiv_get_fields(arxiv_backend=Depends(get_arxiv_search_backend)):
    """返回当前搜索后端支持的 arXiv 查询字段列表。"""
    try:
        return {"fields": arxiv_backend.get_available_fields()}
    except Exception as exc:
        _raise_arxiv_http_error(exc, operation="get fields")


@router.get("/categories")
async def arxiv_get_categories(arxiv_backend=Depends(get_arxiv_search_backend)):
    """返回当前搜索后端支持展示的 arXiv 学科分类列表。"""
    try:
        return {"categories": arxiv_backend.get_subject_categories()}
    except Exception as exc:
        _raise_arxiv_http_error(exc, operation="get categories")


@router.post("/download")
async def arxiv_download(
    arxiv_id: str = Body(...),
    pdf_url: str = Body(...),
    arxiv_api_service=Depends(get_arxiv_api_service),
):
    """下载指定 arXiv 论文 PDF 到本地。"""
    try:
        # 下载含重试等待和 PDF 校验，期间仍需响应认证、限流与健康检查。
        filepath = await run_in_threadpool(arxiv_api_service.download_pdf, pdf_url, arxiv_id)
        return {"status": "success", "filepath": filepath}
    except Exception as exc:
        _raise_arxiv_http_error(exc, operation="download")
