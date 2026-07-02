from __future__ import annotations

"""arXiv 论文检索与下载路由。

该模块负责把用户输入的自然语言检索条件、结构化字段条件、下载请求等，
转换为服务层可执行的调用，并将 arXiv 查询能力暴露给前端。
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from dependencies import get_arxiv_api_service, get_arxiv_service
from services.arxiv.arxiv_query_builder import (
    ArxivSearchValidationError,
    build_arxiv_query_from_structured_params,
    build_arxiv_raw_query,
    validate_arxiv_search_request,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/arxiv", tags=["arxiv"])


@router.post("/search")
async def arxiv_search(
    search_query: Optional[str] = Body(None, description="搜索查询字符串，支持字段前缀语法如 ti:deep learning"),
    id_list: Optional[List[str]] = Body(None, description="arXiv论文ID列表，用于精确匹配"),
    title: Optional[str] = Body(None, description="标题关键词"),
    author: Optional[str] = Body(None, description="作者姓名"),
    abstract: Optional[str] = Body(None, description="摘要关键词"),
    category: Optional[str] = Body(None, description="学科分类代码，如 cs.AI"),
    comment: Optional[str] = Body(None, description="评论关键词"),
    journal_ref: Optional[str] = Body(None, description="期刊引用关键词"),
    report_number: Optional[str] = Body(None, description="报告编号关键词"),
    operator: Optional[str] = Body("AND", description="逻辑操作符：AND 或 OR"),
    max_results: int = Body(10),
    start: int = Body(0),
    sort_by: str = Body("relevance"),
    sort_order: str = Body("descending"),
    submitted_days_ago: Optional[int] = Body(None, description="搜索提交日期在多少天内的文章（本地数据源暂不支持）"),
    arxiv_service=Depends(get_arxiv_service),
):
    """执行 arXiv 检索。

    该接口同时支持两种输入方式：
    1. 原始 search_query / id_list 形式；
    2. title、author、category 等结构化字段形式。
    当结构化字段存在时，优先走结构化查询构建流程，以获得更稳定的查询语义。
    """
    try:
        # 只要出现任意结构化字段，就认为调用方希望由后端拼接更规范的 arXiv 查询表达式。
        structured_fields_present = any([title, author, abstract, category, comment, journal_ref, report_number])
        if structured_fields_present:
            structured = build_arxiv_query_from_structured_params(
                query=search_query,
                title_query=title,
                author_query=author,
                abstract_query=abstract,
                categories=[category] if category else None,
                comment_query=comment,
                journal_ref_query=journal_ref,
                report_number_query=report_number,
                id_list=id_list,
                field_operator=operator if operator and operator.strip() else "AND",
                category_operator="OR",
                submitted_days_ago=submitted_days_ago,
            )
            # 在真正查询前先校验分页、排序、query 组合是否合法，尽量把错误拦在服务边界。
            validate_arxiv_search_request(
                search_query=structured["final_search_query"],
                id_list=structured["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            # 结构化模式下，实际搜索用的是后端整理后的 final_search_query。
            results = arxiv_service.search(
                search_query=structured["final_search_query"],
                id_list=structured["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        else:
            # 原始模式更接近“直通式查询”，适合前端已经自行组织 query 的场景。
            raw_query = build_arxiv_raw_query(
                search_query=search_query,
                id_list=id_list,
                submitted_days_ago=submitted_days_ago,
                append_date_when_query_missing=False,
            )
            validate_arxiv_search_request(
                search_query=raw_query["final_search_query"],
                id_list=raw_query["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            results = arxiv_service.search(
                search_query=raw_query["final_search_query"],
                id_list=raw_query["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )

        return results
    except ArxivSearchValidationError as exc:
        logger.error("Invalid arXiv search query: %s", str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        error_code = str(getattr(exc, "code", "") or "")
        if error_code:
            logger.error("Local arXiv search capability error: %s", str(exc))
            detail_factory = getattr(exc, "to_error_detail", None)
            detail = detail_factory() if callable(detail_factory) else {"code": error_code, "message": str(exc)}
            raise HTTPException(status_code=int(getattr(exc, "status_code", 400)), detail=detail)
        logger.error("Error searching arXiv: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fields")
async def arxiv_get_fields(arxiv_service=Depends(get_arxiv_service)):
    """返回当前支持的 arXiv 查询字段列表。"""
    try:
        fields = arxiv_service.get_available_fields()
        return {"fields": fields}
    except Exception as exc:
        logger.error("Error getting arXiv fields: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/categories")
async def arxiv_get_categories(arxiv_service=Depends(get_arxiv_service)):
    """返回当前支持的 arXiv 学科分类列表。"""
    try:
        categories = arxiv_service.get_subject_categories()
        return {"categories": categories}
    except Exception as exc:
        logger.error("Error getting arXiv categories: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/download")
async def arxiv_download(
    arxiv_id: str = Body(...),
    pdf_url: str = Body(...),
    arxiv_api_service=Depends(get_arxiv_api_service),
):
    """下载指定 arXiv 论文 PDF 到本地。"""
    try:
        filepath = arxiv_api_service.download_pdf(pdf_url, arxiv_id)
        return {"status": "success", "filepath": filepath}
    except Exception as exc:
        logger.error("Error downloading arXiv paper: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/search-and-save")
async def arxiv_search_and_save(
    search_query: str = Body(""),
    id_list: Optional[List[str]] = Body(None),
    max_results: int = Body(10),
    download_pdfs: bool = Body(False),
    arxiv_api_service=Depends(get_arxiv_api_service),
    **kwargs,
):
    """检索论文并按需保存元数据/下载 PDF。

    这是一个更偏“工作流”式的接口：不仅做搜索，
    还把结果持久化到本地或数据库，适合批量导入场景。
    """
    try:
        results = await arxiv_api_service.search_and_save(
            search_query=search_query,
            id_list=id_list,
            max_results=max_results,
            download_pdfs=download_pdfs,
            # 其余扩展参数透传给 service，保持路由层轻量。
            **kwargs,
        )
        return results
    except Exception as exc:
        logger.error("Error in arXiv search and save: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
