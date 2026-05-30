from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from dependencies import get_arxiv_api_service, get_arxiv_service
from services.arxiv_search_service import (
    ArxivSearchValidationError,
    build_arxiv_query_from_structured_params,
    build_arxiv_submitted_date_query,
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
    try:
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
            validate_arxiv_search_request(
                search_query=structured["final_search_query"],
                id_list=structured["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            results = arxiv_service.search(
                search_query=structured["final_search_query"],
                id_list=structured["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        else:
            validate_arxiv_search_request(
                search_query=search_query,
                id_list=id_list,
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            normalized_search_query = search_query
            if submitted_days_ago is not None and submitted_days_ago >= 0 and normalized_search_query and not id_list:
                normalized_search_query = f"({normalized_search_query}) AND {build_arxiv_submitted_date_query(submitted_days_ago)}"
            results = arxiv_service.search(
                search_query=normalized_search_query,
                id_list=id_list,
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
        logger.error("Error searching arXiv: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/fields")
async def arxiv_get_fields(arxiv_service=Depends(get_arxiv_service)):
    try:
        fields = arxiv_service.get_available_fields()
        return {"fields": fields}
    except Exception as exc:
        logger.error("Error getting arXiv fields: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/categories")
async def arxiv_get_categories(arxiv_service=Depends(get_arxiv_service)):
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
    try:
        results = await arxiv_api_service.search_and_save(
            search_query=search_query,
            id_list=id_list,
            max_results=max_results,
            download_pdfs=download_pdfs,
            **kwargs,
        )
        return results
    except Exception as exc:
        logger.error("Error in arXiv search and save: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

