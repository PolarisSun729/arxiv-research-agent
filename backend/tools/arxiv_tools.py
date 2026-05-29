from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from services.arxiv_search_service import (
    ArxivSearchService,
    ArxivSearchValidationError,
    build_arxiv_query_from_structured_params,
    build_arxiv_submitted_date_query,
    validate_arxiv_search_request,
)
from services.database_service import DatabaseService
from services.local_arxiv_service import LocalArxivService
from tools.tool_result import make_tool_error, make_tool_result, make_tool_trace
from utils.config import CORE_CONFIG, get_recommendation_clustering_runtime_config

DATA_SOURCE = CORE_CONFIG["arxiv_data_source"]
ARXIV_PROXY_URL = CORE_CONFIG.get("arxiv_proxy_url", "")

db_service = DatabaseService()
_local_arxiv_service: Optional[LocalArxivService] = None


def _get_recommendation_service():
    from services.arxiv_oai_service import ArxivOaiDatabaseService
    from services.embedding_service import EmbeddingService
    from services.recommendation_service import RecommendationService
    from services.vector_store_service import VectorStoreService

    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService()
    oai_db_service = ArxivOaiDatabaseService()
    return RecommendationService(
        db_service=db_service,
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        get_embedding_config=embedding_service.get_default_embedding_config,
        get_clustering_config=get_recommendation_clustering_runtime_config,
        arxiv_service_factory=lambda: get_arxiv_service(),
        oai_db_service=oai_db_service,
    )


def get_arxiv_service():
    global _local_arxiv_service
    if DATA_SOURCE == "api":
        return ArxivSearchService(proxy_url=ARXIV_PROXY_URL)
    if _local_arxiv_service is None:
        _local_arxiv_service = LocalArxivService()
    return _local_arxiv_service


def _normalize_source_paper(source_paper: Dict[str, Any], fallback_arxiv_id: str) -> Dict[str, Any]:
    authors = source_paper.get("authors", "")
    categories = source_paper.get("categories", "")
    if isinstance(authors, (list, tuple)):
        authors_value = ", ".join([str(item).strip() for item in authors if str(item).strip()])
    else:
        authors_value = str(authors or "").strip()
    if isinstance(categories, (list, tuple)):
        categories_value = ", ".join([str(item).strip() for item in categories if str(item).strip()])
    else:
        categories_value = str(categories or "").strip()

    title = str(source_paper.get("title", "") or "").strip()
    abstract = str(source_paper.get("abstract", "") or source_paper.get("summary", "") or "").strip()
    published_date = str(
        source_paper.get("published_date")
        or source_paper.get("published")
        or source_paper.get("updated")
        or source_paper.get("update_date")
        or source_paper.get("publishedAt")
        or ""
    ).strip()
    url = str(source_paper.get("url") or source_paper.get("abs_url") or source_paper.get("pdf_url") or "").strip()
    arxiv_identifier = str(source_paper.get("arxiv_id") or source_paper.get("id") or fallback_arxiv_id or "").strip()
    if arxiv_identifier.startswith("http"):
        arxiv_identifier = arxiv_identifier.rsplit("/", 1)[-1]

    return {
        "arxiv_id": arxiv_identifier,
        "title": title,
        "authors": authors_value,
        "abstract": abstract,
        "categories": categories_value,
        "published_date": published_date,
        "url": url,
    }


def _build_search_trace(
    *,
    tool_name: str,
    raw_inputs: Dict[str, Any],
    normalized_inputs: Dict[str, Any],
    final_search_query: Optional[str],
    id_list: Optional[List[str]],
    sort_by: str,
    sort_order: str,
    start: int,
    max_results: int,
    submitted_days_ago_applied: Optional[bool] = None,
) -> Dict[str, Any]:
    trace = make_tool_trace(tool_name, inputs=raw_inputs, source=DATA_SOURCE)
    trace.update(
        {
            "raw_inputs": raw_inputs,
            "normalized_inputs": normalized_inputs,
            "final_search_query": final_search_query,
            "id_list": id_list or [],
            "sort_by": sort_by,
            "sort_order": sort_order,
            "start": start,
            "max_results": max_results,
        }
    )
    if submitted_days_ago_applied is not None:
        trace["submitted_days_ago_applied"] = submitted_days_ago_applied
    return trace


def _run_search(
    *,
    tool_name: str,
    raw_inputs: Dict[str, Any],
    normalized_inputs: Dict[str, Any],
    search_query: Optional[str],
    id_list: Optional[List[str]],
    max_results: int,
    start: int,
    sort_by: str,
    sort_order: str,
    submitted_days_ago_applied: Optional[bool] = None,
) -> Dict[str, Any]:
    arxiv_service = get_arxiv_service()
    result = arxiv_service.search(
        search_query=search_query,
        id_list=id_list,
        max_results=max_results,
        start=start,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    papers = result.get("papers", []) if isinstance(result, dict) else []
    return make_tool_result(
        ok=True,
        tool_name=tool_name,
        summary=f"找到 {len(papers)} 篇 arXiv 论文",
        data=result if isinstance(result, dict) else {"result": result},
        trace=_build_search_trace(
            tool_name=tool_name,
            raw_inputs=raw_inputs,
            normalized_inputs=normalized_inputs,
            final_search_query=search_query,
            id_list=id_list,
            sort_by=sort_by,
            sort_order=sort_order,
            start=start,
            max_results=max_results,
            submitted_days_ago_applied=submitted_days_ago_applied,
        ),
    )


def search_arxiv_raw(
    search_query: Optional[str] = None,
    id_list: Optional[List[str]] = None,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    submitted_days_ago: Optional[int] = None,
) -> Dict[str, Any]:
    tool_name = "search_arxiv_raw"
    raw_inputs = {
        "search_query": search_query,
        "id_list": id_list,
        "max_results": max_results,
        "start": start,
        "sort_by": sort_by,
        "sort_order": sort_order,
        "submitted_days_ago": submitted_days_ago,
    }
    try:
        normalized_search_query = " ".join(str(search_query or "").strip().split())
        normalized_id_list = [str(item).strip() for item in (id_list or []) if str(item).strip()]
        final_search_query = normalized_search_query or None
        submitted_days_ago_applied = False
        if submitted_days_ago is not None:
            if submitted_days_ago < 0:
                raise ArxivSearchValidationError(
                    "arxiv_invalid_query: submitted_days_ago must be greater than or equal to 0"
                )
            submitted_days_ago_applied = True
            submitted_date_query = build_arxiv_submitted_date_query(submitted_days_ago)
            final_search_query = f"({final_search_query}) AND {submitted_date_query}" if final_search_query else submitted_date_query

        validate_arxiv_search_request(
            search_query=final_search_query,
            id_list=normalized_id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        normalized_inputs = {
            "search_query": normalized_search_query,
            "id_list": normalized_id_list,
            "submitted_days_ago": submitted_days_ago,
            "submitted_days_ago_applied": submitted_days_ago_applied,
        }
        return _run_search(
            tool_name=tool_name,
            raw_inputs=raw_inputs,
            normalized_inputs=normalized_inputs,
            search_query=final_search_query,
            id_list=normalized_id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
            submitted_days_ago_applied=submitted_days_ago_applied,
        )
    except ArxivSearchValidationError as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="arXiv 参数校验失败",
            data=None,
            trace=_build_search_trace(
                tool_name=tool_name,
                raw_inputs=raw_inputs,
                normalized_inputs={"search_query": search_query, "id_list": id_list},
                final_search_query=None,
                id_list=id_list,
                sort_by=sort_by,
                sort_order=sort_order,
                start=start,
                max_results=max_results,
                submitted_days_ago_applied=False,
            ),
            error=make_tool_error("arxiv_invalid_query", str(exc)),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="arXiv 论文搜索失败",
            data=None,
            trace=_build_search_trace(
                tool_name=tool_name,
                raw_inputs=raw_inputs,
                normalized_inputs={"search_query": search_query, "id_list": id_list},
                final_search_query=None,
                id_list=id_list,
                sort_by=sort_by,
                sort_order=sort_order,
                start=start,
                max_results=max_results,
                submitted_days_ago_applied=False,
            ),
            error=make_tool_error("arxiv_search_failed", str(exc)),
        )


def search_arxiv_structured(
    query: Optional[str] = None,
    title_query: Optional[str] = None,
    author_query: Optional[str] = None,
    abstract_query: Optional[str] = None,
    categories: Optional[List[str]] = None,
    comment_query: Optional[str] = None,
    journal_ref_query: Optional[str] = None,
    report_number_query: Optional[str] = None,
    id_list: Optional[List[str]] = None,
    field_operator: str = "AND",
    category_operator: str = "OR",
    submitted_days_ago: Optional[int] = None,
    max_results: int = 10,
    start: int = 0,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> Dict[str, Any]:
    tool_name = "search_arxiv_structured"
    raw_inputs = {
        "query": query,
        "title_query": title_query,
        "author_query": author_query,
        "abstract_query": abstract_query,
        "categories": categories,
        "comment_query": comment_query,
        "journal_ref_query": journal_ref_query,
        "report_number_query": report_number_query,
        "id_list": id_list,
        "field_operator": field_operator,
        "category_operator": category_operator,
        "submitted_days_ago": submitted_days_ago,
        "max_results": max_results,
        "start": start,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }
    try:
        structured = build_arxiv_query_from_structured_params(
            query=query,
            title_query=title_query,
            author_query=author_query,
            abstract_query=abstract_query,
            categories=categories,
            comment_query=comment_query,
            journal_ref_query=journal_ref_query,
            report_number_query=report_number_query,
            id_list=id_list,
            field_operator=field_operator,
            category_operator=category_operator,
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
        normalized_inputs = dict(structured["normalized_inputs"])
        normalized_inputs.update(
            {
                "max_results": max_results,
                "start": start,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }
        )
        return _run_search(
            tool_name=tool_name,
            raw_inputs=raw_inputs,
            normalized_inputs=normalized_inputs,
            search_query=structured["final_search_query"],
            id_list=structured["id_list"],
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
        )
    except ArxivSearchValidationError as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="arXiv 参数校验失败",
            data=None,
            trace=_build_search_trace(
                tool_name=tool_name,
                raw_inputs=raw_inputs,
                normalized_inputs={"query": query, "id_list": id_list},
                final_search_query=None,
                id_list=id_list,
                sort_by=sort_by,
                sort_order=sort_order,
                start=start,
                max_results=max_results,
                submitted_days_ago_applied=False,
            ),
            error=make_tool_error("arxiv_invalid_query", str(exc)),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="arXiv 论文搜索失败",
            data=None,
            trace=_build_search_trace(
                tool_name=tool_name,
                raw_inputs=raw_inputs,
                normalized_inputs={"query": query, "id_list": id_list},
                final_search_query=None,
                id_list=id_list,
                sort_by=sort_by,
                sort_order=sort_order,
                start=start,
                max_results=max_results,
                submitted_days_ago_applied=False,
            ),
            error=make_tool_error("arxiv_search_failed", str(exc)),
        )


def get_paper_metadata(arxiv_id: str) -> Dict[str, Any]:
    tool_name = "get_paper_metadata"
    trace_inputs = {"arxiv_id": arxiv_id}
    try:
        paper = db_service.get_paper(arxiv_id)
        source = "database"
        if not paper:
            recommendation_service = _get_recommendation_service()
            source_paper = recommendation_service._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
            if not source_paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            paper = recommendation_service._materialize_paper_from_source(source_paper, arxiv_id)
            source = "arxiv"

        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已获取论文 {arxiv_id} 的元数据",
            data={"paper": paper},
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source=source),
        )
    except HTTPException as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary=f"未找到论文 {arxiv_id}",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="database"),
            error=make_tool_error("paper_not_found", exc.detail if hasattr(exc, "detail") else str(exc)),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="获取论文元数据失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs),
            error=make_tool_error("paper_metadata_failed", str(exc)),
        )


def get_paper_or_materialize(arxiv_id: str, paper_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    paper = db_service.get_paper(arxiv_id)
    if paper:
        return paper
    if paper_payload:
        normalized = _normalize_source_paper(paper_payload, arxiv_id)
        return _get_recommendation_service()._materialize_paper_from_source(normalized, arxiv_id)
    recommendation_service = _get_recommendation_service()
    source_paper = recommendation_service._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
    if not source_paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return recommendation_service._materialize_paper_from_source(source_paper, arxiv_id)
