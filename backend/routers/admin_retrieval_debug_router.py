"""管理端检索调试路由（仅管理员）。

对一篇已建索引的论文跑一次单轮检索，返回各路召回、融合、rerank 与最终上下文的逐 chunk 名次，
用来观察多路召回和融合的效果：
- POST /admin/retrieval-debug/runs   传入论文和问题，按给定开关执行一次检索。

和问答走同一条 RetrievalPipeline 和同样的论文目标解析，但不经过研究图，也不生成答案。
改写、HyDE 和 rerank 会调用模型，因此只对管理员开放。接口用同步 def，检索交给线程池执行。
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from dependencies import get_enhanced_retrieval_service, get_paper_catalog_store, get_paper_qa_index_store
from services.paper_evidence_research.dependencies.production_adapters import PaperRetrievalTargetResolver
from services.retrieval.contracts import RetrievalOptions
from services.retrieval.debug_view import build_retrieval_debug_view

router = APIRouter(prefix="/admin/retrieval-debug", tags=["admin-retrieval-debug"])


class RetrievalDebugRequest(BaseModel):
    """开关缺省（None）时沿用检索配置的默认值。"""

    arxiv_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_table_structured_route: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    enable_context_expansion: Optional[bool] = None


@router.post("/runs")
def run_retrieval_debug(
    payload: RetrievalDebugRequest,
    retrieval_service=Depends(get_enhanced_retrieval_service),
    index_store=Depends(get_paper_qa_index_store),
    catalog_store=Depends(get_paper_catalog_store),
):
    """执行一次检索并返回逐 chunk 的名次视图；论文没有可用索引时返回 404。"""
    arxiv_id = payload.arxiv_id.strip()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")
    target = PaperRetrievalTargetResolver(index_store=index_store, catalog_store=catalog_store)(arxiv_id)
    if target is None:
        raise HTTPException(status_code=404, detail="paper index not available")

    started = perf_counter()
    result = retrieval_service.retrieval_pipeline.retrieve(
        user_query=question,
        collection_name=target.collection_name,
        paper_context=target.paper_context,
        options=RetrievalOptions(
            top_k=payload.top_k,
            enable_query_rewrite=payload.enable_query_rewrite,
            enable_hyde=payload.enable_hyde,
            enable_keyword_search=payload.enable_keyword_search,
            enable_table_structured_route=payload.enable_table_structured_route,
            enable_llm_rerank=payload.enable_llm_rerank,
            enable_context_expansion=payload.enable_context_expansion,
            debug=True,
            # 调试页每点一次运行就是一次检索，不为它额外落盘 trace 文件。
            export_trace=False,
            include_stage_chunks=True,
        ),
    )
    elapsed_ms = round((perf_counter() - started) * 1000, 1)
    return {
        "arxiv_id": arxiv_id,
        "title": target.paper_context.get("title") or "",
        "active_build_id": target.paper_context.get("active_build_id") or "",
        "question": question,
        "elapsed_ms": elapsed_ms,
        **build_retrieval_debug_view(result, chunk_key=retrieval_service.fusion_service.chunk_unique_key),
    }
