from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from dependencies import (
    get_database_service,
    get_enhanced_retrieval_service,
    get_generation_service,
    get_paper_qa_service,
    get_vector_store_service,
)
from routers.qa_utils import build_qa_diagnostic, get_latest_retrieval_trace, sanitize_trace_slug

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/paper/{arxiv_id}", tags=["paper-qa"])


class QaRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    debug: Optional[bool] = None


@router.get("/qa-status")
async def get_paper_qa_status(arxiv_id: str, paper_qa_service=Depends(get_paper_qa_service)):
    try:
        return paper_qa_service.get_qa_status(arxiv_id)
    except Exception as exc:
        logger.error("Error getting paper QA status: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qa-diagnose")
async def diagnose_paper_qa(
    arxiv_id: str,
    sample_limit: int = Query(3, ge=0, le=20),
    db_service=Depends(get_database_service),
    vector_store_service=Depends(get_vector_store_service),
):
    try:
        return build_qa_diagnostic(
            db_service=db_service,
            vector_store_service=vector_store_service,
            arxiv_id=arxiv_id,
            sample_limit=sample_limit,
        )
    except Exception as exc:
        logger.error("Error diagnosing paper QA: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/qa-trace/latest")
async def download_latest_qa_trace(
    arxiv_id: str,
    format: str = Query("md"),
    trace_name: Optional[str] = Query(None),
    enhanced_retrieval_service=Depends(get_enhanced_retrieval_service),
):
    try:
        normalized_format = str(format or "md").strip().lower()
        if normalized_format not in {"md", "json"}:
            raise HTTPException(status_code=400, detail="format must be md or json")

        trace_root = Path(str(enhanced_retrieval_service.trace_export_dir))
        paper_dir = trace_root / sanitize_trace_slug(arxiv_id)
        trace_file = None

        if trace_name:
            safe_name = Path(str(trace_name)).name
            if safe_name != trace_name:
                raise HTTPException(status_code=400, detail="Invalid trace_name")
            candidate = paper_dir / safe_name
            if candidate.exists() and candidate.is_file():
                trace_file = candidate
        else:
            trace_file = get_latest_retrieval_trace(enhanced_retrieval_service, arxiv_id, normalized_format)

        if trace_file is None:
            raise HTTPException(status_code=404, detail="No retrieval trace found for this paper")

        return FileResponse(
            path=str(trace_file),
            filename=f"{arxiv_id}_retrieval_trace.{normalized_format}",
            media_type="application/json" if normalized_format == "json" else "text/markdown",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error downloading QA trace: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/create-qa-index")
async def create_paper_qa_index(
    arxiv_id: str,
    loading_method: str = Query("docling"),
    paper_qa_service=Depends(get_paper_qa_service),
):
    try:
        return paper_qa_service.build_qa_index(arxiv_id, loading_method=loading_method)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error creating QA index: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/qa")
async def qa_paper(arxiv_id: str, payload: QaRequest, paper_qa_service=Depends(get_paper_qa_service)):
    try:
        return paper_qa_service.answer_question(arxiv_id, payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error in QA: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/qa/stream")
async def qa_paper_stream(
    arxiv_id: str,
    payload: QaRequest,
    paper_qa_service=Depends(get_paper_qa_service),
    generation_service=Depends(get_generation_service),
):
    question = payload.question.strip()
    logger.info("QA stream request for paper: %s, question: %s", arxiv_id, question)

    _, search_results, qa_context, retrieval_debug = paper_qa_service.build_qa_context(arxiv_id, payload)
    source_payload = paper_qa_service.build_source_payload(search_results)

    def sse_event(event_name: str, data: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_stream():
        try:
            yield sse_event(
                "meta",
                {
                    "status": "started",
                    "arxiv_id": arxiv_id,
                    "question": question,
                    "sources": source_payload,
                    "image_inputs": qa_context["image_inputs"],
                    "asset_metadata": qa_context["asset_metadata"],
                    "retrieval_debug": retrieval_debug,
                },
            )

            for chunk in generation_service.stream_qwen_responses(
                query=question,
                context=qa_context["text_context"],
                model_name="qwen3.6-plus",
                image_inputs=qa_context["image_inputs"],
                asset_metadata=[item for item in qa_context["asset_metadata"] if item.get("chunk_type") == "figure"],
            ):
                if chunk.get("type") == "delta":
                    yield sse_event("delta", {"delta": chunk.get("delta", "")})
                elif chunk.get("type") == "completed":
                    yield sse_event(
                        "done",
                        {
                            "status": "success",
                            "answer": chunk.get("answer", ""),
                            "sources": source_payload,
                            "image_inputs": qa_context["image_inputs"],
                            "asset_metadata": qa_context["asset_metadata"],
                            "retrieval_debug": retrieval_debug,
                            "usage": chunk.get("usage"),
                        },
                    )
                    return

            yield sse_event(
                "done",
                {
                    "status": "success",
                    "answer": "",
                    "sources": source_payload,
                    "image_inputs": qa_context["image_inputs"],
                    "asset_metadata": qa_context["asset_metadata"],
                    "retrieval_debug": retrieval_debug,
                    "usage": None,
                },
            )
        except Exception as exc:
            logger.error("Error in QA stream: %s", str(exc))
            yield sse_event("error", {"status": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
