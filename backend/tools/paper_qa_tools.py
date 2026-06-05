from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from dependencies import get_paper_qa_service as get_dependency_paper_qa_service
from .tool_result import make_tool_error, make_tool_result, make_tool_trace


@lru_cache(maxsize=1)
def _get_paper_qa_service():
    return get_dependency_paper_qa_service()


def check_paper_qa_index(arxiv_id: str) -> Dict[str, Any]:
    tool_name = "check_paper_qa_index"
    trace_inputs = {"arxiv_id": arxiv_id}
    try:
        data = _get_paper_qa_service().get_qa_status(arxiv_id)
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已检查论文 {arxiv_id} 的 QA 索引状态",
            data=data,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="database_service"),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="检查 QA 索引失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="database_service"),
            error=make_tool_error("qa_status_failed", str(exc)),
        )


def build_paper_qa_index(arxiv_id: str, loading_method: str = "docling") -> Dict[str, Any]:
    tool_name = "build_paper_qa_index"
    trace_inputs = {"arxiv_id": arxiv_id, "loading_method": loading_method}
    try:
        data = _get_paper_qa_service().build_qa_index(arxiv_id, loading_method=loading_method)
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已为论文 {arxiv_id} 构建 QA 索引",
            data=data,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="构建 QA 索引失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
            error=make_tool_error("qa_index_failed", str(exc)),
        )


def answer_paper_question(
    arxiv_id: str,
    question: str,
    stricter_grounding: Optional[bool] = None,
    top_k: Optional[int] = None,
    enable_query_rewrite: Optional[bool] = None,
    enable_hyde: Optional[bool] = None,
    enable_keyword_search: Optional[bool] = None,
    enable_llm_rerank: Optional[bool] = None,
    debug: Optional[bool] = None,
) -> Dict[str, Any]:
    tool_name = "answer_paper_question"
    trace_inputs = {
        "arxiv_id": arxiv_id,
        "question": question,
        "stricter_grounding": stricter_grounding,
        "top_k": top_k,
        "enable_query_rewrite": enable_query_rewrite,
        "enable_hyde": enable_hyde,
        "enable_keyword_search": enable_keyword_search,
        "enable_llm_rerank": enable_llm_rerank,
        "debug": debug,
    }
    payload = {
        "question": question,
        # Observer 发现 grounding 不足后会打开该开关，提示下游生成阶段收紧证据引用要求。
        "stricter_grounding": stricter_grounding,
        "top_k": top_k,
        "enable_query_rewrite": enable_query_rewrite,
        "enable_hyde": enable_hyde,
        "enable_keyword_search": enable_keyword_search,
        "enable_llm_rerank": enable_llm_rerank,
        "debug": debug,
    }
    try:
        data = _get_paper_qa_service().answer_question(arxiv_id, payload)
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已回答论文 {arxiv_id} 的问题",
            data=data,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="回答论文问题失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
            error=make_tool_error("paper_qa_answer_failed", str(exc)),
        )
