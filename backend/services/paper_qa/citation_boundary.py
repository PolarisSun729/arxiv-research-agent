"""Paper QA 与 Agent 最终响应共用的引用出站边界。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable

from services.paper_qa.citation_contract import (
    extract_cited_source_ids,
    strip_invalid_citations,
    validate_citations,
)

_CITATION_WARNING = "部分引用未通过校验"


def _source_ids_from_result(paper_qa_result: Mapping[str, Any]) -> list[str]:
    """只从新格式 sources.source_id 读取可引用 ID，不兼容旧 chunk_id。"""
    source_ids: list[str] = []
    sources = paper_qa_result.get("sources")
    if not isinstance(sources, Iterable) or isinstance(sources, (str, bytes, Mapping)):
        return source_ids
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        source_id = str(source.get("source_id") or "").strip()
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def _sanitize_answer(
    answer: Any,
    source_ids: list[str],
    previous_debug: Any = None,
    previous_warning: Any = None,
) -> Dict[str, Any]:
    text = str(answer or "")
    debug = validate_citations(text, source_ids)
    if isinstance(previous_debug, Mapping):
        # 保留 Paper QA 已完成的一次修复轨迹，同时以最终边界的结果为准。
        for key in ("repair_attempted", "repair_succeeded"):
            if key in previous_debug:
                debug[key] = previous_debug[key]
    warning = _CITATION_WARNING if debug["invalid_citations"] else (
        str(previous_warning) if previous_warning else None
    )
    return {
        "answer": strip_invalid_citations(text, source_ids) if warning else text,
        "cited_source_ids": extract_cited_source_ids(text, source_ids),
        "citation_debug": debug,
        "citation_warning": warning,
    }


def sanitize_agent_response_citations(
    *, answer: Any, paper_qa_result: Any
) -> Dict[str, Any]:
    """在 Agent 统一响应出口做确定性清洗，确保没有旧/混合引用漏到前端。"""
    result = dict(paper_qa_result) if isinstance(paper_qa_result, Mapping) else None
    source_ids = _source_ids_from_result(result) if result is not None else []

    nested_answer = result.get("answer") if result is not None else None
    nested = None
    if result is not None and nested_answer is not None:
        nested = _sanitize_answer(
            nested_answer,
            source_ids,
            result.get("citation_debug"),
            result.get("citation_warning"),
        )
        result.update(nested)

    raw_answer = str(answer or "")
    # Paper QA 的答案会被 graph 投影到 state.answer；这里优先复用已经清洗过的嵌套答案。
    if nested is not None and (not raw_answer or raw_answer == str(nested_answer or "")):
        final = nested
    else:
        final = _sanitize_answer(raw_answer, source_ids)

    return {
        "answer": final["answer"],
        "paper_qa_result": result,
        "cited_source_ids": final["cited_source_ids"],
        "citation_debug": final["citation_debug"],
        "citation_warning": final["citation_warning"],
    }
