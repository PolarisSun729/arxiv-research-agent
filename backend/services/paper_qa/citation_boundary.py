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
    """对单个答案文本做一次确定性的引用校验与清洗。

    - 校验结果以「本次」校验为准，但会继承上游已记录的 repair 轨迹，避免 debug 丢失修复信息；
    - 只要存在非法引用就给出统一 warning 并删除非法标记；没有非法引用时沿用上游 warning；
    - cited_source_ids 只提取合法单引用，非法引用不会进入前端的证据高亮列表。
    """
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
    """在 Agent 统一响应出口做确定性清洗，确保没有旧/混合引用漏到前端。

    处理两份答案：
    1. paper_qa_result["answer"]：Paper QA 工具返回的嵌套答案，清洗后回写到 result 副本；
    2. answer：Agent 最终展示给用户的答案，可能就是嵌套答案的投影，也可能是 Agent 自己改写过的文本。
    两者使用同一组 source_id 白名单（来自 paper_qa_result.sources），保证前端证据面板与正文引用一致。
    入参不会被原地修改，返回值中的 paper_qa_result 是浅拷贝。
    """
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
