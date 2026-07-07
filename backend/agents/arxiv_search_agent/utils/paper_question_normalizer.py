"""论文 QA 问题归一化工具。

本模块只处理“目标论文已经解析完成之后”的单篇 QA 问题清洗：
把“第二篇论文 / 最后一篇 / arXiv 2401.xxxxx”这类列表或显式目标引用，
改写成单篇语境下的“这篇论文”，避免最终 QA 模型在单篇上下文里继续寻找不存在的“第二篇”。
"""

from __future__ import annotations

import re
from typing import Any, Mapping

_CHINESE_ORDINAL = "一二三四五六七八九十两"
_LIST_SOURCE_PREFIX = (
    r"(?:(?:当前|页面|展示|列表|论文列表|搜索结果|检索结果|查询结果|推荐|推荐结果)"
    r"(?:里|中|中的|里的|的)?\s*)?"
)
_ORDINAL_TARGET_PATTERN = re.compile(
    rf"{_LIST_SOURCE_PREFIX}(?:这|该|那)?\s*"
    rf"(?:第\s*[{_CHINESE_ORDINAL}0-9]+\s*篇|[{_CHINESE_ORDINAL}0-9]+\s*篇)"
    r"(?:\s*(?:论文|paper))?",
    flags=re.IGNORECASE,
)
_LAST_TARGET_PATTERN = re.compile(
    rf"{_LIST_SOURCE_PREFIX}(?:最后一|最后1|最后|末一|末)\s*篇(?:\s*(?:论文|paper))?",
    flags=re.IGNORECASE,
)
_CONTEXT_TARGET_PATTERN = re.compile(
    r"(?:刚才那篇|刚才这篇|刚才提到的那篇|当前选中论文|当前论文|这篇论文|这篇|该论文|本文)",
    flags=re.IGNORECASE,
)
_ARXIV_TARGET_PATTERN = re.compile(
    r"(?:arxiv\s*(?:id)?\s*[:：]?\s*)?(?:https?://arxiv\.org/(?:abs|pdf)/)?\d{4}\.\d{4,5}(?:v\d+)?",
    flags=re.IGNORECASE,
)


def normalize_single_paper_qa_question(message: Any, paper_reference: Any) -> str:
    """把已解析目标后的用户问题改写为单篇论文语境的问题。

    这里必须依赖 paper_reference 的 final_target_resolved / reference_type：
    只有目标已经唯一落地时，才允许清理“第 N 篇”这类列表引用，避免误删“第二节”
    “第二个实验”等论文内部定位。
    """
    question = _tidy_question(message)
    if not question:
        return ""

    reference = _flatten_reference(paper_reference)
    if not _is_target_resolved(reference):
        return question

    reference_type = str(reference.get("reference_type") or "").strip()
    if reference_type in {"ordinal", "bare_number"}:
        return _replace_target_reference(question, _ORDINAL_TARGET_PATTERN)
    if reference_type == "last_item":
        return _replace_target_reference(question, _LAST_TARGET_PATTERN)
    if reference_type == "context_paper":
        return _replace_target_reference(question, _CONTEXT_TARGET_PATTERN)
    if reference_type == "arxiv_id" or _ARXIV_TARGET_PATTERN.search(question):
        return _replace_target_reference(question, _ARXIV_TARGET_PATTERN)
    return question


def _flatten_reference(paper_reference: Any) -> dict[str, Any]:
    """收敛不同链路里的 paper_ref / reference_hint 形态，供归一化逻辑稳定读取。"""
    if not isinstance(paper_reference, Mapping):
        return {}
    reference = dict(paper_reference)
    nested_hint = reference.get("reference_hint")
    if isinstance(nested_hint, Mapping):
        for key, value in nested_hint.items():
            reference.setdefault(key, value)
    return reference


def _is_target_resolved(reference: Mapping[str, Any]) -> bool:
    """确认目标已经唯一解析；未解析时禁止清理序号引用，避免把问题改坏后误执行 QA。"""
    return bool(
        reference.get("final_target_resolved")
        or reference.get("arxiv_id")
        or (isinstance(reference.get("target"), Mapping) and reference.get("target"))
    )


def _replace_target_reference(question: str, pattern: re.Pattern[str]) -> str:
    normalized = pattern.sub("这篇论文", question, count=1)
    return _tidy_question(normalized)


def _tidy_question(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    # 中文问题内部不需要保留英文空格；保留这一步可以把“第 2 篇”替换后的多余空白收口。
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", text)
    text = re.sub(r"([（(])\s+", r"\1", text)
    text = re.sub(r"\s+([）)])", r"\1", text)
    return text.strip()


__all__ = ["normalize_single_paper_qa_question"]
