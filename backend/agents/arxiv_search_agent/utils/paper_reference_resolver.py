"""论文引用线索提取工具。

这个模块负责把用户口语里的“这篇论文”“第一篇”“arXiv ID 2401.12345”
这类目标引用，解析成统一的引用线索，供后续真正的上下文解析层继续判定。

它只回答“用户文本里出现了哪种引用表达”，不再回答“最终是哪篇论文”。
真正落地到 paper/arxiv_id 必须由后续模块结合页面状态、最近结果和用户选择来完成。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

from .text_utils import _matches_any, _normalize_text

_REFERENCE_HINT_FIELDS = (
    "reference_type",
    "value",
    "confidence",
    "source",
    "requires_context",
    "status",
    "reason",
    "final_target_resolved",
)

_PREFERENCE_ORDINAL_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
    "十三": 13,
    "十四": 14,
    "十五": 15,
    "十六": 16,
    "十七": 17,
    "十八": 18,
    "十九": 19,
    "二十": 20,
    "两": 2,
}

_SPECIAL_ORDINAL_MAP = {
    "最后一": -1,
    "最后1": -1,
    "最后": -1,
    "末一": -1,
    "末": -1,
}


def _safe_int(value: Any, default: int = 0) -> int:
    """安全地把任意值转成 int；失败时返回默认值。

    这里主要服务于序号解析场景，目标是让自然语言里的“第3篇”“三”之类表达
    在转换失败时不要直接抛异常，而是回退到可控默认值。
    """
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _build_reference_hint(
    *,
    reference_type: str,
    value: Any = None,
    confidence: float,
    source: str,
    requires_context: bool,
    reason: Optional[str] = None,
    target: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造统一引用线索，明确禁止在本层落地到具体论文。

    旧实现会在同一个函数里读取 selected_paper/last_papers 并返回 arxiv_id。
    这里保留兼容字段但全部置空，让下游必须显式接入最终目标解析层后才能执行动作。
    """
    status = "unknown" if reference_type == "unknown" else "hint_extracted"
    hint = {
        "status": status,
        "reference_type": reference_type,
        "value": value,
        "confidence": confidence,
        "source": source,
        "requires_context": requires_context,
        "reason": reason,
        "final_target_resolved": False,
        "target": target,
        "paper": None,
        "arxiv_id": None,
        "title": None,
        "matched_from": source,
    }
    hint["reference_hint"] = {key: hint.get(key) for key in _REFERENCE_HINT_FIELDS}
    return hint


def _normalize_context_paper(raw: Any) -> Dict[str, Any]:
    """把上下文中的论文对象规范化成统一字段结构。

    上游不同模块保存论文时可能使用 arxivId/id、published/updated、abs_url/url
    等不同命名，这里统一收口成 node 层可稳定读取的标准字段集合。
    """
    paper = raw if isinstance(raw, Mapping) else {}
    arxiv_id = str(paper.get("arxiv_id") or paper.get("arxivId") or paper.get("id") or "").strip()
    if arxiv_id.startswith("http"):
        # 某些上下文里保存的是 arXiv 页面链接，这里统一裁剪成纯 arXiv ID。
        arxiv_id = arxiv_id.rsplit("/", 1)[-1]

    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors_value = [item.strip() for item in authors.split(",") if item.strip()]
    elif isinstance(authors, (list, tuple, set)):
        authors_value = [str(item).strip() for item in authors if str(item).strip()]
    else:
        authors_value = []

    categories = paper.get("categories", [])
    if isinstance(categories, str):
        categories_value = [item.strip() for item in categories.split(",") if item.strip()]
    elif isinstance(categories, (list, tuple, set)):
        categories_value = [str(item).strip() for item in categories if str(item).strip()]
    else:
        categories_value = []

    abstract = str(paper.get("abstract") or paper.get("summary") or "").strip()
    title = str(paper.get("title") or "").strip()
    published_date = str(
        paper.get("published_date")
        or paper.get("published")
        or paper.get("publishedAt")
        or paper.get("updated")
        or paper.get("updatedAt")
        or ""
    ).strip()
    url = str(paper.get("url") or paper.get("abs_url") or paper.get("absUrl") or paper.get("pdf_url") or paper.get("pdfUrl") or "").strip()

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "summary": abstract,
        "authors": authors_value,
        "categories": categories_value,
        "published_date": published_date,
        "published": published_date,
        "url": url,
        "abs_url": str(paper.get("abs_url") or paper.get("absUrl") or paper.get("url") or "").strip(),
        "pdf_url": str(paper.get("pdf_url") or paper.get("pdfUrl") or "").strip(),
    }


def _merge_context_paper_lists(*paper_lists: Any) -> List[Dict[str, Any]]:
    """合并多路论文列表，并按 arXiv ID/标题去重保序。

    该函数主要用于把 recent/search/papers 等不同上下文字段整合成单一候选集，
    方便后续按序号或显式 ID 做统一定位。
    """
    merged: List[Dict[str, Any]] = []
    seen_keys = set()

    for paper_list in paper_lists:
        if not isinstance(paper_list, list):
            continue
        for raw_paper in paper_list:
            if not isinstance(raw_paper, Mapping):
                continue
            paper = _normalize_context_paper(raw_paper)
            identity = str(paper.get("arxiv_id") or "").strip() or str(paper.get("title") or "").strip().lower()
            if not identity or identity in seen_keys:
                continue
            seen_keys.add(identity)
            merged.append(paper)

    return merged


def _extract_selected_paper(context: Any) -> Optional[Dict[str, Any]]:
    """从上下文中恢复当前已选中的目标论文。

    这里会按多个常见字段名依次尝试，并在必要时从 paper_qa_result 或简化字段
    arxiv_id/title 中补构一个最小论文对象。
    """
    if not isinstance(context, Mapping):
        return None

    for key in (
        "selected_paper",
        "current_paper",
        "active_paper",
        "target_paper",
        "last_target_paper",
        "current_selected_paper",
    ):
        raw_paper = context.get(key)
        if isinstance(raw_paper, Mapping):
            paper = _normalize_context_paper(raw_paper)
            if paper.get("arxiv_id") or paper.get("title"):
                return paper

    paper_qa_result = context.get("paper_qa_result")
    if isinstance(paper_qa_result, Mapping):
        paper = _normalize_context_paper(
            {
                "arxiv_id": paper_qa_result.get("arxiv_id"),
                "title": paper_qa_result.get("title"),
            }
        )
        if paper.get("arxiv_id") or paper.get("title"):
            return paper

    arxiv_id = str(context.get("arxiv_id") or "").strip()
    title = str(context.get("paper_title") or context.get("title") or "").strip()
    if arxiv_id or title:
        paper = _normalize_context_paper({"arxiv_id": arxiv_id, "title": title})
        if paper.get("arxiv_id") or paper.get("title"):
            return paper

    return None


def _extract_last_papers(context: Any) -> List[Dict[str, Any]]:
    """从上下文中提取最近一轮可供引用的论文列表。

    这些结果通常来自搜索返回值或最近浏览记录，是“第一篇/第二篇”这类引用的主要候选池。
    """
    if not isinstance(context, Mapping):
        return []
    return _merge_context_paper_lists(
        context.get("last_papers") or [],
        context.get("papers") or [],
        context.get("search_results") or [],
        context.get("recent_papers") or [],
    )


def _parse_target_reference(message: str) -> Dict[str, Any]:
    """从用户消息里提取论文引用线索。

    支持五类典型表达：
    1. 显式 arXiv ID；
    2. “第一篇 / 第 2 篇”这类序号引用；
    3. “最后一篇”这类相对位置引用；
    4. “这篇论文”这类上下文指代；
    5. 偏好动作里的裸数字引用。
    """
    text = _normalize_text(message)
    if not text:
        return _build_reference_hint(
            reference_type="unknown",
            value=None,
            confidence=0.0,
            source="empty_message",
            requires_context=False,
            reason="用户输入为空，无法提取论文引用线索。",
            target=None,
        )

    # 先处理“最后一篇 / 最后一篇论文”这类相对位置引用。
    # 这类表达只说明相对位置，必须等上下文解析层拿到最近结果列表后才能落地。
    special_ordinal_match = re.search(r"(最后一|最后1|最后|末一|末)\s*(?:篇|个)?(?:论文|paper)?", text)
    if special_ordinal_match:
        raw_value = special_ordinal_match.group(1)
        ordinal = _SPECIAL_ORDINAL_MAP.get(raw_value)
        if ordinal is not None:
            return _build_reference_hint(
                reference_type="last_item",
                value="last",
                confidence=0.88,
                source="last_item_expression",
                requires_context=True,
                target={
                    "target_type": "last_item",
                    "target_value": ordinal,
                    "ordinal": ordinal,
                },
            )

    # 先尝试识别最明确的显式 arXiv ID；即便文本置信度很高，本层也只返回线索，
    # 避免下游绕过最终目标解析层直接触发 QA、下载或偏好写入。
    arxiv_match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", text, flags=re.IGNORECASE)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        return _build_reference_hint(
            reference_type="arxiv_id",
            value=arxiv_id,
            confidence=0.98,
            source="explicit_arxiv_id",
            requires_context=False,
            target={
                "target_type": "arxiv_id",
                "target_value": arxiv_id,
                "arxiv_id": arxiv_id,
            },
        )

    # 再尝试解析“第几篇”形式的序号引用。这里必须要求“第/篇/论文”等目标锚点，
    # 避免把普通数量或章节数字误当成论文列表下标。
    ordinal_match = re.search(
        r"(?:第\s*([一二三四五六七八九十两]{1,3}|[1-9]|1[0-9]|20)\s*(?:篇|个)?(?:论文|paper)?|([一二三四五六七八九十两]{1,3}|[1-9]|1[0-9]|20)\s*(?:篇|个)(?:论文|paper)?)",
        text,
    )
    if ordinal_match:
        raw_value = ordinal_match.group(1) or ordinal_match.group(2)
        ordinal = _PREFERENCE_ORDINAL_MAP.get(raw_value)
        if ordinal is None:
            ordinal = _safe_int(raw_value, default=0)
        if 1 <= ordinal <= 20:
            return _build_reference_hint(
                reference_type="ordinal",
                value=ordinal,
                confidence=0.92,
                source="ordinal_expression",
                requires_context=True,
                target={
                    "target_type": "ordinal",
                    "target_value": ordinal,
                    "ordinal": ordinal,
                },
            )

    # 上下文指代必须放在序号引用之后；真实提问里常见“这第二篇论文/该第 2 篇论文”，
    # 如果先命中“这篇/该论文”，就会掩盖更明确的序号线索。
    if _matches_any(
        text,
        (
            r"这篇",
            r"这个",
            r"这一个",
            r"该论文",
            r"这篇论文",
            r"本文",
            r"当前选中",
            r"当前论文",
            r"当前这篇",
            r"刚才那篇",
            r"刚才这篇",
            r"刚才提到",
        ),
    ):
        return _build_reference_hint(
            reference_type="context_paper",
            value="current",
            confidence=0.76,
            source="contextual_reference",
            requires_context=True,
            target={
                "target_type": "context_paper",
                "target_value": "selected_or_recent",
            },
        )

    bare_match = re.search(r"(?<!\d)([1-9]|1[0-9]|20)(?!\d)", text)
    if bare_match and (text.strip() in {bare_match.group(1), f"第{bare_match.group(1)}", f"第{bare_match.group(1)}篇"} or any(token in text for token in ("喜欢", "不喜欢", "收藏", "标记", "取消", "撤销"))):
        ordinal = _safe_int(bare_match.group(1), default=0)
        if 1 <= ordinal <= 20:
            return _build_reference_hint(
                reference_type="bare_number",
                value=ordinal,
                confidence=0.56,
                source="bare_number",
                requires_context=True,
                target={
                    "target_type": "bare_number",
                    "target_value": ordinal,
                    "ordinal": ordinal,
                },
            )

    return _build_reference_hint(
        reference_type="unknown",
        value=None,
        confidence=0.0,
        source="no_reference",
        requires_context=False,
        reason="用户输入里没有明确论文目标线索。",
        target=None,
    )


def _resolve_paper_reference(message: str, context: Any) -> Dict[str, Any]:
    """返回论文引用线索，不再解析最终目标论文。

    函数名暂时保留给现有调用点复用，但语义已经从“目标论文解析”降级为
    “Reference Hint Extractor”。context 参数仅用于保持签名兼容，不能在这里读取
    selected_paper/last_papers 并绑定具体论文，避免无目标请求被隐式落到默认论文上。
    """
    del context
    hint = _parse_target_reference(message)
    if hint["reference_type"] == "unknown":
        return hint
    return hint


__all__ = [
    "_extract_last_papers",
    "_extract_selected_paper",
    "_merge_context_paper_lists",
    "_normalize_context_paper",
    "_parse_target_reference",
    "_resolve_paper_reference",
]
