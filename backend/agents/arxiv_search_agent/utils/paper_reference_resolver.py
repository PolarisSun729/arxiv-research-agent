"""论文引用解析与上下文目标论文恢复工具。

这个模块负责把用户口语里的“这篇论文”“第一篇”“arXiv ID 2401.12345”
这类目标引用，解析成统一的论文定位结果，供论文阅读、偏好更新等节点复用。

它本质上做的是“从对话上下文恢复用户到底在指哪篇论文”这件事，
因此同时依赖消息文本和 state/context 中保留的最近论文列表、当前选中文献等信息。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

from .text_utils import _matches_any, _normalize_text

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


def _parse_target_reference(message: str) -> Optional[Dict[str, Any]]:
    """从用户消息里解析目标论文引用方式。

    支持三类典型表达：
    1. 显式 arXiv ID；
    2. “这篇论文”这类上下文指代；
    3. “第一篇 / 第 2 篇”这类序号引用。
    """
    text = _normalize_text(message)
    if not text:
        return None

    # 先处理“最后一篇 / 最后一篇论文”这类相对位置引用。
    # 这类表达没有显式数字，如果不优先识别，后续往往会回退成 selected_paper，
    # 从而错误命中当前默认选中的第一篇推荐结果。
    special_ordinal_match = re.search(r"(最后一|最后1|最后|末一|末)\s*(?:篇|个)?(?:论文|paper)?", text)
    if special_ordinal_match:
        raw_value = special_ordinal_match.group(1)
        ordinal = _SPECIAL_ORDINAL_MAP.get(raw_value)
        if ordinal is not None:
            return {
                "target_type": "ordinal",
                "target_value": ordinal,
                "ordinal": ordinal,
            }

    # 先尝试识别最明确的显式 arXiv ID，因为这类引用优先级最高、歧义最小。
    arxiv_match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", text, flags=re.IGNORECASE)
    if arxiv_match:
        return {
            "target_type": "arxiv_id",
            "target_value": arxiv_match.group(1),
            "arxiv_id": arxiv_match.group(1),
        }

    # 再尝试解析“第几篇”形式的序号引用，用于指向上一轮搜索结果中的论文。
    # 这里必须要求“第/篇/论文”等目标锚点，避免“讲一下第二篇”先把“一下”误判成第一篇。
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
            return {
                "target_type": "ordinal",
                "target_value": ordinal,
                "ordinal": ordinal,
            }

    # 上下文指代必须放在序号引用之后；真实提问里常见“这第二篇论文/该第 2 篇论文”，
    # 如果先命中“这篇/该论文”，就会错误回退到默认 selected_paper。
    if _matches_any(
        text,
        (
            r"这篇",
            r"该论文",
            r"这篇论文",
            r"本文",
            r"当前选中",
            r"当前论文",
            r"当前这篇",
        ),
    ):
        return {
            "target_type": "context_paper",
            "target_value": "selected_or_recent",
        }

    bare_match = re.search(r"(?<!\d)([1-9]|1[0-9]|20)(?!\d)", text)
    if bare_match and (text.strip() in {bare_match.group(1), f"第{bare_match.group(1)}", f"第{bare_match.group(1)}篇"} or any(token in text for token in ("喜欢", "不喜欢", "收藏", "标记", "取消", "撤销"))):
        ordinal = _safe_int(bare_match.group(1), default=0)
        if 1 <= ordinal <= 20:
            return {
                "target_type": "ordinal",
                "target_value": ordinal,
                "ordinal": ordinal,
            }

    return None


def _resolve_paper_reference(message: str, context: Any) -> Dict[str, Any]:
    """结合消息文本和上下文，解析出用户真正指向的论文对象。

    这是整个模块的主入口。它会先解析引用类型，再按引用类型去 selected_paper、
    last_papers 或显式 arXiv ID 中定位目标，最终返回统一的 success/failed 结果结构。
    """
    reference = _parse_target_reference(message)
    last_papers = _extract_last_papers(context)
    selected_paper = _extract_selected_paper(context)

    def build_success(paper_payload: Optional[Mapping[str, Any]], *, target: Optional[Dict[str, Any]], matched_from: str) -> Dict[str, Any]:
        # 所有成功分支都通过统一构造器返回，保证字段形态稳定，便于上层节点直接消费。
        normalized_paper = _normalize_context_paper(paper_payload or {})
        resolved_arxiv_id = str(normalized_paper.get("arxiv_id") or "").strip() or None
        resolved_title = str(normalized_paper.get("title") or "").strip() or None
        return {
            "status": "success",
            "reason": None,
            "target": target,
            "paper": normalized_paper,
            "arxiv_id": resolved_arxiv_id,
            "title": resolved_title,
            "matched_from": matched_from,
        }

    if reference is None:
        # 用户没有显式说“哪篇”，则优先回退到当前选中论文；如果最近结果里只有一篇，也可直接默认命中。
        if selected_paper is not None:
            return build_success(selected_paper, target=None, matched_from="selected_paper")
        if len(last_papers) == 1:
            return build_success(last_papers[0], target=None, matched_from="single_recent_paper")
        return {
            "status": "failed",
            "reason": "没有解析到目标论文。请先搜索论文，或直接提供 arXiv ID，或使用“第一篇 / 第二篇”指定搜索结果中的论文。",
            "target": None,
            "paper": None,
            "arxiv_id": None,
            "title": None,
        }

    if reference["target_type"] == "context_paper":
        # “这篇论文”属于纯上下文引用，因此必须依赖 selected_paper 或唯一 recent paper 才能落地。
        if selected_paper is not None:
            return build_success(selected_paper, target=reference, matched_from="selected_paper")
        if len(last_papers) == 1:
            return build_success(last_papers[0], target=reference, matched_from="single_recent_paper")
        return {
            "status": "failed",
            "reason": "当前没有可直接指代的目标论文。请先搜索论文、传入 selected_paper，或用“第一篇 / 第二篇”明确指定。",
            "target": reference,
            "paper": None,
            "arxiv_id": None,
            "title": None,
        }

    if reference["target_type"] == "ordinal":
        ordinal = int(reference["target_value"])
        if not last_papers:
            return {
                "status": "failed",
                "reason": "没有可用的上一轮搜索结果，请先搜索论文，或者直接提供 arXiv ID",
                "target": reference,
                "paper": None,
                "arxiv_id": None,
                "title": None,
            }
        if ordinal < 0:
            ordinal = len(last_papers) + ordinal + 1
            reference = dict(reference)
            reference["resolved_ordinal"] = ordinal
        if ordinal > len(last_papers):
            return {
                "status": "failed",
                "reason": f"上一轮搜索结果只有 {len(last_papers)} 篇，无法选择第 {ordinal} 篇",
                "target": reference,
                "paper": None,
                "arxiv_id": None,
                "title": None,
            }
        # 序号是按用户视角从 1 开始计数，因此内部访问列表时要减 1。
        paper = dict(last_papers[ordinal - 1])
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if not arxiv_id:
            return {
                "status": "failed",
                "reason": "上一轮结果中目标论文缺少 arXiv ID，无法执行偏好动作",
                "target": reference,
                "paper": paper,
                "arxiv_id": None,
                "title": paper.get("title"),
            }
        return build_success(paper, target=reference, matched_from="last_papers")

    arxiv_id = str(reference.get("arxiv_id") or reference.get("target_value") or "").strip()
    if not arxiv_id:
        return {
            "status": "failed",
            "reason": "无法解析 arXiv ID",
            "target": reference,
            "paper": None,
            "arxiv_id": None,
            "title": None,
        }

    # 对显式 arXiv ID，优先尝试在最近论文列表或当前选中论文中补全完整元数据。
    paper = next((paper for paper in last_papers if str(paper.get("arxiv_id") or "").strip() == arxiv_id), None)
    if paper is None and selected_paper is not None and str(selected_paper.get("arxiv_id") or "").strip() == arxiv_id:
        paper = selected_paper
    if paper is not None:
        return build_success(paper, target=reference, matched_from="explicit_arxiv_id")

    return {
        "status": "success",
        "reason": None,
        "target": reference,
        "paper": None,
        "arxiv_id": arxiv_id,
        "title": selected_paper.get("title") if isinstance(selected_paper, Mapping) and str(selected_paper.get("arxiv_id") or "").strip() == arxiv_id else None,
        "matched_from": "explicit_arxiv_id",
    }


__all__ = [
    "_extract_last_papers",
    "_extract_selected_paper",
    "_merge_context_paper_lists",
    "_normalize_context_paper",
    "_parse_target_reference",
    "_resolve_paper_reference",
]
