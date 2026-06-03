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
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _normalize_context_paper(raw: Any) -> Dict[str, Any]:
    paper = raw if isinstance(raw, Mapping) else {}
    arxiv_id = str(paper.get("arxiv_id") or paper.get("arxivId") or paper.get("id") or "").strip()
    if arxiv_id.startswith("http"):
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
    if not isinstance(context, Mapping):
        return []
    return _merge_context_paper_lists(
        context.get("last_papers") or [],
        context.get("papers") or [],
        context.get("search_results") or [],
        context.get("recent_papers") or [],
    )


def _parse_target_reference(message: str) -> Optional[Dict[str, Any]]:
    text = _normalize_text(message)
    if not text:
        return None

    arxiv_match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", text, flags=re.IGNORECASE)
    if arxiv_match:
        return {
            "target_type": "arxiv_id",
            "target_value": arxiv_match.group(1),
            "arxiv_id": arxiv_match.group(1),
        }

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

    ordinal_match = re.search(r"(?:第\s*)?([一二三四五六七八九十两]{1,3}|[1-9]|1[0-9]|20)\s*(?:篇|个)?(?:论文|paper)?", text)
    if ordinal_match:
        raw_value = ordinal_match.group(1)
        ordinal = _PREFERENCE_ORDINAL_MAP.get(raw_value)
        if ordinal is None:
            ordinal = _safe_int(raw_value, default=0)
        if 1 <= ordinal <= 20:
            return {
                "target_type": "ordinal",
                "target_value": ordinal,
                "ordinal": ordinal,
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
    reference = _parse_target_reference(message)
    last_papers = _extract_last_papers(context)
    selected_paper = _extract_selected_paper(context)

    def build_success(paper_payload: Optional[Mapping[str, Any]], *, target: Optional[Dict[str, Any]], matched_from: str) -> Dict[str, Any]:
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
        if ordinal > len(last_papers):
            return {
                "status": "failed",
                "reason": f"上一轮搜索结果只有 {len(last_papers)} 篇，无法选择第 {ordinal} 篇",
                "target": reference,
                "paper": None,
                "arxiv_id": None,
                "title": None,
            }
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
