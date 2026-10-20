"""论文指代：给理解调用列候选清单，再把 LLM 选中的候选落地成具体论文。

"他是怎么做的""第二篇讲了什么"这类指代，正则只能认出固定说法，认不出时还会否决
已经读懂上下文的 LLM。这里改成：代码把上下文里出现过的论文排成带编号的候选清单，
LLM 在追问改写那次调用里顺带判断用户指的是哪个编号，代码只校验编号存在并取回完整记录。
LLM 不可用或输出校验不过时不猜，交给 resolve_paper 追问用户。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .paper_target_resolver import build_paper_context

# 候选太多会拖长提示词，也很少有人指第 20 篇以后的论文。
MAX_REFERENCE_CANDIDATES = 20
REFERENCE_KINDS = ("none", "single", "multiple", "ambiguous", "outside")


def build_reference_candidates(context: Any) -> List[Dict[str, Any]]:
    """按「焦点论文在前、列表论文按原顺序」排出候选，同一篇论文合并来源说明。"""
    paper_context = build_paper_context(context)
    entries: Dict[str, Dict[str, Any]] = {}
    for candidate in paper_context.get("focus_candidates") or []:
        _add_candidate(entries, candidate, str(candidate.get("source_label") or "上下文论文"))
    for source in paper_context.get("list_sources") or []:
        for paper in source.get("papers") or []:
            _add_candidate(entries, paper, f"{source.get('label')}第{paper.get('rank')}篇")
    candidates = list(entries.values())[:MAX_REFERENCE_CANDIDATES]
    for index, candidate in enumerate(candidates, start=1):
        candidate["key"] = f"P{index}"
    return candidates


def _add_candidate(entries: Dict[str, Dict[str, Any]], paper: Mapping[str, Any], label: str) -> None:
    identity = _identity(paper)
    if not identity:
        return
    entry = entries.get(identity)
    if entry is None:
        entries[identity] = {"record": dict(paper), "sources": [label]}
    elif label not in entry["sources"]:
        entry["sources"].append(label)


def _identity(paper: Mapping[str, Any]) -> str:
    arxiv_id = str(paper.get("arxiv_id") or "").strip().lower().split("v")[0]
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    title = " ".join(str(paper.get("title") or "").lower().split())
    return f"title:{title}" if title else ""


def candidates_for_prompt(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """提示词里只放 LLM 判断指代需要的字段；摘要等长文本不放。"""
    rows = []
    for candidate in candidates:
        record = candidate.get("record") if isinstance(candidate.get("record"), Mapping) else {}
        row = {
            "key": candidate.get("key"),
            "title": record.get("title") or None,
            "arxiv_id": record.get("arxiv_id") or None,
            "sources": list(candidate.get("sources") or []),
        }
        rows.append({key: value for key, value in row.items() if value})
    return rows


def ground_paper_reference(raw: Any, candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """校验 LLM 给的指代判断，把候选编号换成完整论文记录。

    返回 {status, kind, keys, papers, error}；status 不是 success 时下游一律追问，不做猜测。
    """
    if not isinstance(raw, Mapping):
        return _invalid("paper_reference is missing or not an object")
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in REFERENCE_KINDS:
        return _invalid(f"unknown paper_reference kind: {kind or '<empty>'}")
    raw_refs = raw.get("refs") if isinstance(raw.get("refs"), list) else []
    keys = list(dict.fromkeys(str(item).strip().upper() for item in raw_refs if str(item).strip()))
    by_key = {str(candidate.get("key")): candidate for candidate in candidates}
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        # 编号不在清单里说明模型编造了论文；宁可追问也不能拿一个不存在的目标去问答。
        return _invalid(f"paper_reference refs not in candidates: {unknown}")
    if kind in {"none", "outside"}:
        keys = []
    elif not keys:
        return _invalid(f"paper_reference kind={kind} without refs")
    elif kind == "single" and len(keys) > 1:
        return _invalid("paper_reference kind=single with multiple refs")
    elif kind == "multiple" and len(keys) == 1:
        kind = "single"
    return {
        "status": "success",
        "kind": kind,
        "keys": keys,
        "papers": [dict(by_key[key]["record"]) for key in keys],
        "error": None,
    }


def _invalid(error: str) -> Dict[str, Any]:
    return {"status": "invalid_output", "kind": None, "keys": [], "papers": [], "error": error}


def unavailable_paper_reference(status: str, error: Optional[str] = None) -> Dict[str, Any]:
    """LLM 没给出指代判断时的占位：resolve_paper 看到它就追问，而不是退回正则。"""
    return {"status": status, "kind": None, "keys": [], "papers": [], "error": error}


__all__ = [
    "MAX_REFERENCE_CANDIDATES",
    "build_reference_candidates",
    "candidates_for_prompt",
    "ground_paper_reference",
    "unavailable_paper_reference",
]
