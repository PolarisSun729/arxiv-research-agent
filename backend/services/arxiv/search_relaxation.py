"""搜索结果偏少时的放宽建议。

只用确定的规则生成候选（放宽时间、展开缩写、去掉某个词、任一词命中），不调用 LLM：
普通搜索的用户要的是可预期的字面检索，按语义改写交给 Agent 搜索。
能便宜计数的后端（本地 OAI 镜像）会给每个候选附上命中数，并丢掉不能带来更多结果的候选。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from services.arxiv.contracts import ArxivSearchError

# 只建议比当前更宽的窗口；没有日期限制时不给时间建议。
RELAXED_TIME_WINDOWS = ((90, "近 90 天"), (365, "近一年"))
# 「去掉某个词」每个词都要计数一次，词太多时不再逐个尝试。
MAX_TERMS_FOR_DROP = 6
# 有计数时只保留命中最多的几个「去掉某个词」，避免提示里候选过多。
MAX_DROP_TERM_SUGGESTIONS = 2

# 常见缩写 → 全称。FTS 用 unicode61 分词、不做词干还原，所以有复数形式的全称要把复数也列出来；
# 列了复数全称的缩写，展开时也会带上缩写的复数（llm → llms）。
ABBREVIATION_EXPANSIONS: Dict[str, List[str]] = {
    "rag": ["retrieval augmented generation"],
    "llm": ["large language model", "large language models"],
    "mllm": ["multimodal large language model", "multimodal large language models"],
    "vlm": ["vision language model", "vision language models"],
    "kg": ["knowledge graph", "knowledge graphs"],
    "gnn": ["graph neural network", "graph neural networks"],
    "rl": ["reinforcement learning"],
    "rlhf": ["reinforcement learning from human feedback"],
    "nlp": ["natural language processing"],
    "cot": ["chain of thought"],
    "moe": ["mixture of experts"],
    "qa": ["question answering"],
    "ner": ["named entity recognition"],
}

_PLAIN_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_BOOLEAN_OPERATORS = {"AND", "OR", "ANDNOT"}

# 计数函数：(search_query, submitted_days_ago) -> 命中总数。
CountFn = Callable[[Optional[str], Optional[int]], int]


def split_plain_terms(search_query: Optional[str]) -> List[str]:
    """查询只由普通词组成时返回词列表，否则返回空列表。

    带字段前缀、引号、括号或布尔操作符的查询是用户刻意写的语法，改写它的含义不可预期，不给写法建议。
    """
    terms = str(search_query or "").split()
    if not terms:
        return []
    if any(term in _BOOLEAN_OPERATORS or not _PLAIN_TERM.fullmatch(term) for term in terms):
        return []
    return terms


def _abbreviation_clause(term: str) -> Optional[str]:
    lowered = term.lower()
    abbreviation = lowered
    if abbreviation not in ABBREVIATION_EXPANSIONS and lowered.endswith("s"):
        abbreviation = lowered[:-1]
    full_forms = ABBREVIATION_EXPANSIONS.get(abbreviation)
    if not full_forms:
        return None
    variants = [abbreviation]
    if len(full_forms) > 1:
        variants.append(f"{abbreviation}s")
    variants.extend(f'"{form}"' for form in full_forms)
    return "(" + " OR ".join(variants) + ")"


def build_relaxation_candidates(
    *,
    search_query: Optional[str],
    submitted_days_ago: Optional[int],
    allow_query_rewrite: bool,
) -> List[Dict[str, Any]]:
    """按「放宽时间 → 展开缩写 → 去掉某个词 → 任一词命中」的顺序生成候选。

    每个候选都带完整的 search_query 和 submitted_days_ago，前端直接替换表单即可重新搜索。
    allow_query_rewrite 为 False（例如用了标题、作者等结构化字段）时只给时间建议。
    """
    candidates: List[Dict[str, Any]] = []

    if submitted_days_ago is not None:
        for days, label in RELAXED_TIME_WINDOWS:
            if days > submitted_days_ago:
                candidates.append(
                    {
                        "kind": "widen_time",
                        "label": label,
                        "search_query": search_query,
                        "submitted_days_ago": days,
                    }
                )

    terms = split_plain_terms(search_query) if allow_query_rewrite else []
    if not terms:
        return candidates

    def rewrite(query: str, kind: str, label: str) -> None:
        candidates.append(
            {"kind": kind, "label": label, "search_query": query, "submitted_days_ago": submitted_days_ago}
        )

    for index, term in enumerate(terms):
        clause = _abbreviation_clause(term)
        if clause:
            expanded = [*terms[:index], clause, *terms[index + 1 :]]
            full_form = ABBREVIATION_EXPANSIONS.get(term.lower()) or ABBREVIATION_EXPANSIONS[term.lower()[:-1]]
            # 括号子句旁边的普通词必须用显式 AND 连接，本地解析器才会按顶层布尔结构拆分。
            rewrite(" AND ".join(expanded), "expand_abbreviation", f"{term} → {full_form[0]}")

    if 2 <= len(terms) <= MAX_TERMS_FOR_DROP:
        for index, term in enumerate(terms):
            rewrite(" ".join([*terms[:index], *terms[index + 1 :]]), "drop_term", f"去掉「{term}」")

    if len(terms) >= 2:
        rewrite(" OR ".join(terms), "any_term", "任一词命中")

    return candidates


def suggest_search_relaxations(
    *,
    search_query: Optional[str],
    submitted_days_ago: Optional[int],
    allow_query_rewrite: bool,
    count: Optional[CountFn],
) -> Dict[str, Any]:
    """生成放宽建议；有计数函数时附上命中数，并只保留比当前命中更多的候选。

    没有计数函数（远程 arXiv API 有限流，逐个计数太慢）时原样返回候选，total 为 None。
    单个候选计数失败（例如本地镜像不支持改写后的语法）只跳过该候选，不影响其他建议。
    """
    candidates = build_relaxation_candidates(
        search_query=search_query,
        submitted_days_ago=submitted_days_ago,
        allow_query_rewrite=allow_query_rewrite,
    )
    if count is None:
        return {"baseline_total": None, "suggestions": [{**item, "total": None} for item in candidates]}

    baseline_total = count(search_query, submitted_days_ago)
    counted: List[Dict[str, Any]] = []
    for item in candidates:
        try:
            total = count(item["search_query"], item["submitted_days_ago"])
        except ArxivSearchError:
            continue
        if total > baseline_total:
            counted.append({**item, "total": total})

    drop_terms = sorted((item for item in counted if item["kind"] == "drop_term"), key=lambda item: -item["total"])
    kept_drop_terms = {id(item) for item in drop_terms[:MAX_DROP_TERM_SUGGESTIONS]}
    suggestions = [item for item in counted if item["kind"] != "drop_term" or id(item) in kept_drop_terms]
    return {"baseline_total": baseline_total, "suggestions": suggestions}
