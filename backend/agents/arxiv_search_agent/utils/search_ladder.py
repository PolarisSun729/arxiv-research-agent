"""阶梯检索：把 ArxivSearchSpec 编译成由严到宽的一组 arXiv 查询。

为什么需要阶梯：解析阶段给出的主题经常是多词短语（例如 "retrieval optimization"），
直接按短语精确匹配很容易 0 命中；过去靠 rewrite→search 的重规划循环来放宽，
但改写工具并不真正放宽条件，最后只会把内部错误码抛给用户。

阶梯在一次 search_arxiv 工具调用内完成，规则固定、可解释：
- L0 strict：概念之间 AND，概念内部主术语与同义写法 OR，多词术语按短语匹配；
- L1 drop_optional_terms：去掉补充词，只保留核心概念；
- L2 loosen_phrases：多词短语改为“所有词都出现即可”；
- L3 drop_inferred_categories：去掉系统推断/默认的分类范围。
用户明确给出的时间窗、分类和主题概念不会被自动放宽。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..schemas import ArxivSearchSpec, SearchConcept

LEVEL_LABELS: Dict[str, str] = {
    "strict": "严格匹配",
    "drop_optional_terms": "去掉补充词",
    "loosen_phrases": "短语改为分词匹配",
    "drop_inferred_categories": "放开推断的分类范围",
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class LadderLevel:
    """阶梯中的一级查询；query 不含时间条件，时间窗由后端工具统一追加。"""

    level: str
    label: str
    query: str
    categories: List[str] = field(default_factory=list)


def text_tokens(value: Any) -> List[str]:
    """抽取英文/数字 token；本地 FTS 和 arXiv API 都只按这类 token 检索。"""
    return [token.lower() for token in _TOKEN_PATTERN.findall(str(value or ""))]


def _text_clause(text: str, *, phrase: bool, field_prefix: str = "all") -> str:
    tokens = text_tokens(text)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return f"{field_prefix}:{tokens[0]}"
    if phrase:
        return f'{field_prefix}:"{" ".join(tokens)}"'
    return "(" + " AND ".join(f"{field_prefix}:{token}" for token in tokens) + ")"


def _concept_clause(concept: SearchConcept, *, phrase: bool) -> str:
    """主术语和同义写法之间是 OR：命中任意一种写法都算命中该概念。"""
    variants: List[str] = []
    for text in [concept.term, *concept.aliases]:
        clause = _text_clause(text, phrase=phrase)
        if clause and clause not in variants:
            variants.append(clause)
    if not variants:
        return ""
    if len(variants) == 1:
        return variants[0]
    return "(" + " OR ".join(variants) + ")"


def _has_multiword_variant(concepts: Sequence[SearchConcept]) -> bool:
    return any(len(text_tokens(text)) > 1 for concept in concepts for text in [concept.term, *concept.aliases])


def effective_concepts(spec: ArxivSearchSpec) -> List[SearchConcept]:
    """旧调用方只给 query 时，把整个 query 当成一个用户概念，保证阶梯同样可用。"""
    if spec.concepts:
        return list(spec.concepts)
    if spec.query and text_tokens(spec.query):
        return [SearchConcept(term=spec.query, source="user")]
    return []


def _build_query(
    concepts: Sequence[SearchConcept],
    optional_terms: Sequence[str],
    *,
    phrase: bool,
    title_query: Optional[str],
    abstract_query: Optional[str],
) -> str:
    clauses = [_concept_clause(concept, phrase=phrase) for concept in concepts]
    clauses.extend(_text_clause(term, phrase=phrase) for term in optional_terms)
    # 标题/摘要字段只有用户明确要求时才会出现在 spec 中，属于用户约束，不参与放宽。
    if title_query:
        clauses.append(_text_clause(title_query, phrase=True, field_prefix="ti"))
    if abstract_query:
        clauses.append(_text_clause(abstract_query, phrase=True, field_prefix="abs"))
    return " AND ".join(clause for clause in clauses if clause)


def build_search_ladder(spec: ArxivSearchSpec) -> List[LadderLevel]:
    """生成去重后的阶梯；查询与上一级完全相同的级别会被跳过。"""
    concepts = effective_concepts(spec)
    optional_terms = list(spec.optional_terms or [])
    categories = list(spec.categories or [])
    sources = dict(spec.constraint_sources or {})
    categories_relaxable = bool(categories) and sources.get("categories", "default") != "user"

    plans = [
        ("strict", optional_terms, True, categories),
        ("drop_optional_terms", [], True, categories),
        ("loosen_phrases", [], False, categories),
        ("drop_inferred_categories", [], False, [] if categories_relaxable else categories),
    ]
    levels: List[LadderLevel] = []
    for level, level_optional_terms, phrase, level_categories in plans:
        if level == "loosen_phrases" and not _has_multiword_variant(concepts):
            continue
        query = _build_query(
            concepts,
            level_optional_terms,
            phrase=phrase,
            title_query=spec.title_query,
            abstract_query=spec.abstract_query,
        )
        if not query:
            # 没有任何可检索的英文主题词时不能只按分类+时间返回“随便几篇”。
            continue
        candidate = LadderLevel(level=level, label=LEVEL_LABELS[level], query=query, categories=list(level_categories))
        if levels and levels[-1].query == candidate.query and levels[-1].categories == candidate.categories:
            continue
        levels.append(candidate)
    return levels


def compose_search_query(level: LadderLevel) -> str:
    """拼出不含时间条件的完整 search_query；分类之间固定为 OR。"""
    if not level.categories:
        return level.query
    category_clause = " OR ".join(f"cat:{category}" for category in level.categories)
    if len(level.categories) > 1:
        category_clause = f"({category_clause})"
    return f"({level.query}) AND {category_clause}"


def paper_key(paper: Mapping[str, Any]) -> str:
    for key in ("arxiv_id", "id", "entry_id", "abs_url", "title"):
        value = str(paper.get(key) or "").strip().lower()
        if value:
            return value
    return ""


__all__ = [
    "LEVEL_LABELS",
    "LadderLevel",
    "build_search_ladder",
    "compose_search_query",
    "effective_concepts",
    "paper_key",
    "text_tokens",
]
