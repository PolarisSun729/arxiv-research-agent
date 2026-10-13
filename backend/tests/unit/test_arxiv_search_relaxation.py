from typing import Dict, Optional, Tuple

import pytest

from services.arxiv.contracts import ArxivSearchError
from services.arxiv.search_relaxation import (
    MAX_DROP_TERM_SUGGESTIONS,
    build_relaxation_candidates,
    split_plain_terms,
    suggest_search_relaxations,
)


def _queries_by_kind(candidates, kind):
    return [item["search_query"] for item in candidates if item["kind"] == kind]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("knowledge graph rag", ["knowledge", "graph", "rag"]),
        ("gpt-4 v1.5", ["gpt-4", "v1.5"]),
        ("", []),
        (None, []),
        # 用户刻意写的语法不改写：字段前缀、引号、括号、布尔操作符。
        ("ti:rag", []),
        ('"knowledge graph"', []),
        ("(rag)", []),
        ("rag OR kg", []),
    ],
)
def test_split_plain_terms_only_accepts_bare_words(query: Optional[str], expected) -> None:
    assert split_plain_terms(query) == expected


def test_candidates_cover_time_abbreviation_drop_and_any_term_in_order() -> None:
    candidates = build_relaxation_candidates(
        search_query="knowledge graph rag",
        submitted_days_ago=30,
        allow_query_rewrite=True,
    )

    assert [item["kind"] for item in candidates] == [
        "widen_time",
        "widen_time",
        "expand_abbreviation",
        "drop_term",
        "drop_term",
        "drop_term",
        "any_term",
    ]
    # 放宽时间保留原查询，放宽写法保留原时间窗口，前端可以直接整体替换表单。
    assert [(item["search_query"], item["submitted_days_ago"]) for item in candidates[:2]] == [
        ("knowledge graph rag", 90),
        ("knowledge graph rag", 365),
    ]
    assert all(item["submitted_days_ago"] == 30 for item in candidates[2:])
    # 括号子句旁的普通词必须用显式 AND 连接，否则本地解析器不会按布尔结构拆分。
    assert _queries_by_kind(candidates, "expand_abbreviation") == [
        'knowledge AND graph AND (rag OR "retrieval augmented generation")'
    ]
    assert _queries_by_kind(candidates, "drop_term") == ["graph rag", "knowledge rag", "knowledge graph"]
    assert _queries_by_kind(candidates, "any_term") == ["knowledge OR graph OR rag"]


def test_abbreviation_with_plural_full_form_also_matches_plural_abbreviation() -> None:
    # FTS 不做词干还原，单复数都要列出来；复数缩写也能识别并展开成同一组形式。
    for term in ("llm", "LLMs"):
        candidates = build_relaxation_candidates(
            search_query=term, submitted_days_ago=None, allow_query_rewrite=True
        )
        assert _queries_by_kind(candidates, "expand_abbreviation") == [
            '(llm OR llms OR "large language model" OR "large language models")'
        ]


def test_time_candidates_only_widen_existing_window() -> None:
    def time_days(days):
        candidates = build_relaxation_candidates(search_query="rag", submitted_days_ago=days, allow_query_rewrite=True)
        return [item["submitted_days_ago"] for item in candidates if item["kind"] == "widen_time"]

    assert time_days(30) == [90, 365]
    assert time_days(180) == [365]
    assert time_days(365) == []
    # 没有日期限制时已经是最宽的时间范围。
    assert time_days(None) == []


def test_structured_or_single_word_queries_skip_inapplicable_rewrites() -> None:
    structured = build_relaxation_candidates(
        search_query="knowledge graph", submitted_days_ago=30, allow_query_rewrite=False
    )
    assert {item["kind"] for item in structured} == {"widen_time"}

    single = build_relaxation_candidates(search_query="transformers", submitted_days_ago=None, allow_query_rewrite=True)
    assert single == []


def test_drop_term_is_skipped_when_query_has_too_many_terms() -> None:
    candidates = build_relaxation_candidates(
        search_query="a b c d e f g", submitted_days_ago=None, allow_query_rewrite=True
    )
    assert _queries_by_kind(candidates, "drop_term") == []
    assert _queries_by_kind(candidates, "any_term") == ["a OR b OR c OR d OR e OR f OR g"]


def _fake_count(totals: Dict[Tuple[Optional[str], Optional[int]], int], failing: Tuple[str, ...] = ()):
    def count(search_query: Optional[str], submitted_days_ago: Optional[int]) -> int:
        if search_query in failing:
            raise ArxivSearchError("unsupported", query=search_query or "", reason="unsupported_syntax")
        return totals.get((search_query, submitted_days_ago), 0)

    return count


def test_suggestions_keep_only_candidates_with_more_results_and_top_drop_terms() -> None:
    totals = {
        ("knowledge graph rag", 30): 25,
        ("knowledge graph rag", 90): 109,
        # 放宽到一年也没有更多结果时，这个候选没有意义。
        ("knowledge graph rag", 365): 25,
        ('knowledge AND graph AND (rag OR "retrieval augmented generation")', 30): 33,
        ("graph rag", 30): 40,
        ("knowledge rag", 30): 96,
        ("knowledge graph", 30): 293,
        ("knowledge OR graph OR rag", 30): 3315,
    }

    result = suggest_search_relaxations(
        search_query="knowledge graph rag",
        submitted_days_ago=30,
        allow_query_rewrite=True,
        count=_fake_count(totals),
    )

    assert result["baseline_total"] == 25
    assert [(item["label"], item["total"]) for item in result["suggestions"]] == [
        ("近 90 天", 109),
        ("rag → retrieval augmented generation", 33),
        # 只保留命中最多的两个「去掉某个词」，保持原有顺序。
        ("去掉「graph」", 96),
        ("去掉「rag」", 293),
        ("任一词命中", 3315),
    ]
    assert sum(item["kind"] == "drop_term" for item in result["suggestions"]) == MAX_DROP_TERM_SUGGESTIONS


def test_suggestions_skip_candidates_whose_count_fails() -> None:
    expanded = '(rag OR "retrieval augmented generation")'
    result = suggest_search_relaxations(
        search_query="rag",
        submitted_days_ago=30,
        allow_query_rewrite=True,
        count=_fake_count({("rag", 30): 1, ("rag", 90): 5, ("rag", 365): 9}, failing=(expanded,)),
    )

    assert [item["kind"] for item in result["suggestions"]] == ["widen_time", "widen_time"]


def test_suggestions_without_counter_return_all_candidates_without_totals() -> None:
    result = suggest_search_relaxations(
        search_query="knowledge graph rag",
        submitted_days_ago=30,
        allow_query_rewrite=True,
        count=None,
    )

    assert result["baseline_total"] is None
    assert len(result["suggestions"]) == 7
    assert all(item["total"] is None for item in result["suggestions"])
