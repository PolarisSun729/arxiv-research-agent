from __future__ import annotations

import sys

from tests.helpers.agent_runtime import load_agent_test_modules


load_agent_test_modules()
search_spec_builder = sys.modules["backend.agents.arxiv_search_agent.utils.search_spec_builder"]


def test_recent_topic_search_keeps_relevance_sorting() -> None:
    spec = search_spec_builder._build_spec_from_rules("帮我找最近 7 天关于 RAG 的 5 篇论文")

    assert spec is not None
    assert spec.query == "RAG"
    assert spec.submitted_days_ago == 7
    assert spec.max_results == 5
    # “最近 7 天”承担时间过滤职责；排序仍按相关性，避免偶然提到关键词的最新论文排在主题论文前面。
    assert spec.sort_by == "relevance"
    assert spec.sort_order == "descending"
