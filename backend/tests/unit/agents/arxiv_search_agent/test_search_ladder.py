from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules

load_agent_test_modules()

schemas = importlib.import_module("backend.agents.arxiv_search_agent.schemas")
ladder = importlib.import_module("backend.agents.arxiv_search_agent.utils.search_ladder")
builder = importlib.import_module("backend.agents.arxiv_search_agent.utils.search_spec_builder")
search_adapters = importlib.import_module("backend.agents.arxiv_search_agent.tool_adapters.search")

ArxivSearchSpec = schemas.ArxivSearchSpec
PROFILE = {"positive_topics": ["RAG retrieval optimization", "knowledge graph"]}
MESSAGE = "帮我找最近 7 天关于 RAG 的 5 篇论文"


def _ground(raw: dict, message: str = MESSAGE, profile: dict | None = PROFILE):
    spec = builder._normalize_and_validate_spec(raw)
    return builder._ground_search_spec(spec, message, raw_payload=raw, research_profile=profile)


def test_grounding_strips_profile_words_the_user_never_said() -> None:
    spec, warnings = _ground(
        {
            "intent": "arxiv_search",
            "concepts": [{"term": "RAG retrieval optimization", "aliases": ["retrieval-augmented generation"], "evidence": "RAG"}],
            "optional_terms": ["retrieval optimization"],
            "submitted_days_ago": 7,
            "max_results": 5,
            "sort_by": "submittedDate",
        }
    )

    assert [concept.term for concept in spec.concepts] == ["RAG"]
    assert spec.concepts[0].aliases == ["retrieval-augmented generation"]
    assert spec.optional_terms == []
    assert spec.query == "RAG"
    assert warnings
    # 原话里的时间和数量是用户约束；没说“最新”就按相关性排序。
    assert spec.constraint_sources["submitted_days_ago"] == "user"
    assert spec.constraint_sources["max_results"] == "user"
    assert spec.constraint_sources["categories"] == "default"
    assert spec.sort_by == "relevance"


def test_grounding_handles_legacy_query_payload() -> None:
    spec, _ = _ground({"intent": "arxiv_search", "query": "RAG retrieval optimization", "submitted_days_ago": 7})

    assert [concept.term for concept in spec.concepts] == ["RAG"]


def test_grounding_trusts_translation_of_chinese_evidence() -> None:
    spec, _ = _ground(
        {
            "intent": "arxiv_search",
            "concepts": [{"term": "retrieval optimization", "evidence": "检索优化"}],
        },
        message="找一些检索优化方面的论文",
    )

    assert [concept.term for concept in spec.concepts] == ["retrieval optimization"]


def test_profile_fills_topic_only_when_user_gave_none() -> None:
    spec, warnings = _ground({"intent": "arxiv_search", "concepts": []}, message="帮我找最近 7 天的论文")

    assert [(concept.term, concept.source) for concept in spec.concepts] == [("RAG retrieval optimization", "profile")]
    assert any("研究画像" in item for item in warnings)


def test_profile_fill_never_uses_excluded_topics() -> None:
    # 排除主题排在列表前面也不能被拿来补全检索词；只有排除主题时就不补全。
    only_negative = {"negative_topics": ["diffusion models"]}
    spec, warnings = _ground({"intent": "arxiv_search", "concepts": []}, message="帮我找最近 7 天的论文", profile=only_negative)

    assert spec.concepts == []
    assert not any("研究画像" in item for item in warnings)

    with_recent = {"negative_topics": ["diffusion models"], "recent_topics": ["graph retrieval"]}
    spec, _ = _ground({"intent": "arxiv_search", "concepts": []}, message="帮我找最近 7 天的论文", profile=with_recent)

    assert [(concept.term, concept.source) for concept in spec.concepts] == [("graph retrieval", "profile")]


def test_sentence_without_topic_is_never_turned_into_keywords() -> None:
    # LLM 判定本句没有主题时，不能再把"我感觉目前""扩大一下"之类的话切成检索词。
    message = "我感觉目前论文数量还是不够，请你扩大一下搜索范围，然后重新搜索一下"
    spec, _ = _ground({"intent": "arxiv_search", "concepts": []}, message=message, profile=None)

    assert spec.concepts == []
    assert not spec.query


def test_user_named_categories_are_locked() -> None:
    spec, _ = _ground({"intent": "arxiv_search", "concepts": [{"term": "RAG", "evidence": "RAG"}]}, message="找 cs.IR 下的 RAG 论文")

    assert spec.categories == ["cs.IR"]
    assert spec.constraint_sources["categories"] == "user"
    levels = ladder.build_search_ladder(spec)
    assert all(level.categories == ["cs.IR"] for level in levels)


def test_ladder_relaxes_from_strict_to_broad() -> None:
    spec = ArxivSearchSpec(
        intent="arxiv_search",
        concepts=[{"term": "LLM agents", "aliases": ["language model agents"]}],
        optional_terms=["memory"],
        categories=["cs.CL", "cs.AI"],
        submitted_days_ago=7,
        constraint_sources={"categories": "default", "submitted_days_ago": "user"},
    )

    levels = ladder.build_search_ladder(spec)

    assert [level.level for level in levels] == ["strict", "drop_optional_terms", "loosen_phrases", "drop_inferred_categories"]
    assert ladder.compose_search_query(levels[0]) == '((all:"llm agents" OR all:"language model agents") AND all:memory) AND (cat:cs.CL OR cat:cs.AI)'
    assert "all:memory" not in levels[1].query
    assert "(all:llm AND all:agents)" in levels[2].query
    assert levels[3].categories == []


def test_search_adapter_merges_levels_until_enough_papers() -> None:
    calls = []

    def fake_invoke(tool_name: str, **kwargs):
        calls.append(kwargs)
        papers = [] if len(calls) == 1 else [{"arxiv_id": f"2401.0000{len(calls)}", "title": "RAG"}]
        return {"ok": True, "tool_name": tool_name, "data": {"papers": papers, "total_results": len(papers)}, "trace": {}}

    adapter = search_adapters.SearchArxivAdapter(fake_invoke)
    spec = ArxivSearchSpec(
        intent="arxiv_search",
        concepts=[{"term": "RAG agents"}],
        categories=["cs.CL"],
        submitted_days_ago=7,
        max_results=2,
        constraint_sources={"categories": "default", "submitted_days_ago": "user"},
    )

    # 全量跑时其他测试文件会重载 schemas 模块，这里拿到的 ArxivSearchSpec 可能已不是适配器引用的那个类；传 dict 让适配器自己校验。
    output = adapter._run(search_adapters.SearchArxivInput(search_spec=spec.model_dump()))

    assert len(calls) == 3
    # 时间窗是用户约束，每一级都原样保留。
    assert all(call["submitted_days_ago"] == 7 for call in calls)
    assert [paper["match_level"] for paper in output.papers] == ["loosen_phrases", "drop_inferred_categories"]
    assert output.relaxation["exhausted"] is False
    assert output.relaxation["locked_constraints"] == ["submitted_days_ago"]


def test_requested_count_above_limit_is_capped_and_flagged() -> None:
    limit = schemas.get_agent_max_search_results()
    spec, _ = _ground(
        {"intent": "arxiv_search", "concepts": [{"term": "RAG", "evidence": "RAG"}], "max_results": limit + 50},
        message=f"帮我找关于 RAG 的 {limit + 50} 篇论文",
    )

    assert spec.max_results == limit
    assert spec.constraint_sources["max_results"] == "user_capped"


def test_requested_count_within_limit_is_kept() -> None:
    spec, _ = _ground(
        {"intent": "arxiv_search", "concepts": [{"term": "RAG", "evidence": "RAG"}], "max_results": 100},
        message="帮我找最近2年内关于 RAG 的100篇论文",
    )

    assert spec.max_results == min(100, schemas.get_agent_max_search_results())
    assert spec.constraint_sources["max_results"] == "user"


def test_capped_count_is_explained_in_answer() -> None:
    notes = search_adapters._relaxation_notes({"requested": 100, "count_capped": True, "attempts": []}, 100)

    assert any("单次检索上限" in note and "100 篇" in note for note in notes)


def test_empty_answer_lists_attempts_and_suggests_wider_time_window() -> None:
    relaxation = {
        "attempts": [
            {"level": "strict", "label": "严格匹配", "query": "all:rag", "categories": ["cs.CL"], "submitted_days_ago": 7, "ok": True, "total_results": 0, "new_papers": 0},
            {"level": "drop_inferred_categories", "label": "放开推断的分类范围", "query": "all:rag", "categories": [], "submitted_days_ago": 7, "ok": True, "total_results": 0, "new_papers": 0},
        ],
        "requested": 5,
        "returned": 0,
        "exhausted": True,
        "locked_constraints": ["submitted_days_ago"],
    }

    answer = search_adapters._empty_result_answer(relaxation)

    assert "1. 严格匹配" in answer and "2. 放开推断的分类范围" in answer
    assert "我没有自动放宽" in answer
    assert "最近 30 天" in answer


def test_personalize_reorders_only_within_same_match_level() -> None:
    adapter = search_adapters.PersonalizePaperResultsAdapter()
    papers = [
        {"arxiv_id": "a", "title": "RAG for code", "match_level": "strict"},
        {"arxiv_id": "b", "title": "Knowledge graph RAG", "match_level": "strict"},
        {"arxiv_id": "c", "title": "Knowledge graph agents", "match_level": "drop_inferred_categories"},
    ]

    output = adapter._run(
        search_adapters.PersonalizePaperResultsInput(arxiv_results={"papers": papers}, research_profile=PROFILE)
    )

    assert [paper["arxiv_id"] for paper in output.ranked_papers] == ["b", "a", "c"]
    assert output.ranked_papers[0]["reason"].startswith("matched_profile_topics")


def test_grounding_demotes_extra_concepts_to_optional_terms() -> None:
    message = "调研最近2年 Agent 的工具怎么编排、怎样表达和生成"
    spec, warnings = _ground(
        {
            "intent": "arxiv_search",
            "concepts": [
                {"term": "agent", "evidence": "Agent"},
                {"term": "tool use", "aliases": ["tool calling", "function calling"], "evidence": "工具"},
                {"term": "orchestration", "evidence": "编排"},
                {"term": "representation", "evidence": "表达"},
                {"term": "generation", "evidence": "生成"},
            ],
        },
        message=message,
        profile=None,
    )

    assert [concept.term for concept in spec.concepts] == ["agent", "tool use", "orchestration"]
    assert spec.optional_terms == ["representation", "generation"]
    assert any("降为补充词" in warning for warning in warnings)


def test_answer_names_dropped_optional_terms() -> None:
    relaxation = {
        "attempts": [
            {"level": "strict", "ok": True, "new_papers": 0},
            {"level": "drop_optional_terms", "label": "去掉补充词", "ok": True, "new_papers": 12},
        ],
        "requested": 10,
        "optional_terms": ["orchestration", "representation"],
    }

    notes = search_adapters._relaxation_notes(relaxation, 10)

    assert any("orchestration、representation" in note for note in notes)
